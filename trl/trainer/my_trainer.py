import unsloth
import os
# os.environ['UNSLOTH_VLLM_STANDBY'] = '1'
import torch
from accelerate import logging
from accelerate.utils import gather
from datasets import Dataset, load_dataset
from torch import nn
from trl import GRPOConfig
from trl.rewards.accuracy_rewards import accuracy_reward
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational
)
from trl.extras.profiling import profiling_context, profiling_decorator
from unsloth import FastLanguageModel, is_bfloat16_supported


logger = logging.get_logger(__name__)


class MyTrainer(GRPOTrainer):
    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        print('reward 계산중')
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func


def shuffle_and_select_data(
    dataset: Dataset,
    num_samples: int | None = None
) -> Dataset:
    dataset = dataset.shuffle()

    if num_samples is not None:
        assert num_samples <= len(dataset)
        dataset = dataset.select(range(num_samples))

    return dataset


def main():
    model_name = 'Qwen/Qwen3-4B-Instruct-2507'
    max_prompt_length = 1024
    max_completion_length = 4096
    max_length = max_prompt_length + max_completion_length
    lora_rank = 16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_length,
        load_in_4bit = False,
        full_finetuning = False,
        fast_inference = True,
        gpu_memory_utilization = 0.8,
        max_lora_rank = lora_rank
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha = lora_rank,
        use_gradient_checkpointing = 'unsloth'
    )

    MATH_PROMPT = r""" Please reason step by step, and put your final answer within \boxed{}."""

    dataset = load_dataset('openai/gsm8k', 'main')['train']
    dataset = dataset.map(lambda x: {
            'prompt': [
                {'role': 'user', 'content': x['question'] + MATH_PROMPT}
            ], 
            'solution': x['answer'].split('####')[1]
        }
    )
    dataset = shuffle_and_select_data(dataset, 1600)

    config = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        use_vllm=True,
        beta=0.0,
        num_iterations=1,
        epsilon_high=0.28,
        scale_rewards='group',
        loss_type='dapo',
        mask_truncated_completions=True,
        top_entropy_quantile=1.0,
        log_completions='rich',
        bf16=True,
        output_dir='output',
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        torch_empty_cache_steps=1,
        learning_rate=1e-6,
        max_steps=50,
        lr_scheduler_type='linear',
        warmup_ratio=0.0,
        logging_steps=0.1,
        report_to='wandb',
        project='my_project',
        gradient_checkpointing=False,
        eval_strategy='no',
        per_device_eval_batch_size=32,
        bf16_full_eval=True,
        eval_steps=0.1,
        eval_on_start=False
    )
    trainer = MyTrainer(
        model=model,
        reward_funcs=accuracy_reward,
        args=config,
        train_dataset=dataset,
        # eval_dataset=,
        processing_class = tokenizer
    )
    trainer.train()


if __name__ == '__main__':
    main()