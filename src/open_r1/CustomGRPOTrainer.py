# Copyright 2025 The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file defines a customized GRPOTrainer that performs regeneration of completions
# for prompts whose reward standard deviation is below a minimum threshold.
#


import os
import random
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Sampler

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
from datasets import Dataset, IterableDataset
from packaging import version

# Import data utilities from trl.data_utils
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template

# Import the original GRPOTrainer from trl (adjust the import path if needed)
from trl.trainer.grpo_trainer import GRPOTrainer

if is_wandb_available():
    import wandb

from vllm import LLM, SamplingParams
from unittest.mock import patch
import time

class CustomGRPOTrainer(GRPOTrainer):
    """
    Customized GRPOTrainer that regenerates completions for prompts if the reward standard
    deviation is below a minimum threshold. Regeneration is performed on a per-prompt group basis,
    using a random temperature chosen from a provided range. The number of regeneration attempts per
    prompt is logged via Wandb and included in the logging table.
    """
    def __init__(
        self,
        *args,
        regen_temp_range: Optional[list[float]] = None,
        regen_min_std: float = 0.1,
        regen_max_attempts: int = 12,
        **kwargs,
    ):
        """
        Initialize the CustomGRPOTrainer.
        
        Args:
            regen_temp_range (list[float], optional): Temperature range for regeneration. Default: [1.1, 1.2, 1.3].
            regen_min_std (float): Minimum reward standard deviation required; if below, regeneration is triggered.
            regen_max_attempts (int): Maximum number of regeneration attempts per prompt.
            disable_prefix_caching_test (bool): disables prefix caching for testing purpose.
        """
        super().__init__(*args, **kwargs)
        self.regen_temp_range = regen_temp_range if regen_temp_range is not None else [0.7, 0.8, 0.9, 1.0]
        self.regen_min_std = regen_min_std
        self.regen_max_attempts = regen_max_attempts
        self.generation_count = 0
        self.regeneration_count = 0

        # Print the regen_temp_range, min_std and max_attempts for debugging
        print(f"Regeneration temperature range: {self.regen_temp_range}")
        print(f"Minimum reward standard deviation for regeneration: {self.regen_min_std}")
        print(f"Maximum regeneration attempts per prompt: {self.regen_max_attempts}")

        # Ensure sampling_params is initialized on all processes.
        if self.args.use_vllm and not hasattr(self, "sampling_params"):
            from vllm import SamplingParams
            self.sampling_params = SamplingParams(
                temperature=self.args.temperature, max_tokens=self.max_completion_length
            )

    def _generate_completions(self, prompts_text, prompt_ids, prompt_mask, temperature):
        """
        Generate completions for a list of prompts at the given temperature.
        
        Args:
            prompts_text (list[str]): List of prompt texts.
            prompt_ids (torch.Tensor): Tokenized prompt IDs.
            prompt_mask (torch.Tensor): Attention mask for the prompts.
            temperature (float): Temperature for generation.
        
        Returns:
            dict: Contains tokenized prompt IDs, generated completion IDs, completion mask, and decoded completions.
        """
        device = self.accelerator.device
        if self.args.use_vllm:
            orig_temp = self.sampling_params.temperature
            self.sampling_params.temperature = temperature

            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # For initial generation, gather prompts across processes.
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                start_time = time.time()
                outputs = self.llm.generate(
                    all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False
                )
                end_time = time.time()
                self.generation_count += 1
                if self.generation_count % 10 == 0:
                    print(f"Main Generation Step {self.generation_count}: Time taken: {end_time - start_time:.2f} seconds")

                # Flatten outputs: for each prompt, get token_ids from each generation.
                completion_ids_list = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids_list = [None] * len(all_prompts_text)
            completion_ids_list = broadcast_object_list(completion_ids_list, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids_list = completion_ids_list[process_slice]
            # Convert each list of token IDs to tensor and pad.
            completion_ids_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in completion_ids_list]

            completion_ids = self._pad_tensors(completion_ids_tensors, self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            self.sampling_params.temperature = orig_temp
        else:
            with self.unwrap_model_for_generation_context():
                gen_config = GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.processing_class.pad_token_id,
                )
                prompt_completion_ids = self.model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=gen_config
                )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
        # Compute the completion mask (stop at the first EOS token).
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completions_text": completions_text,
        }

    def _generate_completions_regeneration(self, prompt_text, prompt_ids, prompt_mask, temperature):
        """
        Generate completions during regeneration for a single prompt.
        Unlike _generate_completions(), this function does not gather prompts across processes.
        Instead, it uses a modified sampling parameter to generate multiple completions in one call.
        
        Args:
            prompt_text (list[str]): List with a single prompt text.
            prompt_ids (torch.Tensor): Tokenized prompt IDs for the prompt.
            prompt_mask (torch.Tensor): Attention mask for the prompt.
            temperature (float): Temperature for generation.
            
        Returns:
            dict: Contains tokenized prompt IDs, generated completion IDs, completion mask, and decoded completions.
        """
        device = self.accelerator.device
        if self.args.use_vllm:
            # Create a copy of the sampling parameters so as not to modify the original.
            from copy import copy
            sampling_params = copy(self.sampling_params)
            sampling_params.n = self.num_generations
            sampling_params.temperature = temperature

            
            start_time = time.time()
            # Directly call llm.generate with the original prompt_text (no replication needed).
            outputs = self.llm.generate(prompt_text, sampling_params=sampling_params, use_tqdm=False)
            end_time = time.time()
            self.regeneration_count += 1
            if self.regeneration_count % 10 == 0:
                print(f"Regeneration Step {self.regeneration_count}: Time taken: {end_time - start_time:.2f} seconds")


            # Flatten outputs.
            completion_ids_list = [out.token_ids for completions in outputs for out in completions.outputs]
            completion_ids_tensors = [torch.tensor(ids, dtype=torch.long, device=device) for ids in completion_ids_list]

            completion_ids = self._pad_tensors(completion_ids_tensors, self.processing_class.pad_token_id)
            
            # Replicate prompt_ids to match the batch size of completions.
            prompt_ids = prompt_ids.repeat(self.num_generations, 1)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # For non-vLLM, assume prompt_ids already corresponds to the replicated prompt.
            prompt_completion_ids = self.model.generate(
                prompt_ids, attention_mask=prompt_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.processing_class.pad_token_id,
                )
            )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Compute completion mask.
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "completions_text": completions_text,
        }



    def _pad_tensors(self, tensors, padding_value):
        """
        Pad a list of 1D tensors to the same length.
        
        Args:
            tensors (list[torch.Tensor]): List of 1D tensors.
            padding_value (int): Padding token ID.
        
        Returns:
            torch.Tensor: A tensor with all inputs padded to the same length.
        """
        max_len = max(t.size(0) for t in tensors)
        padded = [torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=padding_value) for t in tensors]
        return torch.stack(padded)

    def unwrap_model_for_generation_context(self):
        """
        Unwrap the model for generation (needed for vLLM).
        
        Returns:
            The unwrapped model.
        """
        return self.accelerator.unwrap_model(self.model)

    def _compute_rewards_for_group(
        self, prompt: Union[str, list], completions: list[str], is_conv: bool, reward_kwargs: dict = None
    ) -> torch.Tensor:
        """
        Compute rewards for a single prompt group (a prompt with multiple completions).
        If is_conv is True, transform completions into the expected conversational format.
        
        Args:
            prompt (Union[str, list]): The prompt (or conversation) for the group.
            completions (list[str]): List of generated completions (decoded strings).
            is_conv (bool): True if the input is conversational.
            reward_kwargs (dict, optional): Extra kwargs for custom reward functions.
        
        Returns:
            torch.Tensor: Tensor of shape (num_generations,) with computed rewards.
        """
        device = self.accelerator.device
        if is_conv:
            bootstrap = prompt[-1]["content"] if isinstance(prompt, list) and prompt and prompt[-1].get("role") == "assistant" else ""
            transformed_completions = [[{"role": "assistant", "content": bootstrap + comp}] for comp in completions]
            completions = transformed_completions

        rewards_all = []
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, nn.Module):
                if is_conv:
                    messages = [{"messages": prompt + comp} for comp in completions]
                    texts = [apply_chat_template(message, reward_processing_class)["text"] for message in messages]
                else:
                    texts = [prompt + comp for comp in completions]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                with torch.inference_mode():
                    out = reward_func(**reward_inputs).logits[:, 0]
                reward_values = out
            else:
                if reward_kwargs is None:
                    reward_kwargs = {}
                output_reward_func = reward_func(prompts=[prompt] * len(completions), completions=completions, **reward_kwargs)
                reward_values = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            rewards_all.append(reward_values * self.reward_weights[i].to(device))
        rewards = sum(rewards_all)
        return rewards

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare inputs for a training step, including generation, reward computation,
        and regeneration logic.

        Steps:
          1. Process and tokenize prompts.
          2. Extract extra keys (e.g., "solution") for reward functions and gather them globally.
          3. Generate initial completions.
          4. Gather completions, prompts, and extra reward kwargs globally.
          5. On the main process, check reward diversity for each prompt group and regenerate completions if needed,
             using _generate_completions_regeneration().
          6. Broadcast updated completions and regeneration counts.
          7. Re-encode completions, compute per-token log probabilities, decode completions,
             and if conversational, transform them as in the original GRPOTrainer.
          8. Compute rewards and advantages.
          9. Log completions, solutions, rewards, and regen counts via Wandb (logging table built on each process,
             but only main logs to Wandb).

        Returns:
            dict: Prepared tensors for training.
        """
        device = self.accelerator.device

        # 1. Process prompts.
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # 2. Extract extra keys for reward functions (e.g., "solution") and gather them globally.
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        local_reward_kwargs_batch = {key: [example[key] for example in inputs] for key in keys}
        global_reward_kwargs_batch = {}
        for key in keys:
            global_reward_kwargs_batch[key] = gather_object(local_reward_kwargs_batch[key])

        # 3. Generate initial completions.
        gen_out = self._generate_completions(prompts_text, prompt_ids, prompt_mask, temperature=self.args.temperature)
        local_completions_text = gen_out["completions_text"]

        # 4. Gather completions and prompts globally.
        global_completions_text = gather_object(local_completions_text)
        global_prompts_text = gather_object(prompts_text)

        # Determine if inputs[0] is conversational.
        is_conv = is_conversational(inputs[0])

        # 5. On the main process, check reward diversity and regenerate if needed.
        if self.accelerator.is_main_process:
            total = len(global_completions_text)
            num_prompt_groups = total // self.num_generations
            grouped_completions = [
                global_completions_text[i * self.num_generations : (i + 1) * self.num_generations]
                for i in range(num_prompt_groups)
            ]
            grouped_prompts = [
                global_prompts_text[i * self.num_generations] for i in range(num_prompt_groups)
            ]
            global_reward_kwargs = {key: global_reward_kwargs_batch[key] for key in global_reward_kwargs_batch}
            regen_counts = [0] * num_prompt_groups
            for i in range(num_prompt_groups):
                group_start = i * self.num_generations
                group_end = (i + 1) * self.num_generations
                reward_kwargs_group = {key: global_reward_kwargs[key][group_start:group_end] for key in global_reward_kwargs}
                rewards = self._compute_rewards_for_group(grouped_prompts[i], grouped_completions[i], is_conv, reward_kwargs=reward_kwargs_group)
                std_reward = rewards.std().item()
                attempts = 0
                while std_reward < self.regen_min_std and attempts < self.regen_max_attempts:
                    new_temp = random.choice(self.regen_temp_range)
                    new_prompt_inputs = self.processing_class(
                        [grouped_prompts[i]], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                    )
                    new_prompt_inputs = Trainer._prepare_inputs(self, new_prompt_inputs)
                    new_prompt_ids = new_prompt_inputs["input_ids"].to(device)
                    new_prompt_mask = new_prompt_inputs["attention_mask"].to(device)
                    # Use the new regeneration function here.
                    new_gen_out = self._generate_completions_regeneration([grouped_prompts[i]], new_prompt_ids, new_prompt_mask, temperature=new_temp)
                    new_completions = new_gen_out["completions_text"]
                    new_rewards = self._compute_rewards_for_group(grouped_prompts[i], new_completions, is_conv, reward_kwargs=reward_kwargs_group)
                    std_reward = new_rewards.std().item()
                    grouped_completions[i] = new_completions
                    attempts += 1
                regen_counts[i] = attempts
            global_regen_counts_flat = []
            for count in regen_counts:
                global_regen_counts_flat.extend([count] * self.num_generations)
            updated_global_completions_text = [comp for group in grouped_completions for comp in group]
        else:
            expected_total = self.accelerator.num_processes * len(prompts)
            updated_global_completions_text = [None] * expected_total
            global_regen_counts_flat = [None] * expected_total

        # 6. Broadcast updated completions and regeneration counts.
        updated_global_completions_text = broadcast_object_list(updated_global_completions_text, from_process=0)
        global_regen_counts_flat = broadcast_object_list(global_regen_counts_flat, from_process=0)

        num_local_prompts = len(prompts)
        start_idx = self.accelerator.process_index * num_local_prompts
        end_idx = (self.accelerator.process_index + 1) * num_local_prompts
        local_completions_text = updated_global_completions_text[start_idx:end_idx]

        # 7. Re-encode completions and compute per-token log probabilities.
        
        # manually add eos tokens to completions
        local_completions_text = [comp + self.processing_class.eos_token for comp in local_completions_text]
        
        # change padding size to right and add eos token
        completion_encodings = self.processing_class(
            local_completions_text, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
        )

        completion_ids = completion_encodings["input_ids"].to(device)
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        if is_eos.any():
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # 8. Decode completions and transform if conversational, as in original GRPOTrainer.
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if isinstance(prompt, list) and prompt and prompt[-1].get("role") == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # 9. Compute rewards.
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": prompts[idx] + comp} for idx, comp in enumerate(completions)]
                    texts = [apply_chat_template(message, reward_processing_class)["text"] for message in messages]
                else:
                    texts = [prompts[idx] + comp for idx, comp in enumerate(completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # 10. Log completions, solutions, rewards, and regen counts via Wandb.
        
        
        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        
        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd
            if "solution" in inputs[0]:
                local_solutions = [x["solution"] for x in inputs]
            else:
                local_solutions = ["" for _ in inputs]
            global_solutions = gather_object(local_solutions)
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "solution": global_solutions,
                "reward": rewards.tolist(),
                "regen_count": global_regen_counts_flat if global_regen_counts_flat is not None else [-1] * len(rewards),
            }
            df = pd.DataFrame(table)
            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
