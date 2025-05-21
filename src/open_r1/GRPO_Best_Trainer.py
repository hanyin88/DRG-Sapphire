# custom_grpo_trainer.py
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
# Key modifications:
# 1) For each prompt group, if the std of rewards is below a configurable min_std,
#    we regenerate its completions (using a random temperature from a configurable range)
#    up to a maximum number of attempts.
# 2) The number of regeneration attempts per prompt is logged via Wandb and included in the logging table.
# 3) vLLM generation (if enabled) is used as in the original code.
# 4) Multi-GPU synchronization is handled using gather_object and broadcast_object_list.
# 5) The _compute_rewards_for_group() function now accepts an extra parameter indicating whether
#    the input is conversational. When true, it transforms the completions into the expected format.
# 6) Data utilities are imported from trl.data_utils.
# 7) Extra reward-key inputs (e.g., "solution") are gathered globally so that the main process
#    can handle regeneration for the entire batch.
# 8) After computing ref_per_token_logps, we decode completions and, if conversational,
#    transform them as in the original GRPOTrainer.
# 9) A new method, _generate_completions_regeneration(), is used during regeneration so that
#    we generate completions for a single prompt by replicating it self.num_generations times,
#    rather than relying on gather_object.
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

import math

class GRPO_Best_Trainer(GRPOTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Apply cosine decay to beta: decay from original self.beta to 0 over the total training steps.
        # We assume that self.args.max_steps is the total number of training steps.
        global_step = self.state.global_step
        total_steps = self.state.max_steps
        decay_factor = 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
        current_beta = self.beta * decay_factor

        # Debug prints for tracking beta decay and training progress only every 25 steps.
        if global_step % 25 == 0:
            print(f"[DEBUG] Global step: {global_step}, Total steps: {total_steps}, "
                f"Decay factor: {decay_factor:.4f}, Current beta: {current_beta:.4f}")

        # Compute the loss:
        # Multiply the advantages (broadcasted) with the log probability differences and apply the current beta for KL term.
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        per_token_loss = -(per_token_loss - current_beta * per_token_kl)

        # per x, this is the DAPO loss on a 2 d tensor: https://github.com/sail-sg/understand-r1-zero/issues/10
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # print loss every 10 steps
        if self.accelerator.is_main_process and self.state.global_step % 25 == 0:
            print(f"Step {self.state.global_step}: DAPO Loss = {loss.item()}")

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss
