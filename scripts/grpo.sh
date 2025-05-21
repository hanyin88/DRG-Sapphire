#!/bin/bash

output_dir_sft='outputs/sft_all_50_50'

per_device_train_batch_size=4
gradient_accumulation_steps=32
num_processes=4
num_generations=8
grpo_learning_rate="3e-6"

LOG_FILE="log/grpo_all_50_50.log"
run_name="grpo_all_50_50"
output_dir="outputs/grpo_all_50_50"

grpo_dataset_name="data/split_training_all/50_50/grpo_train_merged"

# Run the training script with command-line arguments

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/zero3.yaml \
    --num_processes $num_processes \
    --main_process_port 29503 \
    src/open_r1/grpo.py \
    --config recipes/grpo_config.yaml \
    --model_name_or_path $output_dir_sft \
    --vllm_gpu_memory_utilization 0.7 \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataset_name $grpo_dataset_name \
    --num_generations $num_generations \
    --lr_scheduler_type "constant_with_warmup" \
    --use_liger True  \
    --temperature 1.0 \
    --learning_rate $grpo_learning_rate \
    --max_completion_length 4096 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 250 \
    --output_dir "$output_dir" \
    --run_name "$run_name" > "$LOG_FILE" 2>&1
