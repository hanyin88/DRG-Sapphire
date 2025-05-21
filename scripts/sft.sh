#!/bin/bash

# Example script to run SFT for 50-50 split on the full dataset

sft_learning_rate="4e-5"
epo_sft=3

LOG_FILE_sft="logs/sft_all_50_50.log"
run_name_sft="sft_all_50_50"
output_dir_sft="outputs/sft_all_50_50"

sft_dataset_name="data/split_training_all/50_50/sft_split.csv"

accelerate launch --config_file recipes/zero3.yaml  \
    --num_processes 4 \
    --main_process_port 29524 \
    src/open_r1/sft.py \
    --config recipes/sft_config.yaml \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name $sft_dataset_name \
    --gradient_accumulation_steps 8 \
    --learning_rate $sft_learning_rate \
    --num_train_epochs $epo_sft \
    --packing False \
    --max_seq_length 12846 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing False\
    --bf16 \
    --save_only_model \
    --save_steps 250 \
    --logging_steps 1 \
    --output_dir "$output_dir_sft" \
    --run_name "$run_name_sft" > "$LOG_FILE_sft" 2>&1


    