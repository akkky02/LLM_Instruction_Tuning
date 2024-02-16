#!/bin/bash

python3 finetune.py \
--model_name_or_path "meta-llama/Llama-2-7b-hf" \
--dataset "yahma/alpaca-cleaned" \
--num_train_epochs 1 \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--target_modules 'q_proj', 'v_proj' \
--report_to "wandb" \
--run_name "lama2_7b_base_adapter" \
--output_dir "./experiments" \
--optim "paged_adamw_32bit" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 128 \
--learning_rate 3e-4 \
--weight_decay 0.01 \
--gradient_accumulation_steps 128 \
--gradient_checkpointing \
--do_train \
--warmup_steps 10 \
--logging_steps 1 \
--save_strategy "steps" \
--save_steps 10000 \
--save_total_limit 3 \
--push_to_hub \
--hub_model_id "MAdAiLab/lama2_7b_base_adapter" \
--hub_strategy "every_save"

