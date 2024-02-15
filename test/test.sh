#!/bin/bash

python3 test.py \
--model_name_or_path "meta-llama/Llama-2-7b-hf" \
--max_train_samples 10 \
--dataset "yahma/alpaca-cleaned" \
--double_quant \
--quant_type "nf4" \
--bits 4 \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 \
# --max_memory_MB 6000 \
--report_to "wandb" \
--output_dir "./output" \
--optim "paged_adamw_32bit" \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--max_steps 10 \
--learning_rate 2e-4 \
--weight_decay 0.0 \
--learning_rate 0.0002 \
--max_grad_norm 0.3 \
--gradient_checkpointing \
--do_train \
--lr_scheduler_type "constant" \
--warmup_steps 10 \
--logging_steps 10 \
--save_strategy "steps" \
--save_steps 250 \
--save_total_limit 3
