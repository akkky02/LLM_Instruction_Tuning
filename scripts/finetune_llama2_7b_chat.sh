#!/bin/bash

python3 finetune.py \
    --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "yahma/alpaca-cleaned" \
    --num_train_epochs 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules 'q_proj', 'v_proj' \
    --report_to "wandb" \
    --run_name "llama2_7b_chat_adapter" \
    --output_dir "./experiments/MAdAiLab/llama2_7b_chat_adapter" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --do_train \
    --warmup_steps 100 \
    --optimizer "AdamW" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 3 \
    --push_to_hub \
    --hub_model_id "MAdAiLab/llama2_7b_chat_adapter" \
    --hub_strategy "checkpoint"

# Define your model name and adapter here
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
ADAPTER="MAdAiLab/llama2_7b_chat_adapter"

# Execute your first Python script with the specified arguments
python3 merge_eval.py --model_name "$MODEL_NAME" --adapter "$ADAPTER"

# Run the second script using the variable values
lm_eval --model hf \
    --model_args "pretrained=${ADAPTER}_merged_final" \
    --tasks truthfulqa_mc1,truthfulqa_mc2,arithmetic_2ds,arithmetic_4ds,blimp_causative,mmlu_global_facts \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "./outputs/${ADAPTER}_merged_final" \
    --log_samples
