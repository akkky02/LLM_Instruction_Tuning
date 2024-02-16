#!/bin/bash

# Define your model name and adapter here
MODEL_NAME="meta-llama/Llama-2-7b-hf"
ADAPTER="MAdAiLab/llama2_7b_base_adapter"

# Execute your first Python script with the specified arguments
# python3 merge_eval.py --model_name "$MODEL_NAME" --adapter "$ADAPTER"

# Run the second script using the variable values
lm_eval --model hf \
    --model_args "pretrained=${ADAPTER}_merged_final" \
    --tasks truthfulqa_mc1,truthfulqa_mc2,arithmetic_2ds,arithmetic_4ds,blimp_causative,mmlu_global_facts \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path ./outputs/${ADAPTER}_merged_final \
    --log_samples
