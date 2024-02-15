#bin/bash

python3 new_train.py \
--model_name_or_path "meta-llama/Llama-2-7b-hf" \
--output_dir "llama2_finetuned" \
--dataset_name "yahma/alpaca-cleaned" \
--num_train_epochs 1 \
--r 16 \
--lora_alpha 64 \
--target_modules "all-linear" \
--lora_dropout 0.1 \
--bias "none" \
--task_type "CAUSAL_LM" \