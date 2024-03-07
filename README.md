# LLM_Instruction_Tuning - LoRA and QLoRA Experiments for Llama2 7B

## Introduction

This repository contains the code, data, and experiments for fine-tuning the Llama2 7B language model using LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) techniques. The experiments cover various NLP tasks, including TruthfulQA MC1, TruthfulQA MC2, Arithmetic 2ds, Arithmetic 4ds, BLiMP Causative, MMLU Global Facts.

The main objectives of this project are:

1. Evaluate the performance of the Llama2 7B model when fine-tuned using LoRA and QLoRA techniques.
2. Explore the impact of different configurations, such as rank sizes, alphas, optimization algorithms, and quantization formats, on the model's performance across various NLP tasks.
3. Investigate the trade-offs between computational efficiency, memory usage, and performance when employing LoRA and QLoRA techniques.

The experiments are conducted using the [Hugging Face Transformers](https://huggingface.co/transformers/) library and the [Alpaca 52k cleaned dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned). The results of the experiments are available in the `/outputs` directory.

This project can be used as a reference for fine-tuning various large language models using LoRA and QLoRA techniques on different dataset and for evaluating the performance of these models across various NLP tasks.
## Requirements

- Python 3.10+
- PyTorch 1.12.1
- Hugging Face Transformers 4.26.0
- NVIDIA GPU (RTX A6000, L40, A100 80GB PCIe, or RTX 6000 Ada Generation)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/akkky02/LLM_Instruction_Tuning.git
cd LLM_Instruction_Tuning
```

2. Create a virtual environment and install the required packages:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

To train the Llama2 7B model using LoRA or QLoRA, run the following command:

```bash
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
    --hub_model_id "MAdAiLab/llama2_7b_base_adapter" \
    --hub_strategy "checkpoint"
```

Replace the command-line arguments with the desired settings for your experiment, such as dataset, model_name, rank sizes, alphas, quantization formats, and optimization algorithms.

### Evaluation

After training, use the following command to evaluate the model's performance on various NLP tasks:

```bash
MODEL_NAME="meta-llama/Llama-2-7b-hf"
ADAPTER="MAdAiLab/llama2_7b_base_adapter"

python3 merge_eval.py --model_name "$MODEL_NAME" --adapter "$ADAPTER"

lm_eval --model hf \
    --model_args "pretrained=${ADAPTER}_merged_final" \
    --tasks truthfulqa_mc1,truthfulqa_mc2,arithmetic_2ds,arithmetic_4ds,blimp_causative,mmlu_global_facts \
    --device cuda:0 \
    --batch_size auto:4 \
    --output_path "./outputs/${ADAPTER}_merged_final" \
    --log_samples
```

Adjust the `--tasks` argument based on the specific tasks you want to evaluate.

### Runpod Integration

To run these experiments on the Runpod platform, follow these steps:

1. Sign up for a Runpod account at [https://runpod.io](https://runpod.io).
2. Create a new Runpod instance and select the desired GPU configuration (e.g., NVIDIA RTX A6000, NVIDIA L40, NVIDIA A100 80GB PCIe, or NVIDIA RTX 6000 Ada Generation).
3. Once your Runpod instance is ready, follow the steps in the "Installation" section to set up the environment.
4. Clone this repository into your Runpod instance and follow the "Usage" section to perform training and evaluation.
5. Remember to terminate the Runpod instance after completing your experiments to stop incurring costs.

## Results

The results of the experiments are available in the `results/` directory. Each subdirectory contains the evaluation metrics for a specific LoRA or QLoRA configuration applied to the Llama2 7B model.
For a detailed report and analysis, please refer to our [Evaluation of Llama 2 7b model with LoRA and QLoRA using Huggingface ecosystem ](https://akshat-patil.notion.site/Evaluation-of-Llama-2-7b-model-with-LoRA-and-QLoRA-using-Huggingface-ecosystem-fb714bf2a8c74b2186b4f13879a576f5?pvs=4).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Llama2 7B model](https://huggingface.co/meta-llama/Llama-2-7b-hf) - The language model used in these experiments.
- [Alpaca 52k cleaned dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) - The dataset used for fine-tuning the model.

## References
- [Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of Experiments](https://lightning.ai/pages/community/lora-insights/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Quantized Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2212.00143)
