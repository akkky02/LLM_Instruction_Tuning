import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
# WANDB_API_KEY = os.getenv("WANDB_API_KEY")

def merge_push_eval(model_name, adapter):
    # compute_dtype = getattr(torch, "float32")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload(progressbar=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.push_to_hub(f'{adapter}_merged_final', safe_serialization=True)
    tokenizer.push_to_hub(f'{adapter}_merged_final', safe_serialization=True)   

    # return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge, push, and evaluate a model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--adapter", type=str, required=True, help="Name of the adapter")
    args = parser.parse_args()
    
    merge_push_eval(args.model_name, args.adapter)
