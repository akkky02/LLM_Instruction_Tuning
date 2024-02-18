from collections import defaultdict
from functools import partial
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Sequence, Union
import numpy as np
from tqdm import tqdm
import bitsandbytes as bnb
import pandas as pd
from packaging import version
from packaging.version import parse

import torch
import transformers
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from dotenv import load_dotenv, find_dotenv
import subprocess


load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
os.environ["WANDB_PROJECT"] = "LLM_IFT_Experiments"



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf")

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=100,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."
        },
    )
    dataset: str = field(
        default='yahma/alpaca-cleaned',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    bnb_4bit_use_double_quant: bool = field(default=False, metadata={"help": "Compress the quantization statistics through double quantization."})
    bnb_4bit_quant_type: str = field(default="fp4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    load_in_4bit: bool = field(default=False, metadata={"help": "Load the model in 4-bit mode."})
    lora_r: int = field(default=8, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj'], metadata={"help": "Lora apply to these modules. eg ['gate_proj', 'q_proj', 'v_proj', 'down_proj', 'o_proj', 'up_proj', 'k_proj']"})
    report_to: str = field(default='wandb', metadata={"help": "To use wandb or something else for reporting."})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optimizer: str = field(default='AdamW', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    logging_steps: int = field(default=1, metadata={"help": 'The frequency of update steps after which to log the loss'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=3, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})

## Download and prepare a model checkpoint
def load_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def create_prompt_formats(sample):
    """
    Format various fields of the sample ('instruction', 'context', 'response')
    Then concatenate them using two newline characters
    :param sample: Sample dictionnary
    """

    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    # END_KEY = "### End"

    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['response']}"
    # end = f"{END_KEY}"

    parts = [part for part in [blurb, instruction, input_context, response] if part]

    formatted_prompt = "\n\n".join(parts)

    sample["text"] = formatted_prompt
    return sample

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length

def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, input_dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """

    # Add prompt to each sample
    print("Preprocessing dataset...")
    dataset = input_dataset.map(create_prompt_formats)#, batched=True)

    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text"],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

# Create bitsandbytes config

def create_bnb_config(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype= torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
    )
    return bnb_config

def create_peft_config(training_args):
    """
    Create Parameter-Efficient Fine-Tuning config for your model
    :param modules: Names of the modules to apply Lora to
    """
    config = LoraConfig(
        r=training_args.lora_r,  # dimension of the updated matrices
        lora_alpha=training_args.lora_alpha,  # parameter for scaling
        target_modules=training_args.target_modules,  # list of module names to apply Lora to
        lora_dropout=training_args.lora_dropout,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )

    return config

# SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def train(model, tokenizer, dataset, output_dir, training_args):
    # Apply preprocessing to the model to prepare it by
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    # model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Get lora module names
    modules = find_all_linear_names(model)
    print(f"Modules: {modules}")

    # Create PEFT config for these modules and wrap the model to PEFT
    peft_config = create_peft_config(training_args)
    model = get_peft_model(model, peft_config)

    # Print information about the percentage of trainable parameters
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs

    ### SOURCE https://github.com/artidoro/qlora/blob/main/qlora.py
    # Verifying the datatypes before training

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total+= v
    for k, v in dtypes.items():
        print(k, v, v/total)

    do_train = True

    # Launch training
    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    ###

    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    # trainer.tokenizer.push_to_hub(output_dir, use_temp_dir=True, commit_message="Add tokenizer")
    # trainer.push_to_hub(use_temp_dir=True, commit_message="Add model")
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

def main():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments
    ))
    model_args, data_args, training_args = hfparser.parse_args_into_dataclasses()#return_remaining_strings=True)
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    print(args)

    # Load model and tokenizer
    model_name = args.model_name_or_path
    bnb_config = create_bnb_config(args)
    model, tokenizer = load_model(model_name, bnb_config)

    # Load dataset
    dataset = load_dataset(args.dataset, split='train')
    dataset = dataset.rename_columns({'output': 'response', 'input': 'context', 'instruction': 'instruction'})

    # Preprocess
    max_length = get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, 56, dataset)

    # Train
    train(model, tokenizer, dataset, training_args.output_dir ,training_args)
    bnb_config.to_json_file(training_args.output_dir)

if __name__ == "__main__":
    main()