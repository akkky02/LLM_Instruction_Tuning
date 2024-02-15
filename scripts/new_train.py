import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union
import datasets
import torch
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)




load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
from huggingface_hub.hf_api import HfFolder; HfFolder.save_token(HF_TOKEN)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='meta-llama/Llama-2-7b-hf',
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    # token: str = field(
    #     default=HF_TOKEN,
    #     metadata={
    #         "help": (
    #             "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
    #             "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
    #         )
    #     },
    # )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="yahma/alpaca-cleaned", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class QLoraArgs():
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    """
    r: int = field(default=16, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default='all-linear',
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
            ),
        },
    )
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "Lora dropout"})
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"}
    )
    task_type: Optional[str]  = field(default='CAUSAL_LM', metadata={"help": "Task type"})
    load_in_4bit: bool = field(default=True, metadata={"help": "Load in 4bit"})
    bnb_4bit_use_double_quant : bool = field(default=True, metadata={"help": "Use double quant"})
    bnb_4bit_quant_type : str = field(default="nf4", metadata={"help": "Quant type"})
    # bnb_4bit_compute_dtype : float = field(default=torch.bfloat16, metadata={"help": "Compute dtype"})

        
def load_model(model_name, bnb_config):
    # Load model and tokenizer
    n_gpus = torch.cuda.device_count()
    max_memory = f'{40960}MB'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", 
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_prompt_formats(sample):
    # Format prompt
    INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"
    
    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
    input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
    response = f"{RESPONSE_KEY}\n{sample['response']}"
    
    parts = [part for part in [blurb, instruction, input_context, response] if part]
    formatted_prompt = "\n\n".join(parts)
    
    sample["text"] = formatted_prompt
    return sample


def get_max_length(model):
    # Get max sequence length
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    # Tokenize batch
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

def preprocess_dataset(tokenizer, max_length, seed, dataset):
    # Preprocess dataset
    print("Preprocessing dataset...")
    
    dataset = dataset.map(create_prompt_formats, batched=True) 
    
    _preprocess_batch = partial(preprocess_batch, tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.map(_preprocess_batch, batched=True, remove_columns=["instruction", "context", "response", "text"])
    
    dataset = dataset.filter(lambda x: len(x["input_ids"]) < max_length)
    
    dataset = dataset.shuffle(seed=seed)
    return dataset

# def create_bnb_config(qlora_args):
#     # Create Bits&Bytes config
#     return BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )

# def create_peft_config(modules):
#     # Create PEFT config
#     return LoraConfig(
#         r=16,  
#         lora_alpha=64,  
#         target_modules=modules,
#         lora_dropout=0.1,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )

# def find_all_linear_names(model):
#     # Get names of modules to apply PEFT to
#     lora_modules = set()
#     for name, module in model.named_modules():
#         if isinstance(module, bnb.nn.Linear4bit):
#             names = name.split('.')
#             lora_modules.add(names[0] if len(names) == 1 else names[-1])

#     if 'lm_head' in lora_modules:
#         lora_modules.remove('lm_head')
        
#     return list(lora_modules)

def print_trainable_parameters(model, use_4bit=False):
    # Print number of trainable parameters
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
            
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    if use_4bit:
        trainable_params /= 2
        
    print(f"all params: {all_params:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_params}")

def train(model, tokenizer, dataset, training_args, qlora_args):
    # Training
    
    # Prepare model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(qlora_args)
    model = get_peft_model(model, peft_config)
    
    print_trainable_parameters(model)
    
    # Define trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False
    
    # Train
    print("Training...")
    train_result = trainer.train()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)

    # Save model
    print("Saving model...") 
    os.makedirs(training_args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, QLoraArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, qlora_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, qlora_args = parser.parse_args_into_dataclasses()


    # Load model and tokenizer
    model_name = model_args.model_name_or_path
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_args.load_in_4bit,
        bnb_4bit_use_double_quant=qlora_args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qlora_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model, tokenizer = load_model(model_name, bnb_config)

    # Load dataset
    dataset = load_dataset(data_args.dataset_name, split="train")
    dataset = dataset.rename_columns({'output': 'response', 'input': 'context', 'instruction': 'instruction'})

    # Preprocess
    max_length = get_max_length(model)
    dataset = preprocess_dataset(tokenizer, max_length, 56, dataset)

    # Train
    # output_dir = "llama2_finetuned" 
    train(model, tokenizer, dataset, training_args, qlora_args)
    
    
if __name__ == "__main__":
    main()