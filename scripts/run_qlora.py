import os
import sys
import subprocess
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset as TorchDataset

import bitsandbytes as bnb
from huggingface_hub import login
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_dataset, DatasetDict, Dataset

# Attempt to upgrade and install required packages
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
except:
    print("Failed to install flash-attn")

# Custom dataset class for handling input data
class CustomDataset(TorchDataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_text = item.get('inputs', item.get('input', item.get('text', '')))
        output_text = item.get('outputs', item.get('output', ''))

        full_text = f"{input_text}\n{output_text}".strip()

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

# Function to find all linear layer names in the model
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

# Function to create a PEFT (Parameter-Efficient Fine-Tuning) model
def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Find modules to quantize
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    # Configure LoRA
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Get PEFT model
    model = get_peft_model(model, peft_config)

    # Adjust model parameters
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            pass
        elif "norm" in name:
            module.to(torch.float32)
        elif "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                pass
        else:
            pass

    model.print_trainable_parameters()
    return model

# Function to manually load dataset from JSON files
def load_dataset_manually(dataset_path):
    print(f"Attempting to load dataset from: {dataset_path}")
    try:
        # Check for train and test dataset files
        train_file = os.path.join(dataset_path, "train_dataset.json")
        test_file = os.path.join(dataset_path, "test_dataset.json")
        dataset_dict = {}

        def load_json_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Dataset.from_list(data)

        if os.path.exists(train_file):
            dataset_dict['train'] = load_json_file(train_file)
        if os.path.exists(test_file):
            dataset_dict['test'] = load_json_file(test_file)

        if not dataset_dict:
            # If no train/test files, check for a single dataset.json file
            dataset_file = os.path.join(dataset_path, "dataset.json")
            if os.path.exists(dataset_file):
                dataset = load_json_file(dataset_file)
                print(f"Loaded dataset with {len(dataset)} samples")
                return dataset
            else:
                raise FileNotFoundError("No dataset files found in the specified path.")

        print(f"Loaded dataset with splits: {dataset_dict.keys()}")
        for split, data in dataset_dict.items():
            print(f"{split} split has {len(data)} samples")

        return DatasetDict(dataset_dict)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

# Main training function
def training_function(script_args, training_args):
    # Load model configuration
    config = AutoConfig.from_pretrained(script_args.model_id)
    config.use_flash_attention = script_args.use_flash_attn
    config.use_flash_attention_2 = script_args.use_flash_attn

    # Load and process dataset
    print(f"Contents of {script_args.dataset_path}:")
    for item in os.listdir(script_args.dataset_path):
        print(item)

    try:
        print("Attempting to load dataset...")
        dataset_or_dict = load_dataset_manually(script_args.dataset_path)
        print("Successfully loaded dataset")

        if isinstance(dataset_or_dict, DatasetDict):
            dataset_dict = dataset_or_dict
        elif isinstance(dataset_or_dict, Dataset):
            # Split the dataset into train and test if not already split
            dataset_dict = dataset_or_dict.train_test_split(test_size=0.1)
        else:
            raise ValueError("Unexpected dataset type")

        print(f"Dataset splits: {dataset_dict.keys()}")

        if 'train' not in dataset_dict:
            raise KeyError("No 'train' split found in the dataset")

        train_dataset_raw = dataset_dict['train']
        print(f"Using 'train' split with {len(train_dataset_raw)} samples")

        if len(train_dataset_raw) == 0:
            raise ValueError("Training dataset is empty")

    except Exception as e:
        print(f"Error loading or processing dataset: {str(e)}")
        raise

    print(f"Dataset info: {train_dataset_raw}")

    print("First few samples of the dataset:")
    for i, sample in enumerate(train_dataset_raw.select(range(min(5, len(train_dataset_raw))))):
        print(f"Sample {i}:")
        print(json.dumps(sample, indent=2))
        print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # Create custom datasets
    train_dataset = CustomDataset(train_dataset_raw, tokenizer, max_length=512)  # Adjust max_length as needed
    eval_dataset = None
    if 'test' in dataset_dict:
        eval_dataset = CustomDataset(dataset_dict['test'], tokenizer, max_length=512)

    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        device_map="auto",
        quantization_config=bnb_config,
        config=config,
        trust_remote_code=script_args.trust_remote_code
    )

    # Create PEFT model
    model = create_peft_model(
        model, gradient_checkpointing=training_args.gradient_checkpointing, bf16=training_args.bf16
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Use train_dataset directly
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()

    # Save the model
    sagemaker_save_dir = "/opt/ml/model/"
    if script_args.merge_adapters:
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model = model.merge_and_unload()
        model.save_pretrained(sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB")
    else:
        trainer.model.save_pretrained(sagemaker_save_dir, safe_serialization=True)

    tokenizer.save_pretrained(sagemaker_save_dir)

# Define script arguments
@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_path: str = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
        default=None,
    )
    hf_token: Optional[str] = field(default=None, metadata={"help": "Hugging Face token for authentication"})
    trust_remote_code: bool = field(
        metadata={"help": "Whether to trust remote code."},
        default=False,
    )
    use_flash_attn: bool = field(
        metadata={"help": "Whether to use Flash Attention."},
        default=False,
    )
    merge_adapters: bool = field(
        metadata={"help": "Whether to merge weights for LoRA."},
        default=False,
    )

# Main function
def main():
    # Parse arguments
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.bf16 = False
    training_args.fp16 = True
    set_seed(training_args.seed)

    # Login to Hugging Face Hub if token is provided
    token = os.environ.get("HF_TOKEN") or script_args.hf_token
    if token:
        print(f"Logging into the Hugging Face Hub with token {token[:10]}...")
        login(token=token)

    # Start training
    training_function(script_args, training_args)

if __name__ == "__main__":
    main()