from dataclasses import dataclass, field
import os
import sys
import subprocess
from transformers import AutoConfig
from typing import Optional

# Upgrade flash attention and install required packages
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", "requirements.txt"])
except:
    print("flash-attn failed to install")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from datasets import load_from_disk
import torch

import bitsandbytes as bnb
from huggingface_hub import login


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # Prepare int-4 model for training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Get LoRA target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # Upcast the layer norms in float 32
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            pass  # Leave the Lora layers as they are
        elif "norm" in name:
            module.to(torch.float32)
        elif "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                pass  # Leave embedding layers as they are
        else:
            pass  # Leave all other modules as they are

    model.print_trainable_parameters()
    return model


def training_function(script_args, training_args):
    config = AutoConfig.from_pretrained(script_args.model_id)
    config.use_flash_attention = script_args.use_flash_attn
    config.use_flash_attention_2 = script_args.use_flash_attn

    # Load dataset
    dataset = load_from_disk(script_args.dataset_path)

    # Load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

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

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    # Start training
    trainer.train()

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

    # Save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, padding_side="left")
    tokenizer.save_pretrained(sagemaker_save_dir)


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


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.bf16 = False
    training_args.fp16 = True
    set_seed(training_args.seed)

    token = os.environ.get("HF_TOKEN") or script_args.hf_token
    if token:
        print(f"Logging into the Hugging Face Hub with token {token[:10]}...")
        login(token=token)

    training_function(script_args, training_args)


if __name__ == "__main__":
    main()