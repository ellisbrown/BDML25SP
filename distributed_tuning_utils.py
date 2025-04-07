import os
import json
import torch
import time
import math
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
import glob
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_from_disk

# Set up logging
def setup_logging():
    """Set up logging configuration"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/distributed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

# Common argument parsing function
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed fine-tuning of LLaMA on 2 GPUs")

    # Paths
    parser.add_argument("--model_path", type=str, default="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted",
                        help="Path to the LLaMA 3B model")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed text data")
    parser.add_argument("--output_dir", type=str, default="./distributed-finetuned",
                        help="Directory to save fine-tuned model and outputs")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache",
                        help="Directory to cache datasets")
    parser.add_argument("--deepspeed_config", type=str, default="./ds_config.json",
                        help="Path to DeepSpeed configuration file")

    # Distributed training parameters
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by DeepSpeed launcher)")

    # Training parameters
    parser.add_argument("--train_test_split", type=float, default=0.9,
                        help="Ratio of train/test split (e.g., 0.9 for 90% training)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--per_device_batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Optimization features
    parser.add_argument("--flash_attention", action="store_true", default=True,
                        help="Use Flash Attention for training")
    parser.add_argument("--sdpa_attention", action="store_true", default=True,
                        help="Use SDPA Attention for training")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of modules to apply LoRA to")

    # Precision and quantization options
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit precision")
    parser.add_argument("--use_fp16", action="store_true", default=False,
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                        help="Use mixed precision training (BF16)")
    parser.add_argument("--use_double_quant", action="store_true", default=True,
                        help="Use double quantization for 4-bit training")

    # Optimization flags
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")

    # Pipeline parallelism specific
    parser.add_argument("--num_stages", type=int, default=2,
                        help="Number of pipeline stages (for pipeline parallelism)")

    # Logging and evaluation
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency during training (steps)")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation frequency during training (steps)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Model saving frequency during training (steps)")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep")

    # Parse arguments
    args = parser.parse_args()

    # Convert comma-separated target modules to list
    args.lora_target_modules = args.lora_target_modules.split(",")

    return args

# Function to load and prepare datasets
def prepare_datasets(args, tokenizer):
    """Load datasets from processed text files and tokenize them"""
    # Get split info
    split_info_path = os.path.join(args.data_dir, "split_info.json")

    if os.path.exists(split_info_path):
        # Load from split_info.json
        logging.info(f"Loading data split from {split_info_path}")
        with open(split_info_path, 'r') as f:
            split_info = json.load(f)

        train_files = [os.path.join(args.data_dir, f) for f in split_info["train_files"]]
        test_files = [os.path.join(args.data_dir, f) for f in split_info["test_files"]]
    else:
        # Just get all text files and split them
        logging.info("No split_info.json found, getting all text files")
        txt_files = glob.glob(os.path.join(args.data_dir, "*.txt"))
        random.seed(args.seed)
        random.shuffle(txt_files)
        split_idx = int(len(txt_files) * args.train_test_split)
        train_files = txt_files[:split_idx]
        test_files = txt_files[split_idx:]

    logging.info(f"Training on {len(train_files)} files, testing on {len(test_files)} files")

    # Create datasets from text files
    train_texts = []
    for file_path in tqdm(train_files, desc="Reading training files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Split text into smaller chunks that fit within max_length
                chunks = [text[i:i+args.max_length*4] for i in range(0, len(text), args.max_length*4)]
                train_texts.extend(chunks)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    test_texts = []
    for file_path in tqdm(test_files, desc="Reading test files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = [text[i:i+args.max_length*4] for i in range(0, len(text), args.max_length*4)]
                test_texts.extend(chunks)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    logging.info(f"Created {len(train_texts)} training examples and {len(test_texts)} test examples")

    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    test_dataset = Dataset.from_dict({"text": test_texts})

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True
        )

    # Use map for tokenization
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,  # Use 1 for distributed training to avoid issues
        remove_columns=["text"],
        desc="Tokenizing training dataset"
    )

    test_tokenized = test_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
        desc="Tokenizing test dataset"
    )

    return train_tokenized, test_tokenized

# Configure model with LoRA and quantization
def configure_model_base(args):
    """Base configuration for the LLaMA model with LoRA and quantization"""
    # Configure quantization
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=args.use_double_quant,
            bnb_4bit_quant_type="nf4"
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Load the model with quantization
    model_kwargs = dict(
        quantization_config=quantization_config,
        device_map="auto",  # Will be overridden by DeepSpeed
        trust_remote_code=True,
    )

    if args.flash_attention:
        model_kwargs.update(dict(
            attn_implementation="flash_attention_2"
        ))
    elif args.sdpa_attention:
        model_kwargs.update(dict(
            attn_implementation="sdpa"
        ))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled")

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Log number of trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

    return model, tokenizer

# Evaluate model with perplexity
def evaluate_perplexity(args, model, eval_dataset, tokenizer):
    """Evaluate the model using perplexity metric"""
    model.eval()

    # Create data collator and dataloader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Limit evaluation samples for faster evaluation
    max_eval_samples = 256
    if len(eval_dataset) > max_eval_samples:
        logging.info(f"Limiting evaluation to {max_eval_samples} samples")
        indices = list(range(len(eval_dataset)))
        random.seed(args.seed)
        random.shuffle(indices)
        indices = indices[:max_eval_samples]
        eval_dataset = eval_dataset.select(indices)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.item()
            total_loss += loss * batch["input_ids"].size(0) * batch["input_ids"].size(1)
            total_tokens += batch["input_ids"].size(0) * batch["input_ids"].size(1)

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity

# Save training statistics
def save_training_stats(args, total_time, epoch_times, perplexity, strategy):
    """Save training statistics to a file"""
    if args.local_rank == 0:  # Only save on main process
        stats_file = os.path.join(args.output_dir, "training_stats.txt")
        with open(stats_file, "w") as f:
            f.write(f"Parallelism type: {strategy}\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {total_time:.2f} seconds\n")
            f.write(f"Average time per epoch: {sum(epoch_times)/len(epoch_times):.2f} seconds\n")
            f.write(f"Per-device batch size: {args.per_device_batch_size}\n")
            f.write(f"Total batch size: {args.per_device_batch_size * 2}\n")  # 2 GPUs
            f.write(f"Final perplexity: {perplexity:.2f}\n")

        logging.info(f"Training stats saved to {stats_file}")
