import gc
import os
import glob
import json
import random
import math
import torch
import numpy as np
from tqdm import tqdm
import PyPDF2
import logging
import time
import psutil
import GPUtil
import argparse
import hashlib
import wandb
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_from_disk

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on climate documents with memory optimizations")

    # Paths
    parser.add_argument("--model_path", type=str, default="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B",
                        help="Path to the LLaMA 3B model")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed text data")
    parser.add_argument("--output_dir", type=str, default="./llama-finetuned",
                        help="Directory to save fine-tuned model and outputs")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache",
                        help="Directory to cache datasets")

    # Training parameters
    parser.add_argument("--train_test_split", type=float, default=0.9,
                        help="Ratio of train/test split (e.g., 0.9 for 90% training)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Memory optimization parameters
    parser.add_argument("--start_batch_size", type=int, default=1,
                        help="Minimum batch size to try")
    parser.add_argument("--max_batch_size", type=int, default=128,
                        help="Maximum batch size to try")
    parser.add_argument("--safe_batch_size", type=int, default=8,
                        help="Safe batch size to try first")
    parser.add_argument("--base_grad_accum", type=int, default=32,
                        help="Base number for gradient accumulation steps")
    # flash attention:
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

    # Quantization and precision options
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit precision")
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--use_bf16", action="store_true", default=False,
                        help="Use mixed precision training (BF16)")
    parser.add_argument("--compile_model", action="store_true", default=False,
                        help="Compile model with torch.compile()")

    # Optimization flags
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    parser.add_argument("--use_8bit_optimizer", action="store_true", default=True,
                        help="Use 8-bit optimizer (paged_adamw_8bit)")
    parser.add_argument("--use_double_quant", action="store_true", default=True,
                        help="Use double quantization for 4-bit training")

    # Logging and evaluation
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency during training (steps)")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation frequency during training (steps)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Model saving frequency during training (steps)")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep")

    # Weights & Biases arguments
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llama-finetuning",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/username")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Custom name for this run in W&B")
    parser.add_argument("--wandb_tags", type=str, default="",
                        help="Comma-separated list of tags for W&B run")

    # Devices
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for training (e.g., cuda:0)")

    args = parser.parse_args()

    # Convert comma-separated target modules to list
    args.lora_target_modules = args.lora_target_modules.split(",")

    return args

# Set up global args
args = parse_args()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("training.log"),
        logging.FileHandler(f"logs/training_{timestamp}.log"),
        logging.StreamHandler()
    ]
)

# Function to monitor GPU usage
def log_gpu_usage(log_to_wandb=False):
    """Log GPU memory usage."""
    gpu_metrics = {}
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            if int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) == gpu.id:
                gpu_info.append(f"GPU {gpu.id}: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
                # Add GPU metrics to dict for wandb
                gpu_metrics[f"gpu_{gpu.id}_memory_used_mb"] = gpu.memoryUsed
                gpu_metrics[f"gpu_{gpu.id}_memory_util_pct"] = gpu.memoryUtil * 100

        if gpu_info:
            logging.info("\nGPU Memory Usage: " + ", ".join(gpu_info))
    except Exception as e:
        logging.warning(f"Failed to log GPU usage: {e}")

    # Log RAM usage
    ram = psutil.virtual_memory()
    ram_used_gb = ram.used / 1e9
    ram_total_gb = ram.total / 1e9
    ram_percent = ram.percent

    logging.info(f"RAM Usage: {ram_used_gb:.1f}GB/{ram_total_gb:.1f}GB ({ram_percent:.1f}%)")

    # Add RAM metrics to dict for wandb
    gpu_metrics["ram_used_gb"] = ram_used_gb
    gpu_metrics["ram_util_pct"] = ram_percent

    # Log to wandb if enabled
    if log_to_wandb and args.use_wandb and wandb.run is not None:
        wandb.log(gpu_metrics)

# Create directories if they don't exist
def create_directories(data_dir, output_dir, cache_dir):
    """Create necessary directories."""
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"Created directories: {data_dir}, {output_dir}, {cache_dir}")

# STEP 1: Data Processing - Extract text from PDFs
def extract_text_from_pdfs(pdf_dir, output_dir):
    """Extract text from all PDFs in the directory and save to txt files."""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files")

    for pdf_path in tqdm(pdf_files, desc="Extracting text from PDFs"):
        try:
            filename = os.path.basename(pdf_path).replace('.pdf', '.txt')
            output_path = os.path.join(output_dir, filename)

            # Skip if already processed
            if os.path.exists(output_path):
                continue

            # Extract text using PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text() + "\n\n"

            # Save the extracted text
            with open(output_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")

    return glob.glob(os.path.join(output_dir, "*.txt"))

# STEP 2: Split data into train and test sets
def split_train_test(file_list, train_ratio=0.9):
    """Split files into training and testing sets."""
    random.shuffle(file_list)
    split_idx = int(len(file_list) * train_ratio)
    train_files = file_list[:split_idx]
    test_files = file_list[split_idx:]
    return train_files, test_files

# Function to hash file paths for caching
def hash_file_paths(file_paths):
    """Create a hash from a list of file paths to use as a cache key."""
    # Sort to ensure consistent hash regardless of order
    file_paths = sorted(file_paths)
    # Create a string of all paths and their modification times for better caching
    paths_string = ""
    for path in file_paths:
        mtime = os.path.getmtime(path)
        paths_string += f"{path}:{mtime};"
    # Create a hash
    hash_obj = hashlib.md5(paths_string.encode())
    return hash_obj.hexdigest()

# Create raw text dataset with caching
def create_text_dataset(file_paths, cache_dir):
    """Create dataset of raw texts from files with caching."""
    # Create a hash of the file paths to use as cache key
    files_hash = hash_file_paths(file_paths)
    cache_path = os.path.join(cache_dir, f"raw_dataset_{files_hash}")

    # Check if cached dataset exists
    if os.path.exists(cache_path):
        logging.info(f"Loading raw text dataset from cache: {cache_path}")
        return load_from_disk(cache_path)

    logging.info("Cache not found, creating raw text dataset from files...")
    texts = []

    for file_path in tqdm(file_paths, desc="Reading text files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Process the text in chunks to avoid loading everything in memory
                paragraphs = text.split("\n\n")
                current_chunk = ""

                for para in paragraphs:
                    # If adding this paragraph would make the chunk too long, save current chunk and start a new one
                    if len(current_chunk) + len(para) > args.max_length * 4:  # Rough character estimate
                        if current_chunk:
                            texts.append(current_chunk)
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para if current_chunk else para

                # Add the last chunk if it's not empty
                if current_chunk:
                    texts.append(current_chunk)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    # Create dataset from texts
    dataset = Dataset.from_dict({"text": texts})

    # Save to cache
    logging.info(f"Saving raw text dataset to cache: {cache_path}")
    dataset.save_to_disk(cache_path)

    return dataset

# Prepare and tokenize datasets with caching
def prepare_datasets(train_files, test_files, tokenizer, cache_dir):
    """Prepare and tokenize datasets with caching."""
    # Create raw datasets or load from cache
    train_raw_cache_dir = os.path.join(cache_dir, "raw_train")
    eval_raw_cache_dir = os.path.join(cache_dir, "raw_eval")
    os.makedirs(train_raw_cache_dir, exist_ok=True)
    os.makedirs(eval_raw_cache_dir, exist_ok=True)

    train_dataset = create_text_dataset(train_files, train_raw_cache_dir)
    eval_dataset = create_text_dataset(test_files, eval_raw_cache_dir)

    # Hash tokenizer configuration to include in cache key
    tokenizer_config = f"{tokenizer.name_or_path}_{args.max_length}"
    tokenizer_hash = hashlib.md5(tokenizer_config.encode()).hexdigest()

    # Check for cached tokenized datasets
    train_cache = os.path.join(cache_dir, f"tokenized_train_{tokenizer_hash}")
    eval_cache = os.path.join(cache_dir, f"tokenized_eval_{tokenizer_hash}")

    if os.path.exists(train_cache) and os.path.exists(eval_cache):
        logging.info("Loading tokenized datasets from cache")
        train_tokenized = load_from_disk(train_cache)
        eval_tokenized = load_from_disk(eval_cache)
    else:
        logging.info("Tokenizing datasets")
        # Define tokenization function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_special_tokens_mask=True
            )

        # Use datasets.map for efficient tokenization
        train_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=max(1, os.cpu_count() // 2),  # Use parallel processing but avoid using all cores
            remove_columns=["text"],
            desc="Tokenizing training dataset"
        )

        eval_tokenized = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=max(1, os.cpu_count() // 2),
            remove_columns=["text"],
            desc="Tokenizing evaluation dataset"
        )

        # Save tokenized datasets to cache
        logging.info(f"Saving tokenized datasets to cache: {train_cache} and {eval_cache}")
        train_tokenized.save_to_disk(train_cache)
        eval_tokenized.save_to_disk(eval_cache)

    return train_tokenized, eval_tokenized

# STEP 4: Configure model with memory optimizations
def configure_model_for_fine_tuning():
    """Configure LLaMA model with memory optimizations."""

    # Configure quantization
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=args.use_double_quant,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True  # Add this for CPU offloading
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Load the model with quantization
    model_kwargs = dict(
        quantization_config=quantization_config,
        device_map="auto",
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
    model.print_trainable_parameters()

    # Add after configuring model with LoRA in configure_model_for_fine_tuning()
    if torch.cuda.is_available() and hasattr(torch, 'compile') and args.compile_model:
        model = torch.compile(model, mode="reduce-overhead")
        logging.info("Model compiled with torch.compile()")

    return model, tokenizer

# STEP 5: Configure training arguments
def get_training_args(batch_size, gradient_accumulation_steps):
    """Get training arguments with memory optimizations."""
    # Configure reporting based on wandb settings
    report_to = ["none"]
    if args.use_wandb:
        report_to = ["wandb"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        run_name=args.wandb_run_name if args.use_wandb else None,
        label_names=["labels"],  # Add this line to silence Peft warning
    )

    # Set precision flags
    if args.use_fp16:
        training_args.fp16 = True
    if args.use_bf16:
        training_args.bf16 = True

    # Set optimizer
    if args.use_8bit_optimizer:
        training_args.optim = "paged_adamw_8bit"

    return training_args

# Create a custom trainer that adds perplexity to metrics
class PplxTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log(self, logs, start_time=None):
        """
        Add perplexity to logs
        """
        if "loss" in logs:
            logs["pplx"] = torch.exp(torch.tensor(logs["loss"])).item()
        if "eval_loss" in logs:
            logs["eval_pplx"] = torch.exp(torch.tensor(logs["eval_loss"])).item()

        super().log(logs, start_time)



# STEP 6: Train the model
def train_model(model, tokenizer, train_dataset, eval_dataset, batch_size=1, gradient_accumulation_steps=8):
    """Train the model with memory optimizations."""
    print(f"Training with batch size: {batch_size}, grad accum: {gradient_accumulation_steps}")

    # Limit evaluation samples to save time
    max_eval_samples = 256
    if len(eval_dataset) > max_eval_samples:
        logging.info(f"Limiting evaluation samples to {max_eval_samples} during training (actual: {len(eval_dataset)})")
        # eval_dataset = eval_dataset.select(range(max_eval_samples))
        # random sample
        eval_dataset = eval_dataset.select(random.sample(range(len(eval_dataset)), max_eval_samples))

    # Log to wandb
    use_wandb = args.use_wandb and wandb.run is not None
    if use_wandb:
        wandb.log({
            "train/batch_size": batch_size,
            "train/grad_accum_steps": gradient_accumulation_steps,
            "train/effective_batch_size": batch_size * gradient_accumulation_steps,
            "train/learning_rate": args.learning_rate,
            "train/num_epochs": args.num_epochs,
            "train/train_samples": len(train_dataset),
            "train/eval_samples": len(eval_dataset)
        })

    # Create custom callback to log GPU usage periodically
    class GPULoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Log GPU usage every few steps
            if state.global_step % 10 == 0:
                log_gpu_usage(log_to_wandb=True)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            # Log evaluation metrics to wandb
            if use_wandb:
                # Add eval/ prefix to all metrics for better organization in wandb
                wandb_metrics = {f"eval/{k}": v for k, v in metrics.items()}
                wandb.log(wandb_metrics)

    # Use DataCollator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Configure training arguments
    training_args = get_training_args(batch_size, gradient_accumulation_steps)

    # Initialize Trainer
    trainer = PplxTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[GPULoggingCallback()],
    )

    # Train the model
    train_result = trainer.train()

    # Log final training metrics
    if args.use_wandb and wandb.run is not None:
        metrics = train_result.metrics
        wandb.log({f"train/{k}": v for k, v in metrics.items()})

    # Run final evaluation
    eval_metrics = trainer.evaluate()
    logging.info(f"Final evaluation metrics: {eval_metrics}")

    # Save the model
    trainer.save_model(args.output_dir)
    return trainer, metrics, eval_metrics

# STEP 7: Evaluate the model using perplexity
def evaluate_perplexity(model, tokenizer, eval_dataset):
    """Evaluate the model using perplexity metric."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    # Create a DataCollator for efficient batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create a dataloader for evaluation
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=4,  # Use small batch size for evaluation
        collate_fn=data_collator
    )

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating perplexity"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss.item()
            total_loss += loss * batch["input_ids"].size(0) * batch["input_ids"].size(1)
            total_tokens += batch["input_ids"].size(0) * batch["input_ids"].size(1)

    # Calculate perplexity
    perplexity = math.exp(total_loss / total_tokens)
    print(f"Perplexity: {perplexity:.2f}")

    # Log perplexity to wandb if enabled
    if args.use_wandb and wandb.run is not None:
        wandb.log({"eval/pplx": perplexity})

    return perplexity

# STEP 8: Find maximum batch size through binary search
def find_max_batch_size(model, tokenizer, train_dataset, eval_dataset):
    """Binary search to find maximum batch size with improved memory cleanup."""
    # CRITICAL: Do thorough memory cleanup before starting
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # Reset peak stats for clean measurement

    # Force a full GPU memory check
    initial_memory_allocated = torch.cuda.memory_allocated(0)
    initial_memory_reserved = torch.cuda.memory_reserved(0)
    logging.info(f"Initial GPU memory state - Allocated: {initial_memory_allocated/1e9:.2f}GB, Reserved: {initial_memory_reserved/1e9:.2f}GB")

    min_batch = args.start_batch_size
    max_batch = args.max_batch_size
    optimal_batch = min_batch
    optimal_batch = min_batch

    logging.info("Starting binary search for maximum batch size")
    logging.info(f"Initial search range: {min_batch} to {max_batch}")

    # Log to wandb
    if args.use_wandb and wandb.run is not None:
        wandb.log({
            "batch_search/min_batch": min_batch,
            "batch_search/max_batch": max_batch,
            "batch_search/start_time": time.time()
        })

    # First try a conservative batch size to establish a baseline
    safe_batch_size = args.safe_batch_size
    logging.info(f"Testing a safe batch size of {safe_batch_size} first")
    try:
        result = test_batch_size(model, tokenizer, train_dataset, eval_dataset, safe_batch_size)

        if result == "success":
            min_batch = safe_batch_size
            logging.info(f"Safe batch size {safe_batch_size} works! Continuing search...")

            # Log to wandb
            if args.use_wandb and wandb.run is not None:
                wandb.log({
                    "batch_search/safe_batch_size_success": True,
                    "batch_search/current_optimal": optimal_batch
                })
        else:
            # If safe batch size fails, fall back to batch size 1
            min_batch = 1
            logging.warning(f"Safe batch size {safe_batch_size} failed. Falling back to batch size 1")
    except Exception as e:
        logging.warning(f"Even safe batch size {safe_batch_size} failed: {e}")
        logging.info("Falling back to batch size 1")

        # Log to wandb
        if args.use_wandb and wandb.run is not None:
            wandb.log({
                "batch_search/safe_batch_size_success": False,
                "batch_search/fallback_batch_size": 1
            })
        return 1

    # Binary search
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        if mid_batch == optimal_batch:
            # Already tested this batch size
            min_batch = mid_batch + 1
            continue

        # Test this batch size
        grad_accum_steps = max(1, args.base_grad_accum // mid_batch)  # Ensure at least 1
        result = test_batch_size(model, tokenizer, train_dataset, eval_dataset, mid_batch, grad_accum_steps)

        logs = {
            "batch_search/tried_batch_size": mid_batch,
            "batch_search/gradient_accumulation": grad_accum_steps,
            "batch_search/effective_batch": mid_batch * grad_accum_steps,
            "batch_search/current_optimal": optimal_batch,
            "batch_search/min_bound": min_batch,
            "batch_search/max_bound": max_batch,
        }

        if result == "success":
            optimal_batch = mid_batch
            min_batch = mid_batch + 1
            logs["batch_search/batch_success"] = True
        else:
            # If failed, reduce the max batch size
            max_batch = mid_batch - 1
            logs.update({
                "batch_search/batch_success": False,
                "batch_search/oom_error": result == "oom",
            })

        # Log to wandb
        if args.use_wandb and wandb.run is not None:
            wandb.log(logs)

    logging.info(f"Maximum batch size found: {optimal_batch}")

    # Final batch search results to wandb
    if args.use_wandb and wandb.run is not None:
        wandb.log({
            "batch_search/final_batch_size": optimal_batch,
            "batch_search/final_grad_accum": max(1, args.base_grad_accum // optimal_batch),
            "batch_search/final_effective_batch": optimal_batch * max(1, args.base_grad_accum // optimal_batch),
            "batch_search/end_time": time.time()
        })

    return optimal_batch


def test_batch_size(model, tokenizer, train_dataset, eval_dataset, batch_size, grad_accum_steps=1):
    result = "fail"

    try:
        logging.info(f"Trying batch size: {batch_size} with gradient accumulation steps: {grad_accum_steps}")
        log_gpu_usage(log_to_wandb=True)

        # Test if this batch size works by running one training step
        training_args = get_training_args(batch_size, grad_accum_steps)
        trainer = PplxTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        trainer.train_dataloader = trainer.get_train_dataloader()
        batch = next(iter(trainer.train_dataloader))
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Try one forward and backward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # If successful, this batch size works
        logging.info(f"Batch size {batch_size} works!")
        result = "success"
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "not enough memory" in str(e):
            # If OOM error, try smaller batch size
            logging.info(f"OOM error at batch size {batch_size}")
            result = "oom"
        else:
            # For other errors, log and try smaller batch size
            logging.warning(f"Error at batch size {batch_size}: {e}")
            result = "fail"
    except Exception as e:
        # For other errors, log and try smaller batch size
        logging.warning(f"Error at batch size {batch_size}: {e}")
        result = "fail"

    loss = None
    del loss
    outputs = None
    del outputs
    batch = None
    del batch
    trainer = None
    del trainer
    training_args = None
    del training_args

    # Always clean up GPU memory after each test
    model.zero_grad(set_to_none=True)
    gc.collect()

    torch.cuda.empty_cache()

    # Brief pause to let GPU recover
    time.sleep(2)

    log_gpu_usage(log_to_wandb=True)

    return result

def main():
    """Main function to run the fine-tuning pipeline."""
    start_time = time.time()
    logging.info("Starting LLaMA fine-tuning with memory optimizations")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Log all arguments
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else None
        run_name = args.wandb_run_name if args.wandb_run_name else f"llama-ft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        logging.info(f"Initializing Weights & Biases with project: {args.wandb_project}, run name: {run_name}")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=wandb_tags,
            config=vars(args)  # Track all arguments as config
        )

        # Log hardware info to wandb
        if torch.cuda.is_available():
            wandb.config.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
            })

    try:
        # Create necessary directories
        create_directories(args.data_dir, args.output_dir, args.cache_dir)

        # Log initial GPU usage
        log_gpu_usage(log_to_wandb=args.use_wandb)

        # Load preprocessed files from the data directory
        logging.info("Step 1: Loading preprocessed text files...")
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
            train_files, test_files = split_train_test(txt_files, args.train_test_split)

            # Save the split information for future runs
            split_info = {
                "train_files": [os.path.basename(f) for f in train_files],
                "test_files": [os.path.basename(f) for f in test_files]
            }
            with open(split_info_path, 'w') as f:
                json.dump(split_info, f)

        logging.info(f"Training on {len(train_files)} files, testing on {len(test_files)} files")

        # Initialize model and tokenizer with memory optimizations
        logging.info("Step 2: Configuring model with memory optimizations...")
        model, tokenizer = configure_model_for_fine_tuning()
        log_gpu_usage()

        # Create datasets with caching
        logging.info("Step 3: Creating and tokenizing datasets with caching...")
        train_dataset, eval_dataset = prepare_datasets(
            train_files,
            test_files,
            tokenizer,
            cache_dir=args.cache_dir
        )
        logging.info(f"Train dataset size: {len(train_dataset)} samples")
        logging.info(f"Eval dataset size: {len(eval_dataset)} samples")

        # Find maximum batch size
        logging.info("Step 4: Finding maximum batch size...")
        max_batch_size = find_max_batch_size(model, tokenizer,
            train_dataset.select(range(256)),
            eval_dataset.select(range(256)),
        )
        log_gpu_usage()

        # Train the model with the optimal batch size
        logging.info(f"Step 5: Training model with batch size {max_batch_size}...")
        grad_accum_steps = max(1, args.base_grad_accum // max_batch_size)  # Ensure at least 1
        trainer, train_metrics, eval_metrics = train_model(model, tokenizer, train_dataset, eval_dataset, max_batch_size, grad_accum_steps)
        log_gpu_usage()

        # Evaluate the model
        logging.info("Step 6: Evaluating model...")
        perplexity = evaluate_perplexity(model, tokenizer, eval_dataset)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Format time for logging
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

        logging.info(f"Training complete in {time_str}!")
        logging.info(f"Maximum batch size: {max_batch_size}")
        logging.info(f"Gradient accumulation steps: {grad_accum_steps}")
        logging.info(f"Effective batch size: {max_batch_size * grad_accum_steps}")
        logging.info(f"Final perplexity: {perplexity:.2f}")

        # Final metrics to wandb
        if args.use_wandb and wandb.run is not None:
            wandb.log({
                "train/final_pplx": perplexity,
                "train/max_batch_size": max_batch_size,
                "train/grad_accum_steps": grad_accum_steps,
                "train/effective_batch_size": max_batch_size * grad_accum_steps,
                "train/total_time_seconds": elapsed_time
            })

            # Log final model as artifact
            model_artifact = wandb.Artifact(
                name=f"finetuned-llama-{wandb.run.id}",
                type="model",
                description="LoRA finetuned LLaMA model"
            )
            model_artifact.add_dir(args.output_dir)
            wandb.log_artifact(model_artifact)

        # Save training stats
        with open(os.path.join(args.output_dir, "training_stats.txt"), "w") as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maximum batch size: {max_batch_size}\n")
            f.write(f"Gradient accumulation steps: {grad_accum_steps}\n")
            f.write(f"Effective batch size: {max_batch_size * grad_accum_steps}\n")
            f.write(f"Final perplexity: {perplexity:.2f}\n")
            f.write(f"Total training time: {time_str}\n")
            # Save all arguments
            f.write("\nArguments:\n")
            for arg in vars(args):
                f.write(f"  {arg}: {getattr(args, arg)}\n")

        # Finish wandb run
        if args.use_wandb and wandb.run is not None:
            wandb.finish()

        return max_batch_size, grad_accum_steps, perplexity

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        # Log the error to wandb
        if args.use_wandb and wandb.run is not None:
            wandb.log({"error": str(e)})
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main()