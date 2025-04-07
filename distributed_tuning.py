import os
import json
import torch
import time
import math
import random
import logging
import argparse
import deepspeed
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/distributed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

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
    parser.add_argument("--parallelism_type", type=str, choices=["dp", "tp", "pp"], default="dp",
                        help="Type of parallelism to use (data, tensor, or pipeline)")
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

def create_deepspeed_config(args):
    """Create or update DeepSpeed config based on args"""
    # If config file exists, load it
    if os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
    else:
        # Create a basic DeepSpeed config
        ds_config = {
            "train_batch_size": args.per_device_batch_size * 2,  # 2 GPUs
            "train_micro_batch_size_per_gpu": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "steps_per_print": args.logging_steps,
        }

        # Add ZeRO optimization for Data Parallelism
        if args.parallelism_type == "dp":
            ds_config["zero_optimization"] = {
                "stage": 1,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8
            }

        # Add precision settings
        if args.use_fp16:
            ds_config["fp16"] = {"enabled": True}
        elif args.use_bf16:
            ds_config["bf16"] = {"enabled": True}

        # Add optimization settings
        ds_config["gradient_clipping"] = 1.0
        ds_config["wall_clock_breakdown"] = True
        ds_config["zero_allow_untested_optimizer"] = True

    return ds_config

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

# Configure model with memory optimizations
def configure_model(args):
    """Configure LLaMA model with LoRA and quantization"""
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
        device_map="auto",  # DeepSpeed will handle device placement
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

# Define training loop
def train(args, model, train_dataset, eval_dataset, tokenizer):
    """Train the model with DeepSpeed"""
    # DeepSpeed engine and data loader setup
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Create DeepSpeed config
    ds_config = create_deepspeed_config(args)

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        config=ds_config
    )

    # Track elapsed time for each epoch
    epoch_times = []

    # Get total training steps
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )

    total_steps = len(train_dataloader)
    logging.info(f"Total training steps per epoch: {total_steps}")

    # Training loop
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model_engine.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            # Forward pass
            outputs = model_engine(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})

            # Log loss
            if step % args.logging_steps == 0 and args.local_rank == 0:
                logging.info(f"Epoch: {epoch+1}/{args.num_epochs}, Step: {step}/{total_steps}, Loss: {loss.item():.4f}")

            # Evaluate periodically
            if step % args.eval_steps == 0 and args.local_rank == 0 and step > 0:
                eval_loss, perplexity = evaluate(args, model_engine, eval_dataset, tokenizer)
                logging.info(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                model_engine.train()  # Back to training mode

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        # Evaluate at the end of each epoch
        if args.local_rank == 0:
            eval_loss, perplexity = evaluate(args, model_engine, eval_dataset, tokenizer)
            logging.info(f"Epoch {epoch+1} completed - Time: {epoch_time:.2f}s, Loss: {total_loss/total_steps:.4f}, Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Save model
        if args.local_rank == 0:
            output_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            model_engine.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")

    # Log average epoch time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    logging.info(f"Average time per epoch: {avg_epoch_time:.2f}s")

    return model_engine, epoch_times

# Evaluate model with perplexity
def evaluate(args, model, eval_dataset, tokenizer):
    """Evaluate the model using perplexity metric"""
    model.eval()

    # Create data collator and dataloader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=data_collator,
        shuffle=False
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

def main():
    """Main function for distributed training"""
    # Parse arguments
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize distributed environment
    deepspeed.init_distributed()

    # Get local rank from environment variable (set by DeepSpeed launcher)
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    # Create output directory
    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

    logging.info(f"Starting distributed training with {args.parallelism_type} parallelism")
    logging.info(f"Local rank: {args.local_rank}")

    # Configure model
    model, tokenizer = configure_model(args)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(args, tokenizer)
    logging.info(f"Train dataset size: {len(train_dataset)} samples")
    logging.info(f"Eval dataset size: {len(eval_dataset)} samples")

    # Train the model
    start_time = time.time()
    model, epoch_times = train(args, model, train_dataset, eval_dataset, tokenizer)
    total_time = time.time() - start_time

    # Final evaluation
    if args.local_rank == 0:
        eval_loss, perplexity = evaluate(args, model, eval_dataset, tokenizer)
        logging.info(f"Final evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Log training statistics
        logging.info(f"Training completed in {total_time:.2f} seconds")
        for i, epoch_time in enumerate(epoch_times):
            logging.info(f"Epoch {i+1} time: {epoch_time:.2f} seconds")

        # Save training stats to file
        stats_file = os.path.join(args.output_dir, "training_stats.txt")
        with open(stats_file, "w") as f:
            f.write(f"Parallelism type: {args.parallelism_type}\n")
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {total_time:.2f} seconds\n")
            f.write(f"Average time per epoch: {sum(epoch_times)/len(epoch_times):.2f} seconds\n")
            f.write(f"Per-device batch size: {args.per_device_batch_size}\n")
            f.write(f"Total batch size: {args.per_device_batch_size * 2}\n")  # 2 GPUs
            f.write(f"Final perplexity: {perplexity:.2f}\n")

    logging.info("Distributed training complete!")

if __name__ == "__main__":
    main()