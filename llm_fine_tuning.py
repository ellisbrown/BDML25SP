"""
LLaMA Fine-Tuning with Memory Optimizations
- Data processing from climate document PDFs
- LoRA + Quantization + Gradient Accumulation & Checkpointing
- Maximizing batch size on a single GPU
"""

import os
import glob
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
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

# Constants
MODEL_PATH = "/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B"  # Path to LLaMA 3B model
DATA_PATH = "/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"   # Path to climate PDFs
OUTPUT_DIR = "./llama-finetuned"
TXT_DIR = "./extracted_texts"
TRAIN_TEST_SPLIT = 0.9  # 90% for training, 10% for testing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# Create directory for extracted texts if it doesn't exist
os.makedirs(TXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to monitor GPU usage
def log_gpu_usage():
    """Log GPU memory usage."""
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            if int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) == gpu.id:
                gpu_info.append(f"GPU {gpu.id}: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")

        if gpu_info:
            logging.info("GPU Memory Usage: " + ", ".join(gpu_info))
    except Exception as e:
        logging.warning(f"Failed to log GPU usage: {e}")

    # Log RAM usage
    ram = psutil.virtual_memory()
    logging.info(f"RAM Usage: {ram.used/1e9:.1f}GB/{ram.total/1e9:.1f}GB ({ram.percent:.1f}%)")

# STEP 1: Data Processing - Extract text from PDFs
def extract_text_from_pdfs(pdf_dir, output_dir):
    """Extract text from all PDFs in the directory and save to txt files."""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

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
            print(f"Error processing {pdf_path}: {e}")

    return glob.glob(os.path.join(output_dir, "*.txt"))

# STEP 2: Split data into train and test sets
def split_train_test(file_list, train_ratio=0.9):
    """Split files into training and testing sets."""
    random.shuffle(file_list)
    split_idx = int(len(file_list) * train_ratio)
    train_files = file_list[:split_idx]
    test_files = file_list[split_idx:]
    return train_files, test_files

# STEP 3: Create datasets for training and evaluation
def create_dataset(file_paths, tokenizer, max_length=512):
    """Create dataset from text files."""
    texts = []

    for file_path in tqdm(file_paths, desc="Creating dataset"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            # Split text into chunks of max_length tokens
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens), max_length):
                chunk = tokens[i:i + max_length]
                if len(chunk) > 100:  # Skip very short chunks
                    texts.append(tokenizer.decode(chunk))

    return Dataset.from_dict({"text": texts})

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the text examples."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True
    )

# STEP 4: Configure model with memory optimizations
def configure_model_for_fine_tuning():
    """Configure LLaMA model with memory optimizations."""

    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Load the model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,                     # Rank dimension
        lora_alpha=32,           # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Apply to attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

# STEP 5: Configure training arguments
def get_training_args(batch_size, gradient_accumulation_steps):
    """Get training arguments with memory optimizations."""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        fp16=True,                         # Use mixed precision training
        logging_steps=10,
        num_train_epochs=3,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        optim="paged_adamw_8bit",         # Use 8-bit Adam optimizer for memory efficiency
    )

# STEP 6: Train the model
def train_model(model, tokenizer, train_dataset, eval_dataset, batch_size=1, gradient_accumulation_steps=8):
    """Train the model with memory optimizations."""
    print(f"Training with batch size: {batch_size}, grad accum: {gradient_accumulation_steps}")

    # Use DataCollator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Configure training arguments
    training_args = get_training_args(batch_size, gradient_accumulation_steps)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(OUTPUT_DIR)
    return trainer

# STEP 7: Evaluate the model using perplexity
def evaluate_perplexity(model, tokenizer, test_dataset):
    """Evaluate the model using perplexity metric."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for item in tqdm(test_dataset, desc="Evaluating perplexity"):
            inputs = tokenizer(item["text"], return_tensors="pt").to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    # Calculate perplexity
    perplexity = math.exp(total_loss / total_tokens)
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity

# STEP 8: Find maximum batch size through binary search
def find_max_batch_size(model, tokenizer, train_dataset, eval_dataset):
    """Binary search to find maximum batch size."""
    min_batch = 1
    max_batch = 128  # Start with a high upper bound for H100
    optimal_batch = 1

    logging.info("Starting binary search for maximum batch size")
    logging.info(f"Initial search range: {min_batch} to {max_batch}")

    # First try a conservative batch size to establish a baseline
    safe_batch_size = 8
    logging.info(f"Testing a safe batch size of {safe_batch_size} first")
    try:
        grad_accum_steps = max(1, 32 // safe_batch_size)
        training_args = get_training_args(safe_batch_size, grad_accum_steps)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset.select(range(min(20, len(train_dataset)))),
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )
        trainer.train_dataloader = trainer.get_train_dataloader()
        batch = next(iter(trainer.train_dataloader))
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Try one forward and backward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # If successful, we know this works
        optimal_batch = safe_batch_size
        min_batch = safe_batch_size
        logging.info(f"Safe batch size {safe_batch_size} works! Continuing search...")
    except Exception as e:
        logging.warning(f"Even safe batch size {safe_batch_size} failed: {e}")
        logging.info("Falling back to batch size 1")
        return 1

    # Release memory before continuing
    torch.cuda.empty_cache()

    # Binary search
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        if mid_batch == optimal_batch:
            # Already tested this batch size
            min_batch = mid_batch + 1
            continue

        grad_accum_steps = max(1, 32 // mid_batch)  # Ensure at least 1
        try:
            logging.info(f"Trying batch size: {mid_batch} with gradient accumulation steps: {grad_accum_steps}")
            log_gpu_usage()

            # Test if this batch size works by running one training step
            training_args = get_training_args(mid_batch, grad_accum_steps)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset.select(range(min(20, len(train_dataset)))),
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
            optimal_batch = mid_batch
            logging.info(f"Batch size {mid_batch} works!")
            min_batch = mid_batch + 1
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "not enough memory" in str(e):
                # If OOM error, try smaller batch size
                logging.info(f"OOM error at batch size {mid_batch}")
                max_batch = mid_batch - 1
            else:
                # For other errors, log and try smaller batch size
                logging.warning(f"Error at batch size {mid_batch}: {e}")
                max_batch = mid_batch - 1

        # Always clean up GPU memory after each test
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        # Brief pause to let GPU recover
        time.sleep(2)

    logging.info(f"Maximum batch size found: {optimal_batch}")
    return optimal_batch

def main():
    """Main function to run the fine-tuning pipeline."""
    start_time = time.time()
    logging.info("Starting LLaMA fine-tuning with memory optimizations")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device count: {torch.cuda.device_count()}")
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        # Log initial GPU usage
        log_gpu_usage()

        # Extract text from PDFs
        logging.info("Step 1: Extracting text from PDFs...")
        txt_files = extract_text_from_pdfs(DATA_PATH, TXT_DIR)

        # Split data
        logging.info("Step 2: Splitting into train and test sets...")
        train_files, test_files = split_train_test(txt_files, TRAIN_TEST_SPLIT)
        logging.info(f"Training on {len(train_files)} files, testing on {len(test_files)} files")

        # Initialize model and tokenizer with memory optimizations
        logging.info("Step 3: Configuring model with memory optimizations...")
        model, tokenizer = configure_model_for_fine_tuning()
        log_gpu_usage()

        # Create datasets
        logging.info("Step 4: Creating datasets...")
        train_dataset = create_dataset(train_files, tokenizer)
        eval_dataset = create_dataset(test_files, tokenizer)
        logging.info(f"Train dataset size: {len(train_dataset)} samples")
        logging.info(f"Eval dataset size: {len(eval_dataset)} samples")

        # Find maximum batch size
        logging.info("Step 5: Finding maximum batch size...")
        max_batch_size = find_max_batch_size(model, tokenizer, train_dataset, eval_dataset)
        log_gpu_usage()

        # Train the model with the optimal batch size
        logging.info(f"Step 6: Training model with batch size {max_batch_size}...")
        grad_accum_steps = max(1, 32 // max_batch_size)  # Ensure at least 1
        trainer = train_model(model, tokenizer, train_dataset, eval_dataset, max_batch_size, grad_accum_steps)
        log_gpu_usage()

        # Evaluate the model
        logging.info("Step 7: Evaluating model...")
        perplexity = evaluate_perplexity(model, tokenizer, eval_dataset)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"Training complete in {int(hours)}h {int(minutes)}m {int(seconds)}s!")
        logging.info(f"Maximum batch size: {max_batch_size}")
        logging.info(f"Gradient accumulation steps: {grad_accum_steps}")
        logging.info(f"Effective batch size: {max_batch_size * grad_accum_steps}")
        logging.info(f"Final perplexity: {perplexity:.2f}")

        # Save training stats
        with open(os.path.join(OUTPUT_DIR, "training_stats.txt"), "w") as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Maximum batch size: {max_batch_size}\n")
            f.write(f"Gradient accumulation steps: {grad_accum_steps}\n")
            f.write(f"Effective batch size: {max_batch_size * grad_accum_steps}\n")
            f.write(f"Final perplexity: {perplexity:.2f}\n")
            f.write(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")

        return max_batch_size, grad_accum_steps, perplexity

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()