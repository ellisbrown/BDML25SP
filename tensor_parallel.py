import os
import torch
import torch.distributed as dist
import time
import logging
from datetime import datetime
from functools import partial

# Import your existing functions
from llm_fine_tuning import (
    parse_args, configure_model_for_fine_tuning, prepare_datasets,
    evaluate_perplexity, log_gpu_usage, get_training_args
)

from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# Try to import tensor parallelism from different libraries
try:
    from torch.distributed.tensor.parallel import parallelize_module
    has_native_tp = True
except ImportError:
    has_native_tp = False
    
try:
    import deepspeed
    has_deepspeed = True
except ImportError:
    has_deepspeed = False

class TimingCallback(TrainerCallback):
    """Callback to time each epoch"""
    def __init__(self):
        self.epoch_start_time = None
        self.epoch_times = []
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        logging.info(f"Starting epoch {state.epoch}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            logging.info(f"Epoch {state.epoch} completed in {epoch_time:.2f} seconds")
            
            # Log to wandb if enabled
            if args.report_to == ["wandb"]:
                import wandb
                wandb.log({"train/epoch_time": epoch_time})

# Helper function to identify modules for tensor parallelism
def get_layer_modules_for_tp(model):
    """
    Identifies attention modules for tensor parallelism
    """
    tp_modules = []
    
    # Find all query, key, value projection layers
    for name, module in model.named_modules():
        if any(substr in name for substr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            tp_modules.append(name)
    
    return tp_modules

# Functions to apply tensor parallelism
def apply_native_tensor_parallel(model, devices):
    """Apply tensor parallelism using PyTorch native API"""
    if not has_native_tp:
        raise ImportError("Native tensor parallelism requires PyTorch 2.0+")
    
    tp_modules = get_layer_modules_for_tp(model)
    tp_size = len(devices)
    
    # Create the tensor parallel environment
    pg = dist.new_group(ranks=list(range(tp_size)))
    
    # Use column parallelism for the model
    parallelize_module(
        module=model,
        device_mesh=devices,
        parallelize_plan="column",
        tp_mesh_dim=0
    )
    
    return model

def apply_custom_tensor_parallel(model, rank, world_size):
    """
    Implements a simple custom tensor parallelism by manually 
    sharding specific layers across GPUs
    """
    # For each transformer layer, split attention matrices
    # This is a simplified version - actual implementation would require
    # deeper integration with model architecture
    for name, param in model.named_parameters():
        if any(substr in name for substr in ['q_proj', 'k_proj', 'v_proj']):
            # Split parameter along output dimension
            split_size = param.size(0) // world_size
            start_idx = rank * split_size
            end_idx = (rank + 1) * split_size if rank < world_size - 1 else param.size(0)
            
            # Only keep the relevant portion of the parameter
            with torch.no_grad():
                new_param = param[start_idx:end_idx].clone()
                param.data = new_param
    
    return model

# Tensor Parallel Training function
def train_with_tensor_parallelism(args):
    """
    Implementation of Tensor Parallelism
    """
    # Initialize process group for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    logging.info(f"Initializing process: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # Initialize the distributed environment
    if local_rank != -1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if local_rank != -1 else "cuda:0")
    
    # Configure model and tokenizer
    model, tokenizer = configure_model_for_fine_tuning(args)
    
    # Apply tensor parallelism to the model
    if has_native_tp:
        logging.info("Using PyTorch native tensor parallelism")
        devices = [torch.device(f"cuda:{i}") for i in range(world_size)]
        model = apply_native_tensor_parallel(model, devices)
    else:
        logging.info("Using custom tensor parallelism implementation")
        model = apply_custom_tensor_parallel(model, rank, world_size)
    
    # Load datasets
    from glob import glob
    import json
    
    # Load preprocessed files from the data directory
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
        txt_files = glob(os.path.join(args.data_dir, "*.txt"))
        train_files, test_files = split_train_test(txt_files, args.train_test_split)
    
    train_dataset, eval_dataset = prepare_datasets(
        train_files, test_files, tokenizer, args.cache_dir
    )
    
    # Configure data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Get training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        # Distributed training specific
        local_rank=local_rank,
        fp16=args.use_fp16,
        bf16=args.use_bf16,
        optim="paged_adamw_8bit" if args.use_8bit_optimizer else "adamw_torch",
        # Logging and reporting
        report_to=["wandb"] if args.use_wandb else ["none"],
        run_name=args.wandb_run_name if args.use_wandb else None,
    )
    
    # Create timing callback
    timing_callback = TimingCallback()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[timing_callback],
    )
    
    # Start training
    start_time = time.time()
    train_result = trainer.train()
    total_train_time = time.time() - start_time
    
    # Log training results only on main process
    if rank == 0:
        logging.info(f"Training completed in {total_train_time:.2f} seconds")
        logging.info(f"Average epoch time: {sum(timing_callback.epoch_times)/len(timing_callback.epoch_times):.2f} seconds")
        
        for i, epoch_time in enumerate(timing_callback.epoch_times):
            logging.info(f"Epoch {i+1} time: {epoch_time:.2f} seconds")
        
        # Run evaluation
        eval_metrics = trainer.evaluate()
        perplexity = evaluate_perplexity(model, tokenizer, eval_dataset)
        
        logging.info(f"Final perplexity: {perplexity:.2f}")
        logging.info(f"Final evaluation metrics: {eval_metrics}")
        
        # Save the model (only on main process)
        trainer.save_model(args.output_dir)
        
        # Save training stats
        with open(os.path.join(args.output_dir, "training_stats.txt"), "w") as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {total_train_time:.2f} seconds\n")
            f.write(f"Average epoch time: {sum(timing_callback.epoch_times)/len(timing_callback.epoch_times):.2f} seconds\n")
            f.write(f"Final perplexity: {perplexity:.2f}\n")
            f.write(f"World size: {world_size}\n")
            f.write(f"Parallelism Strategy: Tensor Parallelism\n")
            
            # Save individual epoch times
            f.write("\nEpoch Times:\n")
            for i, epoch_time in enumerate(timing_callback.epoch_times):
                f.write(f"Epoch {i+1}: {epoch_time:.2f} seconds\n")
            
            # Save all arguments
            f.write("\nArguments:\n")
            for arg in vars(args):
                f.write(f"  {arg}: {getattr(args, arg)}\n")
    
    # Clean up process group
    if local_rank != -1:
        dist.destroy_process_group()
    
    return timing_callback.epoch_times if timing_callback.epoch_times else [total_train_time]

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Add batch size argument if not present (for backward compatibility)
    if not hasattr(args, "batch_size"):
        args.batch_size = 16  # Default batch size for distributed training
    
    # Train with tensor parallelism
    epoch_times = train_with_tensor_parallelism(args)
    
    # Log final stats
    logging.info(f"Tensor Parallelism Training Complete!")
    logging.info(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds")
