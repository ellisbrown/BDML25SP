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

# Try to import pipeline parallelism
try:
    from torch.distributed.pipeline.sync import Pipe
    has_pipe = True
except ImportError:
    has_pipe = False

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

# Helper functions to split model for pipeline parallelism
def split_model_for_pipeline(model, num_partitions=2):
    """
    Split the LLaMA model into partitions for pipeline parallelism
    """
    import torch.nn as nn

    # For transformer models, each layer is a good partition point
    num_layers = len(model.model.layers)
    layers_per_partition = num_layers // num_partitions

    partitions = []

    # Create the first partition with embeddings and first set of layers
    class FirstPartition(nn.Module):
        def __init__(self, model, end_idx):
            super().__init__()
            self.embed_tokens = model.model.embed_tokens
            self.layers = nn.ModuleList(model.model.layers[:end_idx])

        def forward(self, x, attention_mask=None):
            hidden_states = self.embed_tokens(x)
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            return hidden_states, attention_mask

    # Create middle partitions (if more than 2 partitions)
    class MiddlePartition(nn.Module):
        def __init__(self, model, start_idx, end_idx):
            super().__init__()
            self.layers = nn.ModuleList(model.model.layers[start_idx:end_idx])

        def forward(self, args):
            hidden_states, attention_mask = args
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            return hidden_states, attention_mask

    # Create the last partition with final layers and LM head
    class LastPartition(nn.Module):
        def __init__(self, model, start_idx):
            super().__init__()
            self.layers = nn.ModuleList(model.model.layers[start_idx:])
            self.norm = model.model.norm
            self.lm_head = model.lm_head

        def forward(self, args):
            hidden_states, attention_mask = args
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            return logits

    # Create partitions based on number requested
    if num_partitions == 2:
        partitions.append(FirstPartition(model, layers_per_partition))
        partitions.append(LastPartition(model, layers_per_partition))
    else:
        # Handle more partitions if needed
        partitions.append(FirstPartition(model, layers_per_partition))

        for i in range(1, num_partitions - 1):
            start_idx = i * layers_per_partition
            end_idx = (i + 1) * layers_per_partition
            partitions.append(MiddlePartition(model, start_idx, end_idx))

        partitions.append(LastPartition(model, (num_partitions - 1) * layers_per_partition))

    return partitions

# Pipeline Parallel Training function
def train_with_pipeline_parallelism(args):
    """
    Implementation of Pipeline Parallelism
    """
    if not has_pipe:
        raise ImportError("Pipeline parallelism requires PyTorch's pipeline API")

    # Initialize process group for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    logging.info(f"Initializing process: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # Initialize the distributed environment
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # Configure model and tokenizer
    model, tokenizer = configure_model_for_fine_tuning(args)

    # Split model for pipeline parallelism
    model_partitions = split_model_for_pipeline(model, world_size)

    # Assign partitions to devices and create pipeline
    devices = [torch.device(f"cuda:{i}") for i in range(world_size)]
    for i, partition in enumerate(model_partitions):
        partition.to(devices[i])

    # Create pipeline model
    chunks = 4  # Number of micro-batches for pipeline parallelism
    pipeline_model = Pipe(
        nn.Sequential(*model_partitions),
        chunks=chunks,
        checkpoint="never"  # Don't use activation checkpointing for pipeline
    )

    # Define a custom forward pass for the pipeline model
    class PipelineModel(nn.Module):
        def __init__(self, pipeline_model):
            super().__init__()
            self.pipeline_model = pipeline_model

        def forward(self, input_ids, attention_mask=None, labels=None):
            # Forward pass through pipeline
            logits = self.pipeline_model(input_ids)

            # Compute loss if labels are provided
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                return {"loss": loss, "logits": logits}

            return {"logits": logits}

    # Create the pipeline model
    pipelined_model = PipelineModel(pipeline_model)

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
        from llm_fine_tuning import split_train_test
        train_files, test_files = split_train_test(txt_files, args.train_test_split)

    train_dataset, eval_dataset = prepare_datasets(
        train_files, test_files, tokenizer, args.cache_dir
    )

    # Configure data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Get training arguments, adjust batch size to account for chunking
    adjusted_batch_size = args.batch_size // chunks
    if adjusted_batch_size < 1:
        adjusted_batch_size = 1
        logging.warning(f"Batch size too small for chunks. Setting to 1, but this may cause issues.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=adjusted_batch_size,
        per_device_eval_batch_size=adjusted_batch_size,
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
        model=pipelined_model,
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

        # Run evaluation on original model to get perplexity
        perplexity = evaluate_perplexity(model, tokenizer, eval_dataset)

        logging.info(f"Final perplexity: {perplexity:.2f}")

        # Save the original model (only on main process)
        # Note: saving the pipeline model would be complex, so we save the original model
        model.save_pretrained(args.output_dir)

        # Save training stats
        with open(os.path.join(args.output_dir, "training_stats.txt"), "w") as f:
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total training time: {total_train_time:.2f} seconds\n")
            f.write(f"Average epoch time: {sum(timing_callback.epoch_times)/len(timing_callback.epoch_times):.2f} seconds\n")
            f.write(f"Final perplexity: {perplexity:.2f}\n")
            f.write(f"World size: {world_size}\n")
            f.write(f"Parallelism Strategy: Pipeline Parallelism\n")
            f.write(f"Number of chunks (micro-batches): {chunks}\n")

            # Save individual epoch times
            f.write("\nEpoch Times:\n")
            for i, epoch_time in enumerate(timing_callback.epoch_times):
                f.write(f"Epoch {i+1}: {epoch_time:.2f} seconds\n")

            # Save all arguments
            f.write("\nArguments:\n")
            for arg in vars(args):
                f.write(f"  {arg}: {getattr(args, arg)}\n")

    # Clean up process group
    if world_size > 1:
        dist.destroy_process_group()

    return timing_callback.epoch_times if timing_callback.epoch_times else [total_train_time]

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Add batch size argument if not present (for backward compatibility)
    if not hasattr(args, "batch_size"):
        args.batch_size = 16  # Default batch size for distributed training

    # Train with pipeline parallelism
    epoch_times = train_with_pipeline_parallelism(args)

    # Log final stats
    logging.info(f"Pipeline Parallelism Training Complete!")
    logging.info(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds")
