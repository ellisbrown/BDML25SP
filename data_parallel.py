print("data_parallel.py")
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
import logging
from datetime import datetime
from functools import partial

# Import your existing functions
from llm_fine_tuning import (
    parse_args, prepare_datasets,
    evaluate_perplexity, log_gpu_usage, get_training_args,
    split_train_test
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

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

# Modified configuration function
def configure_model_for_distributed(args, local_rank):
    """Configure model with quantization for distributed training"""
    logging.info("Configuring model for distributed training")

    # Configure quantization
    quantization_config = None
    if args.load_in_4bit:
        logging.info("Loading in 4-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=args.use_double_quant,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif args.load_in_8bit:
        logging.info("Loading in 8-bit precision")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    # Attention implementation
    attn_implementation = None
    if args.flash_attention:
        logging.info("Using Flash Attention")
        attn_implementation = "flash_attention_2"
    elif args.sdpa_attention:
        logging.info("Using SDPA Attention")
        attn_implementation = "sdpa"

    # Load tokenizer
    logging.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Set device map for proper distribution
    # For data parallelism, we want the model on the specific local GPU
    device_map = {"": local_rank} if local_rank != -1 else "auto"

    # Load model with quantization - IMPORTANT: do not use DDP here yet
    logging.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        attn_implementation=attn_implementation if attn_implementation else None,
        trust_remote_code=True,
    )

    # Prepare model for k-bit training
    logging.info("Preparing model for k-bit training")
    model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        logging.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    # Apply LoRA to the model
    logging.info("Applying LoRA to model")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Don't wrap in DDP here - we'll do it separately later if needed
    return model, tokenizer

# Data Parallel Training function
def train_with_data_parallelism(args):
    """
    Implementation of Data Parallelism using PyTorch DDP and Hugging Face Accelerate
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

    # Configure model and tokenizer (already handles device placement)
    model, tokenizer = configure_model_for_distributed(args, local_rank)

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
        train_files, test_files, tokenizer, args.cache_dir, args
    )

    # Use DistributedSampler for data parallelism
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if local_rank != -1 else None

    # Configure data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Now wrap with DDP if needed (AFTER LoRA is applied)
    if local_rank != -1:
        # Convert any BitsAndBytes modules if needed
        # Find modules not compatible with DDP
        non_ddp_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                non_ddp_params.append(name)

        if non_ddp_params:
            logging.info(f"Found {len(non_ddp_params)} parameters not compatible with DDP. These will not be synced.")

        # Create DDP model with appropriate settings
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # Get training arguments with added distributed settings
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

    # If using distributed, set the train sampler
    if train_sampler is not None:
        trainer.train_dataloader = trainer.get_train_dataloader()

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
        perplexity = evaluate_perplexity(model, tokenizer, eval_dataset, args)

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
            f.write(f"Parallelism Strategy: Data Parallelism\n")

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

    # Train with data parallelism
    epoch_times = train_with_data_parallelism(args)

    # Log final stats
    logging.info(f"Data Parallelism Training Complete!")
    logging.info(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f} seconds")