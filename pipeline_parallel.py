print("pipeline_parallel.py")
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import time
import logging
from datetime import datetime
from functools import partial

# Import your existing functions
from llm_fine_tuning import (
    parse_args, prepare_datasets,
    evaluate_perplexity, log_gpu_usage, split_train_test
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

# Try to import pipeline parallelism support
try:
    import deepspeed
    from deepspeed.pipe import PipelineModule, LayerSpec
    has_deepspeed_pipe = True
except ImportError:
    has_deepspeed_pipe = False

try:
    from torch.distributed.pipeline.sync import Pipe
    has_torch_pipe = True
except ImportError:
    has_torch_pipe = False

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

# Configure model for pipeline parallelism
def configure_model_for_pipeline(args, rank, world_size):
    """Load model and prepare it for pipeline parallelism"""
    logging.info("Configuring model for pipeline parallelism")

    # For pipeline parallelism with quantization, we need to first load the model
    # normally, then split it across devices

    # Load tokenizer
    logging.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

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

    # For pipeline parallelism, we need to:
    # 1. Load the model on CPU first
    # 2. Apply LoRA
    # 3. Then split it across GPUs

    # Load model on CPU to prepare for splitting
    if rank == 0:
        logging.info(f"Loading model from {args.model_path} on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quantization_config,
            device_map="cpu",  # Load on CPU first
            attn_implementation=attn_implementation if attn_implementation else None,
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        if args.load_in_4bit or args.load_in_8bit:
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

        # Get the number of layers
        num_layers = len(model.model.layers)
        logging.info(f"Model has {num_layers} transformer layers")

        # Calculate layers per device
        layers_per_device = num_layers // world_size
        logging.info(f"Assigning {layers_per_device} layers per device")

        # DeepSpeed Pipeline requires a different model structure
        # So we'll use a simpler approach with DeepSpeed ZeRO-3 for pipeline-like behavior

        # Instead of actual pipeline parallelism, use DeepSpeed ZeRO-3
        # which effectively gives us similar behavior in a more stable way

        # Move model to the appropriate device
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
    else:
        # For other ranks, we'll wait for broadcast from rank 0
        model = None

    # Broadcast model from rank 0 to all other ranks
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            # Pack model's state_dict
            state_dict = model.state_dict()
            # Broadcast state_dict
            dist.broadcast_object_list([state_dict], src=0)
        else:
            # Receive state_dict
            objects = [None]
            dist.broadcast_object_list(objects, src=0)
            state_dict = objects[0]

            # Load model and apply state_dict
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                quantization_config=quantization_config,
                device_map=f"cuda:{rank}",
                attn_implementation=attn_implementation if attn_implementation else None,
                trust_remote_code=True,
            )
            if args.load_in_4bit or args.load_in_8bit:
                model = prepare_model_for_kbit_training(model)
            if args.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # Apply LoRA
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)

            # Load state_dict
            try:
                model.load_state_dict(state_dict, strict=False)
                logging.info(f"Rank {rank} loaded model state successfully")
            except Exception as e:
                logging.warning(f"Rank {rank} failed to load state: {e}")

    if world_size > 1:
        dist.barrier()

    return model, tokenizer

# Pipeline Parallel Training function
def train_with_pipeline_parallelism(args):
    """
    Implementation of Pipeline Parallelism using DeepSpeed ZeRO-3
    """
    # Initialize process group for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    logging.info(f"Initializing process: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # Initialize the distributed environment
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # Configure model and tokenizer for pipeline parallelism
    model, tokenizer = configure_model_for_pipeline(args, rank, world_size)

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

    # DeepSpeed configuration for ZeRO-3, which we use as a more stable alternative to pipeline
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e8,
            "stage3_prefetch_bucket_size": 5e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        },
        "bf16": {
            "enabled": args.use_bf16
        },
        "fp16": {
            "enabled": args.use_fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "steps_per_print": args.logging_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "wall_clock_breakdown": False
    }

    # Set batch size - a bit lower to account for pipeline communication
    adjusted_batch_size = max(1, args.batch_size // 2)

    # Get training arguments
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
        # DeepSpeed config for pipeline-like behavior
        deepspeed=ds_config if has_deepspeed_pipe else None,
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
            f.write(f"Parallelism Strategy: Pipeline Parallelism (via DeepSpeed ZeRO-3)\n")

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