print("distributed_training_main.py")
import os
import argparse
import logging
import time
import torch
from datetime import datetime


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/distributed_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

def get_available_gpus():
    """Get the number of available GPUs"""
    return torch.cuda.device_count()

def parse_args_distributed():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Distributed LLM fine-tuning")

    # Basic paths
    parser.add_argument("--model_path", type=str, default="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted",
                        help="Path to the LLaMA 3B model")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed text data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/llama-finetuned-distributed",
                        help="Directory to save fine-tuned model and outputs")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache",
                        help="Directory to cache datasets")

    # Training parameters
    parser.add_argument("--train_test_split", type=float, default=0.9,
                        help="Ratio of train/test split (e.g., 0.9 for 90% training)")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU")

    # Distributed training parameters
    parser.add_argument("--parallelism_type", type=str, choices=["data", "tensor", "pipeline", "all"],
                        default="data", help="Type of parallelism to use")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (defaults to all available)")

    # Memory optimization parameters (keeping these from Assignment 1)
    parser.add_argument("--flash_attention", action="store_true", default=True,
                        help="Use Flash Attention for training")
    parser.add_argument("--sdpa_attention", action="store_true", default=True,
                        help="Use SDPA Attention for training")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated list of modules to apply LoRA to")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit precision")
    parser.add_argument("--load_in_8bit", action="store_true", default=False,
                        help="Load model in 8-bit precision")
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--use_bf16", action="store_true", default=False,
                        help="Use mixed precision training (BF16)")
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
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="llama-finetuning-distributed",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity/username")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Custom name for this run in W&B")
    parser.add_argument("--wandb_tags", type=str, default="",
                        help="Comma-separated list of tags for W&B run")

    args = parser.parse_args()

    print(f"Arguments: {args}")

    # Convert comma-separated target modules to list
    args.lora_target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else []

    # Set number of GPUs if not specified
    if args.num_gpus is None:
        args.num_gpus = get_available_gpus()
        logging.info(f"Using all available GPUs: {args.num_gpus}")
    else:
        if args.num_gpus > get_available_gpus():
            logging.warning(f"Requested {args.num_gpus} GPUs but only {get_available_gpus()} available")
            args.num_gpus = get_available_gpus()

    return args

def run_distributed_training(args):
    """Run the appropriate distributed training based on the parallelism type"""

    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Print GPU information
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Number of available GPUs: {get_available_gpus()}")
    logging.info(f"Using {args.num_gpus} GPUs")

    # Initialize wandb if enabled
    if args.use_wandb:
        import wandb

        wandb_tags = args.wandb_tags.split(",") if args.wandb_tags else []
        wandb_tags.append(args.parallelism_type)

        run_name = args.wandb_run_name if args.wandb_run_name else f"llama-dist-{args.parallelism_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=wandb_tags,
            config=vars(args)
        )

    # Set environment variables for distributed training
    os.environ["WORLD_SIZE"] = str(args.num_gpus)

    # Create output directory for specific parallelism type
    parallelism_output_dir = os.path.join(args.output_dir, args.parallelism_type)
    os.makedirs(parallelism_output_dir, exist_ok=True)
    args.output_dir = parallelism_output_dir

    start_time = time.time()

    # Run appropriate parallelism strategy
    if args.parallelism_type == "data" or args.parallelism_type == "all":
        from data_parallel import train_with_data_parallelism

        logging.info("Running Data Parallelism")
        data_output_dir = os.path.join(args.output_dir, "data_parallel")
        os.makedirs(data_output_dir, exist_ok=True)

        # Save args
        data_args = args
        data_args.output_dir = data_output_dir

        # Run data parallelism
        data_epoch_times = train_with_data_parallelism(data_args)

        logging.info(f"Data Parallelism - Average epoch time: {sum(data_epoch_times)/len(data_epoch_times):.2f} seconds")

    if args.parallelism_type == "tensor" or args.parallelism_type == "all":
        from tensor_parallel import train_with_tensor_parallelism

        logging.info("Running Tensor Parallelism")
        tensor_output_dir = os.path.join(args.output_dir, "tensor_parallel")
        os.makedirs(tensor_output_dir, exist_ok=True)

        # Save args
        tensor_args = args
        tensor_args.output_dir = tensor_output_dir

        # Run tensor parallelism
        tensor_epoch_times = train_with_tensor_parallelism(tensor_args)

        logging.info(f"Tensor Parallelism - Average epoch time: {sum(tensor_epoch_times)/len(tensor_epoch_times):.2f} seconds")

    if args.parallelism_type == "pipeline" or args.parallelism_type == "all":
        from pipeline_parallel import train_with_pipeline_parallelism

        logging.info("Running Pipeline Parallelism")
        pipeline_output_dir = os.path.join(args.output_dir, "pipeline_parallel")
        os.makedirs(pipeline_output_dir, exist_ok=True)

        # Save args
        pipeline_args = args
        pipeline_args.output_dir = pipeline_output_dir

        # Run pipeline parallelism
        pipeline_epoch_times = train_with_pipeline_parallelism(pipeline_args)

        logging.info(f"Pipeline Parallelism - Average epoch time: {sum(pipeline_epoch_times)/len(pipeline_epoch_times):.2f} seconds")

    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time:.2f} seconds")

    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({"total_training_time": total_time})
        wandb.finish()

    return 0

if __name__ == "__main__":
    args = parse_args_distributed()
    run_distributed_training(args)
