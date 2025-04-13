import os
import time
import torch
import os
import logging
import numpy as np
import random
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

import distributed_tuning_utils as utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

def main():
    """Main function for Tensor Parallel distributed training using PyTorch native TP"""

    # Parse arguments
    args = utils.parse_args()

    if args.debug:
        args.log_dir += "_debug"
        args.output_dir += "_debug"
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    utils.setup_logging(args.log_dir)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize PyTorch distributed environment
    torch.distributed.init_process_group(backend="nccl")

    # Get local rank and world size
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = torch.distributed.get_world_size()

    # Create output directory
    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

    logging.info(f"Starting Tensor Parallel distributed training")
    logging.info(f"Local rank: {args.local_rank}, World size: {world_size}")

    # Use 2D device mesh for DP and TP
    device_mesh = init_device_mesh("cuda", (1, world_size), mesh_dim_names=("dp", "tp"))
    torch.cuda.set_device(args.local_rank)

    print(f"Rank: {args.local_rank}, Device: {torch.cuda.current_device()}")

    # Configure base model without quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for better compatibility
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing enabled")

    # Move model to correct device
    model = model.to(f"cuda:{args.local_rank}")

    # Apply a simplified tensor parallelism plan (similar to Fred's approach)
    model = parallelize_module(model, device_mesh=device_mesh["tp"], parallelize_plan={
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
    })

    # Adjust the attention head counts in each layer
    for layer_id, transformer_block in enumerate(model.model.layers):
        attn_layer = transformer_block.self_attn
        if hasattr(attn_layer, "config"):
            if hasattr(attn_layer.config, "num_attention_heads"):
                attn_layer.config.num_attention_heads = attn_layer.config.num_attention_heads // device_mesh["tp"].size(0)
            if hasattr(attn_layer.config, "num_key_value_heads"):
                attn_layer.config.num_key_value_heads = max(1, attn_layer.config.num_key_value_heads // device_mesh["tp"].size(0))

    # Wrap with FSDP for better handling of parameters
    model = FSDP(
        model,
        device_mesh=device_mesh["dp"],
        use_orig_params=False,  # Critical change!
        forward_prefetch=False
    )

    # Continue with your code for preparing datasets, optimizer, etc.

    # Prepare datasets
    train_dataset, eval_dataset = utils.prepare_datasets(args, tokenizer)
    logging.info(f"Train dataset size: {len(train_dataset)} samples")
    logging.info(f"Eval dataset size: {len(eval_dataset)} samples")

    # create a subset of 100 samples for quick testing
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.debug_size))

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # Create data loader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )

    total_steps = len(train_dataloader)
    logging.info(f"Total training steps per epoch: {total_steps}")

    # Track elapsed time for each epoch
    epoch_times = []

    # Training loop
    start_time = time.time()

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(f"cuda:{args.local_rank}") for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": avg_loss})

            # Log loss
            if step % args.logging_steps == 0:
                logging.info(f"Epoch: {epoch+1}/{args.num_epochs}, Step: {step}/{total_steps}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

            # Evaluate periodically
            if step % args.eval_steps == 0 and step > 0:
                eval_loss, perplexity = utils.evaluate_perplexity(args, model, eval_dataset, tokenizer)
                logging.info(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                model.train()  # Back to training mode

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        # Evaluate at the end of each epoch
        eval_loss, perplexity = utils.evaluate_perplexity(args, model, eval_dataset, tokenizer)
        if args.local_rank == 0:
            logging.info(f"Epoch {epoch+1} completed - Time: {epoch_time:.2f}s, Loss: {total_loss/total_steps:.4f}, Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Save model at the end of each epoch
        if args.local_rank == 0:
            output_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")

    # Calculate total training time
    total_time = time.time() - start_time
    logging.info(f"Total training time: {total_time:.2f} seconds")

    # Final evaluation
    eval_loss, perplexity = utils.evaluate_perplexity(args, model, eval_dataset, tokenizer)
    if args.local_rank == 0:
        logging.info(f"Final evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Log training statistics
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        logging.info(f"Training completed in {total_time:.2f} seconds")
        logging.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

        # Save training stats
        utils.save_training_stats(args, total_time, epoch_times, perplexity, "tensor_parallel")

    logging.info("Tensor Parallel distributed training complete!")

    # Clean up
    if args.local_rank == 0:
        logging.info("Cleaning up...")
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        logging.info("Cleanup complete!")
    else:
        # Wait for other processes to finish
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        logging.info("Cleanup complete!")

    # Destroy the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()