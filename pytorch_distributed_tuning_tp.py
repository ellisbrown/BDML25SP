import os
import time
import torch
import logging
import numpy as np
import random
from tqdm import tqdm
import torch.distributed as dist

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
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

def main():
    """Main function for Tensor Parallel distributed training using PyTorch native TP"""

    # Parse arguments
    args = utils.parse_args()

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

    # Initialize DeviceMesh for tensor parallelism
    tp_mesh = init_device_mesh("cuda", (world_size,))

    # Configure base model
    model, tokenizer = utils.configure_model_base(args, device_map="cpu")

    # Move the model to meta device to avoid full initialization on all GPUs
    if hasattr(model, "to_meta"):
        model.to_meta()

    # Create the tensor parallel plan for the model
    # This plan targets Llama model architecture specifically
    # Get all transformer blocks
    transformer_blocks = model.model.model.layers

    # Parallelize each transformer block
    for layer_id, transformer_block in enumerate(transformer_blocks):
        # Create a plan for this specific layer
        layer_tp_plan = create_llama_tp_plan(transformer_block)

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.self_attn
        if hasattr(attn_layer, "num_heads"):
            attn_layer.num_heads = attn_layer.num_heads // tp_mesh.size(0)
        elif hasattr(attn_layer, "n_heads"):
            attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size(0)
            if hasattr(attn_layer, "n_kv_heads"):
                attn_layer.n_kv_heads = max(1, attn_layer.n_kv_heads // tp_mesh.size(0))

        # Apply tensor parallelism to this layer
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )

    # Parallelize embedding and output layers
    model = parallelize_module(
        model,
        tp_mesh,
        {
            # Handle Llama embedding specifically
            "model.embed_tokens": RowwiseParallel(),
            # Handle output layer
            "lm_head": ColwiseParallel(),
        }
    )

    # Prepare datasets
    train_dataset, eval_dataset = utils.prepare_datasets(args, tokenizer)
    logging.info(f"Train dataset size: {len(train_dataset)} samples")
    logging.info(f"Eval dataset size: {len(eval_dataset)} samples")

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
            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": avg_loss})

            # Log loss
            if step % args.logging_steps == 0 and args.local_rank == 0:
                logging.info(f"Epoch: {epoch+1}/{args.num_epochs}, Step: {step}/{total_steps}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

            # Evaluate periodically
            if step % args.eval_steps == 0 and args.local_rank == 0 and step > 0:
                eval_loss, perplexity = utils.evaluate_perplexity(args, model, eval_dataset, tokenizer)
                logging.info(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                model.train()  # Back to training mode

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        # Evaluate at the end of each epoch
        if args.local_rank == 0:
            eval_loss, perplexity = utils.evaluate_perplexity(args, model, eval_dataset, tokenizer)
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

    # Final evaluation
    if args.local_rank == 0:
        eval_loss, perplexity = utils.evaluate_perplexity(args, model, eval_dataset, tokenizer)
        logging.info(f"Final evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Log training statistics
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        logging.info(f"Training completed in {total_time:.2f} seconds")
        logging.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

        # Save training stats
        utils.save_training_stats(args, total_time, epoch_times, perplexity, "tensor_parallel")

    logging.info("Tensor Parallel distributed training complete!")

def create_llama_tp_plan(transformer_block):
    """Create a tensor parallel plan for a Llama transformer block"""
    layer_tp_plan = {}

    # Check if we have self_attn (transformer) or attention (Llama)
    attn_name = "self_attn" if hasattr(transformer_block, "self_attn") else "attention"

    # Handle attention layers
    layer_tp_plan[f"{attn_name}.q_proj"] = ColwiseParallel()
    layer_tp_plan[f"{attn_name}.k_proj"] = ColwiseParallel()
    layer_tp_plan[f"{attn_name}.v_proj"] = ColwiseParallel()
    layer_tp_plan[f"{attn_name}.o_proj"] = RowwiseParallel()

    # Find input/output norm layers
    input_norm_name = None
    if hasattr(transformer_block, "input_layernorm"):
        input_norm_name = "input_layernorm"
    elif hasattr(transformer_block, "attention_norm"):
        input_norm_name = "attention_norm"

    # Apply SequenceParallel to the norm layers if found
    if input_norm_name:
        layer_tp_plan[input_norm_name] = SequenceParallel()

    # Find MLP/feedforward layers
    if hasattr(transformer_block, "mlp"):
        mlp_name = "mlp"
        layer_tp_plan[f"{mlp_name}.gate_proj"] = ColwiseParallel()
        layer_tp_plan[f"{mlp_name}.down_proj"] = RowwiseParallel()
        layer_tp_plan[f"{mlp_name}.up_proj"] = ColwiseParallel()
    elif hasattr(transformer_block, "feed_forward"):
        mlp_name = "feed_forward"
        layer_tp_plan[f"{mlp_name}.w1"] = ColwiseParallel()
        layer_tp_plan[f"{mlp_name}.w2"] = RowwiseParallel()
        layer_tp_plan[f"{mlp_name}.w3"] = ColwiseParallel()

    # Find post-attention norm layer
    post_norm_name = None
    if hasattr(transformer_block, "post_attention_layernorm"):
        post_norm_name = "post_attention_layernorm"
    elif hasattr(transformer_block, "ffn_norm"):
        post_norm_name = "ffn_norm"

    # Apply SequenceParallel to post norm if found
    if post_norm_name:
        layer_tp_plan[post_norm_name] = SequenceParallel()

    return layer_tp_plan

if __name__ == "__main__":
    main()