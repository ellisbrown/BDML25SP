import os
import time
import torch
import deepspeed
from tqdm import tqdm
import logging
import numpy as np
import random
from transformers import DataCollatorForLanguageModeling

import distributed_tuning_utils as utils

def main():
    """Main function for Data Parallel distributed training"""
    # Set up logging
    utils.setup_logging()

    # Parse arguments
    args = utils.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize DeepSpeed distributed environment
    deepspeed.init_distributed()

    # Get local rank from environment variable (set by DeepSpeed launcher)
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    # Create output directory
    if args.local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

    logging.info(f"Starting Data Parallel distributed training")
    logging.info(f"Local rank: {args.local_rank}")

    # Configure model with LoRA and quantization
    logging.info(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    logging.info(f"RANK: {os.environ.get('RANK')}")
    logging.info(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

    model, tokenizer = utils.configure_model_base(args, device_map={"": int(os.environ.get("LOCAL_RANK", 0))})

    # Prepare datasets
    train_dataset, eval_dataset = utils.prepare_datasets(args, tokenizer)
    logging.info(f"Train dataset size: {len(train_dataset)} samples")
    logging.info(f"Eval dataset size: {len(eval_dataset)} samples")

    # Load DeepSpeed config
    if os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            import json
            ds_config = json.load(f)
    else:
        raise ValueError(f"DeepSpeed config file not found: {args.deepspeed_config}")

    # Train with Data Parallelism
    start_time = time.time()

    # When using deepspeed launcher, we should not pass the config again
    # The config is already passed via the command line
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters())
    )

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create dataloader
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
            avg_loss = total_loss / (step + 1)
            progress_bar.set_postfix({"loss": avg_loss})

            # Log loss
            if step % args.logging_steps == 0 and args.local_rank == 0:
                logging.info(f"Epoch: {epoch+1}/{args.num_epochs}, Step: {step}/{total_steps}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

            # Evaluate periodically
            if step % args.eval_steps == 0 and args.local_rank == 0 and step > 0:
                eval_loss, perplexity = utils.evaluate_perplexity(args, model_engine, eval_dataset, tokenizer)
                logging.info(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                model_engine.train()  # Back to training mode

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        # Evaluate at the end of each epoch
        if args.local_rank == 0:
            eval_loss, perplexity = utils.evaluate_perplexity(args, model_engine, eval_dataset, tokenizer)
            logging.info(f"Epoch {epoch+1} completed - Time: {epoch_time:.2f}s, Loss: {total_loss/total_steps:.4f}, Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Save model at the end of each epoch
        if args.local_rank == 0:
            output_dir = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            os.makedirs(output_dir, exist_ok=True)
            model_engine.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model saved to {output_dir}")

    # Calculate total training time
    total_time = time.time() - start_time

    # Final evaluation
    if args.local_rank == 0:
        eval_loss, perplexity = utils.evaluate_perplexity(args, model_engine, eval_dataset, tokenizer)
        logging.info(f"Final evaluation - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

        # Log training statistics
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        logging.info(f"Training completed in {total_time:.2f} seconds")
        logging.info(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

        # Save training stats
        utils.save_training_stats(args, total_time, epoch_times, perplexity, "data_parallel")

    logging.info("Data Parallel distributed training complete!")

if __name__ == "__main__":
    main()
