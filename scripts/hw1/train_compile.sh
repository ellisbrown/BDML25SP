#!/bin/bash
set -e

# Script to run LLaMA fine-tuning with memory optimizations
# Usage: ./run_training.sh

echo "Starting LLaMA fine-tuning with memory optimizations..."

# Configure paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"  # Path to the converted model
DATA_DIR="./processed_data"
OUTPUT_DIR="./llama-finetuned"
LOGS_DIR="./logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Set GPU device
export CUDA_VISIBLE_DEVICES=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the fine-tuning script
python llm_fine_tuning.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --max_length 512 \
    --seed 42 \
    --start_batch_size 2 \
    --max_batch_size 32 \
    --safe_batch_size 2 \
    --base_grad_accum 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj"  \
    --load_in_4bit \
    --use_bf16 \
    --use_gradient_checkpointing \
    --use_8bit_optimizer \
    --compile_model \
    --logging_steps 1 \
    --eval_steps 20 \
    --save_steps 500 \
    --save_total_limit 1 \
    --device "cuda:0" \
    --use_wandb \
    --wandb_project "bdml25sp" \
    --wandb_entity "ellisbrown" \
    --wandb_run_name "hw1_finetuning"

echo "Fine-tuning complete! See $OUTPUT_DIR/training_stats.txt for results."