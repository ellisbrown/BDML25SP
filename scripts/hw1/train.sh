#!/bin/bash

# Script to run LLaMA fine-tuning with memory optimizations
# Usage: ./run_training.sh

echo "Starting LLaMA fine-tuning with memory optimizations..."

# Configure paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"  # Path to the converted model
DATA_DIR="./processed_data"
OUTPUT_DIR="./llama-finetuned"

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the fine-tuning script
python llm_fine_tuning.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --max_length 512 \
    --seed 42 \
    --start_batch_size 4 \
    --max_batch_size 128 \
    --safe_batch_size 8 \
    --base_grad_accum 32 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --load_in_4bit \
    --use_fp16 \
    --use_gradient_checkpointing \
    --use_8bit_optimizer \
    --use_double_quant \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 1 \
    --device "cuda:0"

echo "Fine-tuning complete! See $OUTPUT_DIR/training_stats.txt for results."