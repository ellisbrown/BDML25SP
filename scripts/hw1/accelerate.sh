#!/bin/bash
set -e

echo "Starting LLaMA fine-tuning with Accelerate..."

# Configure paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"
DATA_DIR="./processed_data"
OUTPUT_DIR="./checkpoints/llama-finetuned-accelerate"
LOGS_DIR="./logs"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Set GPU device
GPU_IDS=1
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# First run 'accelerate config' once to set up configuration
# Then run with accelerate launch
accelerate launch \
    --config_file accelerate_config.yaml \
    --gpu_ids $GPU_IDS \
    llm_fine_tuning.py \
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
    --use_double_quant \
    --flash_attention \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 500 \
    --save_total_limit 1 \
    --device "cuda:0" \
    --use_wandb \
    --wandb_project "bdml25sp" \
    --wandb_entity "ellisbrown" \
    --wandb_run_name hw1_finetuning_accelerate

echo "Fine-tuning complete! See $OUTPUT_DIR/training_stats.txt for results."