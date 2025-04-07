#!/bin/bash
set -e

echo "Starting LLaMA fine-tuning with Data Parallelism..."

# Configure paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"
DATA_DIR="./processed_data"
OUTPUT_DIR="./checkpoints/llama-finetuned-distributed/data_parallel"
LOGS_DIR="./logs/hw2"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Use all available GPUs
export CUDA_VISIBLE_DEVICES=0,1


# First approach - use torchrun for more reliable setup
# This is often more stable than accelerate for complex models with quantization
torchrun --nproc_per_node=2 \
    distributed_training_main.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --parallelism_type "data" \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --max_length 512 \
    --seed 42 \
    --batch_size 16 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
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
    --use_wandb \
    --wandb_project "bdml25sp" \
    --wandb_entity "ellisbrown" \
    --wandb_run_name "hw2_data_parallel"

# Alternative approach using accelerate if needed
# Uncomment to use this instead
# accelerate launch \
#     --config_file accelerate_config_distributed.yaml \
#     distributed_training_main.py \
#     --model_path "$MODEL_PATH" \
#     --data_dir "$DATA_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --parallelism_type "data" \
#     --learning_rate 2e-4 \
#     --num_epochs 1 \
#     --max_length 512 \
#     --seed 42 \
#     --batch_size 16 \
#     --lora_r 8 \
#     --lora_alpha 32 \
#     --lora_dropout 0.1 \
#     --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
#     --load_in_4bit \
#     --use_bf16 \
#     --use_gradient_checkpointing \
#     --use_8bit_optimizer \
#     --use_double_quant \
#     --flash_attention \
#     --logging_steps 10 \
#     --eval_steps 50 \
#     --save_steps 500 \
#     --save_total_limit 1 \
#     --use_wandb \
#     --wandb_project "bdml25sp" \
#     --wandb_entity "ellisbrown" \
#     --wandb_run_name "hw2_data_parallel"

echo "Data Parallel training complete! See $OUTPUT_DIR/training_stats.txt for results."
