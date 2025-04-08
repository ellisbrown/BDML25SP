#!/bin/bash
set -e
echo "Starting distributed training with Data Parallelism..."

# Configure paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"
DATA_DIR="./processed_data"
OUTPUT_DIR="./checkpoints/hw2/dp"
LOGS_DIR="./logs/hw2/dp"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Run DeepSpeed with 2 GPUs for Data Parallelism
deepspeed --num_gpus=2 \
    distributed_tuning_dp.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --deepspeed_config "./ds_config_dp.json" \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --max_length 512 \
    --per_device_batch_size 56 \
    --gradient_accumulation_steps 1 \
    --seed 42 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --load_in_8bit \
    --use_bf16 \
    --use_gradient_checkpointing \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500

echo "Data Parallel training complete! See $OUTPUT_DIR/training_stats.txt for results."
