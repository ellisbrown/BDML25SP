#!/bin/bash
set -e
echo "Starting distributed training with Data Parallelism..."

# Configure paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"
DATA_DIR="./processed_data"
OUTPUT_DIR="./checkpoints/hw2/dp_no_lora"
LOGS_DIR="./logs/hw2/dp_no_lora"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"


# Run DeepSpeed with 2 GPUs for Data Parallelism
    # --num_gpus=2 \
deepspeed \
    --master_port 12341 \
    --include localhost:2,3 \
    distributed_tuning_dp.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOGS_DIR" \
    --deepspeed_config "./ds_config_dp.json" \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --max_length 512 \
    --per_device_batch_size 56 \
    --gradient_accumulation_steps 1 \
    --seed 42 \
    --use_bf16 \
    --use_gradient_checkpointing \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500

echo "Data Parallel training complete! See $OUTPUT_DIR/training_stats.txt for results."
