#!/bin/bash
set -e
echo "Starting distributed training with PyTorch Tensor Parallelism..."

# Configure paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"
DATA_DIR="./processed_data"
OUTPUT_DIR="./checkpoints/hw2/pytorch_tp"
LOGS_DIR="./logs/hw2/pytorch_tp"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOGS_DIR"

# Run PyTorch distributed with 2 GPUs for Tensor Parallelism
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=12344 \
    pytorch_distributed_tuning_tp.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOGS_DIR" \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --max_length 512 \
    --per_device_batch_size 8 \
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

echo "PyTorch Tensor Parallel training complete! See $OUTPUT_DIR/training_stats.txt for results."
