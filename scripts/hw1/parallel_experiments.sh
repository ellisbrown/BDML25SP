#!/bin/bash

# This script runs multiple experiments in parallel across available GPUs
# Usage: ./parallel-experiments.sh

# Base paths
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B"
DATA_PATH="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"

# Create master experiment directory
EXPERIMENTS_DIR="./experiments"
mkdir -p "$EXPERIMENTS_DIR"

# Function to run an experiment on a specific GPU
# Parameters:
#   $1: GPU ID (0-7)
#   $2: Experiment name
#   $3: Experiment flags (additional args to pass to the script)
run_on_gpu() {
    GPU_ID=$1
    EXP_NAME=$2
    EXP_FLAGS=$3

    OUTPUT_DIR="$EXPERIMENTS_DIR/$EXP_NAME"
    LOG_FILE="$EXPERIMENTS_DIR/${EXP_NAME}_gpu${GPU_ID}.log"

    mkdir -p "$OUTPUT_DIR"

    echo "Starting experiment '$EXP_NAME' on GPU $GPU_ID"
    echo "Logs will be written to: $LOG_FILE"

    # Run the experiment on the specified GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python llm_fine_tuning.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --txt_dir "./extracted_texts" \
        --device "cuda:0" \
        $EXP_FLAGS > "$LOG_FILE" 2>&1 &

    # Store the PID to monitor later
    echo $! > "$OUTPUT_DIR/process_pid.txt"
}

# Run different experiment configurations in parallel on different GPUs
# GPU 0: Default configuration with all optimizations
run_on_gpu 0 "exp1_all_optimizations" "--load_in_4bit --use_fp16 --use_gradient_checkpointing --use_8bit_optimizer --use_double_quant"

# GPU 1: LoRA only (no quantization)
run_on_gpu 1 "exp2_lora_only" "--use_fp16 --use_gradient_checkpointing"

# GPU 2: 8-bit quantization
run_on_gpu 2 "exp3_8bit_quantization" "--load_in_8bit --use_fp16 --use_gradient_checkpointing --use_8bit_optimizer"

# GPU 3: Higher LoRA rank (r=16)
run_on_gpu 3 "exp4_higher_lora_rank" "--lora_r 16 --lora_alpha 64 --load_in_4bit --use_fp16 --use_gradient_checkpointing --use_8bit_optimizer --use_double_quant"

# GPU 4: BF16 precision
run_on_gpu 4 "exp5_bf16_precision" "--load_in_4bit --use_bf16 --use_gradient_checkpointing --use_8bit_optimizer --use_double_quant"

# GPU 5: Maximum batch size focus (shorter epochs)
run_on_gpu 5 "exp6_max_batch_size" "--num_epochs 1 --start_batch_size 16 --max_batch_size 256 --safe_batch_size 32 --load_in_4bit --use_fp16 --use_gradient_checkpointing --use_8bit_optimizer --use_double_quant"

# GPU 6: Higher learning rate
run_on_gpu 6 "exp7_higher_lr" "--learning_rate 5e-4 --load_in_4bit --use_fp16 --use_gradient_checkpointing --use_8bit_optimizer --use_double_quant"

# GPU 7: Different target modules
run_on_gpu 7 "exp8_different_modules" "--lora_target_modules q_proj,v_proj --load_in_4bit --use_fp16 --use_gradient_checkpointing --use_8bit_optimizer --use_double_quant"

echo "All experiments launched! To monitor progress, use:"
echo "  tail -f $EXPERIMENTS_DIR/*/gpu*.log"
echo ""
echo "To check GPU usage:"
echo "  nvidia-smi"
echo ""
echo "To see which experiments are still running:"
echo "  ps -ef | grep python"
