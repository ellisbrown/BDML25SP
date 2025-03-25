#!/bin/bash

# This script provides different configurations for memory optimization experiments
# You can run individual configurations with:
# source experiment-configs.sh && run_experiment1

# Base paths and environment setup - same for all experiments
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B"
DATA_PATH="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"
DEVICE="cuda:0"
export CUDA_VISIBLE_DEVICES=0

# Experiment 1: Default configuration (All optimizations enabled)
run_experiment1() {
    echo "Running Experiment 1: Default configuration with all optimizations"

    # Set output directories
    OUTPUT_DIR="./experiments/exp1_all_optimizations"
    TXT_DIR="./extracted_texts"
    mkdir -p "$OUTPUT_DIR"

    python llm_fine_tuning.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --txt_dir "$TXT_DIR" \
        --load_in_4bit \
        --use_fp16 \
        --use_gradient_checkpointing \
        --use_8bit_optimizer \
        --use_double_quant \
        --device $DEVICE
}

# Experiment 2: LoRA only (no quantization)
run_experiment2() {
    echo "Running Experiment 2: LoRA only (no quantization)"

    # Set output directories
    OUTPUT_DIR="./experiments/exp2_lora_only"
    TXT_DIR="./extracted_texts"
    mkdir -p "$OUTPUT_DIR"

    python llm_fine_tuning.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --txt_dir "$TXT_DIR" \
        --use_fp16 \
        --use_gradient_checkpointing \
        --device $DEVICE
}

# Experiment 3: 8-bit Quantization (instead of 4-bit)
run_experiment3() {
    echo "Running Experiment 3: 8-bit Quantization"

    # Set output directories
    OUTPUT_DIR="./experiments/exp3_8bit_quantization"
    TXT_DIR="./extracted_texts"
    mkdir -p "$OUTPUT_DIR"

    python llm_fine_tuning.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --txt_dir "$TXT_DIR" \
        --load_in_8bit \
        --use_fp16 \
        --use_gradient_checkpointing \
        --use_8bit_optimizer \
        --device $DEVICE
}

# Experiment 4: Higher LoRA rank for better model quality (uses more memory)
run_experiment4() {
    echo "Running Experiment 4: Higher LoRA rank (r=16)"

    # Set output directories
    OUTPUT_DIR="./experiments/exp4_higher_lora_rank"
    TXT_DIR="./extracted_texts"
    mkdir -p "$OUTPUT_DIR"

    python llm_fine_tuning.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --txt_dir "$TXT_DIR" \
        --lora_r 16 \
        --lora_alpha 64 \
        --load_in_4bit \
        --use_fp16 \
        --use_gradient_checkpointing \
        --use_8bit_optimizer \
        --use_double_quant \
        --device $DEVICE
}

# Experiment 5: BF16 precision instead of FP16 (may be more stable but use more memory)
run_experiment5() {
    echo "Running Experiment 5: BF16 precision"

    # Set output directories
    OUTPUT_DIR="./experiments/exp5_bf16_precision"
    TXT_DIR="./extracted_texts"
    mkdir -p "$OUTPUT_DIR"

    python llm_fine_tuning.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --txt_dir "$TXT_DIR" \
        --load_in_4bit \
        --use_bf16 \
        --use_gradient_checkpointing \
        --use_8bit_optimizer \
        --use_double_quant \
        --device $DEVICE
}

# Experiment 6: Focus on finding maximum batch size (shorter epochs)
run_experiment6() {
    echo "Running Experiment 6: Maximum batch size focus"

    # Set output directories
    OUTPUT_DIR="./experiments/exp6_max_batch_size"
    TXT_DIR="./extracted_texts"
    mkdir -p "$OUTPUT_DIR"

    python llm_fine_tuning.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --txt_dir "$TXT_DIR" \
        --num_epochs 1 \
        --start_batch_size 16 \
        --max_batch_size 256 \
        --safe_batch_size 32 \
        --load_in_4bit \
        --use_fp16 \
        --use_gradient_checkpointing \
        --use_8bit_optimizer \
        --use_double_quant \
        --device $DEVICE
}

# Run all experiments
run_all_experiments() {
    run_experiment1
    run_experiment2
    run_experiment3
    run_experiment4
    run_experiment5
    run_experiment6
}

echo "Available experiments:"
echo "  run_experiment1 - Default configuration with all optimizations"
echo "  run_experiment2 - LoRA only (no quantization)"
echo "  run_experiment3 - 8-bit Quantization"
echo "  run_experiment4 - Higher LoRA rank (r=16)"
echo "  run_experiment5 - BF16 precision"
echo "  run_experiment6 - Maximum batch size focus"
echo "  run_all_experiments - Run all experiments sequentially"
echo ""
echo "Run an experiment by executing: source experiment-configs.sh && run_experiment1"