#!/bin/bash

# Master script to run data preprocessing and LLaMA fine-tuning
# Usage: ./run_all.sh [gpu_id]

# Set default GPU ID
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Configuration
PDF_DIR="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B"
DATA_DIR="./processed_data"
OUTPUT_DIR="./llama-finetuned"

# # Create virtual environment if it doesn't exist
# if [ ! -d "llm_env" ]; then
#     echo "Creating new virtual environment..."
#     python -m venv llm_env
# fi

# # Activate virtual environment
# source llm_env/bin/activate

# # Install required packages
# echo "Installing required packages..."
# pip install torch transformers datasets peft tqdm bitsandbytes accelerate PyPDF2 psutil gputil

# Step 1: Preprocess data (if not already done)
if [ ! -f "$DATA_DIR/split_info.json" ]; then
    echo "=== Step 1: Preprocessing data ==="
    python preprocess_data.py \
        --pdf_dir "$PDF_DIR" \
        --output_dir "$DATA_DIR" \
        --train_ratio 0.9 \
        --num_workers -1  # Use all available CPU cores
else
    echo "=== Step 1: Skipping preprocessing (already done) ==="
fi

# Step 2: Fine-tune LLaMA
echo "=== Step 2: Fine-tuning LLaMA with memory optimizations ==="
python llm_fine_tuning.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --load_in_4bit \
    --use_fp16 \
    --use_gradient_checkpointing \
    --use_8bit_optimizer \
    --use_double_quant \
    --device "cuda:0"

echo "=== All done! ==="
echo "Trained model saved to: $OUTPUT_DIR"
echo "Training stats: $OUTPUT_DIR/training_stats.txt"