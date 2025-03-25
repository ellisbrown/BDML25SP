#!/bin/bash

# Set default values for your environment
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B"
DATA_PATH="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"
OUTPUT_DIR="./llama-finetuned"
TXT_DIR="./extracted_texts"
DEVICE="cuda:0"

# Create virtual environment if it doesn't exist
if [ ! -d "llm_env" ]; then
    echo "Creating new virtual environment..."
    python -m venv llm_env
fi

# Activate virtual environment
source llm_env/bin/activate

# Install required packages
pip install torch==2.0.1 transformers==4.31.0 datasets==2.13.1 peft==0.4.0 tqdm bitsandbytes==0.40.2 accelerate==0.21.0 PyPDF2 psutil gputil

# Extract dataset if needed
if [ ! -d "$DATA_PATH" ] || [ -z "$(ls -A $DATA_PATH)" ]; then
    echo "Extracting climate text dataset..."
    unzip -q /root/bdml25sp/datasets/BDML25SP/climate_text_dataset.zip -d /root/bdml25sp/datasets/BDML25SP/
fi

# Set environment variables for one GPU
export CUDA_VISIBLE_DEVICES=0

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TXT_DIR"

# Set configuration variables
# You can edit these to try different configurations
TRAIN_TEST_SPLIT=0.9
LEARNING_RATE=2e-4
NUM_EPOCHS=3
MAX_LENGTH=512
SEED=42

# Memory optimization parameters
START_BATCH_SIZE=1
MAX_BATCH_SIZE=128
SAFE_BATCH_SIZE=8
BASE_GRAD_ACCUM=32

# LoRA parameters
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

# Execute the Python script with all arguments
python llm_fine_tuning.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --txt_dir "$TXT_DIR" \
    --train_test_split $TRAIN_TEST_SPLIT \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --max_length $MAX_LENGTH \
    --seed $SEED \
    --start_batch_size $START_BATCH_SIZE \
    --max_batch_size $MAX_BATCH_SIZE \
    --safe_batch_size $SAFE_BATCH_SIZE \
    --base_grad_accum $BASE_GRAD_ACCUM \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_target_modules $LORA_TARGET_MODULES \
    --load_in_4bit \
    --use_fp16 \
    --use_gradient_checkpointing \
    --use_8bit_optimizer \
    --use_double_quant \
    --device $DEVICE

# Print completion message
echo "Training completed! Check $OUTPUT_DIR/training_stats.txt for results."