#!/bin/bash

# Script to run the improved PDF preprocessing with optimal settings
# Usage: ./run_preprocess.sh

echo "Starting PDF preprocessing with optimized settings..."

# Configuration
PDF_DIR="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"
OUTPUT_DIR="./processed_data"
NUM_WORKERS=48      # Use a reasonable number of processes
CHUNK_SIZE=5        # Process PDFs in small batches
MIN_CHARS=500       # Minimum text size to consider successful
TRAIN_RATIO=0.9     # 90% for training, 10% for testing

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the preprocessing script
python preprocess_data.py \
    --pdf_dir "$PDF_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "$NUM_WORKERS" \
    --chunk_size "$CHUNK_SIZE" \
    --min_chars "$MIN_CHARS" \
    --train_ratio "$TRAIN_RATIO"

echo "Preprocessing complete! Check $OUTPUT_DIR/split_info.json for dataset details."