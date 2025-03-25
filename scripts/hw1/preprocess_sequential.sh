#!/bin/bash

# Script to run the ultra-reliable preprocessing
# Usage: ./run_simple_preprocess.sh

echo "Starting simple PDF preprocessing..."

# Configuration
PDF_DIR="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"
OUTPUT_DIR="./processed_data"
MIN_CHARS=500       # Minimum text size to consider successful
TIMEOUT=5          # Maximum time to spend on each PDF (in seconds)
TRAIN_RATIO=0.9     # 90% for training, 10% for testing

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the preprocessing script
python preprocess_sequential.py \
    --pdf_dir "$PDF_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --min_chars "$MIN_CHARS" \
    --timeout "$TIMEOUT" \
    --train_ratio "$TRAIN_RATIO"

echo "Preprocessing complete! Check $OUTPUT_DIR/split_info.json for dataset details."