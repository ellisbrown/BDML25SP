#!/bin/bash

# Script to run the ultra-simple preprocessing
# Usage: ./run_ultra_simple.sh

echo "Starting ultra-simple PDF preprocessing..."

# Configuration
PDF_DIR="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset"
OUTPUT_DIR="./processed_data"
PAGE_LIMIT=30       # Maximum pages to process per PDF
MIN_CHARS=500       # Minimum text size to consider successful
TRAIN_RATIO=0.9     # 90% for training, 10% for testing

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the preprocessing script
python preprocess_sequential.py \
    --pdf_dir "$PDF_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --page_limit "$PAGE_LIMIT" \
    --min_chars "$MIN_CHARS" \
    --train_ratio "$TRAIN_RATIO"

echo "Preprocessing complete! Check $OUTPUT_DIR/split_info.json for dataset details."