#!/bin/bash

# Script to convert LLaMA weights to Hugging Face format
# Usage: ./convert_model.sh

echo "Converting LLaMA model to Hugging Face format..."

# Set paths
ORIGINAL_MODEL_DIR="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B"
CONVERTED_MODEL_DIR="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted"

# Check if conversion script exists, download if not
if [ ! -f "convert_llama_weights_to_hf.py" ]; then
    echo "Downloading LLaMA conversion script..."
    wget https://github.com/huggingface/transformers/raw/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
fi

# Install required packages if needed
pip install sentencepiece protobuf

# Run the conversion script
echo "Running conversion script..."
python convert_llama_weights_to_hf.py \
    --input_dir $ORIGINAL_MODEL_DIR \
    --output_dir $CONVERTED_MODEL_DIR \
    --model_size 3B \
    --llama_version 3.2

echo "Conversion complete! The converted model is available at: $CONVERTED_MODEL_DIR"
