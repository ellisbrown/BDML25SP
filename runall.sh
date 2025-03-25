#!/bin/bash

# # Create virtual environment if it doesn't exist
# if [ ! -d "llm_env" ]; then
#     echo "Creating new virtual environment..."
#     python -m venv llm_env
# fi

# # Activate virtual environment
# source llm_env/bin/activate

# # Install required packages
# pip install torch==2.0.1 transformers==4.31.0 datasets==2.13.1 peft==0.4.0 tqdm bitsandbytes==0.40.2 accelerate==0.21.0 PyPDF2

# # Extract dataset if needed
# if [ ! -d "/root/bdml25sp/datasets/BDML25SP/climate_text_dataset" ] || [ -z "$(ls -A /root/bdml25sp/datasets/BDML25SP/climate_text_dataset)" ]; then
#     echo "Extracting climate text dataset..."
#     unzip -q /root/bdml25sp/datasets/BDML25SP/climate_text_dataset.zip -d /root/bdml25sp/datasets/BDML25SP/
# fi

# Set environment variables for one GPU
export CUDA_VISIBLE_DEVICES=0

# Create output directory
mkdir -p llama-finetuned

# Run the Python script
python llm_fine_tuning.py