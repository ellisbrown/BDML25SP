#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting RAG Pipeline Script..."

# --- Configuration ---
# Adjust these paths based on your environment (especially on HPC)
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted" # Or /scratch/BDML25SP/Llama3.2-3B-converted
DATA_DIR="./processed_data"
OUTPUT_DIR="./output/rag_output_ivfpq" # Use a different output dir for this index type
EVAL_QUERIES="./eval_queries.json" # Path to your evaluation queries JSON file

# RAG Parameters
EMBEDDING_MODEL="all-MiniLM-L6-v2"
FAISS_INDEX_TYPE="IndexIVFPQ" # Use IVF with Product Quantization
TOP_K=5 # Retrieve top 5 chunks
CHUNK_SIZE=300
CHUNK_OVERLAP=50
TEMPLATE_VERSION="simple" # Or "complex"

# Generation Parameters
MAX_NEW_TOKENS=150
TEMPERATURE=0.6
TOP_P=0.9
# LOAD_IN_4BIT="--load_in_4bit" # Uncomment to enable 4-bit quantization for the generator

# System Parameters
SEED=42
# Automatically detect GPU or use CPU
# DEVICE="cuda:0"
DEVICE="cuda"
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU not detected, using CPU for generation."
    DEVICE="cpu"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to: $OUTPUT_DIR"

# --- Run the Python Script ---
python rag.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --eval_queries_file "$EVAL_QUERIES" \
    --embedding_model "$EMBEDDING_MODEL" \
    --faiss_index_type "$FAISS_INDEX_TYPE" \
    --top_k "$TOP_K" \
    --chunk_size "$CHUNK_SIZE" \
    --chunk_overlap "$CHUNK_OVERLAP" \
    --template_version "$TEMPLATE_VERSION" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --generation_temperature "$TEMPERATURE" \
    --generation_top_p "$TOP_P" \
    --seed "$SEED" \
    --device "$DEVICE" \
    ${LOAD_IN_4BIT:-} # Only adds --load_in_4bit if the variable is set

echo "RAG Pipeline Script finished."
echo "Check logs in ./logs/ and results in $OUTPUT_DIR"
