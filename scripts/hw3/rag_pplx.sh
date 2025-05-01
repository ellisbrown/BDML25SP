#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting RAG Perplexity Evaluation Script..."

# --- Configuration ---
# Adjust these paths based on your environment (especially on HPC)
MODEL_PATH="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted" # Or /scratch/BDML25SP/Llama3.2-3B-converted
DATA_DIR="./processed_data"
OUTPUT_DIR="./rag_perplexity_output_ivfpq" # Specific output dir for this run
CACHE_DIR="./dataset_cache" # Cache dir for datasets

# Data & Chunking Parameters
TRAIN_TEST_SPLIT=0.9
CHUNK_SIZE=300
CHUNK_OVERLAP=50

# RAG Parameters
EMBEDDING_MODEL="all-MiniLM-L6-v2"
FAISS_INDEX_TYPE="IndexIVFPQ" # Use IVF with Product Quantization
TOP_K=5 # Retrieve top 5 train chunks for context

# Evaluation Parameters
MAX_LENGTH=512 # Max sequence length for LLaMA model input

# System Parameters
SEED=42
# Automatically detect GPU or use CPU for the LLaMA model
DEVICE="cuda"
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU not detected, using CPU for LLaMA model."
    DEVICE="cpu"
fi

# Create output and cache directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"
echo "Output will be saved to: $OUTPUT_DIR"
echo "Dataset cache directory: $CACHE_DIR"

# --- Run the Python Script ---
# Ensure the python script is named rag_pplx.py or adjust the name below
python rag_pplx.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --train_test_split "$TRAIN_TEST_SPLIT" \
    --chunk_size "$CHUNK_SIZE" \
    --chunk_overlap "$CHUNK_OVERLAP" \
    --embedding_model "$EMBEDDING_MODEL" \
    --faiss_index_type "$FAISS_INDEX_TYPE" \
    --top_k "$TOP_K" \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --device "$DEVICE" \
    # Add --use_gpu_for_embeddings if desired and SentenceTransformer supports it well

echo "RAG Perplexity Evaluation Script finished."
echo "Check logs in ./logs/ and results in $OUTPUT_DIR"
