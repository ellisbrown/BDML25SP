#!/bin/bash
set -e

echo "Starting all LLaMA distributed fine-tuning experiments..."

# get path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Create summary file
SUMMARY_FILE="logs/hw2/distributed_training_summary.txt"
echo "Distributed Training Summary" > $SUMMARY_FILE
echo "=========================" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "GPUs used: $(nvidia-smi --list-gpus | wc -l)" >> $SUMMARY_FILE
echo "=========================" >> $SUMMARY_FILE

# Run Data Parallelism
echo "Running Data Parallelism experiment..."
bash $SCRIPT_DIR/run_data_parallel.sh
DATA_TIME=$(grep "Average epoch time" ./checkpoints/llama-finetuned-distributed/data_parallel/training_stats.txt | awk '{print $4}' 2>/dev/null || echo "N/A")
DATA_PERPLEXITY=$(grep "Final perplexity" ./checkpoints/llama-finetuned-distributed/data_parallel/training_stats.txt | awk '{print $3}' 2>/dev/null || echo "N/A")
echo "Data Parallelism:" >> $SUMMARY_FILE
echo "  Average Epoch Time: $DATA_TIME seconds" >> $SUMMARY_FILE
echo "  Final Perplexity: $DATA_PERPLEXITY" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Run Tensor Parallelism
echo "Running Tensor Parallelism experiment..."
bash $SCRIPT_DIR/run_tensor_parallel.sh
TENSOR_TIME=$(grep "Average epoch time" ./checkpoints/llama-finetuned-distributed/tensor_parallel/training_stats.txt | awk '{print $4}' 2>/dev/null || echo "N/A")
TENSOR_PERPLEXITY=$(grep "Final perplexity" ./checkpoints/llama-finetuned-distributed/tensor_parallel/training_stats.txt | awk '{print $3}' 2>/dev/null || echo "N/A")
echo "Tensor Parallelism:" >> $SUMMARY_FILE
echo "  Average Epoch Time: $TENSOR_TIME seconds" >> $SUMMARY_FILE
echo "  Final Perplexity: $TENSOR_PERPLEXITY" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Run Pipeline Parallelism
echo "Running Pipeline Parallelism experiment..."
bash $SCRIPT_DIR/run_pipeline_parallel.sh
PIPELINE_TIME=$(grep "Average epoch time" ./checkpoints/llama-finetuned-distributed/pipeline_parallel/training_stats.txt | awk '{print $4}' 2>/dev/null || echo "N/A")
PIPELINE_PERPLEXITY=$(grep "Final perplexity" ./checkpoints/llama-finetuned-distributed/pipeline_parallel/training_stats.txt | awk '{print $3}' 2>/dev/null || echo "N/A")
echo "Pipeline Parallelism:" >> $SUMMARY_FILE
echo "  Average Epoch Time: $PIPELINE_TIME seconds" >> $SUMMARY_FILE
echo "  Final Perplexity: $PIPELINE_PERPLEXITY" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Calculate the fastest approach
echo "Performance Summary:" >> $SUMMARY_FILE
echo "=========================" >> $SUMMARY_FILE

# Find the fastest approach
FASTEST="Unknown"
FASTEST_TIME=99999999

# Function to check if value is numeric
is_numeric() {
    if [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        return 0
    else
        return 1
    fi
}

# Add check to handle case where times might be N/A
if is_numeric "$DATA_TIME"; then
    if (( $(echo "$DATA_TIME < $FASTEST_TIME" | bc -l) )); then
        FASTEST="Data Parallelism"
        FASTEST_TIME=$DATA_TIME
    fi
fi

if is_numeric "$TENSOR_TIME"; then
    if (( $(echo "$TENSOR_TIME < $FASTEST_TIME" | bc -l) )); then
        FASTEST="Tensor Parallelism"
        FASTEST_TIME=$TENSOR_TIME
    fi
fi

if is_numeric "$PIPELINE_TIME"; then
    if (( $(echo "$PIPELINE_TIME < $FASTEST_TIME" | bc -l) )); then
        FASTEST="Pipeline Parallelism"
        FASTEST_TIME=$PIPELINE_TIME
    fi
fi

echo "Fastest approach: $FASTEST with $FASTEST_TIME seconds per epoch" >> $SUMMARY_FILE

echo "All experiments completed! See $SUMMARY_FILE for a summary of results."
echo "Detailed results are available in the respective training_stats.txt files in each output directory."