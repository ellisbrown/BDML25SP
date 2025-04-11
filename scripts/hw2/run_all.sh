#!/bin/bash
set -e

echo "Running all distributed training experiments (DP, TP, PP)..."

# Make directories
mkdir -p ./logs/hw2
mkdir -p ./checkpoints/hw2

# Data Parallelism
echo "================================================"
echo "Starting Data Parallelism experiment..."
echo "================================================"
bash scripts/hw2/dp.sh

# Wait for system resources to stabilize
sleep 30

# Tensor Parallelism
echo "================================================"
echo "Starting Tensor Parallelism experiment..."
echo "================================================"
bash scripts/hw2/tp.sh

# Wait for system resources to stabilize
sleep 30

# Pipeline Parallelism
echo "================================================"
echo "Starting Pipeline Parallelism experiment..."
echo "================================================"
bash scripts/hw2/pp.sh

# Generate comparison report
echo "================================================"
echo "Generating performance comparison report..."
echo "================================================"

# Extract epoch times from training stats
DP_TIME=$(grep "Average time per epoch" ./checkpoints/hw2/dp/training_stats.txt | awk '{print $5}')
TP_TIME=$(grep "Average time per epoch" ./checkpoints/hw2/tp/training_stats.txt | awk '{print $5}')
PP_TIME=$(grep "Average time per epoch" ./checkpoints/hw2/pp/training_stats.txt | awk '{print $5}')

# Extract perplexity from training stats
DP_PPL=$(grep "Final perplexity" ./checkpoints/hw2/dp/training_stats.txt | awk '{print $3}')
TP_PPL=$(grep "Final perplexity" ./checkpoints/hw2/tp/training_stats.txt | awk '{print $3}')
PP_PPL=$(grep "Final perplexity" ./checkpoints/hw2/pp/training_stats.txt | awk '{print $3}')

# Determine most efficient strategy
if (( $(echo "$DP_TIME < $TP_TIME" | bc -l) )) && (( $(echo "$DP_TIME < $PP_TIME" | bc -l) )); then
    FASTEST="Data Parallelism"
    FASTEST_TIME=$DP_TIME
elif (( $(echo "$TP_TIME < $PP_TIME" | bc -l) )); then
    FASTEST="Tensor Parallelism"
    FASTEST_TIME=$TP_TIME
else
    FASTEST="Pipeline Parallelism"
    FASTEST_TIME=$PP_TIME
fi

# Calculate speedup relative to Data Parallelism (baseline)
DP_SPEEDUP="1.00"
TP_SPEEDUP=$(echo "scale=2; $DP_TIME / $TP_TIME" | bc)
PP_SPEEDUP=$(echo "scale=2; $DP_TIME / $PP_TIME" | bc)

# Create comparison report
cat > ./checkpoints/hw2/comparison_report.txt << EOL
================================================
Distributed Training Comparison Report
================================================

Strategy          | Time per Epoch (s) | Speedup vs DP | Perplexity
-----------------+-------------------+--------------+------------
Data Parallelism  | ${DP_TIME}        | ${DP_SPEEDUP}        | ${DP_PPL}
Tensor Parallelism| ${TP_TIME}        | ${TP_SPEEDUP}        | ${TP_PPL}
Pipeline Parallel | ${PP_TIME}        | ${PP_SPEEDUP}        | ${PP_PPL}

The most efficient strategy in terms of time per epoch is:
${FASTEST} with ${FASTEST_TIME} seconds per epoch.

EOL

echo "All experiments completed! See ./checkpoints/hw2/comparison_report.txt for results."
