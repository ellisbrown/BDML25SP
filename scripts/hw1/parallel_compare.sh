#!/bin/bash

# This script collects and compares results from all completed experiments
# Usage: ./compare-results.sh

EXPERIMENTS_DIR="./experiments"
RESULTS_FILE="./experiment_results_summary.txt"

# ANSI color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Experiment Results Comparison =====${NC}"
echo ""

# Check if there are any experiments
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo -e "${RED}No experiments directory found. Run parallel-experiments.sh first.${NC}"
    exit 1
fi

# Create a header for the results file
echo "Experiment Results Summary - $(date)" > "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Initialize arrays for data
declare -a exp_names
declare -a batch_sizes
declare -a effective_batch_sizes
declare -a perplexities
declare -a training_times

# Collect results from all completed experiments
echo -e "${CYAN}Collecting results from completed experiments...${NC}"
for EXP_DIR in $EXPERIMENTS_DIR/*/; do
    if [ -d "$EXP_DIR" ]; then
        EXP_NAME=$(basename "$EXP_DIR")
        STATS_FILE="$EXP_DIR/training_stats.txt"

        if [ -f "$STATS_FILE" ]; then
            echo -e "  Found results for ${GREEN}$EXP_NAME${NC}"

            # Extract key metrics
            BATCH_SIZE=$(grep "Maximum batch size:" "$STATS_FILE" | awk '{print $3}')
            GRAD_ACCUM=$(grep "Gradient accumulation steps:" "$STATS_FILE" | awk '{print $3}')
            EFFECTIVE_BATCH=$(grep "Effective batch size:" "$STATS_FILE" | awk '{print $3}')
            PERPLEXITY=$(grep "Final perplexity:" "$STATS_FILE" | awk '{print $3}')

            # Extract training time
            TRAINING_TIME=$(grep "Total training time:" "$STATS_FILE" | sed 's/Total training time: //')

            # Store in arrays
            exp_names+=("$EXP_NAME")
            batch_sizes+=("$BATCH_SIZE")
            effective_batch_sizes+=("$EFFECTIVE_BATCH")
            perplexities+=("$PERPLEXITY")
            training_times+=("$TRAINING_TIME")

            # Add detailed results to the summary file
            echo "Experiment: $EXP_NAME" >> "$RESULTS_FILE"
            echo "-----------------------------------------" >> "$RESULTS_FILE"
            echo "Maximum batch size: $BATCH_SIZE" >> "$RESULTS_FILE"
            echo "Gradient accumulation steps: $GRAD_ACCUM" >> "$RESULTS_FILE"
            echo "Effective batch size: $EFFECTIVE_BATCH" >> "$RESULTS_FILE"
            echo "Final perplexity: $PERPLEXITY" >> "$RESULTS_FILE"
            echo "Training time: $TRAINING_TIME" >> "$RESULTS_FILE"
            echo "" >> "$RESULTS_FILE"

            # Add experiment configuration
            echo "Configuration:" >> "$RESULTS_FILE"
            grep -A 50 "Arguments:" "$STATS_FILE" >> "$RESULTS_FILE"
            echo "" >> "$RESULTS_FILE"
            echo "" >> "$RESULTS_FILE"
        fi
    fi
done

# Sort experiments by effective batch size (highest first)
# Create a temporary array for sorting
for ((i=0; i<${#exp_names[@]}; i++)); do
    sorted_data[$i]="${effective_batch_sizes[$i]}|${exp_names[$i]}|${batch_sizes[$i]}|${perplexities[$i]}|${training_times[$i]}"
done

# Sort the array
IFS=$'\n' sorted_data=($(sort -rn -t '|' -k1 <<<"${sorted_data[*]}"))
unset IFS

# Display comparative table of results
echo ""
echo -e "${CYAN}Comparative Results (Sorted by Effective Batch Size):${NC}"
echo -e "${YELLOW}---------------------------------------------------------------------${NC}"
printf "%-25s %-15s %-15s %-15s %s\n" "Experiment" "Batch Size" "Effective Batch" "Perplexity" "Training Time"
echo -e "${YELLOW}---------------------------------------------------------------------${NC}"

for data in "${sorted_data[@]}"; do
    IFS='|' read -ra PARTS <<< "$data"
    printf "%-25s %-15s %-15s %-15s %s\n" "${PARTS[1]}" "${PARTS[2]}" "${PARTS[0]}" "${PARTS[3]}" "${PARTS[4]}"
done

echo -e "${YELLOW}---------------------------------------------------------------------${NC}"

# Find the best experiment for batch size (highest effective batch)
if [ ${#sorted_data[@]} -gt 0 ]; then
    IFS='|' read -ra BEST_BATCH <<< "${sorted_data[0]}"
    echo -e "${GREEN}Best configuration for batch size: ${BEST_BATCH[1]} (Effective batch: ${BEST_BATCH[0]})${NC}"

    # Find the best experiment for perplexity (lowest value)
    best_ppl_idx=0
    best_ppl=${perplexities[0]}

    for ((i=1; i<${#perplexities[@]}; i++)); do
        if (( $(echo "${perplexities[$i]} < $best_ppl" | bc -l) )); then
            best_ppl=${perplexities[$i]}
            best_ppl_idx=$i
        fi
    done

    echo -e "${GREEN}Best configuration for perplexity: ${exp_names[$best_ppl_idx]} (Perplexity: ${perplexities[$best_ppl_idx]})${NC}"
fi

echo ""
echo -e "${BLUE}Detailed results written to: ${YELLOW}$RESULTS_FILE${NC}"
echo -e "${BLUE}==========================================${NC}"
