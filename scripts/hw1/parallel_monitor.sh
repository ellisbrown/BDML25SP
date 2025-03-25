#!/bin/bash

# This script monitors parallel experiments running on multiple GPUs
# Usage: ./monitor-experiments.sh

EXPERIMENTS_DIR="./experiments"

# ANSI color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Parallel Experiments Monitor =====${NC}"
echo ""

# Check if there are any experiments
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo -e "${RED}No experiments directory found. Run parallel-experiments.sh first.${NC}"
    exit 1
fi

# Get GPU usage information
echo -e "${CYAN}GPU Usage:${NC}"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | while read -r line; do
    IFS=',' read -ra GPU_INFO <<< "$line"
    GPU_ID=$(echo ${GPU_INFO[0]} | xargs)
    GPU_NAME=$(echo ${GPU_INFO[1]} | xargs)
    GPU_UTIL=$(echo ${GPU_INFO[2]} | xargs)
    GPU_MEM_USED=$(echo ${GPU_INFO[3]} | xargs)
    GPU_MEM_TOTAL=$(echo ${GPU_INFO[4]} | xargs)

    # Calculate memory usage percentage
    MEM_USED_VAL=$(echo $GPU_MEM_USED | sed 's/[^0-9]*//g')
    MEM_TOTAL_VAL=$(echo $GPU_MEM_TOTAL | sed 's/[^0-9]*//g')
    MEM_PERCENT=$((MEM_USED_VAL * 100 / MEM_TOTAL_VAL))

    echo -e "  GPU $GPU_ID ($GPU_NAME): ${YELLOW}$GPU_UTIL${NC}, Memory: $GPU_MEM_USED / $GPU_MEM_TOTAL ($MEM_PERCENT%)"
done

echo ""
echo -e "${CYAN}Experiment Status:${NC}"

# List all experiment directories
for EXP_DIR in $EXPERIMENTS_DIR/*/; do
    if [ -d "$EXP_DIR" ]; then
        EXP_NAME=$(basename "$EXP_DIR")

        # Check if there's a PID file and if the process is still running
        PID_FILE="$EXP_DIR/process_pid.txt"
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null; then
                STATUS="${GREEN}Running${NC}"

                # Try to extract batch size and epoch information from the log file
                LOG_FILE=$(find "$EXPERIMENTS_DIR" -name "${EXP_NAME}_gpu*.log" | head -n 1)
                if [ -f "$LOG_FILE" ]; then
                    # Extract the latest batch size info
                    BATCH_SIZE=$(grep "Maximum batch size found:" "$LOG_FILE" | tail -n 1 | awk '{print $NF}')

                    # Extract training progress
                    EPOCH_INFO=$(grep -o "Epoch [0-9]*/[0-9]*" "$LOG_FILE" | tail -n 1)

                    if [ ! -z "$BATCH_SIZE" ]; then
                        STATUS="$STATUS - Batch size: $BATCH_SIZE"
                    fi

                    if [ ! -z "$EPOCH_INFO" ]; then
                        STATUS="$STATUS - $EPOCH_INFO"
                    fi
                fi
            else
                # Process not running - check if completed successfully
                if [ -f "$EXP_DIR/training_stats.txt" ]; then
                    STATUS="${CYAN}Completed${NC}"
                    PERPLEXITY=$(grep "Final perplexity:" "$EXP_DIR/training_stats.txt" | awk '{print $3}')
                    BATCH_SIZE=$(grep "Maximum batch size:" "$EXP_DIR/training_stats.txt" | awk '{print $3}')
                    if [ ! -z "$PERPLEXITY" ] && [ ! -z "$BATCH_SIZE" ]; then
                        STATUS="$STATUS - Batch size: $BATCH_SIZE, Perplexity: $PERPLEXITY"
                    fi
                else
                    STATUS="${RED}Failed${NC}"
                fi
            fi
        else
            STATUS="${YELLOW}Unknown${NC}"
        fi

        echo -e "  $EXP_NAME: $STATUS"
    fi
done

echo ""
echo -e "${CYAN}Recent Log Output:${NC}"
for EXP_DIR in $EXPERIMENTS_DIR/*/; do
    if [ -d "$EXP_DIR" ]; then
        EXP_NAME=$(basename "$EXP_DIR")
        LOG_FILE=$(find "$EXPERIMENTS_DIR" -name "${EXP_NAME}_gpu*.log" | head -n 1)

        if [ -f "$LOG_FILE" ]; then
            echo -e "${YELLOW}=== Last 5 lines from $EXP_NAME ===${NC}"
            tail -n 5 "$LOG_FILE"
            echo ""
        fi
    fi
done

echo -e "${BLUE}==========================================${NC}"
echo -e "Run this script again to refresh status: ${GREEN}./monitor-experiments.sh${NC}"
