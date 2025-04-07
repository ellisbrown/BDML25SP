#!/bin/bash
set -e

echo "Starting LLaMA distributed training experiments on 8-GPU node..."

# Create directories
mkdir -p logs
mkdir -p checkpoints/llama-finetuned-distributed

# Create timestamp for unique run identification
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Remove any existing PID file
rm -f experiment_pids.txt

# Function to run experiment
run_experiment() {
    local EXP_NAME=$1
    local GPU_IDS=$2
    local SCRIPT=$3
    local EXTRA_ARGS=$4
    local LOG_FILE="logs/${EXP_NAME}_${TIMESTAMP}.log"
    
    echo "Starting $EXP_NAME on GPUs $GPU_IDS..."
    
    # Export GPU IDs for this experiment
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    
    # Determine number of GPUs for this experiment
    local NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    
    # If this is a data parallel experiment, we need to create a custom accelerate config
    if [[ "$SCRIPT" == *"data_parallel"* ]]; then
        local CONFIG_FILE="accelerate_config_${EXP_NAME}.yaml"
        cp accelerate_config_distributed.yaml "$CONFIG_FILE"
        # Update num_processes in config file
        sed -i "s/num_processes: 2/num_processes: ${NUM_GPUS}/g" "$CONFIG_FILE"
        # Add to extra args
        EXTRA_ARGS="--config_file $CONFIG_FILE $EXTRA_ARGS"
    fi
    
    # Create a custom output directory
    local OUTPUT_DIR="./checkpoints/llama-finetuned-distributed/${EXP_NAME}_${TIMESTAMP}"
    mkdir -p "$OUTPUT_DIR"
    EXTRA_ARGS="--output_dir $OUTPUT_DIR $EXTRA_ARGS"
    
    # Add wandb run name
    EXTRA_ARGS="--wandb_run_name hw2_${EXP_NAME}_${TIMESTAMP} $EXTRA_ARGS"
    
    # Execute the script
    bash $SCRIPT $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
    
    echo "Started $EXP_NAME job (PID: $!). Log: $LOG_FILE"
    echo "$EXP_NAME PID: $!" >> experiment_pids.txt
}

# Now we can run experiments with different GPU configurations
# Option 1: 4 separate experiments with 2 GPUs each
run_experiment "data_parallel_2gpu" "0,1" "./run_data_parallel.sh" ""
run_experiment "tensor_parallel_2gpu" "2,3" "./run_tensor_parallel.sh" ""
run_experiment "pipeline_parallel_2gpu" "4,5" "./run_pipeline_parallel.sh" ""
run_experiment "data_parallel_large_batch" "6,7" "./run_data_parallel.sh" "--batch_size 32"

# Option 2: 2 experiments with 4 GPUs each
# Uncomment these and comment out the above if you want to try 4-GPU experiments
# run_experiment "data_parallel_4gpu" "0,1,2,3" "./run_data_parallel.sh" ""
# run_experiment "pipeline_parallel_4gpu" "4,5,6,7" "./run_pipeline_parallel.sh" ""

# Option 3: Single experiment with all 8 GPUs
# Uncomment this and comment out the above if you want to try an 8-GPU experiment
# run_experiment "data_parallel_8gpu" "0,1,2,3,4,5,6,7" "./run_data_parallel.sh" "--batch_size 64"

echo "All experiments launched in background. Use 'tail -f logs/*.log' to monitor progress."
echo "PIDs are stored in experiment_pids.txt"

# Create a monitoring function
watch_experiments() {
    echo "Watching for experiment completion..."
    while true; do
        if [ ! -f experiment_pids.txt ]; then
            echo "No experiments running."
            break
        fi
        
        all_done=true
        while read pid_line; do
            exp_name=$(echo $pid_line | cut -d' ' -f1)
            pid=$(echo $pid_line | cut -d' ' -f3)
            if ps -p $pid > /dev/null; then
                all_done=false
                echo "$exp_name (PID: $pid) is still running..."
            else
                echo "$exp_name (PID: $pid) has completed!"
            fi
        done < experiment_pids.txt
        
        if $all_done; then
            echo "All experiments have completed!"
            break
        fi
        
        sleep 60  # Check every minute
    done
    
    # Generate a summary when all experiments are done
    echo "Generating summary report..."
    {
        echo "# Distributed Training Summary"
        echo "Date: $(date)"
        echo "Experiments from run: $TIMESTAMP"
        echo
        echo "| Experiment | GPUs | Parallelism | Batch Size | Epoch Time (s) | Perplexity |"
        echo "|------------|------|-------------|------------|----------------|------------|"
        
        # For each experiment, extract results
        for exp_dir in ./checkpoints/llama-finetuned-distributed/*_"${TIMESTAMP}"; do
            if [ -d "$exp_dir" ]; then
                exp_name=$(basename "$exp_dir" | sed "s/_${TIMESTAMP}//")
                if [ -f "${exp_dir}/training_stats.txt" ]; then
                    # Extract metrics
                    num_gpus=$(grep "World size" "${exp_dir}/training_stats.txt" | awk '{print $3}')
                    parallelism=$(grep "Parallelism Strategy" "${exp_dir}/training_stats.txt" | awk '{print $3}')
                    batch_size=$(grep "batch_size" "${exp_dir}/training_stats.txt" | awk '{print $3}')
                    epoch_time=$(grep "Average epoch time" "${exp_dir}/training_stats.txt" | awk '{print $4}')
                    perplexity=$(grep "Final perplexity" "${exp_dir}/training_stats.txt" | awk '{print $3}')
                    
                    echo "| $exp_name | $num_gpus | $parallelism | $batch_size | $epoch_time | $perplexity |"
                else
                    echo "| $exp_name | N/A | N/A | N/A | Training failed | N/A |"
                fi
            fi
        done
        
        echo
        echo "## Best Performing Configuration"
        # Find the experiment with the lowest epoch time
        best_time=9999999
        best_exp="none"
        
        for exp_dir in ./checkpoints/llama-finetuned-distributed/*_"${TIMESTAMP}"; do
            if [ -d "$exp_dir" ] && [ -f "${exp_dir}/training_stats.txt" ]; then
                epoch_time=$(grep "Average epoch time" "${exp_dir}/training_stats.txt" | awk '{print $4}')
                if (( $(echo "$epoch_time < $best_time" | bc -l) )); then
                    best_time=$epoch_time
                    best_exp=$(basename "$exp_dir" | sed "s/_${TIMESTAMP}//")
                fi
            fi
        done
        
        if [ "$best_exp" != "none" ]; then
            exp_dir="./checkpoints/llama-finetuned-distributed/${best_exp}_${TIMESTAMP}"
            num_gpus=$(grep "World size" "${exp_dir}/training_stats.txt" | awk '{print $3}')
            parallelism=$(grep "Parallelism Strategy" "${exp_dir}/training_stats.txt" | awk '{print $3}')
            batch_size=$(grep "batch_size" "${exp_dir}/training_stats.txt" | awk '{print $3}')
            
            echo "The best performing configuration was **$best_exp** with:"
            echo "- $num_gpus GPUs using $parallelism"
            echo "- Batch size: $batch_size"
            echo "- Average epoch time: $best_time seconds"
            perplexity=$(grep "Final perplexity" "${exp_dir}/training_stats.txt" | awk '{print $3}')
            echo "- Final perplexity: $perplexity"
        else
            echo "No successful experiments found."
        fi
    } > "distributed_summary_${TIMESTAMP}.md"
    
    echo "Summary report generated: distributed_summary_${TIMESTAMP}.md"
}

# Start the monitoring in background
watch_experiments &

echo "Experiment monitor started in background."
echo "You can safely disconnect. Experiments will continue to run."
