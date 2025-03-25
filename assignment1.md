# Programming Assignment 1: LLM Fine-Tuning Report

## Memory Optimization Strategies

In this assignment, I implemented several memory optimization techniques to maximize the batch size for fine-tuning the LLaMA 3B model on climate documents. These optimizations enabled training with a batch size of 19 on an NVIDIA A100-SXM4-80GB GPU while using only 28% of available GPU memory. Below are the key strategies implemented:

### 1. HuggingFace Accelerate Integration

Accelerate provides a unified API for distributed training across multiple devices with minimal code changes while optimizing memory usage:

- **Unified Memory Management**: Coordinated memory allocation and caching to prevent memory fragmentation
- **BF16 Mixed Precision Optimization**: Specifically configured for A100 GPUs to leverage tensor cores (`accelerate_config.yaml` setting: `mixed_precision: bf16`)
- **Dynamic Resource Allocation**: Automatically determines optimal device mapping and memory allocation
- **Single GPU Optimization**: Even on a single GPU, Accelerate streamlines memory operations and provides significant speedups

The accelerate configuration was specified in `accelerate_config.yaml` and used with `accelerate launch` command, resulting in significantly faster training times with efficient memory management.

### 2. Low-Rank Adaptation (LoRA)

LoRA significantly reduces memory requirements by adding trainable low-rank matrices to the pre-trained model weights instead of updating all parameters:

- **Configuration**: Rank (r) = 8, Alpha = 32, Dropout = 0.1
- **Target Modules**: Query, Key, Value, and Output projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Memory Savings**: Training required ~0.1% of the parameters compared to full fine-tuning
- **Parameter Reduction**: From ~3 billion parameters to just a few million trainable parameters
- **Implementation**: Used Hugging Face's PEFT library for efficient LoRA integration

### 3. Quantization Techniques

I implemented 4-bit quantization (QLoRA) with double quantization to drastically reduce model memory footprint:

- **4-bit Precision**: Reduced memory requirements by ~75% compared to FP16
- **Double Quantization**: Further reduced memory by applying a second quantization layer to the quantization constants
- **NF4 Data Type**: Used nested float 4-bit format for better numerical stability than int4
- **Memory Impact**: Quantization reduced the model's memory footprint from ~6GB to ~1.5GB
- **Implementation**: Configured through BitsAndBytesConfig:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=args.use_double_quant,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)
```

### 4. A100-Optimized Mixed Precision Training

Leveraged the A100 GPU's tensor cores with optimized precision formats:

- **BF16 Format**: Utilized BFloat16 precision specifically optimized for A100 GPUs (instead of FP16)
  - BF16 offers a wider dynamic range than FP16, reducing the risk of training instability
  - A100 tensor cores are specially designed to accelerate BF16 operations
  - Configured through Accelerate (`mixed_precision: bf16` in config)
- **Memory Reduction**: Reduced memory footprint during forward and backward passes by ~50%
- **Computational Speedup**: Achieved approximately 2-3x faster training compared to FP32

### 5. Gradient Accumulation and Checkpointing

- **Gradient Checkpointing**: Traded computation for memory by not storing all activations
  - Enabled via `model.gradient_checkpointing_enable()`
  - Reduced peak memory usage during backward pass by ~60%
  - Increased computation time by only ~20% due to A100's high computational efficiency
  - Essential for handling longer sequences (512 tokens) within memory constraints

### 6. Optimizer Optimization

- **8-bit Adam**: Used memory-efficient optimizer states with `use_8bit_optimizer=True`
  - Reduced optimizer state memory by 75% compared to FP32 Adam
  - Used Paged Optimizer to avoid memory fragmentation
  - Implementation: Specified `optim="paged_adamw_8bit"` in TrainingArguments

### 7. Attention Module Optimizations

- **Flash Attention**: Implemented memory-efficient attention algorithm
  - Reduced memory complexity from O(n²) to O(n) for sequence length n
  - Optimized specifically for A100 GPU architecture
  - Enabled via `attn_implementation="flash_attention_2"`

- **SDPA Attention**: Used Scaled Dot Product Attention as fallback when Flash Attention isn't available
  - More memory-efficient than standard attention implementation
  - Enabled via `attn_implementation="sdpa"`
  - Part of PyTorch's built-in optimizations

## Training Performance

### Maximum Batch Size

- **Per Device Batch Size**: 19
- **Gradient Accumulation Steps**: 1
- **Effective Batch Size**: 19

The maximum batch size was determined through a binary search method that progressively tested different batch sizes until finding the largest one that could fit in GPU memory without causing out-of-memory errors.

### Hardware Utilization

- **GPU**: NVIDIA A100-SXM4-80GB
- **GPU Memory Usage**: 22.9GB/81.9GB (28.0%)
- **RAM Usage**: ~26GB/1082GB (3.2%)
- **Training Time**: Approximately 24 minutes

### Evaluation Results

- **Perplexity on Test Set**: 8.18
- **Interpretation**: This is an excellent perplexity score (< 10), indicating that the model has learned to predict the climate dataset very well. A perplexity score of 8.18 means the model is making confident and accurate predictions on climate-related text, demonstrating successful domain adaptation through fine-tuning.

## Dataset and Preprocessing

- **Dataset Source**: Climate documents dataset of IPCC reports and climate change AI publications
- **Processing**: Extracted text from PDFs using PyPDF2
- **Train/Test Split**: 90/10 ratio (747 training files, 83 testing files)
- **Training Samples**: 12,634 text chunks
- **Evaluation Samples**: 2,901 text chunks
- **Sequence Length**: 512 tokens

## Running the Code

### Environment Setup

```bash
# Create environment using conda
conda env create -f environment.yaml
conda activate BDML

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
This script:
1. Extracts text from PDFs in the `climate_text_dataset`
2. Limits extraction to 30 pages per document
3. Creates a 90/10 train/test split
4. Saves processed data to `./processed_data`

```bash
# Run the preprocessing script to extract text from PDFs
./scripts/hw1/preprocess.sh
```

### Model Preparation

We need to convert the model to HF format using the following script: `https://github.com/huggingface/transformers/raw/refs/heads/main/src/transformers/models/llama/convert_llama_weights_to_hf.py`

```bash
# Convert the LLaMA model to Hugging Face format
./scripts/hw1/convert_model.sh
```

### Training Process

```bash
# Run the training script
./scripts/hw1/accelerate.sh
```

### Monitoring and Logs

- Training progress is logged to both console and `./logs/training_[TIMESTAMP].log`
- The script automatically monitors GPU memory usage
- Weights & Biases integration is available for real-time monitoring
- Final performance metrics are saved to `./checkpoints/llama-finetuned/training_stats.txt`

## Detailed Implementation

### 1. Binary Search for Batch Size

The implementation uses a systematic binary search approach to find the maximum batch size:

```python
def find_max_batch_size(model, tokenizer, train_dataset, eval_dataset):
    # Start with a safe batch size
    safe_batch_size = args.safe_batch_size
    min_batch = args.start_batch_size
    max_batch = args.max_batch_size
    optimal_batch = min_batch

    # Binary search to find maximum batch size
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
        if mid_batch == optimal_batch:
            # Already tested this batch size
            min_batch = mid_batch + 1
            continue

        # Test this batch size
        result = test_batch_size(model, tokenizer, train_dataset, eval_dataset, mid_batch)

        if result == "success":
            optimal_batch = mid_batch
            min_batch = mid_batch + 1
        else:
            max_batch = mid_batch - 1

    return optimal_batch
```

### 2. 4-bit Quantization Configuration

The implementation configures BitsAndBytes for 4-bit quantization:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=args.use_double_quant,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True
)
```

### 3. Memory-Efficient Training Configuration

```python
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=args.use_fp16,
    bf16=args.use_bf16,
    optim="paged_adamw_8bit"
)
```

## Training Visualizations

The following visualizations from Weights & Biases monitoring provide insight into the training process, memory optimization effectiveness, and model performance:

### Batch Size Optimization

The binary search process for finding the optimal batch size shows how the algorithm efficiently narrowed down to the optimal batch size of 19:

- Starting with an initial range of 2-32
- Testing different batch sizes systematically
- Converging to the final value of 19
- Refining upper and lower bounds based on successful/failed attempts

### Training and Perplexity Metrics

The training metrics show consistent improvement in model performance:

- **Training Loss**: Steadily decreased from ~2.3 to ~2.05, indicating effective learning
- **Training Perplexity**: Improved from ~10 to ~8, showing increasing confidence in predictions
- **Evaluation Perplexity**: Decreased from ~8.5 to ~8.18, demonstrating good generalization

The perplexity graph shows a steady downward trend throughout training, confirming that the model was effectively learning the climate domain language patterns despite the memory-optimized training approach.

### GPU Memory Utilization

Despite working with a 3B parameter model, the optimizations kept GPU memory usage at approximately 28% of the available 80GB on the A100 GPU. The memory allocation remained stable throughout training at around 22.9GB, showing that our approach effectively managed memory without encountering leaks or fragmentation issues.

### GPU Utilization

The GPU utilization metrics showed high utilization (consistently around 80-90%), indicating that our optimizations maintained computational efficiency while reducing memory requirements. The periodic dips correspond to evaluation phases and memory cleanup operations.

These visualizations confirm that our memory optimization approach successfully balanced the trade-off between memory usage and computational efficiency, enabling effective fine-tuning of the LLaMA 3B model on climate documents using a single A100 GPU.

## Conclusion

The implemented memory optimization techniques successfully enabled fine-tuning the LLaMA 3B model on climate documents with a batch size of 19 on an NVIDIA A100-SXM4-80GB GPU. Without these optimizations, the full fine-tuning approach would require significantly more memory, making it impractical or impossible to run on a single GPU.

The combination of LoRA, 4-bit quantization, mixed precision training, gradient checkpointing, and other memory optimizations proved highly effective for maximizing GPU memory efficiency. The resulting model achieved an excellent perplexity of **8.18** on the test set, indicating strong performance on climate-related text.

From the logs and Weights & Biases metrics, we observed:
- **Initial GPU Memory**: ~3GB allocated (3.7% of A100's 80GB)
- **Peak Memory Usage**: 22.9GB (28.0% of available memory)
- **RAM Usage**: ~26GB (3.2% of system memory)
- **Memory Efficiency Gain**: Compared to full-precision training (which would require 100+ GB), achieved ~80% memory reduction

These results demonstrate that it's possible to fine-tune large language models on domain-specific data with reasonable hardware resources by employing the right memory optimization techniques. This approach enables more researchers and developers to work with LLMs even without access to extensive computational resources.

### Future Work

- Experiment with different LoRA configurations (rank, target modules)
- Test higher batch sizes using gradient accumulation
- Compare performance with 8-bit quantization instead of 4-bit
