# Programming Assignment 1: LLM Fine-Tuning Report

## Memory Optimization Strategies

In this assignment, I implemented several memory optimization techniques to maximize the batch size for fine-tuning the LLaMA 3B model on climate documents. Below are the key strategies tested:

### 1. Low-Rank Adaptation (LoRA)

LoRA significantly reduces memory requirements by adding trainable low-rank matrices to the pre-trained model weights instead of updating all parameters:

- **Configuration**: Rank (r) = 8, Alpha = 32, Dropout = 0.1
- **Target Modules**: Query, Key, Value, and Output projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Parameter Reduction**: Only ~0.1% of the parameters needed to be updated compared to full fine-tuning
- **Implementation**: Used Hugging Face's PEFT library for efficient LoRA integration

### 2. Quantization

I implemented 4-bit quantization (QLoRA) to drastically reduce model memory footprint:

- **Load In 4-bit**: Reduced memory requirements by ~75% compared to FP16
- **Compute Type**: Used FP16 computation for stability
- **Double Quantization**: Further reduced memory by applying a second level of quantization
- **Implementation**: Used the bitsandbytes library with BitsAndBytesConfig

### 3. Mixed Precision Training

- **FP16 Training**: Enabled mixed precision to reduce memory usage during forward and backward passes
- **BF16 Training**: Additionally enabled BF16 format for better numerical stability with certain operations
- **Implementation**: Activated via TrainingArguments with both `fp16=True` and `bf16=True`

### 4. Gradient Accumulation and Checkpointing

- **Gradient Checkpointing**: Traded computation for memory by not storing all activations
  - Enabled via `model.gradient_checkpointing_enable()`
  - Reduced memory usage during backward pass at the cost of extra computation
  - Essential for handling longer sequences within memory constraints

### 5. Optimizer Optimization

- **8-bit Adam**: Used memory-efficient optimizer states with `use_8bit_optimizer=True`
- **Implementation**: Specified `optim="paged_adamw_8bit"` in TrainingArguments

### 6. Attention Optimizations

- **SDPA Attention**: Used Scaled Dot Product Attention for more memory-efficient transformer operations
- **Flash Attention**: Implemented memory-efficient attention algorithm

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

    # Binary search to find maximum batch size
    while min_batch <= max_batch:
        mid_batch = (min_batch + max_batch) // 2
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

## Conclusion

The implemented memory optimization techniques successfully enabled fine-tuning the LLaMA 3B model on climate documents with a batch size of 19 on an NVIDIA A100-SXM4-80GB GPU. Without these optimizations, the full fine-tuning approach would require significantly more memory, making it impractical or impossible to run on a single GPU.

The combination of LoRA, 4-bit quantization, mixed precision training, gradient checkpointing, and other memory optimizations proved highly effective for maximizing GPU memory efficiency. The resulting model achieved an excellent perplexity of 8.18 on the test set, indicating strong performance on climate-related text.

These results demonstrate that it's possible to fine-tune large language models on domain-specific data with reasonable hardware resources by employing the right memory optimization techniques. This approach enables more researchers and developers to work with LLMs even without access to extensive computational resources.

### Future Work

- Experiment with different LoRA configurations (rank, target modules)
- Test higher batch sizes using gradient accumulation
- Compare performance with 8-bit quantization instead of 4-bit
