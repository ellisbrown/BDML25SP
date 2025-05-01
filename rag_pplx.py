import argparse
import os
import glob
import json
import logging
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import faiss # faiss-cpu or faiss-gpu
import random
import hashlib # For caching keys
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from datasets import Dataset, load_from_disk # For dataset handling like in HW1

# --- Logging Setup ---
# Configure logging to output to both console and a file
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_file = f"logs/hw3/rag_pipeline_{timestamp}.log"
os.makedirs("logs/hw3", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RAG perplexity using LLaMA")

    # Paths
    parser.add_argument("--model_path", type=str, default="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted",
                        help="Path to the base LLaMA 3B model (Hugging Face format)")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed text data (.txt files)")
    parser.add_argument("--output_dir", type=str, default="./rag_perplexity_output",
                        help="Directory to save FAISS index, results, etc.")
    parser.add_argument("--cache_dir", type=str, default="./dataset_cache", # From HW1
                        help="Directory to cache datasets")
    parser.add_argument("--faiss_index_path", type=str, default="./rag_perplexity_output/train_faiss_index.idx",
                        help="Path to save/load the FAISS index (built on train set)")
    parser.add_argument("--train_chunks_path", type=str, default="./rag_perplexity_output/train_chunks.json",
                        help="Path to save/load the ordered list of training chunks")
    parser.add_argument("--test_chunks_path", type=str, default="./rag_perplexity_output/test_chunks.json",
                        help="Path to save/load the list of test chunks")


    # Data & Chunking Parameters
    parser.add_argument("--train_test_split", type=float, default=0.9, # From HW1
                        help="Ratio of train/test split for documents")
    parser.add_argument("--chunk_size", type=int, default=300,
                        help="Target chunk size in tokens")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Overlap between chunks in tokens")

    # RAG Parameters
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model for embeddings")
    parser.add_argument("--use_gpu_for_embeddings", action="store_true", default=True,
                         help="Use GPU for SentenceTransformer encoding if available")
    parser.add_argument("--faiss_index_type", type=str, default="IndexFlatL2",
                        help="FAISS index type (e.g., IndexFlatL2, IndexIVFPQ)")
    # TODO: Add FAISS params for IVFPQ if needed
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of training chunks to retrieve for context")

    # Evaluation Parameters
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for perplexity evaluation")
    parser.add_argument("--max_length", type=int, default=512, # Max length for model input
                        help="Maximum sequence length for the generator model")

    # System Parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device for LLaMA model (e.g., cuda:0 or cpu)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True) # Ensure cache dir exists
    # Adjust index/chunk paths based on output_dir
    args.faiss_index_path = os.path.join(args.output_dir, "train_faiss_index.idx")
    args.train_chunks_path = os.path.join(args.output_dir, "train_chunks.json")
    args.test_chunks_path = os.path.join(args.output_dir, "test_chunks.json")

    return args

# --- Data Loading & Splitting (Adapted from HW1) ---

def split_train_test_files(data_dir, train_ratio=0.9, seed=42):
    """Splits .txt files in a directory into training and testing lists."""
    all_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not all_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")
    random.seed(seed)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    test_files = all_files[split_idx:]
    logging.info(f"Split {len(all_files)} files into {len(train_files)} train and {len(test_files)} test files.")
    return train_files, test_files

def chunk_texts(texts, sources, chunk_size, chunk_overlap, tokenizer):
    """Chunks a list of texts using the tokenizer."""
    all_chunks = []
    all_chunk_sources = []
    for text, source in tqdm(zip(texts, sources), total=len(texts), desc="Chunking texts"):
        if not text.strip():
            continue
        tokens = tokenizer.encode(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            # Skip chunk if it's too short (e.g., less than overlap)
            if len(chunk_tokens) < chunk_overlap and i > 0:
                continue
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                all_chunks.append(chunk_text)
                all_chunk_sources.append(source)
    return all_chunks, all_chunk_sources


def load_and_chunk_split(file_list, chunk_size, chunk_overlap, tokenizer, split_name="train"):
    """Loads texts from files and chunks them."""
    logging.info(f"Loading and chunking {split_name} files...")
    texts = []
    sources = []
    for file_path in tqdm(file_list, desc=f"Reading {split_name} files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
                sources.append(os.path.basename(file_path))
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    chunks, chunk_sources = chunk_texts(texts, sources, chunk_size, chunk_overlap, tokenizer)
    logging.info(f"Created {len(chunks)} chunks for {split_name} split.")
    return chunks, chunk_sources


# --- Embedding and Indexing (Train Set Only) ---
def build_or_load_train_index(train_chunks, train_chunk_sources, embedding_model, index_path, chunks_path, index_type="IndexFlatL2"):
    """Builds/Loads FAISS index for training chunks."""
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        logging.info(f"Loading existing TRAIN FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        logging.info(f"Loading corresponding TRAIN chunks from {chunks_path}")
        with open(chunks_path, 'r') as f:
            saved_data = json.load(f)
        loaded_chunks = saved_data['chunks']
        # loaded_sources = saved_data['sources'] # Optionally load sources
        logging.info(f"Loaded {len(loaded_chunks)} train chunks and index with {index.ntotal} vectors.")
        if len(loaded_chunks) != index.ntotal:
             logging.warning(f"Mismatch between loaded train chunks ({len(loaded_chunks)}) and index size ({index.ntotal}). Rebuilding index.")
             return build_or_load_train_index(train_chunks, train_chunk_sources, embedding_model, index_path + ".rebuild", chunks_path + ".rebuild", index_type)
        return index, loaded_chunks
    else:
        logging.info(f"Building new TRAIN FAISS index ({index_type})...")
        if not train_chunks:
             logging.error("No training chunks provided to build index.")
             return None, []

        # Determine embedding dimension
        sample_embedding = embedding_model.encode([train_chunks[0]], convert_to_numpy=True)
        d = sample_embedding.shape[1]
        logging.info(f"Embedding dimension: {d}")

        # Create FAISS index
        if index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(d)
        elif index_type == "IndexIVFPQ":
             nlist = 100; m = 8; nbits = 8 # Example params
             quantizer = faiss.IndexFlatL2(d)
             index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
             logging.info(f"Using IndexIVFPQ (nlist={nlist}, m={m}, nbits={nbits}). Training...")
             num_train_samples = max(nlist * 50, len(train_chunks) // 10, 10000)
             train_indices = random.sample(range(len(train_chunks)), min(num_train_samples, len(train_chunks)))
             embeddings_for_training = embedding_model.encode(
                 [train_chunks[i] for i in train_indices], convert_to_numpy=True, show_progress_bar=True
             )
             if embeddings_for_training.shape[0] < nlist:
                 logging.warning(f"IVFPQ training samples ({embeddings_for_training.shape[0]}) < nlist ({nlist}).")
             index.train(embeddings_for_training) # Corrected call
             logging.info("Index training complete.")
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

        # Encode all training chunks
        logging.info(f"Encoding all {len(train_chunks)} training chunks...")
        chunk_embeddings = embedding_model.encode(
            train_chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=128
        )

        # Add embeddings to index
        logging.info(f"Adding {chunk_embeddings.shape[0]} embeddings to the index...")
        index.add(chunk_embeddings) # No need for n_vectors with numpy input
        logging.info(f"Index built successfully with {index.ntotal} vectors.")

        # Save index and chunks
        logging.info(f"Saving train index to {index_path}")
        faiss.write_index(index, index_path)
        logging.info(f"Saving train chunk list to {chunks_path}")
        with open(chunks_path, 'w') as f:
             json.dump({'chunks': train_chunks, 'sources': train_chunk_sources}, f)

        return index, train_chunks

# --- Retrieval (Query with Test Chunk, Retrieve from Train Index) ---
def retrieve_train_context(query_chunk_text, train_index, embedding_model, train_chunks_list, top_k=3):
    """Retrieves top_k train chunks relevant to the query (test) chunk."""
    start_time = time.time()
    # Encode the query chunk (which is a test chunk text)
    query_vec = embedding_model.encode([query_chunk_text], convert_to_numpy=True)
    # Search the training index
    distances, indices = train_index.search(query_vec, top_k)
    retrieval_time = time.time() - start_time

    # Optional: Assert that distances are sorted (FAISS should handle this)
    if len(distances[0]) > 1 and not np.all(np.diff(distances[0]) >= -1e-6): # Allow for small floating point inaccuracies
         logging.warning(f"Distances may not be monotonically increasing. Check index behavior. Distances: {distances[0]}")
         # It's generally safe to proceed as FAISS aims to return the nearest first.

    retrieved_train_chunks = [train_chunks_list[i] for i in indices[0]]
    return retrieved_train_chunks, retrieval_time

# --- Perplexity Evaluation ---
def calculate_perplexity_for_chunk(context_chunks, target_chunk, model, tokenizer, device, max_length):
    """Calculates perplexity of the target_chunk given context_chunks."""

    # Construct the input sequence: context + target
    # Reverse context order so most relevant is last (closest to target)
    context = "\n\n".join(context_chunks[::-1])
    full_text = context + "\n\n" + target_chunk # Separator might matter

    # Tokenize the full sequence
    inputs = tokenizer(full_text, return_tensors="pt", truncation=False) # Don't truncate yet

    # --- Length Management ---
    total_tokens = inputs.input_ids.shape[1]
    if total_tokens > max_length:
        # Truncate from the beginning (context part)
        keep_tokens = max_length
        truncated_input_ids = inputs.input_ids[:, -keep_tokens:]
        truncated_attention_mask = inputs.attention_mask[:, -keep_tokens:]
        logging.warning(f"Input exceeds max_length ({total_tokens} > {max_length}). Truncating from left to {keep_tokens} tokens.")
    else:
        truncated_input_ids = inputs.input_ids
        truncated_attention_mask = inputs.attention_mask

    # --- Identify Target Token Span ---
    # Re-tokenize the target chunk *separately* to find its length
    # Important: Use add_special_tokens=False to avoid extra tokens messing up length calculation
    target_tokens = tokenizer.encode(target_chunk, add_special_tokens=False)
    target_len = len(target_tokens)

    # The target tokens should be at the *end* of the truncated input
    # Check if the target chunk was fully preserved after potential truncation
    if total_tokens > max_length:
        # Decode the end of the truncated input to verify target presence
        potential_target_in_truncated = tokenizer.decode(truncated_input_ids[0, -target_len:], skip_special_tokens=True)
        # This check might not be perfect due to tokenization artifacts, but gives an idea
        if not target_chunk.strip().endswith(potential_target_in_truncated.strip()):
             logging.error(f"Target chunk seems to be truncated. Cannot reliably calculate its perplexity. Target: '{target_chunk[:50]}...', End of Input: '{potential_target_in_truncated[:50]}...'")
             return None, None # Cannot calculate perplexity if target is cut off

    # Define start and end index for loss calculation (relative to truncated_input_ids)
    loss_start_index = truncated_input_ids.shape[1] - target_len
    loss_end_index = truncated_input_ids.shape[1] - 1 # Loss is calculated up to the second-to-last token prediction

    # Move to device
    input_ids = truncated_input_ids.to(device)
    attention_mask = truncated_attention_mask.to(device)
    labels = input_ids.clone() # Labels are the input tokens themselves for Causal LM

    # --- Model Forward Pass ---
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    # --- Calculate Loss ONLY on Target Tokens ---
    logits = outputs.logits

    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Select the logits and labels corresponding to the target chunk
    # The prediction for token `i` is in `shift_logits[:, i-1, :]`
    # The actual token `i` is `shift_labels[:, i-1]`
    # We want loss for predictions from `loss_start_index` up to `loss_end_index - 1`
    target_logits = shift_logits[:, loss_start_index-1:loss_end_index, :]
    target_labels = shift_labels[:, loss_start_index-1:loss_end_index]

    if target_logits.shape[1] == 0 or target_labels.shape[1] == 0:
        logging.warning("Target sequence length for loss calculation is zero. Skipping perplexity calculation.")
        return None, None

    # Calculate cross-entropy loss for the target sequence
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    target_loss = loss_fct(target_logits.view(-1, target_logits.size(-1)), target_labels.view(-1))

    # Calculate perplexity
    perplexity = torch.exp(target_loss)

    return target_loss.item(), perplexity.item()


# --- Model Loading ---
def load_llama_model(model_path, device):
    """Loads the base LLaMA model and tokenizer."""
    logging.info(f"Loading base LLaMA model from: {model_path}")
    # No quantization for perplexity calculation baseline usually
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", # Automatically distribute across available GPUs/CPU
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )
    model.eval() # Set to evaluation mode

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logging.info("Base LLaMA model and tokenizer loaded.")
    return model, tokenizer

# --- Main Workflow ---
def main():
    """Executes the RAG perplexity evaluation pipeline."""
    args = parse_args()
    logging.info(f"Starting RAG Perplexity Evaluation with args: {args}")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Load Models ---
    logging.info("Loading LLaMA model and tokenizer...")
    model, tokenizer = load_llama_model(args.model_path, args.device)

    logging.info(f"Loading embedding model: {args.embedding_model}")
    embed_device = 'cuda' if args.use_gpu_for_embeddings and torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(args.embedding_model, device=embed_device)
    logging.info(f"Embedding model loaded on device: {embed_device}")

    # --- Load, Split, and Chunk Data ---
    # Check if pre-chunked data exists
    if os.path.exists(args.train_chunks_path) and os.path.exists(args.test_chunks_path):
        logging.info("Loading pre-chunked train and test data...")
        with open(args.train_chunks_path, 'r') as f:
            train_data = json.load(f)
            train_chunks = train_data['chunks']
            train_chunk_sources = train_data['sources']
        with open(args.test_chunks_path, 'r') as f:
            test_data = json.load(f)
            test_chunks = test_data['chunks']
            # test_chunk_sources = test_data['sources'] # Optional
    else:
        logging.info("Pre-chunked data not found. Loading, splitting, and chunking documents...")
        train_files, test_files = split_train_test_files(args.data_dir, args.train_test_split, args.seed)
        train_chunks, train_chunk_sources = load_and_chunk_split(train_files, args.chunk_size, args.chunk_overlap, tokenizer, "train")
        test_chunks, _ = load_and_chunk_split(test_files, args.chunk_size, args.chunk_overlap, tokenizer, "test") # Sources not needed for test eval here

        # Save chunked data
        logging.info(f"Saving train chunks to {args.train_chunks_path}")
        with open(args.train_chunks_path, 'w') as f:
            json.dump({'chunks': train_chunks, 'sources': train_chunk_sources}, f)
        logging.info(f"Saving test chunks to {args.test_chunks_path}")
        with open(args.test_chunks_path, 'w') as f:
             json.dump({'chunks': test_chunks, 'sources': []}, f) # Save empty sources if not needed

    if not train_chunks or not test_chunks:
        logging.error("Failed to load/create train or test chunks. Exiting.")
        return

    # --- Build/Load Train Index ---
    train_index, train_chunks_in_index = build_or_load_train_index(
        train_chunks, train_chunk_sources, embedding_model,
        args.faiss_index_path, args.train_chunks_path, args.faiss_index_type
    )
    if not train_index:
        logging.error("Failed to build/load training index. Exiting.")
        return
    # Ensure we use the chunks corresponding to the index
    train_chunks = train_chunks_in_index

    # --- Evaluation Loop ---
    results = []
    total_perplexity_rag = 0
    total_perplexity_base = 0
    valid_rag_count = 0
    valid_base_count = 0
    total_retrieval_time = 0

    logging.info(f"Starting perplexity evaluation on {len(test_chunks)} test chunks...")
    # Consider batching the evaluation for efficiency if needed
    for i, test_chunk in enumerate(tqdm(test_chunks, desc="Evaluating Test Chunks")):
        # 1. Retrieve context from training index based on test chunk
        context_chunks, retrieval_time = retrieve_train_context(
            test_chunk, train_index, embedding_model, train_chunks, args.top_k
        )
        total_retrieval_time += retrieval_time

        # 2. Calculate Perplexity with RAG context
        loss_rag, ppl_rag = calculate_perplexity_for_chunk(
            context_chunks, test_chunk, model, tokenizer, args.device, args.max_length
        )

        # 3. Calculate Perplexity without RAG context (Baseline)
        loss_base, ppl_base = calculate_perplexity_for_chunk(
            [], test_chunk, model, tokenizer, args.device, args.max_length # Empty context
        )

        # Store results
        results.append({
            "test_chunk_index": i,
            "retrieval_time_s": retrieval_time,
            "perplexity_rag": ppl_rag,
            "loss_rag": loss_rag,
            "perplexity_base": ppl_base,
            "loss_base": loss_base,
            # "test_chunk_text": test_chunk[:100] + "..." # Optional: store snippet
        })

        if ppl_rag is not None:
            total_perplexity_rag += ppl_rag
            valid_rag_count += 1
        if ppl_base is not None:
            total_perplexity_base += ppl_base
            valid_base_count += 1

        # Log progress periodically
        if (i + 1) % 50 == 0:
             logging.info(f"Processed {i+1}/{len(test_chunks)} chunks...")
             if valid_rag_count > 0:
                 logging.info(f"  Current Avg PPL (RAG): {total_perplexity_rag / valid_rag_count:.2f}")
             if valid_base_count > 0:
                 logging.info(f"  Current Avg PPL (Base): {total_perplexity_base / valid_base_count:.2f}")


    # --- Final Metrics and Reporting ---
    avg_retrieval_time = total_retrieval_time / len(test_chunks) if test_chunks else 0
    avg_perplexity_rag = total_perplexity_rag / valid_rag_count if valid_rag_count > 0 else None
    avg_perplexity_base = total_perplexity_base / valid_base_count if valid_base_count > 0 else None

    print("=" * 50)
    logging.info("Evaluation Summary:")
    logging.info(f"  Evaluated {len(test_chunks)} test chunks.")
    logging.info(f"  Average Retrieval Time per Chunk: {avg_retrieval_time:.4f} seconds")
    if avg_perplexity_rag is not None:
        logging.info(f"  Average Perplexity (RAG Context): {avg_perplexity_rag:.2f} ({valid_rag_count}/{len(test_chunks)} valid)")
    else:
        logging.info("  Average Perplexity (RAG Context): Not Calculated")
    if avg_perplexity_base is not None:
        logging.info(f"  Average Perplexity (Base Model): {avg_perplexity_base:.2f} ({valid_base_count}/{len(test_chunks)} valid)")
    else:
        logging.info("  Average Perplexity (Base Model): Not Calculated")
    print("=" * 50)

    # --- Comparison ---
    logging.info("Comparison Points:")
    logging.info(f"  - RAG Perplexity ({avg_perplexity_rag:.2f} if calculated) vs. Base Model Perplexity ({avg_perplexity_base:.2f} if calculated)")
    logging.info("    Lower RAG perplexity indicates the retrieved context helped the model predict the test chunk.")
    # Load HW1 fine-tuned perplexity if available for comparison
    hw1_ppl = 8.18 # Example value from your HW1 report
    logging.info(f"  - Compare with Fine-Tuned Model Perplexity (HW1): {hw1_ppl:.2f}")
    logging.info("    This compares domain adaptation (fine-tuning) vs. context injection (RAG) for improving predictions on test data.")
    logging.info(f"  - Inference Time: RAG involves retrieval ({avg_retrieval_time:.4f}s avg) + model forward pass per chunk.")
    logging.info("    Compare total RAG time per chunk vs. fine-tuned model inference time per chunk (needs measurement).")


    # Save detailed results
    results_path = os.path.join(args.output_dir, "rag_perplexity_results.json")
    logging.info(f"Saving detailed results to {results_path}")
    with open(results_path, 'w') as f:
         # Convert potential numpy types to standard types for JSON
        serializable_results = []
        for r in results:
            serializable_r = {}
            for k, v in r.items():
                if isinstance(v, np.floating): serializable_r[k] = float(v)
                elif isinstance(v, np.integer): serializable_r[k] = int(v)
                elif v is None: serializable_r[k] = None
                else: serializable_r[k] = v
            serializable_results.append(serializable_r)
        json.dump(serializable_results, f, indent=2)

    # Save summary
    summary_path = os.path.join(args.output_dir, "perplexity_summary.txt")
    logging.info(f"Saving summary to {summary_path}")
    with open(summary_path, 'w') as f:
        f.write("RAG Perplexity Evaluation Summary\n")
        f.write("="*30 + "\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Evaluated Test Chunks: {len(test_chunks)}\n")
        f.write(f"Embedding Model: {args.embedding_model}\n")
        f.write(f"FAISS Index Type: {args.faiss_index_type}\n")
        f.write(f"Retrieved Chunks (k): {args.top_k}\n")
        f.write(f"Generator Model: {args.model_path}\n")
        f.write(f"Max Sequence Length: {args.max_length}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Avg Retrieval Time: {avg_retrieval_time:.4f} s\n")
        f.write(f"Avg Perplexity (RAG): {avg_perplexity_rag:.2f if avg_perplexity_rag is not None else 'N/A'} ({valid_rag_count}/{len(test_chunks)} valid)\n")
        f.write(f"Avg Perplexity (Base): {avg_perplexity_base:.2f if avg_perplexity_base is not None else 'N/A'} ({valid_base_count}/{len(test_chunks)} valid)\n")
        f.write("="*30 + "\n")
        f.write("Arguments Used:\n")
        for arg, value in vars(args).items():
            f.write(f"  {arg}: {value}\n")

    logging.info("RAG perplexity evaluation finished.")


if __name__ == "__main__":
    main()
