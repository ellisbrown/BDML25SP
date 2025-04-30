import argparse
import os
import glob
import json
import logging
import time
import torch
import numpy as np
import faiss # faiss-cpu or faiss-gpu
import random
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# --- Logging Setup ---
# Configure logging to output to both console and a file
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_file = f"logs/rag_pipeline_{timestamp}.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# --- Argument Parsing (Adapted from Assignment 1 & 3) ---
def parse_args():
    """Parses command-line arguments for the RAG pipeline."""
    parser = argparse.ArgumentParser(description="Build and evaluate a RAG system using LLaMA")

    # Paths
    parser.add_argument("--model_path", type=str, default="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted", # Or /scratch/BDML25SP/Llama3.2-3B-converted on HPC
                        help="Path to the base LLaMA 3B model (Hugging Face format)")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed text data (.txt files)")
    parser.add_argument("--output_dir", type=str, default="./rag_output",
                        help="Directory to save FAISS index, results, and potentially other outputs")
    parser.add_argument("--faiss_index_path", type=str, default="./rag_output/faiss_index.idx",
                        help="Path to save/load the FAISS index")
    parser.add_argument("--documents_path", type=str, default="./rag_output/documents.json",
                        help="Path to save/load the ordered list of document texts corresponding to the index")

    # RAG Parameters
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", # Example from Assignment 3
                        help="Name of the SentenceTransformer model to use for embeddings")
    parser.add_argument("--use_gpu_for_embeddings", action="store_true", default=True,
                         help="Use GPU for SentenceTransformer encoding if available")
    parser.add_argument("--faiss_index_type", type=str, default="IndexFlatL2", # Example from Assignment 3
                        help="Type of FAISS index to create (e.g., IndexFlatL2, IndexIVFPQ)")
    # TODO: Add more FAISS params if using complex indices like IVFPQ (nlist, nprobe, m)
    # parser.add_argument("--faiss_nlist", type=int, default=100, help="Number of clusters for IVF indices")
    # parser.add_argument("--faiss_m", type=int, default=8, help="Number of subquantizers for PQ indices")
    # parser.add_argument("--faiss_nbits", type=int, default=8, help="Number of bits per subquantizer for PQ indices")
    parser.add_argument("--chunk_size", type=int, default=300, # Typical size mentioned
                        help="Target chunk size in tokens (used if re-chunking)")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Overlap between chunks (used if re-chunking)")
    parser.add_argument("--top_k", type=int, default=3, # Retrieve top 3 most relevant docs
                        help="Number of documents/chunks to retrieve for context")

    # Generation Parameters
    parser.add_argument("--max_new_tokens", type=int, default=100, # Generate up to 100 new tokens
                        help="Maximum number of new tokens for the generator")
    parser.add_argument("--generation_temperature", type=float, default=0.7,
                         help="Temperature for generation sampling")
    parser.add_argument("--generation_top_p", type=float, default=0.9,
                         help="Top-p for generation sampling")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, # Optional: Quantize generator model
                        help="Load generator model in 4-bit precision")

    # Evaluation Parameters
    parser.add_argument("--eval_queries_file", type=str, default="eval_queries.json",
                        help="JSON file containing queries for evaluation (list of strings)")

    # System Parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for generation (e.g., cuda:0 or cpu)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args

# --- Data Loading & Chunking ---
def load_and_chunk_documents(data_dir, chunk_size, chunk_overlap, tokenizer):
    """
    Loads text documents from .txt files and splits them into chunks.
    NOTE: This is a placeholder for chunking. You might want to use
          LangChain's RecursiveCharacterTextSplitter or similar for
          more robust chunking based on the tokenizer.
    """
    logging.info(f"Loading and chunking documents from: {data_dir}")
    all_files = glob.glob(os.path.join(data_dir, "*.txt"))
    chunks = []
    chunk_sources = [] # Keep track of the source file for each chunk

    if not all_files:
        logging.error(f"No .txt files found in {data_dir}. Ensure preprocessing was completed.")
        return [], []

    for file_path in tqdm(all_files, desc="Reading and chunking documents"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    continue

                # --- Basic Placeholder Chunking ---
                # This simple split might not respect token boundaries well.
                # Consider using tokenizer-aware splitting.
                tokens = tokenizer.encode(content)
                for i in range(0, len(tokens), chunk_size - chunk_overlap):
                    chunk_tokens = tokens[i : i + chunk_size]
                    chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    if chunk_text.strip(): # Avoid empty chunks
                        chunks.append(chunk_text)
                        chunk_sources.append(os.path.basename(file_path))
                # --- End Placeholder Chunking ---

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")

    logging.info(f"Loaded and chunked into {len(chunks)} chunks.")
    return chunks, chunk_sources

# --- Embedding and Indexing ---
def build_or_load_index(chunks, embedding_model, index_path, documents_path, chunk_sources, index_type="IndexFlatL2"):
    """
    Builds a FAISS index from document chunks or loads it if it exists.
    Also saves/loads the corresponding chunk texts.
    """
    if os.path.exists(index_path) and os.path.exists(documents_path):
        logging.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        logging.info(f"Loading corresponding chunks from {documents_path}")
        with open(documents_path, 'r') as f:
            saved_data = json.load(f)
        loaded_chunks = saved_data['chunks']
        # loaded_sources = saved_data['sources'] # Optionally load sources
        logging.info(f"Loaded {len(loaded_chunks)} chunks and index with {index.ntotal} vectors.")
        # Optional: Verify consistency
        if len(loaded_chunks) != index.ntotal:
             logging.warning(f"Mismatch between loaded chunks ({len(loaded_chunks)}) and index size ({index.ntotal}). Rebuilding index.")
             # Force rebuild if inconsistent
             return build_or_load_index(chunks, embedding_model, index_path + ".rebuild", documents_path + ".rebuild", chunk_sources, index_type)
        return index, loaded_chunks # Return loaded chunks as they correspond to the index

    else:
        logging.info(f"Building new FAISS index ({index_type})...")
        # Determine embedding dimension
        logging.info("Determining embedding dimension...")
        # Encode a small sample to get dimension
        sample_embedding = embedding_model.encode([chunks[0]] if chunks else ["test"], convert_to_numpy=True)
        d = sample_embedding.shape[1]
        logging.info(f"Detected embedding dimension: {d}")

        # Create FAISS index based on type
        if index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(d)
            logging.info("Using IndexFlatL2.")
        elif index_type == "IndexIVFPQ":
             # Example for IndexIVFPQ - parameters would need to be args
             # nlist = args.faiss_nlist
             # m = args.faiss_m
             # nbits = args.faiss_nbits
             nlist = 100 # Number of clusters (Voronoi cells)
             m = 8      # Number of subquantizers
             nbits = 8  # bits per subquantizer index (8 bits = 256 centroids per subquantizer)
             quantizer = faiss.IndexFlatL2(d) # Base index for clustering
             index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
             logging.info(f"Using IndexIVFPQ (nlist={nlist}, m={m}, nbits={nbits}). Training required...")
             # --- Training the Index (Required for IVF indices) ---
             # Needs representative embeddings. Encode a subset or all chunks.
             # Ensure enough training data, typically > nlist * 39 samples recommended by Faiss docs
             num_train_samples = max(nlist * 50, len(chunks) // 10, 10000) # Heuristic for training size
             logging.info(f"Encoding {min(num_train_samples, len(chunks))} chunks for index training...")
             train_indices = random.sample(range(len(chunks)), min(num_train_samples, len(chunks)))
             embeddings_for_training = embedding_model.encode(
                 [chunks[i] for i in train_indices],
                 convert_to_numpy=True,
                 show_progress_bar=True
             )
             if embeddings_for_training.shape[0] < nlist:
                 logging.warning(f"Number of training samples ({embeddings_for_training.shape[0]}) is less than nlist ({nlist}). This might lead to poor clustering. Consider more data or smaller nlist.")
             index.train(embeddings_for_training)
             logging.info("Index training complete.")
             # Set nprobe (number of clusters to search) - higher means more accurate but slower
             # index.nprobe = 10 # Example value, could be an argument
        # TODO: Add other index types like HNSW if needed
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

        # Encode all chunks for adding to the index
        logging.info(f"Encoding all {len(chunks)} chunks for indexing...")
        chunk_embeddings = embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=128 # Adjust batch size based on GPU memory
        )

        logging.info(f"Adding {len(chunk_embeddings)} embeddings to the index...")
        index.add(chunk_embeddings)
        logging.info(f"Index built successfully with {index.ntotal} vectors.")

        # Save the index and the corresponding chunks/sources
        logging.info(f"Saving index to {index_path}")
        faiss.write_index(index, index_path)
        logging.info(f"Saving chunk list to {documents_path}")
        with open(documents_path, 'w') as f:
             json.dump({'chunks': chunks, 'sources': chunk_sources}, f) # Save sources too

        return index, chunks # Return the chunks used to build this index

# --- Retrieval ---
def retrieve_chunks(query, index, embedding_model, chunks_list, top_k=3):
    """Retrieves top_k chunks relevant to the query."""
    # logging.info(f"Retrieving top {top_k} chunks for query: '{query[:50]}...'")
    start_time = time.time()
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k) # Search returns distances and indices
    retrieval_time = time.time() - start_time

    retrieved_chunks_content = [chunks_list[i] for i in indices[0]]
    # logging.info(f"Retrieved indices: {indices[0]} in {retrieval_time:.4f} seconds.")
    # logging.info(f"Retrieved distances: {distances[0]}") # Smaller L2 distance is better
    return retrieved_chunks_content, retrieval_time

# --- Generation ---
def load_generator_model(model_path, load_in_4bit=False):
    """Loads the LLaMA model and tokenizer for generation."""
    logging.info(f"Loading generator model from: {model_path}")

    quantization_config = None
    if load_in_4bit:
         logging.info("Using 4-bit quantization for generator model.")
         quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True, # From Assignment 1
            bnb_4bit_quant_type="nf4"      # From Assignment 1
         )

    # Determine torch dtype based on availability and args
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", # Automatically distribute across available GPUs/CPU
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True # May be needed for some LLaMA versions/variants
    )

    # Set pad token if not set (common issue)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logging.info("Generator model and tokenizer loaded.")
    return model, tokenizer

def generate_answer(query, retrieved_chunks, model, tokenizer, device, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generates an answer based on the query and retrieved context."""
    start_time = time.time()
    context = "\n\n".join(retrieved_chunks) # Join chunks with double newline

    # --- Prompt Engineering ---
    # Simple prompt template, can be refined
    prompt = f"""Use the following context to answer the question below. If the context doesn't contain the answer, say "I cannot answer the question based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

    # --- Tokenization and Context Length Management ---
    # Ensure the total prompt length doesn't exceed model limits
    # A common strategy: prioritize query, then fill with context
    query_tokens = tokenizer.encode(f"\n\nQuestion: {query}\n\nAnswer:", add_special_tokens=False)
    prompt_template_tokens = tokenizer.encode("Context:\n\n", add_special_tokens=False) # Approx length of template parts
    max_context_len_tokens = tokenizer.model_max_length - len(query_tokens) - len(prompt_template_tokens) - max_new_tokens - 50 # Conservative buffer

    context_tokens = tokenizer.encode(context, add_special_tokens=False)

    if len(context_tokens) > max_context_len_tokens:
        logging.warning(f"Context length ({len(context_tokens)} tokens) exceeds limit ({max_context_len_tokens}). Truncating.")
        # Truncate context (from the beginning, assuming later chunks are more relevant - adjust if needed)
        truncated_context_tokens = context_tokens[-max_context_len_tokens:]
        truncated_context = tokenizer.decode(truncated_context_tokens)
        logging.info(f"Truncated context to {len(truncated_context_tokens)} tokens.")
        # Rebuild prompt with truncated context
        prompt = f"""Use the following context to answer the question below. If the context doesn't contain the answer, say "I cannot answer the question based on the provided context."

Context:
{truncated_context}

Question: {query}

Answer:"""
    else:
        logging.info(f"Context length ({len(context_tokens)} tokens) fits within limit.")


    # Tokenize the final prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device) # Do not truncate here, handled above

    if inputs['input_ids'].shape[1] >= tokenizer.model_max_length:
         logging.error(f"Final prompt length ({inputs['input_ids'].shape[1]}) still exceeds model max length ({tokenizer.model_max_length}) after attempting truncation. Generation might fail.")
         # Handle this case - maybe return an error message or try more aggressive truncation
         return "Error: Prompt too long for model.", 0.0, prompt

    # --- Generation ---
    # Generation settings
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True if temperature > 0 else False, # Use sampling only if temperature > 0
        "pad_token_id": tokenizer.eos_token_id # Avoid padding warning during generation
    }

    logging.info("Generating answer...")
    with torch.no_grad(): # Inference doesn't require gradient tracking
        outputs = model.generate(**inputs, **generation_config)
    generation_time = time.time() - start_time

    # Decode the generated tokens, skipping the prompt part
    # outputs[0] contains the full sequence (prompt + generated)
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    logging.info(f"Answer generated in {generation_time:.2f} seconds.")
    return answer.strip(), generation_time, prompt # Return prompt for inspection

# --- Main Workflow ---
def main():
    """Executes the RAG pipeline steps."""
    args = parse_args()
    logging.info(f"Starting RAG pipeline with args: {args}")

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Load Models ---
    # Load embedding model first (needed for dimension check and potentially chunking)
    logging.info(f"Loading embedding model: {args.embedding_model}")
    embed_device = 'cuda' if args.use_gpu_for_embeddings and torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(args.embedding_model, device=embed_device)
    logging.info(f"Embedding model loaded on device: {embed_device}")

    # Load generator model and tokenizer (needed for chunking if using token-based chunker)
    generator_model, generator_tokenizer = load_generator_model(args.model_path, args.load_in_4bit)

    # --- Prepare Data and Index ---
    # Load/Chunk Documents & Build/Load Index
    # This logic handles loading saved index/chunks or building them if they don't exist
    # We pass the generator_tokenizer for potential token-based chunking
    # NOTE: If build_or_load_index loads from file, it returns the loaded chunks
    index, chunks_in_index = build_or_load_index(
        [], # Pass empty list initially, load_and_chunk_documents will be called inside if needed
        embedding_model,
        args.faiss_index_path,
        args.documents_path,
        [], # Pass empty sources initially
        args.faiss_index_type
    )
    # If the index was just built, chunks_in_index contains the chunks used.
    # If loaded, it also contains the chunks loaded from documents_path.
    if not index or not chunks_in_index:
         logging.info("Index or chunks list is empty after build/load attempt. Attempting to build from scratch.")
         # Explicitly load and chunk first if loading failed or wasn't attempted
         loaded_chunks, loaded_sources = load_and_chunk_documents(args.data_dir, args.chunk_size, args.chunk_overlap, generator_tokenizer)
         if not loaded_chunks:
              logging.error("Failed to load or chunk documents. Exiting.")
              return
         # Now try building the index again with the loaded chunks
         index, chunks_in_index = build_or_load_index(
              loaded_chunks,
              embedding_model,
              args.faiss_index_path,
              args.documents_path,
              loaded_sources,
              args.faiss_index_type
         )
         if not index or not chunks_in_index:
              logging.error("Failed to build index even after explicit loading/chunking. Exiting.")
              return


    # --- Evaluation / Interaction Loop ---
    queries = []
    if os.path.exists(args.eval_queries_file):
        logging.info(f"Loading evaluation queries from {args.eval_queries_file}")
        try:
            with open(args.eval_queries_file, 'r') as f:
                queries = json.load(f) # Expecting a list of strings
            if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
                logging.error(f"Evaluation queries file ({args.eval_queries_file}) should contain a JSON list of strings.")
                queries = [] # Reset if format is wrong
        except json.JSONDecodeError:
            logging.error(f"Could not decode JSON from {args.eval_queries_file}.")
            queries = []
    else:
        logging.warning(f"Evaluation queries file not found: {args.eval_queries_file}.")

    if not queries:
         logging.warning("No evaluation queries loaded. Using example query.")
         queries = ["What is the impact of climate change on agriculture?"] # Example query

    results = []
    total_retrieval_time = 0
    total_generation_time = 0
    total_queries = len(queries)

    logging.info(f"Starting evaluation loop for {total_queries} queries...")
    for i, query in enumerate(queries):
        print("-" * 50)
        logging.info(f"Processing query {i+1}/{total_queries}: {query}")

        # a. Retrieve Chunks
        retrieved_chunks, retrieval_time = retrieve_chunks(
            query,
            index,
            embedding_model,
            chunks_in_index, # Use the chunks corresponding to the loaded/built index
            args.top_k
        )
        total_retrieval_time += retrieval_time
        logging.info(f"Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.4f} seconds.")

        # b. Generate Answer
        answer, generation_time, prompt_used = generate_answer(
            query,
            retrieved_chunks,
            generator_model,
            generator_tokenizer,
            args.device,
            args.max_new_tokens,
            args.generation_temperature,
            args.generation_top_p
        )
        total_generation_time += generation_time

        # Print and Store Results
        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
        # print(f"\nPrompt used:\n{prompt_used}\n") # Optional: print the full prompt for debugging

        results.append({
            "query": query,
            "answer": answer,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "total_time_s": retrieval_time + generation_time,
            "retrieved_context": retrieved_chunks, # Optionally store context
            "prompt": prompt_used, # Optionally store prompt
        })
        print("-" * 50)

    # --- Performance Summary ---
    if total_queries > 0:
        avg_retrieval_time = total_retrieval_time / total_queries
        avg_generation_time = total_generation_time / total_queries
        avg_total_time = (total_retrieval_time + total_generation_time) / total_queries
        print("=" * 50)
        logging.info("Performance Summary:")
        logging.info(f"  Processed {total_queries} queries.")
        logging.info(f"  Average Retrieval Time: {avg_retrieval_time:.4f} seconds")
        logging.info(f"  Average Generation Time: {avg_generation_time:.3f} seconds")
        logging.info(f"  Average Total Time per Query: {avg_total_time:.3f} seconds")
        print("=" * 50)

        # Save results to JSON
        results_path = os.path.join(args.output_dir, "rag_results.json")
        logging.info(f"Saving detailed results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save performance summary to text file
        summary_path = os.path.join(args.output_dir, "performance_summary.txt")
        logging.info(f"Saving performance summary to {summary_path}")
        with open(summary_path, 'w') as f:
             f.write("RAG Performance Summary\n")
             f.write("="*25 + "\n")
             f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
             f.write(f"Processed Queries: {total_queries}\n")
             f.write(f"Embedding Model: {args.embedding_model}\n")
             f.write(f"FAISS Index Type: {args.faiss_index_type}\n")
             f.write(f"Retrieved Chunks (k): {args.top_k}\n")
             f.write(f"Generator Model: {args.model_path}\n")
             f.write(f"Generator Max New Tokens: {args.max_new_tokens}\n")
             f.write("-" * 25 + "\n")
             f.write(f"Average Retrieval Time: {avg_retrieval_time:.4f} s\n")
             f.write(f"Average Generation Time: {avg_generation_time:.3f} s\n")
             f.write(f"Average Total Time per Query: {avg_total_time:.3f} s\n")
             f.write("="*25 + "\n")
             f.write("Arguments Used:\n")
             for arg, value in vars(args).items():
                  f.write(f"  {arg}: {value}\n")


    # --- Comparison (Placeholder) ---
    # TODO: Implement comparison with fine-tuned model results
    # 1. Load fine-tuned model inference times (if logged previously)
    # 2. Load fine-tuned model answers (if generated previously for the same queries)
    # 3. Compare avg_total_time with fine-tuned inference time
    # 4. Compare answer quality (requires manual review or metrics like ROUGE/BLEU if reference answers exist)
    logging.info("Comparison with fine-tuned model (Assignment 1) is pending implementation.")

    logging.info("RAG pipeline finished.")


if __name__ == "__main__":
    main()

"""

**Key Improvements & Notes:**

1.  **Chunking:** Added a `load_and_chunk_documents` function. It currently uses a very basic placeholder method based on token counts. **You should replace this with a more robust method**, possibly using `RecursiveCharacterTextSplitter` from LangChain or a similar library that respects sentence boundaries and uses the `generator_tokenizer` for accurate token counting.
2.  **Index/Chunk Consistency:** The `build_or_load_index` function now saves and loads the list of `chunks` (and their `sources`) alongside the FAISS index. This ensures that the indices retrieved from the index correspond correctly to the text chunks being used for context. It also includes a basic consistency check.
3.  **IVFPQ Index:** Added basic support for `IndexIVFPQ` as an example, including the necessary `train` step. You'll need to tune parameters like `nlist`, `m`, `nbits`, and potentially `nprobe` based on your dataset size and performance requirements.
4.  **Prompt Engineering & Context Management:** The `generate_answer` function now includes a more explicit prompt template and basic logic to truncate the context from the left if the combined prompt exceeds the model's maximum length. This is crucial to avoid errors. You might want to refine the prompt template further.
5.  **Error Handling:** Added basic checks for file existence (e.g., `eval_queries.json`) and JSON format.
6.  **Logging & Saving:** Improved logging, including saving a performance summary and detailed results to files in the `output_dir`.
7.  **Modularity:** Functions are more clearly defined for each step (loading/chunking, indexing, retrieval, generation).
8.  **Placeholders:** Still includes `# TODO:` comments for areas like advanced chunking, more index types, and the final comparison logic.

Remember to install the necessary libraries: `transformers`, `torch`, `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`), `tqdm`, `numpy`, `datasets` (if using advanced chunker
"""