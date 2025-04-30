import argparse
import os
import glob
import json
import logging
import time
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing (Adapted from Assignment 1) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Build and evaluate a RAG system using LLaMA")

    # Paths
    parser.add_argument("--model_path", type=str, default="/root/bdml25sp/datasets/BDML25SP/Llama3.2-3B-converted", # Or /scratch/BDML25SP/Llama3.2-3B-converted on HPC
                        help="Path to the base LLaMA 3B model (Hugging Face format)")
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="Directory containing preprocessed text data (.txt files)")
    parser.add_argument("--output_dir", type=str, default="./rag_output",
                        help="Directory to save FAISS index and potentially other outputs")
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
    # Add more FAISS params if using complex indices like IVFPQ (nlist, nprobe, m)
    parser.add_argument("--chunk_size", type=int, default=300, # Typical size mentioned
                        help="Target chunk size in tokens (used if re-chunking)")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Overlap between chunks (used if re-chunking)")
    parser.add_argument("--top_k", type=int, default=3, # Retrieve top 3 most relevant docs
                        help="Number of documents to retrieve for context")

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
                        help="JSON file containing queries for evaluation")


    # System Parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for generation (e.g., cuda:0 or cpu)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    return args

# --- Data Loading ---
def load_documents(data_dir):
    """Loads text documents from .txt files in the specified directory."""
    logging.info(f"Loading documents from: {data_dir}")
    all_files = glob.glob(os.path.join(data_dir, "*.txt"))
    documents = []
    doc_sources = [] # Keep track of the source file for each doc/chunk

    if not all_files:
        logging.error(f"No .txt files found in {data_dir}. Ensure preprocessing was completed.")
        return [], []

    for file_path in tqdm(all_files, desc="Reading documents"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Simple loading: treat each file as one document
                # TODO: Implement chunking here if needed, based on args.chunk_size/overlap
                # For now, one file = one document
                content = f.read()
                if content.strip(): # Avoid empty documents
                    documents.append(content)
                    doc_sources.append(os.path.basename(file_path))
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    logging.info(f"Loaded {len(documents)} documents.")
    return documents, doc_sources

# --- Embedding and Indexing ---
def build_or_load_index(documents, embedding_model, index_path, index_type="IndexFlatL2"):
    """Builds a FAISS index from documents or loads it if it exists."""
    if os.path.exists(index_path):
        logging.info(f"Loading existing FAISS index from {index_path}")
        index = faiss.read_index(index_path)
        return index
    else:
        logging.info(f"Building new FAISS index ({index_type})...")
        # Determine embedding dimension
        test_embedding = embedding_model.encode(["test sentence"], convert_to_numpy=True)
        d = test_embedding.shape[1]
        logging.info(f"Detected embedding dimension: {d}")

        # Create FAISS index
        # TODO: Add support for more complex index types like IndexIVFPQ based on args.faiss_index_type
        if index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(d)
        elif index_type == "IndexIVFPQ":
             # Example for IndexIVFPQ - parameters would need to be args
             nlist = 100  # Number of clusters
             m = 8      # Number of subquantizers
             nbits = 8  # bits per subquantizer index
             quantizer = faiss.IndexFlatL2(d)
             index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
             # Training required for IVFPQ
             logging.info("Training IVFPQ index...")
             # Use a subset of embeddings for training if dataset is large
             embeddings_for_training = embedding_model.encode(documents[:max(10000, len(documents)//10)], convert_to_numpy=True, show_progress_bar=True)
             index.train(embeddings_for_training)
             logging.info("Index training complete.")
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

        logging.info("Encoding documents...")
        doc_embeddings = embedding_model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        logging.info(f"Adding {len(doc_embeddings)} embeddings to the index...")
        index.add(doc_embeddings)
        logging.info(f"Index built successfully with {index.ntotal} vectors.")

        logging.info(f"Saving index to {index_path}")
        faiss.write_index(index, index_path)
        return index

# --- Retrieval ---
def retrieve_documents(query, index, embedding_model, documents, top_k=3):
    """Retrieves top_k documents relevant to the query."""
    # logging.info(f"Retrieving top {top_k} documents for query: '{query}'")
    query_vec = embedding_model.encode([query], convert_to_numpy=True) #
    distances, indices = index.search(query_vec, top_k) #

    retrieved_docs = [documents[i] for i in indices[0]] #
    # logging.info(f"Retrieved indices: {indices[0]}")
    # logging.info(f"Retrieved distances: {distances[0]}")
    return retrieved_docs

# --- Generation ---
def load_generator_model(model_path, load_in_4bit=False):
    """Loads the LLaMA model and tokenizer for generation."""
    logging.info(f"Loading generator model from: {model_path}")

    quantization_config = None
    if load_in_4bit:
         logging.info("Using 4-bit quantization for generator model.")
         quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 # Or torch.bfloat16 if supported and preferred
         )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto", # Automatically distribute across available GPUs/CPU
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if available
        quantization_config=quantization_config,
        trust_remote_code=True # Needed for some LLaMA versions
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logging.info("Generator model and tokenizer loaded.")
    return model, tokenizer

def generate_answer(query, retrieved_docs, model, tokenizer, device, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generates an answer based on the query and retrieved context."""
    context = " ".join(retrieved_docs) #

    # Truncate context if it's too long to fit within model's max length minus query and response buffer
    # This is a simple truncation strategy; more sophisticated methods exist.
    tokenizer.truncation_side = 'left' # Truncate from the left (remove older context first)
    max_context_tokens = tokenizer.model_max_length - len(tokenizer.encode(query)) - max_new_tokens - 50 # Leave buffer
    context_tokens = tokenizer.encode(context, max_length=max_context_tokens, truncation=True)
    truncated_context = tokenizer.decode(context_tokens)

    prompt = f"""Context: {truncated_context}

Question: {query}

Answer:""" #

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device) #

    # Generation settings
    generation_config = {
        "max_new_tokens": max_new_tokens, #
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True, # Use sampling for more diverse answers
        "pad_token_id": tokenizer.eos_token_id # Avoid padding warning
    }

    logging.info("Generating answer...")
    start_time = time.time()
    with torch.no_grad(): # No need to track gradients during inference
        outputs = model.generate(**inputs, **generation_config) #
    end_time = time.time()

    generation_time = end_time - start_time
    # Decode the generated tokens, skipping the prompt part
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True) #

    logging.info(f"Answer generated in {generation_time:.2f} seconds.")
    return answer, generation_time, prompt # Return prompt for inspection

# --- Main Workflow ---
def main():
    args = parse_args()
    logging.info(f"Starting RAG pipeline with args: {args}")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1. Load Documents
    if os.path.exists(args.documents_path) and os.path.exists(args.faiss_index_path):
         logging.info(f"Loading documents list from {args.documents_path}")
         with open(args.documents_path, 'r') as f:
              documents_data = json.load(f)
         documents = documents_data['documents']
         # doc_sources = documents_data['sources'] # Optionally load sources too
    else:
        documents, doc_sources = load_documents(args.data_dir)
        if not documents:
            return # Exit if no documents loaded
        # Save documents list for consistency with the index
        logging.info(f"Saving documents list to {args.documents_path}")
        with open(args.documents_path, 'w') as f:
            # Save sources too if needed later: json.dump({'documents': documents, 'sources': doc_sources}, f)
             json.dump({'documents': documents}, f)


    # 2. Load Embedding Model
    logging.info(f"Loading embedding model: {args.embedding_model}")
    embed_device = 'cuda' if args.use_gpu_for_embeddings and torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(args.embedding_model, device=embed_device)
    logging.info(f"Embedding model loaded on device: {embed_device}")

    # 3. Build or Load FAISS Index
    index = build_or_load_index(documents, embedding_model, args.faiss_index_path, args.faiss_index_type)

    # 4. Load Generator Model
    generator_model, generator_tokenizer = load_generator_model(args.model_path, args.load_in_4bit)

    # 5. Example Interaction / Evaluation Loop
    #    Load queries from a file or use example queries
    queries = []
    if os.path.exists(args.eval_queries_file):
        logging.info(f"Loading evaluation queries from {args.eval_queries_file}")
        with open(args.eval_queries_file, 'r') as f:
            queries = json.load(f) # Expecting a list of strings
    else:
        logging.warning(f"Evaluation queries file not found: {args.eval_queries_file}. Using example query.")
        queries = ["What is the impact of climate change on agriculture?"] # Example query

    results = []
    total_retrieval_time = 0
    total_generation_time = 0

    for query in queries:
        print("-" * 50)
        logging.info(f"Processing query: {query}")

        # a. Retrieve
        start_retrieval = time.time()
        retrieved_docs = retrieve_documents(query, index, embedding_model, documents, args.top_k)
        retrieval_time = time.time() - start_retrieval
        total_retrieval_time += retrieval_time
        logging.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f} seconds.")
        # print("--- Retrieved Context ---")
        # for i, doc in enumerate(retrieved_docs):
        #     print(f"Doc {i+1}: {doc[:150]}...") # Print snippet
        # print("-" * 25)

        # b. Generate
        answer, generation_time, prompt = generate_answer(
            query,
            retrieved_docs,
            generator_model,
            generator_tokenizer,
            args.device,
            args.max_new_tokens,
            args.generation_temperature,
            args.generation_top_p
        )
        total_generation_time += generation_time

        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
        # print(f"\nPrompt used:\n{prompt}\n") # Optional: print the full prompt

        results.append({
            "query": query,
            "answer": answer,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            # "retrieved_context": retrieved_docs # Optionally store context
        })

    # 6. Basic Performance Metrics
    num_queries = len(queries)
    if num_queries > 0:
        avg_retrieval_time = total_retrieval_time / num_queries
        avg_generation_time = total_generation_time / num_queries
        avg_total_time = (total_retrieval_time + total_generation_time) / num_queries
        print("=" * 50)
        logging.info("Performance Summary:")
        logging.info(f"  Processed {num_queries} queries.")
        logging.info(f"  Average Retrieval Time: {avg_retrieval_time:.3f} seconds")
        logging.info(f"  Average Generation Time: {avg_generation_time:.3f} seconds")
        logging.info(f"  Average Total Time per Query: {avg_total_time:.3f} seconds")
        print("=" * 50)

        # Save results
        results_path = os.path.join(args.output_dir, "rag_results.json")
        logging.info(f"Saving results to {results_path}")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    # TODO: Add comparison logic here
    # - Load results from fine-tuned model (Assignment 1) if available
    # - Compare inference times (Avg Total Time here vs. fine-tuned model's time)
    # - Compare answer quality (manual inspection, or more advanced metrics if feasible)


if __name__ == "__main__":
    main()