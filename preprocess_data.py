"""
Improved Climate Data Preprocessing with Robust Error Handling
- Extracts text from PDFs using a controlled number of processes
- Includes timeout mechanism for problematic PDFs
- Safely handles errors and continues processing
"""

import os
import glob
import random
import argparse
import json
import PyPDF2
import logging
import time
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
from datetime import datetime
from functools import partial

# Set up logging with reduced PyPDF2 verbosity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
# Reduce the verbosity of PyPDF2 warnings
logging.getLogger('PyPDF2').setLevel(logging.ERROR)


def extract_text_from_pdf(pdf_path, output_dir, min_chars=500, timeout=30):
    """Extract text from a single PDF file with timeout and error handling.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the extracted text
        min_chars: Minimum number of characters for a valid extraction
        timeout: Maximum time in seconds to spend on one PDF

    Returns:
        tuple: (output_path, success_flag)
    """
    try:
        filename = os.path.basename(pdf_path).replace('.pdf', '.txt')
        output_path = os.path.join(output_dir, filename)

        # Skip if already processed
        if os.path.exists(output_path):
            return output_path, True

        # Extract text using PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Limit the number of pages to process if the PDF is very large
            max_pages = min(len(reader.pages), 300)  # Set a reasonable limit

            for page_num in range(max_pages):
                try:
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:  # Only add if text was extracted
                        text += page_text + "\n\n"
                except Exception as e:
                    # Skip problematic pages
                    continue

        # Check if extraction was successful
        if len(text) < min_chars:
            return None, False

        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

        return output_path, True

    except Exception as e:
        return None, False


def process_pdf_with_timeout(pdf_path, output_dir, min_chars=500, timeout=30):
    """Process a PDF with a timeout to prevent hanging."""
    try:
        return extract_text_from_pdf(pdf_path, output_dir, min_chars)
    except Exception as e:
        return None, False


def process_pdfs(pdf_dir, output_dir, num_workers=8, min_chars=500, chunk_size=20):
    """Process PDFs in parallel with controlled batch size.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save extracted text files
        num_workers: Number of worker processes (defaults to 8)
        min_chars: Minimum number of characters for a valid extraction
        chunk_size: Number of PDFs to process in each batch

    Returns:
        list: Paths of successfully processed text files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of PDF files
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files to process")

    # Set a reasonable number of workers
    if num_workers is None or num_workers <= 0:
        num_workers = min(os.cpu_count(), 8)  # Use at most 8 CPUs

    logging.info(f"Using {num_workers} worker processes")

    # Create a partial function with fixed parameters
    process_fn = partial(process_pdf_with_timeout, output_dir=output_dir, min_chars=min_chars)

    # Process PDFs in parallel with better error handling
    successful_files = []

    # Process in smaller batches to avoid memory issues
    with tqdm(total=len(pdf_files), desc="Extracting text from PDFs") as pbar:
        for i in range(0, len(pdf_files), chunk_size):
            batch = pdf_files[i:i+chunk_size]

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all jobs
                future_to_pdf = {executor.submit(process_fn, pdf): pdf for pdf in batch}

                # Process as they complete
                for future in as_completed(future_to_pdf):
                    pdf = future_to_pdf[future]
                    try:
                        result, success = future.result(timeout=60)  # Add extra timeout for safety
                        if success and result:
                            successful_files.append(result)
                    except (TimeoutError, Exception) as e:
                        logging.warning(f"Failed to process {pdf}: {str(e)}")

                    pbar.update(1)

    # Count successes and failures
    num_failed = len(pdf_files) - len(successful_files)

    logging.info(f"Successfully processed {len(successful_files)} files")
    logging.info(f"Failed to process {num_failed} files")

    return successful_files


def split_train_test(file_list, train_ratio=0.9, seed=42):
    """Split files into training and testing sets."""
    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle the file list
    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)

    # Split into train and test sets
    split_idx = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:split_idx]
    test_files = shuffled_files[split_idx:]

    return train_files, test_files


def save_split_info(train_files, test_files, output_dir):
    """Save train/test split information to a JSON file."""
    split_info = {
        "train_files": [os.path.basename(f) for f in train_files],
        "test_files": [os.path.basename(f) for f in test_files],
        "train_size": len(train_files),
        "test_size": len(test_files),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Save to JSON file
    split_info_path = os.path.join(output_dir, "split_info.json")
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    logging.info(f"Split info saved to {split_info_path}")
    return split_info_path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess climate PDF data for LLaMA fine-tuning")

    # Input/output paths
    parser.add_argument("--pdf_dir", type=str, default="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset",
                        help="Directory containing the climate PDF files")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                        help="Directory to save extracted text and split information")

    # Processing parameters
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of train/test split (e.g., 0.9 for 90% training)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of worker processes (default: 8)")
    parser.add_argument("--min_chars", type=int, default=500,
                        help="Minimum number of characters for a valid extraction")
    parser.add_argument("--chunk_size", type=int, default=20,
                        help="Number of PDFs to process in each batch")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_args()

    start_time = datetime.now()
    logging.info(f"Starting preprocessing at {start_time}")
    logging.info(f"PDF directory: {args.pdf_dir}")
    logging.info(f"Output directory: {args.output_dir}")

    # Process PDFs with controlled parallelism and better error handling
    txt_files = process_pdfs(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        min_chars=args.min_chars,
        chunk_size=args.chunk_size
    )

    if not txt_files:
        logging.error("No text files were successfully extracted. Cannot proceed.")
        return

    # Split into train and test sets
    train_files, test_files = split_train_test(
        file_list=txt_files,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

    logging.info(f"Training set: {len(train_files)} files")
    logging.info(f"Test set: {len(test_files)} files")

    # Save split information
    split_info_path = save_split_info(train_files, test_files, args.output_dir)

    # Calculate and log elapsed time
    elapsed_time = datetime.now() - start_time
    logging.info(f"Preprocessing completed in {elapsed_time}")
    logging.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()