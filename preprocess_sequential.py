"""
Ultra-simple and reliable preprocessing script for climate data
- Processes PDFs sequentially to avoid multiprocessing issues
- Uses strict timeouts to handle problematic files
- Saves progress continuously so it can be stopped and resumed
"""

import os
import glob
import random
import argparse
import json
import signal
import logging
import time
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess_sequential.log"),
        logging.StreamHandler()
    ]
)

# Set a timeout for PDF processing
def timeout_handler(signum, frame):
    raise TimeoutError("PDF processing timed out")

def extract_text_with_timeout(pdf_path, timeout=30):
    """Extract text from a PDF with a strict timeout"""
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set alarm for N seconds

    try:
        # Import PyPDF2 here to avoid any module-level issues
        import PyPDF2

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Limit pages to avoid extremely large files
            max_pages = min(len(reader.pages), 300)

            for page_num in range(max_pages):
                try:
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                except Exception as e:
                    # Skip problematic pages quietly
                    continue

        # Cancel the alarm
        signal.alarm(0)
        return text
    except TimeoutError:
        logging.warning(f"Processing timed out for {pdf_path}")
        return ""
    except Exception as e:
        logging.warning(f"Error processing {pdf_path}: {str(e)}")
        return ""
    finally:
        # Ensure alarm is canceled even if an exception occurs
        signal.alarm(0)

def process_pdfs(pdf_dir, output_dir, min_chars=500, timeout=30, progress_file="preprocessing_progress.json"):
    """Process PDFs sequentially with timeouts"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all PDF files
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files to process")

    # Load progress if exists
    processed_files = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get("processed_files", []))
                logging.info(f"Loaded progress: {len(processed_files)} files already processed")
        except Exception as e:
            logging.warning(f"Error loading progress file: {e}")

    # Filter out already processed files
    pdf_files_to_process = [pdf for pdf in pdf_files if pdf not in processed_files]
    logging.info(f"Remaining files to process: {len(pdf_files_to_process)}")

    # Process files sequentially with progress bar
    successful_files = []
    for pdf_path in tqdm(pdf_files_to_process, desc="Processing PDFs"):
        try:
            # Get output path
            filename = os.path.basename(pdf_path).replace('.pdf', '.txt')
            output_path = os.path.join(output_dir, filename)

            # Skip if already processed (double-check)
            if os.path.exists(output_path):
                processed_files.add(pdf_path)
                continue

            # Extract text with timeout
            text = extract_text_with_timeout(pdf_path, timeout)

            # Check if extraction was successful
            if text and len(text) >= min_chars:
                # Save text to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                successful_files.append(output_path)

            # Mark file as processed regardless of success
            processed_files.add(pdf_path)

            # Save progress every 10 files
            if len(processed_files) % 10 == 0:
                with open(progress_file, 'w') as f:
                    json.dump({"processed_files": list(processed_files)}, f)

        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")

    # Get all successfully extracted text files
    all_txt_files = glob.glob(os.path.join(output_dir, "*.txt"))

    # Final progress save
    with open(progress_file, 'w') as f:
        json.dump({"processed_files": list(processed_files)}, f)

    logging.info(f"Successfully processed {len(all_txt_files)} files")
    return all_txt_files

def split_train_test(file_list, train_ratio=0.9, seed=42):
    """Split files into training and testing sets."""
    random.seed(seed)
    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)
    split_idx = int(len(shuffled_files) * train_ratio)
    return shuffled_files[:split_idx], shuffled_files[split_idx:]

def save_split_info(train_files, test_files, output_dir):
    """Save train/test split information to a JSON file."""
    split_info = {
        "train_files": [os.path.basename(f) for f in train_files],
        "test_files": [os.path.basename(f) for f in test_files],
        "train_size": len(train_files),
        "test_size": len(test_files),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    split_info_path = os.path.join(output_dir, "split_info.json")
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    logging.info(f"Split info saved to {split_info_path}")
    return split_info_path

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simple preprocessing of PDF data")
    parser.add_argument("--pdf_dir", type=str, default="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset",
                        help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                        help="Directory to save processed text")
    parser.add_argument("--min_chars", type=int, default=500,
                        help="Minimum characters for valid extraction")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout in seconds for processing each PDF")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio for train/test split")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    start_time = datetime.now()
    logging.info(f"Starting preprocessing at {start_time}")

    # Process PDFs sequentially with timeouts
    txt_files = process_pdfs(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        min_chars=args.min_chars,
        timeout=args.timeout
    )

    if not txt_files:
        logging.error("No text files were successfully extracted.")
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

    elapsed_time = datetime.now() - start_time
    logging.info(f"Preprocessing completed in {elapsed_time}")

if __name__ == "__main__":
    main()