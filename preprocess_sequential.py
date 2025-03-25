"""
Ultra-simple preprocessing script based on working example
- Uses bare minimum approach that's proven to work
- No complex error handling or multiprocessing
- Just extracts text and saves files
"""

import os
import glob
import random
import argparse
import json
import logging
from datetime import datetime
from tqdm import tqdm
from PyPDF2 import PdfReader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

def process_pdfs(pdf_dir, output_dir, page_limit=30, min_chars=500):
    """Process PDFs using the proven successful approach"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all PDF files
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDF files to process")

    # Track successfully processed files
    successful_files = []

    # Process files sequentially with progress bar
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Get output path
            filename = os.path.basename(pdf_path).replace('.pdf', '.txt')
            output_path = os.path.join(output_dir, filename)

            # Skip if already processed
            if os.path.exists(output_path):
                successful_files.append(output_path)
                continue

            # Open PDF - following the approach from your friend's code
            reader = PdfReader(pdf_path)
            text = ""

            # Extract text from each page (up to page_limit)
            for page_num, page in enumerate(reader.pages):
                if page_num >= page_limit:
                    break
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                except Exception as e:
                    # Just log and continue to next page
                    logging.warning(f"Error on page {page_num} of {pdf_path}: {str(e)}")

            # Check if extraction was successful
            if text and len(text) >= min_chars:
                # Save text to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                successful_files.append(output_path)
                logging.info(f"Successfully processed {pdf_path}")
            else:
                logging.warning(f"Extracted text too short for {pdf_path}")

        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")

    # Get all successfully extracted text files
    all_txt_files = glob.glob(os.path.join(output_dir, "*.txt"))
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
    parser = argparse.ArgumentParser(description="Ultra-simple preprocessing of PDF data")
    parser.add_argument("--pdf_dir", type=str, default="/root/bdml25sp/datasets/BDML25SP/climate_text_dataset",
                        help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                        help="Directory to save processed text")
    parser.add_argument("--page_limit", type=int, default=30,
                        help="Maximum number of pages to process per PDF")
    parser.add_argument("--min_chars", type=int, default=500,
                        help="Minimum characters for valid extraction")
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

    # Process PDFs sequentially
    txt_files = process_pdfs(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        page_limit=args.page_limit,
        min_chars=args.min_chars
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