import os
import logging
import argparse
from stemming_utils import StemmingUtils # Import stemming utilities - now handles NLTK resource downloading internally

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def stem_extracted_keyphrases(input_dir, output_dir):
    # Create an output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each file in the input directory
    processed_count = 0

    if os.path.exists(input_dir):
        # Find all keyphrase files
        keyphrase_files = []
        for filename in os.listdir(input_dir):
            if filename.startswith("chapter_") and filename.endswith("_keyphrases.txt"):
                try:
                    # Extract chapter number
                    chapter_num = int(filename.split('_')[1])
                    keyphrase_files.append((chapter_num, filename))
                except (IndexError, ValueError):
                    print(f"Warning: Could not determine chapter number from filename: {filename}")
                    continue

        # Sort by chapter number
        keyphrase_files.sort()

        # Process each file
        for chapter_num, filename in keyphrase_files:
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)

            # Stem phrases in the file
            StemmingUtils.stem_file(input_file, output_file, model_option=args.model_option)

            print(f"Stemmed keyphrases for Chapter {chapter_num}")
            processed_count += 1

    print(f"Completed stemming keyphrases: {processed_count} files processed")
    print(f"Stemmed keyphrases saved to: {output_dir}")


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Processing keyphrases from: {input_dir}")
    print(f"Saving stemmed keyphrases to: {output_dir}")

    stem_extracted_keyphrases(input_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stem extracted keyphrases")
    parser.add_argument("--model_option", type=str, default="keybert", choices=["keybert", "ollama"], help="Run on 'keybert' or 'ollama' model")
    # These are optional — will be overridden if not provided
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing extracted keyphrases")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save stemmed keyphrases")

    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = os.path.join("data", "keyphrases", args.model_option)

    if args.output_dir is None:
        args.output_dir = os.path.join("data", "keyphrases", "stemmed", args.model_option)

    main(args)