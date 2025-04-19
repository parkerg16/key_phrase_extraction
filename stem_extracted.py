import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import stemming utilities - now handles NLTK resource downloading internally
from stemming_utils import StemmingUtils

def stem_extracted_keyphrases(input_dir, output_dir):
    """
    Process all keyphrase files in the input directory,
    stem the phrases, and save to the output directory.
    
    Args:
        input_dir (str): Directory containing extracted keyphrases
        output_dir (str): Directory to save stemmed keyphrases
    """
    # Create output directory
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
            StemmingUtils.stem_file(input_file, output_file)
            
            print(f"Stemmed keyphrases for Chapter {chapter_num}")
            processed_count += 1
    
    print(f"Completed stemming keyphrases: {processed_count} files processed")
    print(f"Stemmed keyphrases saved to: {output_dir}")

def main():
    if len(sys.argv) > 2:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        # Default directories
        method = input("Enter extraction method (standard/ollama): ").strip().lower()
        
        if method == "standard" or method == "":
            input_dir = "key_phrases"
            output_dir = "key_phrases_stemmed"
        elif method == "ollama":
            input_dir = "key_phrases_ollama"
            output_dir = "key_phrases_ollama_stemmed"
        else:
            print(f"Unknown method: {method}. Using 'standard' as default.")
            input_dir = "key_phrases"
            output_dir = "key_phrases_stemmed"
    
    print(f"Processing keyphrases from: {input_dir}")
    print(f"Saving stemmed keyphrases to: {output_dir}")
    
    stem_extracted_keyphrases(input_dir, output_dir)

if __name__ == "__main__":
    main()