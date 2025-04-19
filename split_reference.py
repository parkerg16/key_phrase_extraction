import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import stemming utilities - now handles NLTK resource downloading internally
from stemming_utils import StemmingUtils

def split_index_file(index_file, output_dir, stemmed_output_dir=None):
    """
    Split the index_by_chapter.txt file into individual files, one for each chapter.
    Each file will contain the keyphrases for that chapter.
    Optionally creates stemmed versions in a separate directory.
    
    Args:
        index_file (str): Path to the index file
        output_dir (str): Directory to store regular reference files
        stemmed_output_dir (str, optional): Directory to store stemmed reference files
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    if stemmed_output_dir:
        os.makedirs(stemmed_output_dir, exist_ok=True)
    
    current_chapter = None
    current_phrases = []
    
    print(f"Reading index file: {index_file}")
    
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Chapter"):
                # Save previous chapter if we have one
                if current_chapter is not None and current_phrases:
                    # Regular reference file
                    output_file = os.path.join(output_dir, f"chapter_{current_chapter}_reference.txt")
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        out_f.write('\n'.join(current_phrases))
                    print(f"Created {output_file} with {len(current_phrases)} keyphrases")
                    
                    # Stemmed reference file (if requested)
                    if stemmed_output_dir:
                        stemmed_output_file = os.path.join(stemmed_output_dir, f"chapter_{current_chapter}_reference.txt")
                        stemmed_phrases = StemmingUtils.stem_phrases(current_phrases)
                        with open(stemmed_output_file, 'w', encoding='utf-8') as out_f:
                            out_f.write('\n'.join(stemmed_phrases))
                        print(f"Created stemmed {stemmed_output_file}")
                
                # Start new chapter
                current_chapter = int(line.split()[1])
                current_phrases = []
            else:
                # Add to current chapter's keyphrases
                current_phrases.append(line.strip())
    
    # Save the last chapter
    if current_chapter is not None and current_phrases:
        # Regular reference file
        output_file = os.path.join(output_dir, f"chapter_{current_chapter}_reference.txt")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write('\n'.join(current_phrases))
        print(f"Created {output_file} with {len(current_phrases)} keyphrases")
        
        # Stemmed reference file (if requested)
        if stemmed_output_dir:
            stemmed_output_file = os.path.join(stemmed_output_dir, f"chapter_{current_chapter}_reference.txt")
            stemmed_phrases = StemmingUtils.stem_phrases(current_phrases)
            with open(stemmed_output_file, 'w', encoding='utf-8') as out_f:
                out_f.write('\n'.join(stemmed_phrases))
            print(f"Created stemmed {stemmed_output_file}")
    
    print(f"Completed splitting index file into individual files")
    print(f"Regular keyphrases in: {output_dir}")
    if stemmed_output_dir:
        print(f"Stemmed keyphrases in: {stemmed_output_dir}")

def main():
    if len(sys.argv) > 2:
        index_file = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        # Default paths
        index_file = "../index_by_chapter.txt"
        output_dir = "reference_keyphrases"
    
    # Default stemmed output directory
    stemmed_output_dir = "reference_keyphrases_stemmed"
    
    split_index_file(index_file, output_dir, stemmed_output_dir)

if __name__ == "__main__":
    main()