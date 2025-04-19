import os
import sys
import argparse
import logging

# Allow imports from scripts/stemming
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEMMING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "stemming"))
sys.path.insert(0, STEMMING_DIR)

from stemming_utils import StemmingUtils


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_index_file(index_file, output_dir, stemmed_output_dir):
    """
    Split the index_by_chapter.txt file into individual files per chapter.
    Also generates stemmed versions.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stemmed_output_dir, exist_ok=True)

    current_chapter = None
    current_phrases = []

    logger.info(f" Reading index file: {index_file}")

    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Chapter"):
                if current_chapter is not None and current_phrases:
                    write_chapter(current_chapter, current_phrases, output_dir, stemmed_output_dir)
                current_chapter = int(line.split()[1])
                current_phrases = []
            else:
                current_phrases.append(line)

    # Final chapter write
    if current_chapter is not None and current_phrases:
        write_chapter(current_chapter, current_phrases, output_dir, stemmed_output_dir)

    logger.info(f" Completed writing reference files.")
    logger.info(f"   - Regular saved to: {output_dir}")
    logger.info(f"   - Stemmed saved to: {stemmed_output_dir}")

def write_chapter(chapter_num, phrases, output_dir, stemmed_output_dir):
    base_filename = f"chapter_{chapter_num}_reference.txt"

    # Save regular
    out_path = os.path.join(output_dir, base_filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(phrases))
    logger.info(f"Saved {len(phrases)} phrases to {out_path}")

    # Save stemmed
    stemmed_phrases = StemmingUtils.stem_phrases(phrases)
    stem_path = os.path.join(stemmed_output_dir, base_filename)
    with open(stem_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(stemmed_phrases))
    logger.info(f"Saved stemmed phrases to {stem_path}")

def main(args):
    split_index_file(
        index_file=args.index_file,
        output_dir=args.output_dir,
        stemmed_output_dir=args.stemmed_output_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split index_by_chapter.txt into per-chapter reference files")
    parser.add_argument("--index_file", type=str, default="data/index_by_chapter.txt",
                        help="Path to index_by_chapter.txt")
    parser.add_argument("--output_dir", type=str, default="data/keyphrases/referenced",
                        help="Directory to save reference keyphrases")
    parser.add_argument("--stemmed_output_dir", type=str, default="data/keyphrases/stemmed/referenced",
                        help="Directory to save stemmed reference keyphrases")
    args = parser.parse_args()
    main(args)
