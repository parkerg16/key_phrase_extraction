import os
import argparse
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from colorama import Fore, init

init(autoreset=True)

TEST_CHAPTERS = {'6', '10', '11', '12'}

def main(args):
    # Load either fine-tuned or base model
    if os.path.isdir(args.model):
        print(Fore.CYAN + f"Loading fine-tuned model from: {args.model}")
        model = SentenceTransformer(args.model)
    else:
        print(Fore.YELLOW + f"Loading base model: {args.model}")
        model = SentenceTransformer(args.model)

    kw_model = KeyBERT(model=model)

    os.makedirs(args.output_dir, exist_ok=True)
    chapter_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.txt')])

    for file_name in chapter_files:
        chapter_num = file_name.split('_')[-2]  # e.g., chapter_10_chunk.txt → 10

        if chapter_num not in TEST_CHAPTERS:
            continue

        with open(os.path.join(args.input_dir, file_name), 'r', encoding='utf-8') as f:
            text = f.read()

        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            diversity=0.7,
            top_n=100
        )

        output_file = os.path.join(args.output_dir, f"chapter_{chapter_num}_keyphrases.txt")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for phrase, score in keywords:
                out_f.write(f"{phrase}: {score}\n")

        print(Fore.GREEN + f"✓ Chapter {chapter_num} keyphrases saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyword Extraction for Test Chapters")
    parser.add_argument('--input_dir', type=str, default='data/processed_chunks')
    parser.add_argument('--output_dir', type=str, default='data/keyphrases/test')
    parser.add_argument('--model', type=str, default='distilroberta-base-msmarco-v2', help="Path to model or HuggingFace model name")
    args = parser.parse_args()
    main(args)
