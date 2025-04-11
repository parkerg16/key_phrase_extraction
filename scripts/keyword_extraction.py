import argparse
import os
import re
import torch
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from colorama import Fore, init

init(autoreset=True)

TRAIN_CHAPTERS = {'1', '2', '3', '4', '5', '7', '8', '9', '13', '14', '15', '16', '17', '18', '19'}
TEST_CHAPTERS = {'6', '10', '11', '12'}


def extract_chapter_number(file_name):
    match = re.search(r"(?:chapter[_-]?|ch)(\d+)", file_name, re.IGNORECASE)
    return match.group(1) if match else None


def main(args):
    # Detect device type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running extraction on device: {device}")

    # Load model (fine-tuned or base)
    if os.path.isdir(args.model):
        print(Fore.CYAN + f"Loading fine-tuned model from: {args.model}")
        model = SentenceTransformer(args.model)
    else:
        print(Fore.YELLOW + f"Loading base model: {args.model}")
        model = SentenceTransformer(args.model)

    model = model.to(device)
    kw_model = KeyBERT(model=model)

    # Save the model (optional)
    if args.save_model_path:
        print(Fore.GREEN + f"Saving model to {args.save_model_path}...")
        model.save(args.save_model_path)

    os.makedirs(args.output_dir, exist_ok=True)

    chapter_files = []
    for f in os.listdir(args.input_dir):
        if f.endswith(".txt"):
            chapter_num = extract_chapter_number(f)
            if chapter_num:
                chapter_files.append((chapter_num, f))

    for chapter_num, file_name in chapter_files:
        if args.type == 'train':
            if chapter_num not in TRAIN_CHAPTERS:
                continue
        else:
            if chapter_num not in TEST_CHAPTERS:
                continue

        file_path = os.path.join(args.input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
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
    parser = argparse.ArgumentParser(description="Unified Keyword Extraction for Training or Testing")
    parser.add_argument('--input_dir', type=str, default='data/processed_chunks')
    parser.add_argument('--output_dir', type=str, default='data/keyphrases/train')
    parser.add_argument('--model', type=str, default='distilroberta-base-msmarco-v2', help="Path to model or HuggingFace name")
    parser.add_argument('--type', type=str, choices=['train', 'test'], default='train', help="Run on 'train' or 'test' split")
    parser.add_argument('--save_model_path', type=str, default='', help="Optional: Save model to this path")

    args = parser.parse_args()
    main(args)
