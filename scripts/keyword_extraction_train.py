import argparse
import os
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from colorama import Fore, init

init(autoreset=True)

def main(args):
    # Load either pretrained or previously saved model
    if os.path.isdir(args.model):
        print(Fore.CYAN + f"Loading fine-tuned model from: {args.model}")
        model = SentenceTransformer(args.model)
    else:
        print(Fore.YELLOW + f"Loading base model: {args.model}")
        model = SentenceTransformer(args.model)

    kw_model = KeyBERT(model=model)

    # Save the model (if first time + path provided)
    if args.save_model_path:
        print(Fore.GREEN + f"Saving model to {args.save_model_path}...")
        model.save(args.save_model_path)

    # Directory setup
    os.makedirs(args.output_dir, exist_ok=True)
    chapter_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.txt')])

    for file_name in chapter_files:
        chapter_num = file_name.split('_')[-2]
        if chapter_num not in {'1', '2', '3', '4', '5', '7', '8', '9', '13', '14', '15', '16', '17', '18', '19'}:
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
    parser = argparse.ArgumentParser(description="Keyword Extraction for Training Chapters")
    parser.add_argument('--input_dir', type=str, default='data/processed_chunks')
    parser.add_argument('--output_dir', type=str, default='data/keyphrases/train')
    parser.add_argument('--model', type=str, default='distilroberta-base-msmarco-v2', help="Path to model or HuggingFace name")
    parser.add_argument('--save_model_path', type=str, default='', help="Optional: Save model to this path")

    args = parser.parse_args()
    main(args)
