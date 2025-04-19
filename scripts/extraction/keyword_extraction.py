import os
import argparse
import re
from keybert import KeyBERT
from colorama import Fore, init

init(autoreset=True)

def get_chapter_num(filename):
    try:
        # Extract the number from the ch{num}.txt format
        if filename.startswith("ch") and filename.endswith(".txt"):
            # Extract the part between 'ch' and '.txt'
            return int(filename[2:-4])
        # Also handle older chapter_X_chunk.txt formats for backward compatibility
        elif filename.startswith("chapter_") and '_chunk.txt' in filename:
            return int(filename.split('_')[1])
        else:
            return 999 # Default for unknown formats
    except (IndexError, ValueError):
        return 999

def main(args):
    book_chunks_path = args.input_dir
    key_phrases_path = args.output_dir

    # Load model (fine-tuned or base)
    if os.path.isdir(args.model):
        print(Fore.CYAN + f"Loading fine-tuned model from: {args.model}")
        kw_model = KeyBERT(model=args.model)
    else:
        # Initialize KeyBert with a standard pre-trained model
        # Options include: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'distilbert-base-nli-mean-tokens', 'distilroberta-base', 'paraphrase-multilingual-MiniLM-L12-v2'
        print(Fore.YELLOW + f"Loading base model: {args.model}")
        kw_model = KeyBERT(model=args.model)

        # Save the model (optional)
    if args.save_model_path:
        print(Fore.GREEN + f"Saving model to {args.save_model_path}...")
        kw_model.save(args.save_model_path)

    if not os.path.exists(key_phrases_path):
        os.makedirs(key_phrases_path)
        print(f"Created folder: {key_phrases_path}")

    # Get a sorted list of all .txt files in the folder (sorted by chapter number)
    chapter_files = sorted(
        [f for f in os.listdir(book_chunks_path) if f.endswith(".txt")],
        key=get_chapter_num
    )

    # Print the list of found chapter files
    print(f"Found chapter files: {', '.join(chapter_files)}")

    # Determine actual chapter numbers directly from the file names
    chapter_numbers = []
    for file_name in chapter_files:
        try:
            chapter_num = get_chapter_num(file_name)
            chapter_numbers.append(chapter_num)
        except:
            # Fall back to using the enumeration as the chapter number
            chapter_numbers.append(len(chapter_numbers) + 1)

    # Report which chapters were detected
    print(f"Detected chapters:  {', '.join(map(str,sorted(chapter_numbers)))}")
    print(f"Found {len(chapter_files)} chapter to process.")

    # Iterate over each file and extract keywords
    for i, (file_name, chapter_num) in enumerate(zip(chapter_files, chapter_numbers)):
        # Check if this chapter's keyphrases have already been generated

        output_file = os.path.join(key_phrases_path, f"chapter_{chapter_num}_keyphrases.txt")
        if os.path.exists(output_file):
            print(f"Skipping chapter {chapter_num} ({i+1}/{len(chapter_files)}) - already processed.")
            continue

        print(f"Processing chapter {chapter_num} ({i+1}/{len(chapter_files)})...")

        file_path = os.path.join(book_chunks_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_text = f.read()

        print(f" Extracting keyphrases from chapters {chapter_num} {len(chapter_text)} characters)...")

        # For very large chapters (over 250,000 characters), split into chunks
        if len(chapter_text) > 250000:
            print(f" Chapter {chapter_num} is very large, splitting into chunks...")
            chunk_size = 200000 # Process in 200k chunks
            combined_keywords = []

            for chunk_start in range(0, len(chapter_text), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(chapter_text))
                chunk = chapter_text[chunk_start:chunk_end]
                print(f" Processing chunk {chunk_start} to {chunk_end} of {len(chapter_text)}...")

                try:
                    # Extract keywords from this chunk
                    chunk_keywords = kw_model.extract_keywords(
                        chunk,
                        keyphrase_ngram_range=(1,3),
                        stop_words='english',
                        use_mmr=True,
                        diversity=0.7,
                        top_n=30 # Extract fewer from each chunk
                    )

                    combined_keywords.extend(chunk_keywords)
                    print(f" Extracted {len(chunk_keywords)} keyphrases in chunk")
                except Exception as e:
                    print(f" Error processing chunk: {e}")
                    continue

            keywords = []
            seen_phrases = set()
            for phrase, score in sorted(combined_keywords, key=lambda x: x[1], reverse=True):
                if phrase not in seen_phrases and len(keywords) < 75:
                    keywords.append((phrase, score))
                    seen_phrases.add(phrase)
        else:
            # Normal processing for smaller chapters
            try:
                # Extract keywords from the chapter text
                keywords = kw_model.extract_keywords(
                    chapter_text,
                    keyphrase_ngram_range=(1,3), # Length of phrases (1-3 words)
                    stop_words='english', # Filter common English words
                    use_mmr=True, # Use Maximal Marginal Relevance for diversity
                    diversity=0.7, # Balance between relevance and diversity
                    top_n=75 # Number of phrases to extract (matching Ollama)
                )
            except Exception as e:
                print(f" Error extracting keyphrases from chapter {chapter_num}: {e}")
                print("Generating empty keyphrase file to mark as processed")
                # Create an empty file to mark as processed
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write("")
                continue

        print(Fore.GREEN + f" Found {len(keywords)} keyphrases for chapter {chapter_num}")
        # Only print first 5 for sample
        for phrase, score in keywords[:5]:
            print(f" {phrase}: {score:.4f}")

        # Write the keyphrases
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Only write the keyphrases, not the scores (for consistency with other methods)
            for phrase, score in keywords:
                out_f.write(f"{phrase}\n")

        print(Fore.GREEN + f"Keyphrases written to {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyphrases from text using KeyBert")
    parser.add_argument("--input_dir", type=str, default="data/chapters")
    parser.add_argument("--output_dir", type=str, default="data/keyphrases/keybert")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Path to model or HuggingFace name")
    parser.add_argument("--save_model_path", type=str, default='', help="Optional: Save model to this path")

    args = parser.parse_args()
    main(args)