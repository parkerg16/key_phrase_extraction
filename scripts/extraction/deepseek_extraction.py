import os
import re
import argparse
import requests
from colorama import Fore, init

init(autoreset=True)

# Configuration for DeepSeek API
DEEPSEEK_API = "http://localhost:11434/api/generate"  # Using the same port as Ollama
MODEL_NAME = "deepseek-r1:8b"  # The requested DeepSeek model

def extract_keyphrases(text: str) -> list[str]:
    """
    Sends the text to the DeepSeek API and parses out the keyphrases
    from the 'response' field in the returned JSON.
    """
    prompt = (
        "You are an expert keyphrase extractor. Your ONLY task is to extract important keyphrases from the text below.\n\n"
        "INSTRUCTIONS:\n"
        "1. Extract up to 75 important noun phrases (1-5 words each) from the text\n"
        "2. List them in order of importance\n"
        "3. Use EXACTLY the same wording as appears in the text - do not stem, lemmatize or paraphrase\n"
        "4. Format your response in this exact format (no additional text or explanations):\n\n"
        "Keyphrases: phrase one; phrase two; phrase three; etc\n\n"
        "EXAMPLE INPUT:\n"
        "Text: Machine learning is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.\n\n"
        "EXAMPLE OUTPUT:\n"
        "Keyphrases: machine learning; field of inquiry; methods; artificial intelligence; performance; tasks; data\n\n"
        "Now extract keyphrases from this text:\n\n"
        f"Text: {text}"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        resp = requests.post(DEEPSEEK_API, json=payload)
        resp.raise_for_status()
        
        data = resp.json()
        
        # The completion text is under 'response'
        raw = data.get("response", "")
        
        # Print raw response for debugging (in case the format is unexpected)
        print(Fore.YELLOW + f"Raw response (first 200 chars): {raw[:200]}...")
        
        # Extract the part after 'Keyphrases:' if present (case insensitive)
        if "keyphrases:" in raw.lower():
            kp_str = raw.lower().split("keyphrases:", 1)[1]
        else:
            # Try to find list-like formats if the expected format isn't found
            print(Fore.YELLOW + "Warning: 'Keyphrases:' marker not found, attempting to parse anyway...")
            # Check if it starts with numbered list
            if any(line.strip().startswith(str(i)+'.') for i in range(1, 10) for line in raw.split('\n')):
                # Extract phrases from numbered list
                phrases = []
                for line in raw.split('\n'):
                    line = line.strip()
                    if line and any(line.startswith(f"{i}.") for i in range(1, 100)):
                        phrase = line.split('.', 1)[1].strip()
                        if phrase:
                            phrases.append(phrase)
                return phrases
            else:
                # Fall back to the raw text
                kp_str = raw
        
        # Split on semicolons and clean whitespace
        phrases = [p.strip() for p in kp_str.split(';') if p.strip()]
        
        # If we didn't get any phrases with semicolons, try newlines
        if not phrases and '\n' in kp_str:
            phrases = [p.strip() for p in kp_str.split('\n') if p.strip()]
        
        # If we still have no phrases, try commas
        if not phrases and ',' in kp_str:
            phrases = [p.strip() for p in kp_str.split(',') if p.strip()]
            
        # Remove any numbers/bullets at the beginning of phrases
        cleaned_phrases = []
        for p in phrases:
            # Remove numbering like "1. " or "- " from the beginning
            p = re.sub(r'^\d+\.\s*|\-\s*', '', p)
            if p.strip():
                cleaned_phrases.append(p.strip())
                
        print(Fore.GREEN + f"Successfully extracted {len(cleaned_phrases)} phrases")
        return cleaned_phrases
    
    except requests.exceptions.RequestException as e:
        print(Fore.RED + f"API request error: {e}")
        return []

def get_chapter_num(filename):
    try:
        # Extract the number from the ch{num}.txt format
        if filename.startswith("ch") and filename.endswith(".txt"):
            return int(filename[2:-4])
        # Also handle older chapter_X_chunk.txt formats for backward compatibility
        elif filename.startswith("chapter_") and '_chunk.txt' in filename:
            return int(filename.split('_')[1])
        else:
            return 999  # Default for unknown formats
    except (IndexError, ValueError):
        return 999

def main(args):
    book_chunks_path = args.input_dir
    key_phrases_path = args.output_dir

    # Ensure output directory exists
    os.makedirs(key_phrases_path, exist_ok=True)
    print(Fore.GREEN + f"Output directory: {key_phrases_path}")

    # Get all .txt chapter files
    try:
        # Sort by chapter number for proper numerical order
        chapter_files = sorted(
            [f for f in os.listdir(book_chunks_path) if f.endswith(".txt")],
            key=get_chapter_num
        )

        # Print the list of found chapter files
        print(Fore.CYAN + f"Found chapter files: {', '.join(chapter_files)}")

        # Print which chapters were detected
        chapter_nums = [get_chapter_num(f) for f in chapter_files]
        print(Fore.CYAN + f"Detected chapters: {', '.join(map(str, sorted(chapter_nums)))}")
        print(Fore.CYAN + f"Found {len(chapter_files)} chapters to process.")

    except FileNotFoundError:
        print(Fore.RED + f"Error: Directory '{book_chunks_path}' not found.")
        return

    # Process each chapter file
    for i, file_name in enumerate(chapter_files):
        chapter_num = get_chapter_num(file_name)
        if chapter_num == 999:
            print(Fore.YELLOW + f"Warning: Could not determine chapter number from filename: {file_name}")
            continue

        # Check if this chapter's keyphrases have already been generated
        output_file = os.path.join(key_phrases_path, f"chapter_{chapter_num}_keyphrases.txt")
        if os.path.exists(output_file):
            print(Fore.YELLOW + f"Skipping chapter {chapter_num} ({i+1}/{len(chapter_files)}) - already processed.")
            continue

        print(Fore.CYAN + f"Processing chapter {chapter_num} ({i+1}/{len(chapter_files)})...")

        file_path = os.path.join(book_chunks_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_text = f.read()

        print(f"Extracting keyphrases from chapter {chapter_num} ({len(chapter_text)} characters)...")

        # Extract keyphrases from the text
        phrases = extract_keyphrases(chapter_text)

        # Print to console (first 5 for visibility)
        print(Fore.GREEN + f"Found {len(phrases)} keyphrases for chapter {chapter_num}")
        for phrase in phrases[:5]:  # Only show first 5
            print(f"- {phrase}")

        # Save the keyphrases to a file
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(phrases))

        print(Fore.GREEN + f"Keyphrases for Chapter {chapter_num} saved to {output_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyphrases from text using DeepSeek model")
    parser.add_argument("--input_dir", type=str, default="data/chapters",
                        help="Directory containing chapter text files")
    parser.add_argument("--output_dir", type=str, default="data/keyphrases/deepseek",
                        help="Directory to save extracted keyphrases")
    args = parser.parse_args()
    main(args)