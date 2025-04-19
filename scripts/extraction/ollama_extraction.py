import os
import argparse
import requests

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:instruct"

def extract_keyphrases(text: str) -> list[str]:
    """
    Sends the text to the Ollama API and pares out the keyphrases
    from the 'response' field in the returned JSON.
    """

    prompt = (
        "You are an expert keyphrase extractor. Identify up to 75 contiguous noun phrases (1-5 words) from the text, in order of importance, and output tehm exactly as they appear-no stemming or paraphrasing. "
        "The answer should be listed after 'Keyphrases:' and separated by semicolons (;).\n\n"
        f"Text: {text}"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    resp = requests.post(OLLAMA_API, json=payload)
    resp.raise_for_status()

    data = resp.json()

    # The completion text is under 'response'
    raw = data.get("response", "")

    # Extract the part after 'Keyphrases:' if present
    if "Keyphrases:" in raw:
        kp_str = raw.split("Keyphrases:", 1)[1]
    else :
        kp_str = raw

    # Split on semicolons and clean whitespace
    phrases = [p.strip() for p in kp_str.split(';') if p.strip()]
    return phrases

def main(args):
    book_chunks_path = args.input_dir
    key_phrases_path = args.output_dir

    # Ensure output directory exists
    os.makedirs(key_phrases_path, exist_ok=True)

    # Get all .txt chapter files
    try:
        # Sort by chapter number (extracted from filenames) for proper numerical order
        def get_chapter_num(filename):
            try:
                # Extract the number from the ch{num}.txt format
                if filename.startswith("ch") and filename.endswith(".txt"):
                    return int(filename[2:-4])
                elif filename.startswith("chapter_") and '_chunk.txt' in filename:
                    return int(filename.split('_')[1])
                else:
                    return 999 # Default for unknown formats
            except (IndexError, ValueError):
                return 999 # Large default value for files without proper making

        chapter_files = sorted(
            [f for f in os.listdir(book_chunks_path) if f.endswith(".txt")],
            key=get_chapter_num
        )

        # Print the list of found chapter files
        print(f"Found chapter files: {', '.join(chapter_files)}")

        # Print which chapters were detected
        chapter_nums = [get_chapter_num(f) for f in chapter_files]
        print(f"Detected chapters:  {', '.join(map(str,sorted(chapter_nums)))}")
        print(f"Found {len(chapter_files)} chapter to process.")

    except FileNotFoundError:
        print(f"Error: Directory '{book_chunks_path}' not found.")
        return

    # Process each chapter file
    for file_name in chapter_files:
        # Extract chapter number from the filename
        try:
            # Use the same logic as in get_chapter_num
            if file_name.startswith('ch') and file_name.endswith('.txt'):
                # Extract the part between 'ch' and '.txt'
                chapter_str = file_name[2:-4]
                chapter_num = int(chapter_str)
            elif file_name.startswith('chapter_') and '_chunk.txt' in file_name:
                chapter_num = int(file_name.split('_')[1])
            else:
                print(f"Warning: Could not determine chapter number from filename: {file_name}")
                continue
        except (IndexError, ValueError):
            print(f"Warning: Could not determine chapter number from filename: {file_name}")
            continue

        file_path = os.path.join(book_chunks_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_text = f.read()

        # Extract keyphrases from the text
        phrases = extract_keyphrases(chapter_text)

        # Print to console
        print(f"--- Chapter {chapter_num} Keyphrases ---")
        for phrase in phrases:
            print(phrase)
        print()

        # Save the keyphrases to a file
        output_file = os.path.join(key_phrases_path, f"chapter_{chapter_num}_keyphrases.txt")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(phrases))

        print(f"Keyphrases for Chapter {chapter_num} saved to {output_file}\n")

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Extract keyphrases from text using Ollama API")
    parse.add_argument("--input_dir", type=str, default="data/chapters")
    parse.add_argument("--output_dir", type=str, default="data/keyphrases/ollama")
    args = parse.parse_args()
    main(args)