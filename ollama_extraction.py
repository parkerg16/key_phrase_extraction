import os
import requests

# Configuration
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:instruct"

def extract_keyphrases(text: str) -> list[str]:
    """
    Sends the text to the Ollama API and parses out the keyphrases
    from the 'response' field in the returned JSON.
    """
    prompt = (
        "You are an expert keyphrase extractor. Identify up to 75 contiguous noun phrases (1–5 words) from the text, in order of importance, and output them exactly as they appear—no stemming or paraphrasing. "
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
    else:
        kp_str = raw
    # Split on semicolons and clean whitespace
    phrases = [p.strip() for p in kp_str.split(";") if p.strip()]
    return phrases

def main():
    book_chunks_path = 'pre_chunked'  # Changed from 'book_chunks' to use the pre_chunked folder
    key_phrases_path = 'key_phrases_ollama'

    # Ensure output directory exists
    os.makedirs(key_phrases_path, exist_ok=True)

    # Get all .txt chapter files
    try:
        # Sort by chapter number (extracted from filenames) for proper numerical order
        def get_chapter_num(filename):
            try:
                # Extract the number from ch{num}.txt format
                if filename.startswith('ch') and filename.endswith('.txt'):
                    # Extract the part between 'ch' and '.txt'
                    chapter_str = filename[2:-4]
                    return int(chapter_str)
                # Also handle older chapter_X_chunk.txt format for backward compatibility
                elif filename.startswith('chapter_') and '_chunk.txt' in filename:
                    return int(filename.split('_')[1])
                else:
                    return 999  # Default for unknown formats
            except (IndexError, ValueError):
                return 999  # Large default value for files without proper naming
                
        chapter_files = sorted(
            [f for f in os.listdir(book_chunks_path) if f.endswith('.txt')],
            key=get_chapter_num
        )
        
        # Print the list of found chapter files
        print(f"Found chapter files: {', '.join(chapter_files)}")
        
        # Print which chapters were detected
        chapter_nums = [get_chapter_num(f) for f in chapter_files]
        print(f"Detected chapters: {', '.join(map(str, sorted(chapter_nums)))}")
        print(f"Found {len(chapter_files)} chapters to process")
        
    except FileNotFoundError:
        print(f"Error: '{book_chunks_path}' directory not found.")
        return

    # Process each chapter file
    for file_name in chapter_files:
        # Extract chapter number from filename
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

        # Extract keyphrases via Ollama
        phrases = extract_keyphrases(chapter_text)

        # Print to console
        print(f"--- Chapter {chapter_num} Keyphrases ---")
        for ph in phrases:
            print(ph)
        print()

        # Write to output file
        output_file = os.path.join(key_phrases_path, f"chapter_{chapter_num}_keyphrases.txt")
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(phrases))

        print(f"Keyphrases for Chapter {chapter_num} written to {output_file}\n")

if __name__ == "__main__":
    main()
