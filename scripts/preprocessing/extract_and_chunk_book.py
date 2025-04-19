import os
import re
import argparse
from pdfminer.high_level import extract_text
from colorama import Fore, init
from preprocess_text import process_chapters

init(autoreset=True)


def extract_book_text(book_path, output_text_path):
    if os.path.exists(output_text_path):
        print(Fore.YELLOW + f"{output_text_path} already exists. Skipping PDF extraction.")
        return
    text = extract_text(book_path)
    with open(output_text_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    print(Fore.GREEN + f"PDF text extracted to {output_text_path}")


def split_text_into_chapters(text_path, output_dir, skip_first=True, max_chapters=None):
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    pattern = r'\f\s*Chapter\s*\{?\d+\}?'
    chapters = re.split(pattern, text)
    chapters = [c.strip() for c in chapters if c.strip()]

    if skip_first:
        chapters = chapters[1:]
    if max_chapters:
        chapters = chapters[:max_chapters]

    os.makedirs(output_dir, exist_ok=True)

    for i, chapter in enumerate(chapters, start=1):
        output_file = os.path.join(output_dir, f"chapter_{i}_chunk.txt")
        if not os.path.exists(output_file):
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(chapter)
            print(Fore.CYAN + f"Written Chapter {i} to {output_file}" + Fore.RESET)
        else:
            print(Fore.YELLOW + f"{output_file} already exists. Skipping." + Fore.RESET)

    return output_dir


def main(book_path, output_dir, skip_first=True, max_chapters=None):
    intermediate_text = "data/extracted_text.txt"
    extract_book_text(book_path, intermediate_text)
    chunk_dir = split_text_into_chapters(intermediate_text, output_dir, skip_first, max_chapters)
    process_chapters(chunk_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and chunk textbook PDF into chapter files.")
    parser.add_argument("--book_path", type=str, default="data/raw/new_book.pdf")
    parser.add_argument("--output_dir", type=str, default="data/chunks")
    parser.add_argument("--max_chapters", type=int, default=19)
    parser.add_argument("--skip_first", action="store_true")

    args = parser.parse_args()
    main(args.book_path, args.output_dir, args.skip_first, args.max_chapters)
