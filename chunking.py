import re
import os
from pdfminer.high_level import extract_text
from colorama import Fore, init

# Configurations
init(autoreset=True)
BOOK_PATH = 'book.pdf'
TEXT_OUTPUT = 'extracted_text.txt'


def extract_book(book_path=BOOK_PATH, text_output=TEXT_OUTPUT):
    output_file = text_output
    if os.path.exists(output_file):
        print(Fore.YELLOW + f"{output_file} already exists. Skipping PDF extraction.")
        return

    text = extract_text(book_path)
    with open(TEXT_OUTPUT, "w", encoding="utf-8") as text_file:
        text_file.write(text)

    print(Fore.GREEN + f"PDF Extraction Complete exported to {output_file}")


def chunk_text(text_path=TEXT_OUTPUT):
    with open(text_path, "r", encoding="utf-8") as file:
        text = file.read()

    pattern = r'\f\s*Chapter\s*\{?\d+\}?'
    chapters = re.split(pattern, text)
    chapters = [chapter.strip() for chapter in chapters if chapter.strip()]
    print(f"Total chapters found: {len(chapters)}")

    base_name = os.path.splitext(os.path.basename(BOOK_PATH))[0]
    output_folder = base_name + "_chunks"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(Fore.GREEN + f"Created folder {output_folder}" + Fore.RESET)
    else:
        print(Fore.YELLOW + f"Folder {output_folder} already exists." + Fore.RESET)

    # Write each chapter into its own file
    for i, chapter in enumerate(chapters, start=1):
        output_file = os.path.join(output_folder, f"chapter_{i}_chunk.txt")
        if os.path.exists(output_file):
            print(Fore.YELLOW + f"{output_file} already exists. Skipping this chapter." + Fore.RESET)
            continue

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(chapter)
        print(Fore.CYAN + f"Written Chapter {i} to {output_file}" + Fore.RESET)
        print(f"\n--- Chapter {i} Preview ---")
        print(chapter[:300])  # prints first 300 characters of the chapter as a preview


extract_book()
chunk_text()
