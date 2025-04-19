import re
import os
import io
from colorama import Fore, init
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# Initialize colorama
init(autoreset=True)

# Default configurations
BOOK_PATH = 'book.pdf'
TEXT_OUTPUT = 'extracted_text.txt'


def extract_book(book_path=BOOK_PATH, text_output=TEXT_OUTPUT):
    """Extract text from PDF book using pdfminer.six"""
    output_file = text_output
    if os.path.exists(output_file):
        print(Fore.YELLOW + f"{output_file} already exists. Skipping PDF extraction.")
        return

    print(Fore.CYAN + f"Extracting text from {book_path}...")
    
    # Set up PDF parser and related objects
    output_string = io.StringIO()
    with open(book_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # Process each page
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    
    # Get the extracted text
    text = output_string.getvalue()
    
    # Write to output file
    with open(text_output, "w", encoding="utf-8") as text_file:
        text_file.write(text)

    print(Fore.GREEN + f"PDF Extraction Complete exported to {output_file}")


def chunk_text(text_path=TEXT_OUTPUT, skip_chapters=None, force_overwrite=False):
    """
    Split text into chapter chunks and save to separate files
    
    Args:
        text_path: Path to the extracted text file
        skip_chapters: List of chapter numbers to skip (1-indexed)
        force_overwrite: Whether to overwrite existing chapter files
    """
    if skip_chapters is None:
        skip_chapters = []
    
    # Print skipped chapters for clarity
    if skip_chapters:
        print(Fore.YELLOW + f"Skipping chapters: {', '.join(map(str, skip_chapters))}")
    
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
        # Skip specified chapters
        if i in skip_chapters:
            print(Fore.RED + f"Skipping Chapter {i} as requested." + Fore.RESET)
            continue
            
        output_file = os.path.join(output_folder, f"chapter_{i}_chunk.txt")
        
        # Check if file exists and determine whether to skip
        if os.path.exists(output_file) and not force_overwrite:
            print(Fore.YELLOW + f"{output_file} already exists. Skipping this chapter." + Fore.RESET)
            continue

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(chapter)
        print(Fore.CYAN + f"Written Chapter {i} to {output_file}" + Fore.RESET)
        print(f"\n--- Chapter {i} Preview ---")
        print(chapter[:300])  # prints first 300 characters of the chapter as a preview


# Default behavior when run directly - can be modified in IDE for different options
if __name__ == "__main__":
    # Extract the text from the PDF
    extract_book()
    
    # Chunk the text into chapters
    # Modify this line directly in the IDE to skip different chapters if needed
    # Examples:
    # chunk_text()  # Process all chapters (default behavior)
    # chunk_text(skip_chapters=[3, 5])  # Skip specific chapters if needed
    # chunk_text(force_overwrite=True)  # Overwrite existing files
    chunk_text(force_overwrite=True)  # Process all chapters, overwrite existing files
