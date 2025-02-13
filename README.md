# PDF Book Chapter Chunker

A simple Python tool that extracts text from a PDF book and splits it into individual chapters. This project leverages [pdfminer.six](https://github.com/pdfminer/pdfminer.six) for PDF text extraction, [Colorama](https://pypi.org/project/colorama/) for colorful terminal output, and regular expressions to automatically chunk the text into chapters.

## Features

- **PDF Extraction:** Automatically extracts the entire text from a PDF file.
- **Chapter Chunking:** Splits the extracted text into individual chapters based on form feed and chapter markers (e.g., `\fChapter {1}`, `\fChapter {2}`, etc.).
- **Organized Output:** Saves each chapter as a separate `.txt` file in a folder named after the book (with `_chunks` appended).
- **Configurable:** Easily modify file paths and settings via configuration constants.

## Prerequisites

- Python 3.6 or higher

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/parkerg16/key_phrase_extraction/
   cd key_phrase_extraction
Install dependencies:

**This project includes a requirements.txt file. Install the required packages with:**

  ```bash
  pip install -r requirements.txt
  ```

# Prepare Your PDF:
Place your PDF file (e.g., book.pdf) in the project directory.

Configure Settings:

You can adjust the configuration values at the top of the script (chunking.py) if needed:

BOOK_PATH: Path to your PDF file.
TEXT_OUTPUT: Path where the extracted text will be saved.
Run the Script:

Execute the script to extract text and split it into chapters:

```bash
python chunking.py
```

The script will:

Extract text from your PDF and save it as extracted_text.txt (if it doesn't already exist).
Split the text into chapters based on the pattern \fChapter {number}.
Create a folder (e.g., book_chunks) and output each chapter into its own file (e.g., chapter_1_chunk.txt, chapter_2_chunk.txt, etc.).
Skip any files or folders that already exist.
What Does chunking.py Do?
The chunking.py script is the core of this project. It performs the following tasks:

# PDF Extraction:
Uses pdfminer.six to extract text from the specified PDF file. The extracted text is saved to a file (default: extracted_text.txt).

# Text Chunking:
Reads the extracted text and applies a regular expression to split the text into chapters. Chapters are identified by a form feed (\f) followed by the word "Chapter" (with an optional chapter number in braces).

# File Output:
Creates a directory named after the PDF (with _chunks appended) and writes each chapter's content into separate text files named in the format chapter_X_chunk.txt. If a chapter file already exists, the script skips writing that file.

# Visual Feedback:
Uses Colorama to print colored messages to the console, indicating the progress of PDF extraction and chapter creation, along with a preview of the first 300 characters of each chapter.

Example Terminal Output
After running the script, you might see output like this:

```bash
extracted_text.txt already exists. Skipping PDF extraction.
Folder book_chunks already exists.
Written Chapter 1 to book_chunks/chapter_1_chunk.txt

--- Chapter 1 Preview ---
[First 300 characters of chapter 1...]

Written Chapter 2 to book_chunks/chapter_2_chunk.txt

--- Chapter 2 Preview ---
[First 300 characters of chapter 2...]
...
```
