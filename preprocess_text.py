import os
import re


def preprocess_text(text):
    text = re.sub(r'\[\d+\]', '', text)  # Remove citations
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)  # Remove citations with years
    # text = re.sub(r'\(-\d+\)', '', text)  # Remove negative numbers this removes hyphens on accident so I'm not going to use it
    text = re.sub(r'\b(?:figure|fig\.?)\s*\d+\b', '', text, flags=re.IGNORECASE)  # Remove figures
    text = re.sub(r'\bchapter\s*\d+\b', '', text, flags=re.IGNORECASE)  # Remove Chapters
    text = re.sub(r'\btable\s*\d+\b', '', text, flags=re.IGNORECASE)  # Remove Tables
    text = re.sub(r'\b\d{4}\b', '', text)  # Remove dates
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\d+', '', text)  # Remove all remaining numbers

    # Stripping the white space makes it hard to read the output so we wil disable for now.
    # text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text


def process_chapters(input_folder='new_book_chunks', output_folder='Processed_Chapters'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            input_path = os.path.join(input_folder, file_name)
            with open(input_path, 'r', encoding='utf-8') as file:
                chapter_text = file.read()
            processed_text = preprocess_text(chapter_text)
            output_path = os.path.join(output_folder, file_name)
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write(processed_text)
            print(f"Processed {file_name} and saved to {output_path}")


if __name__ == '__main__':
    process_chapters(input_folder='Chapters')
