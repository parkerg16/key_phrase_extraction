import os
import re

def preprocess_text(text):
    text = re.sub(r'\[\d+\]', '', text)  # Inline citations [3]
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)  # Year-based citations (Smith, 2022)
    text = re.sub(r'\b(?:figure|fig\.?)\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bchapter\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\btable\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{4}\b', '', text)  # Standalone years
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\d+', '', text)  # Remaining numbers
    return text.lower()


def process_chapters(input_folder='data/chunks', output_folder='data/processed_chunks'):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            input_path = os.path.join(input_folder, file_name)
            with open(input_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
            cleaned_text = preprocess_text(raw_text)
            output_path = os.path.join(output_folder, file_name)
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write(cleaned_text)
            print(f"Processed {file_name} → {output_path}")


if __name__ == '__main__':
    process_chapters()
