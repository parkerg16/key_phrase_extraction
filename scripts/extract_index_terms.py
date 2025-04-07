import os
import re
import argparse

def extract_index_terms(input_text_path, output_path):
    with open(input_text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if 'index' not in text.lower():
        print("Warning: 'Index' not found. Double-check the source.")

    start_match = re.search(r'\n\s*Index\s*\n', text, re.IGNORECASE)
    if not start_match:
        raise ValueError("Index section not found.")
    start_idx = start_match.end()

    end_match = re.search(r'\n\s*About the Author\s*\n', text[start_idx:], re.IGNORECASE)
    end_idx = start_idx + end_match.start() if end_match else len(text)

    index_text = text[start_idx:end_idx]
    lines = index_text.split('\n')

    terms = []
    for line in lines:
        line = line.strip()
        if line:
            term = line.split(',')[0].strip()  # Handle: "term, 135-137"
            terms.append(term)

    with open(output_path, 'w', encoding='utf-8') as f:
        for term in terms:
            f.write(term + '\n')

    print(f"Extracted {len(terms)} index terms to {output_path}")
    return terms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract index terms from full textbook text.")
    parser.add_argument('--input', type=str, default='data/extracted_text.txt')
    parser.add_argument('--output', type=str, default='data/index_terms.txt')
    args = parser.parse_args()

    extract_index_terms(args.input, args.output)
