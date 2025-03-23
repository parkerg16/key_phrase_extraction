import os
import re

def extract_index_terms(extracted_text_path):
    with open(extracted_text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Check if "Index" is present for debugging
    if 'index' not in text.lower():
        print("Warning: 'Index' not found in the file. Please check the extracted text.")
    
    # Find start of index, ignoring case and allowing extra spaces
    match = re.search(r'\n\s*Index\s*\n', text, re.IGNORECASE)
    if not match:
        raise ValueError("Index section not found in the extracted text file")
    start_idx = match.end()

    # Find end of index, similarly flexible
    match = re.search(r'\n\s*About the Author\s*\n', text[start_idx:], re.IGNORECASE)
    if match:
        end_idx = start_idx + match.start()
    else:
        end_idx = len(text)

    index_text = text[start_idx:end_idx]
    
    # Process the index text
    lines = index_text.split('\n')
    terms = []
    for line in lines:
        line = line.strip()
        if line:
            term = line.split(',')[0].strip()
            terms.append(term)
    
    # Write terms to file
    with open('index_terms.txt', 'w', encoding='utf-8') as f:
        for term in terms:
            f.write(term + '\n')
    
    return terms

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the file in the same directory
extracted_text_path = os.path.join(script_dir, 'extracted_text.txt')

index_terms = extract_index_terms(extracted_text_path)
print(f"Extracted {len(index_terms)} index terms.")