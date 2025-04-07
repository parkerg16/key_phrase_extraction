import os
import re

with open('index_terms.txt', 'r', encoding='utf-8') as f:
    index_terms = [line.strip() for line in f.readlines()]

book_chunks_path = 'book_chunks'
ground_truth_path = 'ground_truth'

os.makedirs(ground_truth_path, exist_ok=True)

chapter_files = sorted([f for f in os.listdir(book_chunks_path) if f.startswith('chapter') and f.endswith('.txt')])

for chapter_file in chapter_files:
    chapter_num = chapter_file.split('_')[1].split('.')[0]
    chapter_path = os.path.join(book_chunks_path, chapter_file)
    
    with open(chapter_path, 'r', encoding='utf-8') as f:
        chapter_text = f.read().lower()
    
    chapter_ground_truth = []
    for term in index_terms:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        if re.search(pattern, chapter_text):
            chapter_ground_truth.append(term)
    
    output_path = os.path.join(ground_truth_path, f'chapter_{chapter_num}_ground_truth.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for term in chapter_ground_truth:
            f.write(term + '\n')
    
    print(f"Created ground truth for chapter {chapter_num} with {len(chapter_ground_truth)} terms")

print("Ground truth creation complete.")