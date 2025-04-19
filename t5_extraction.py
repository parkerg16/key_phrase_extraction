import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from colorama import Fore, init
import torch

init(autoreset=True)

# Load the pretrained T5 model (fine-tuned for keyphrase extraction)
model_name = "BeIR/trec-covid-keyphrase-t5-base"  # Swap this with your own fine-tuned model if available
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
model.eval()

# Paths
book_chunks_path = 'book_chunks'
key_phrases_path = 'key_phrases_t5'

if not os.path.exists(key_phrases_path):
    os.makedirs(key_phrases_path)
    print(f"Created folder: {key_phrases_path}")

# Get a sorted list of all .txt files in the folder
chapter_files = sorted([f for f in os.listdir(book_chunks_path) if f.endswith('.txt')])

# Function to extract keyphrases using T5
def extract_keyphrases(text, max_input_length=512):
    input_text = f"extract keyphrases: {text.strip().replace('\n', ' ')}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True).cuda()
    outputs = model.generate(
        inputs,
        max_length=64,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [phrase.strip() for phrase in decoded.split(",") if phrase.strip()]

# Process each file
for idx, file_name in enumerate(chapter_files, start=1):
    file_path = os.path.join(book_chunks_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        chapter_text = f.read()

    # Extract keyphrases
    keyphrases = extract_keyphrases(chapter_text)

    print(Fore.GREEN + f"--- Chapter {idx} Keyphrases ---")
    for phrase in keyphrases:
        print(phrase)

    # Write to file
    output_file = os.path.join(key_phrases_path, f"chapter_{idx}_keyphrases.txt")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for phrase in keyphrases:
            out_f.write(f"{phrase}\n")

    print(Fore.RED + f"Keyphrases written to {output_file}\n")
