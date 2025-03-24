import os
from keybert import KeyBERT
from colorama import Fore, init
from sentence_transformers import SentenceTransformer

init(autoreset=True)

# Initialize KeyBERT with the default model
# model = SentenceTransformer("fine_tuned_model")
model = SentenceTransformer("distilroberta-base-msmarco-v2")
kw_model = KeyBERT(model=model)


# Folder containing your chapter text files
book_chunks_path = 'Processed_Chapters'
key_phrases_path = 'key_phrases'

if not os.path.exists(key_phrases_path):
    os.makedirs(key_phrases_path)
    print(f"Created folder: {key_phrases_path}")

# Get a sorted list of all .txt files in the folder
chapter_files = sorted([f for f in os.listdir(book_chunks_path) if f.endswith('.txt')])

# Iterate over each file and extract keywords
for idx, file_name in enumerate(chapter_files, start=1):
    file_path = os.path.join(book_chunks_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        chapter_text = f.read()

    # Extract keywords from the chapter text
    keywords = kw_model.extract_keywords(
        chapter_text,
        keyphrase_ngram_range=(1, 3),  # Length of phrase
        stop_words='english',  # Filter common english
        use_mmr=True,  # Text relation ?
        diversity=0.7,
        top_n=100  # Number of phrases to extract
    )

    print(Fore.GREEN + f"--- Chapter {idx} Keywords ---")
    print(keywords)

    # Write the keyphrases to an output file in the 'key_phrases' folder.
    output_file = os.path.join(key_phrases_path, f"chapter_{idx}_keyphrases.txt")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for phrase, score in keywords:
            out_f.write(f"{phrase}: {score}\n")

    print(Fore.RED + f"Keyphrases written to {output_file}\n")
