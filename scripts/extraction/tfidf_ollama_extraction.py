import os
import argparse
import re
import json
import time
import string
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from colorama import Fore, init

init(autoreset=True)

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:instruct"  # Use the same model as in ollama_extraction.py

def get_chapter_num(filename):
    try:
        # Extract the number from the ch{num}.txt format
        if filename.startswith("ch") and filename.endswith(".txt"):
            # Extract the part between 'ch' and '.txt'
            return int(filename[2:-4])
        # Also handle older chapter_X_chunk.txt formats for backward compatibility
        elif filename.startswith("chapter_") and '_chunk.txt' in filename:
            return int(filename.split('_')[1])
        else:
            return 999 # Default for unknown formats
    except (IndexError, ValueError):
        return 999

def extract_keyphrases_sklearn(text, top_n=150):
    """Extract keyphrases using scikit-learn's TF-IDF implementation"""
    # Clean the text - remove chapter headers and figure references
    text = re.sub(r'CHAPTER \d+', '', text)
    text = re.sub(r'Figure \d+[-\.]\d+', '', text)
    
    # Prepare corpus by splitting into paragraphs
    paragraphs = [p for p in text.split('\n\n') if len(p.strip()) > 50]
    
    # If we don't have enough paragraphs, use sentences instead
    if len(paragraphs) < 5:
        # Simple sentence splitting by common terminators
        sentences = []
        for para in paragraphs:
            sentences.extend([s.strip() for s in re.split(r'[.!?]+', para) if len(s.strip()) > 40])
        paragraphs = sentences
    
    # Ensure we have at least one valid paragraph
    if not paragraphs:
        return []
    
    # Create TF-IDF vectorizer for unigrams
    unigram_vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        max_df=0.85,
        min_df=1,
        stop_words='english'
    )
    
    # Create TF-IDF vectorizer for bigrams
    bigram_vectorizer = TfidfVectorizer(
        ngram_range=(2, 2),
        max_df=0.85,
        min_df=1,
        stop_words='english'
    )
    
    # Create TF-IDF vectorizer for trigrams
    trigram_vectorizer = TfidfVectorizer(
        ngram_range=(3, 3),
        max_df=0.85,
        min_df=1,
        stop_words='english'
    )
    
    try:
        # Extract unigrams
        unigram_matrix = unigram_vectorizer.fit_transform(paragraphs)
        unigram_feature_names = unigram_vectorizer.get_feature_names_out()
        unigram_scores = np.asarray(unigram_matrix.sum(axis=0)).flatten()
        unigram_keywords = [(unigram_feature_names[i], unigram_scores[i]) 
                           for i in unigram_scores.argsort()[::-1] 
                           if len(unigram_feature_names[i]) > 2]  # Filter out very short words
        unigram_keywords = unigram_keywords[:top_n // 3]
    except Exception as e:
        print(f"Error extracting unigrams: {e}")
        unigram_keywords = []
    
    try:
        # Extract bigrams
        bigram_matrix = bigram_vectorizer.fit_transform(paragraphs)
        bigram_feature_names = bigram_vectorizer.get_feature_names_out()
        bigram_scores = np.asarray(bigram_matrix.sum(axis=0)).flatten()
        bigram_keywords = [(bigram_feature_names[i], bigram_scores[i]) 
                          for i in bigram_scores.argsort()[::-1]]
        bigram_keywords = bigram_keywords[:top_n // 3]
    except Exception as e:
        print(f"Error extracting bigrams: {e}")
        bigram_keywords = []
    
    try:
        # Extract trigrams
        trigram_matrix = trigram_vectorizer.fit_transform(paragraphs)
        trigram_feature_names = trigram_vectorizer.get_feature_names_out()
        trigram_scores = np.asarray(trigram_matrix.sum(axis=0)).flatten()
        trigram_keywords = [(trigram_feature_names[i], trigram_scores[i]) 
                           for i in trigram_scores.argsort()[::-1]]
        trigram_keywords = trigram_keywords[:top_n // 3]
    except Exception as e:
        print(f"Error extracting trigrams: {e}")
        trigram_keywords = []
    
    # Combine all keyphrases and sort by score
    all_keywords = unigram_keywords + bigram_keywords + trigram_keywords
    sorted_keywords = sorted(all_keywords, key=lambda x: x[1], reverse=True)
    
    # Make list of unique keyphrases and return top N
    unique_keyphrases = []
    seen = set()
    for phrase, score in sorted_keywords:
        if phrase not in seen and len(unique_keyphrases) < top_n:
            unique_keyphrases.append((phrase, score))
            seen.add(phrase)
    
    return unique_keyphrases

def ollama_api_extract_keyphrases(text, candidates, model=MODEL_NAME, max_retries=3, retry_delay=5):
    """Use Ollama API to extract keyphrases with TF-IDF candidates as guidance"""
    
    # Format the list of candidate keyphrases
    candidate_text = "\n".join([f"- {phrase}" for phrase, _ in candidates])
    
    # Prepare the prompt
    prompt = (
        "You are an expert keyphrase extractor. Below is a text followed by a list of "
        "candidate keyphrases extracted using TF-IDF. Select the 75 most important keyphrases "
        "from these candidates that best represent the main concepts in the text. "
        "Return ONLY the selected keyphrases after 'Keyphrases:' separated by semicolons (;).\n\n"
        f"TEXT:\n{text[:3000]}...\n\n"  # Limit text size but provide enough context
        f"CANDIDATE KEYPHRASES:\n{candidate_text}\n\n"
    )
    
    # Prepare the API request
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    # Send the request with retries
    for attempt in range(max_retries):
        try:
            print(f"  Making API request to Ollama (attempt {attempt+1}/{max_retries})...")
            response = requests.post(OLLAMA_API, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Parse the response
            result = response.json()
            raw_text = result.get("response", "")
            
            # Extract keyphrases from the response
            if "Keyphrases:" in raw_text:
                kp_str = raw_text.split("Keyphrases:", 1)[1]
            else:
                kp_str = raw_text
                
            # Split on semicolons and clean whitespace
            keyphrases = [p.strip() for p in kp_str.split(';') if p.strip()]
            
            print(f"  Successfully extracted {len(keyphrases)} keyphrases via Ollama API")
            return keyphrases
            
        except requests.exceptions.RequestException as e:
            print(f"  Error in API request (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  Failed after {max_retries} attempts. Using TF-IDF candidates directly as fallback.")
                return [phrase for phrase, _ in candidates[:75]]
        except Exception as e:
            print(f"  Unexpected error: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  Failed after {max_retries} attempts. Using TF-IDF candidates directly as fallback.")
                return [phrase for phrase, _ in candidates[:75]]

def main(args):
    book_chunks_path = args.input_dir
    key_phrases_path = args.output_dir

    if not os.path.exists(key_phrases_path):
        os.makedirs(key_phrases_path)
        print(f"Created folder: {key_phrases_path}")

    # Get a sorted list of all .txt files in the folder (sorted by chapter number)
    chapter_files = sorted(
        [f for f in os.listdir(book_chunks_path) if f.endswith(".txt")],
        key=get_chapter_num
    )

    # Print the list of found chapter files
    print(f"Found chapter files: {', '.join(chapter_files)}")

    # Determine actual chapter numbers directly from the file names
    chapter_numbers = []
    for file_name in chapter_files:
        try:
            chapter_num = get_chapter_num(file_name)
            chapter_numbers.append(chapter_num)
        except:
            # Fall back to using the enumeration as the chapter number
            chapter_numbers.append(len(chapter_numbers) + 1)

    # Report which chapters were detected
    print(f"Detected chapters:  {', '.join(map(str,sorted(chapter_numbers)))}")
    print(f"Found {len(chapter_files)} chapters to process.")

    # Iterate over each file and extract keywords
    for i, (file_name, chapter_num) in enumerate(zip(chapter_files, chapter_numbers)):
        # Check if this chapter's keyphrases have already been generated
        output_file = os.path.join(key_phrases_path, f"chapter_{chapter_num}_keyphrases.txt")
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Skipping chapter {chapter_num} ({i+1}/{len(chapter_files)}) - already processed.")
            continue

        print(f"Processing chapter {chapter_num} ({i+1}/{len(chapter_files)})...")

        file_path = os.path.join(book_chunks_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            chapter_text = f.read()

        print(f" Extracting keyphrases from chapter {chapter_num} ({len(chapter_text)} characters)...")

        # Step 1: Get TF-IDF candidates
        print(" Generating TF-IDF candidate keyphrases...")
        tfidf_candidates = extract_keyphrases_sklearn(chapter_text, top_n=150)
        
        # Step 2: Use Ollama to select the best keyphrases from candidates
        print(" Using Ollama to select best keyphrases from candidates...")
        keyphrases = ollama_api_extract_keyphrases(chapter_text, tfidf_candidates, model=args.model)

        # Print summary
        print(Fore.GREEN + f" Selected {len(keyphrases)} keyphrases for chapter {chapter_num}")
        print(" Sample keyphrases:")
        for phrase in keyphrases[:5]:
            print(f"  - {phrase}")

        # Write the keyphrases
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for phrase in keyphrases:
                out_f.write(f"{phrase}\n")

        print(Fore.GREEN + f"Keyphrases written to {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyphrases using TF-IDF + Ollama")
    parser.add_argument("--input_dir", type=str, default="data/chapters")
    parser.add_argument("--output_dir", type=str, default="data/keyphrases/tfidf_ollama")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Ollama model to use")
    
    args = parser.parse_args()
    main(args)