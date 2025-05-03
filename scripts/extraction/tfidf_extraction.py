import os
import argparse
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from colorama import Fore, init

init(autoreset=True)

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

def extract_keyphrases_sklearn(text, top_n=100):
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

        # For very large chapters (over 250,000 characters), split into chunks
        if len(chapter_text) > 250000:
            print(f" Chapter {chapter_num} is very large, splitting into chunks...")
            chunk_size = 200000 # Process in 200k chunks
            combined_keywords = []

            for chunk_start in range(0, len(chapter_text), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(chapter_text))
                chunk = chapter_text[chunk_start:chunk_end]
                print(f" Processing chunk {chunk_start} to {chunk_end} of {len(chapter_text)}...")

                try:
                    # Extract keywords from this chunk
                    chunk_keywords = extract_keyphrases_sklearn(
                        chunk,
                        top_n=40  # Extract more from each chunk
                    )

                    combined_keywords.extend(chunk_keywords)
                    print(f" Extracted {len(chunk_keywords)} keyphrases in chunk")
                except Exception as e:
                    print(f" Error processing chunk: {e}")
                    continue

            # Combine and deduplicate keywords from all chunks
            keywords = []
            seen_phrases = set()
            for phrase, score in sorted(combined_keywords, key=lambda x: x[1], reverse=True):
                if phrase not in seen_phrases and len(keywords) < 100:
                    keywords.append((phrase, score))
                    seen_phrases.add(phrase)
        else:
            # Normal processing for smaller chapters
            try:
                # Extract keywords using sklearn's TF-IDF
                keywords = extract_keyphrases_sklearn(
                    chapter_text,
                    top_n=100  # Number of phrases to extract
                )
            except Exception as e:
                print(f" Error extracting keyphrases from chapter {chapter_num}: {e}")
                print(" Generating empty keyphrase file to mark as processed")
                # Create an empty file to mark as processed
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write("")
                continue

        print(Fore.GREEN + f" Found {len(keywords)} keyphrases for chapter {chapter_num}")
        # Only print first 5 for sample
        for phrase, score in keywords[:5]:
            print(f" {phrase}: {score:.4f}")

        # Write the keyphrases
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Only write the keyphrases, not the scores (for consistency with other methods)
            for phrase, score in keywords:
                out_f.write(f"{phrase}\n")

        print(Fore.GREEN + f"Keyphrases written to {output_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyphrases from text using TF-IDF")
    parser.add_argument("--input_dir", type=str, default="data/chapters")
    parser.add_argument("--output_dir", type=str, default="data/keyphrases/tfidf")
    
    args = parser.parse_args()
    main(args)