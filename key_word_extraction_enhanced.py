import os
import re
import nltk
from keybert import KeyBERT
from colorama import Fore, init
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

init(autoreset=True)

# Initialize KeyBERT with a public pre-trained model
# Using all-MiniLM-L6-v2 which is a good general-purpose model
model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=model)

# Folder containing your chapter text files
book_chunks_path = 'book_chunks'
key_phrases_path = 'key_phrases_enhanced'

# Create output directory if it doesn't exist
if not os.path.exists(key_phrases_path):
    os.makedirs(key_phrases_path)
    print(f"Created folder: {key_phrases_path}")

# Get a sorted list of all .txt files in the folder
chapter_files = sorted([f for f in os.listdir(book_chunks_path) if f.endswith('.txt')])

# Create a custom stopwords list (common words to exclude)
stop_words = list(stopwords.words('english'))
# Add technical stopwords that may not be useful for your domain
custom_stops = [
    'fig', 'figure', 'et', 'al', 'e.g', 'i.e', 'etc', 'colab',
    'using', 'use', 'used', 'uses', 'like', 'make', 'makes',
    'would', 'could', 'should', 'may', 'might', 'can',
    'one', 'two', 'three', 'first', 'second', 'third',
    'example', 'examples', 'see', 'say', 'says', 'said',
    'also', 'usually', 'often', 'sometimes', 'generally',
    'note', 'notice', 'shown', 'show', 'shows', 'showed',
    'chapter', 'section', 'page', 'pages', 'part'
]
stop_words.extend(custom_stops)

# Iterate over each file and extract keywords
for idx, file_name in enumerate(chapter_files, start=1):
    file_path = os.path.join(book_chunks_path, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        chapter_text = f.read()
    
    # Clean text - remove excessive whitespace, special characters, etc.
    chapter_text = re.sub(r'\s+', ' ', chapter_text)  # Replace multiple spaces with single space
    chapter_text = re.sub(r'[^\w\s\-\.]', ' ', chapter_text)  # Keep alphanumeric, spaces, hyphens, periods
    
    # Extract keywords with more varied settings
    
    # Method 1: MMR to maximize diversity
    keywords_mmr = kw_model.extract_keywords(
        chapter_text,
        keyphrase_ngram_range=(1, 3),  # Allow 1-3 word phrases
        stop_words=stop_words,
        use_mmr=True,
        diversity=0.7,
        top_n=100
    )
    
    # Method 2: Higher relevance without MMR
    keywords_max = kw_model.extract_keywords(
        chapter_text,
        keyphrase_ngram_range=(1, 3), 
        stop_words=stop_words,
        use_mmr=False,
        top_n=100
    )
    
    # Method 3: Use maximal sum similarity for more natural phrases
    try:
        # Need more candidates than keywords to return
        keywords_mss = kw_model.extract_keywords(
            chapter_text,
            keyphrase_ngram_range=(1, 3),
            stop_words=stop_words,
            use_maxsum=True,
            nr_candidates=150,  # Increased from 30 to exceed top_n
            top_n=100
        )
    except Exception as e:
        print(f"Warning: Maximal sum similarity failed: {e}")
        keywords_mss = []  # Fallback to empty list if it fails
    
    # Combine all keywords, remove duplicates, and sort by score
    all_keywords = {}
    
    # Add keywords from all methods
    for phrase, score in keywords_mmr:
        all_keywords[phrase] = max(score, all_keywords.get(phrase, 0))
    
    for phrase, score in keywords_max:
        all_keywords[phrase] = max(score, all_keywords.get(phrase, 0))
    
    for phrase, score in keywords_mss:
        all_keywords[phrase] = max(score, all_keywords.get(phrase, 0))
    
    # Filter out phrases with score below threshold
    min_score = 0.1  # Adjust this threshold as needed
    filtered_keywords = {k: v for k, v in all_keywords.items() if v >= min_score}
    
    # Sort by score in descending order
    sorted_keywords = sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top N unique keyphrases
    top_n = 200  # Increase this for more keyphrases
    top_keywords = sorted_keywords[:top_n]
    
    print(Fore.GREEN + f"--- Chapter {idx} Keywords (Enhanced) ---")
    print(f"Found {len(top_keywords)} keyphrases with score >= {min_score}")
    
    # Write the keyphrases to an output file
    output_file = os.path.join(key_phrases_path, f"chapter_{idx}_keyphrases.txt")
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for phrase, score in top_keywords:
            out_f.write(f"{phrase}: {score}\n")
    
    print(Fore.RED + f"Enhanced keyphrases written to {output_file}\n")

print(Fore.GREEN + "Enhanced keyphrase extraction completed!")