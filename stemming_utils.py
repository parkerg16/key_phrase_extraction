import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import logging
import os
from typing import List, Sequence

# Configure NLTK to use a directory within the project for downloads
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

# Download required NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
            except Exception as e:
                logging.warning(f"Failed to download {resource}: {e}")

# Download resources at module import
download_nltk_resources()

logger = logging.getLogger(__name__)

class StemmingUtils:
    _stemmer = PorterStemmer()

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        return [
            tok
            for tok in re.split(r'[^a-zA-Z0-9]', text.lower())
            if tok
        ]

    @classmethod
    def stem_phrase(cls, phrase: str) -> str:
        try:
            words = word_tokenize(phrase.lower())
        except LookupError as e:
            logger.warning("NLTK punkt missing; falling back: %s", e)
            # Try to download the resource again
            try:
                nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
                words = word_tokenize(phrase.lower())
            except Exception:
                # If download fails, use the fallback tokenizer
                words = cls.simple_tokenize(phrase)
        stems = [cls._stemmer.stem(w) for w in words]
        return ' '.join(stems)

    @classmethod
    def stem_phrases(cls, phrases: Sequence[str]) -> List[str]:
        return [cls.stem_phrase(p) for p in phrases]

    @classmethod
    def stem_file(cls, input_path: str, output_path: str) -> None:
        with open(input_path, 'r', encoding='utf-8') as fin:
            lines = [ln.strip() for ln in fin if ln.strip()]
        stemmed = cls.stem_phrases(lines)
        with open(output_path, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(stemmed))