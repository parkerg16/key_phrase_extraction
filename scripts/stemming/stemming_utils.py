import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import logging
import os
import subprocess
import sys
from typing import List, Sequence, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup NLTK resource directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
nltk_data_dir = os.path.join(project_root, 'nltk_data')
local_nltk_data_dir = os.path.join(script_dir, 'nltk_data')

# Ensure directories exist
os.makedirs(nltk_data_dir, exist_ok=True)
os.makedirs(local_nltk_data_dir, exist_ok=True)

# Add our data directories to NLTK's search path
nltk.data.path.insert(0, nltk_data_dir)
nltk.data.path.insert(0, local_nltk_data_dir)

def check_and_download_nltk_resources():
    """Check if required NLTK resources are available, and download them if not."""
    resources_available = True
    
    # Check for punkt
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        resources_available = False
    
    # Check for punkt_tab
    try:
        nltk.data.find('tokenizers/punkt_tab/english')
    except LookupError:
        resources_available = False
    
    # If resources are missing, run the download script
    if not resources_available:
        logger.warning("NLTK resources missing. Running download script...")
        download_script = os.path.join(script_dir, 'download_nltk_resources.py')
        subprocess.run([sys.executable, download_script], check=True)
        
        # Verify resources after download
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab/english')
            logger.info("NLTK resources successfully downloaded!")
        except LookupError as e:
            logger.error(f"Still missing NLTK resources after download attempt: {e}")
    
    return resources_available

# Try to ensure NLTK resources are available
check_and_download_nltk_resources()

logger = logging.getLogger(__name__)

class StemmingUtils:
    _stemmers = {
        "keybert": PorterStemmer(),
        "ollama": SnowballStemmer("english")  # You can customize this
    }

    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        return [tok for tok in re.split(r'[^a-zA-Z0-9]', text.lower()) if tok]

    @classmethod
    def stem_phrase(cls, phrase: str, model_option: str = "keybert") -> str:
        stemmer = cls._stemmers.get(model_option, PorterStemmer())
        try:
            words = word_tokenize(phrase.lower())
        except LookupError as e:
            logger.warning("NLTK punkt missing; using fallback tokenizer. Error: %s", e)
            try:
                nltk.download('punkt', download_dir=nltk_data_dir, quiet=False)
                words = word_tokenize(phrase.lower())
            except:
                words = cls.simple_tokenize(phrase)
        stems = [stemmer.stem(w) for w in words]
        return ' '.join(stems)

    @classmethod
    def stem_phrases(cls, phrases: Sequence[str], model_option: str = "keybert") -> List[str]:
        return [cls.stem_phrase(p, model_option=model_option) for p in phrases]

    @classmethod
    def stem_file(cls, input_path: str, output_path: str, model_option: str = "keybert") -> None:
        with open(input_path, 'r', encoding='utf-8') as fin:
            lines = [ln.strip() for ln in fin if ln.strip()]
        stemmed = cls.stem_phrases(lines, model_option=model_option)
        with open(output_path, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(stemmed))
