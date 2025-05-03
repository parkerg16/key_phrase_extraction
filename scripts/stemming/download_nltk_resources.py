#!/usr/bin/env python3
"""
Script to download required NLTK resources for the keyphrase extraction project.
This resolves the "Resource punkt_tab not found" error.
"""

import os
import sys
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Set up NLTK data directories
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    nltk_data_dir = os.path.join(project_root, "nltk_data")
    stemming_nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    
    os.makedirs(nltk_data_dir, exist_ok=True)
    os.makedirs(stemming_nltk_data_dir, exist_ok=True)
    
    # Add our data directories to NLTK's search path
    nltk.data.path.insert(0, nltk_data_dir)
    nltk.data.path.insert(0, stemming_nltk_data_dir)
    
    # Resources to download
    resources = [
        'punkt',          # Basic tokenizer
        'punkt_tab'       # Tab-delimited version of punkt (needed for some operations)
    ]
    
    # Download each resource
    for resource in resources:
        logger.info(f"Checking for NLTK resource: {resource}")
        try:
            # First check if it's already available
            if resource == 'punkt_tab':
                try:
                    nltk.data.find(f'tokenizers/punkt_tab/english')
                    logger.info(f"Resource {resource} is already available.")
                    continue
                except LookupError:
                    pass
            else:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                    logger.info(f"Resource {resource} is already available.")
                    continue
                except LookupError:
                    pass
            
            # Download to both directories to ensure availability
            logger.info(f"Downloading {resource} to {nltk_data_dir}...")
            nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
            
            logger.info(f"Downloading {resource} to {stemming_nltk_data_dir}...")
            nltk.download(resource, download_dir=stemming_nltk_data_dir, quiet=False)
            
            logger.info(f"Successfully downloaded {resource}")
            
        except Exception as e:
            logger.error(f"Failed to download {resource}: {e}")
            return False
    
    # Verify the downloads
    verification_failed = False
    for resource in resources:
        try:
            if resource == 'punkt_tab':
                nltk.data.find(f'tokenizers/punkt_tab/english')
            else:
                nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"Verified: {resource} is available")
        except LookupError:
            logger.error(f"Verification failed: {resource} still not available")
            verification_failed = True
    
    if verification_failed:
        logger.error("Some resources couldn't be verified. There might still be issues.")
        return False
    else:
        logger.info("All NLTK resources successfully downloaded and verified!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)