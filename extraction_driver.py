import os
import subprocess
import sys

PYTHON = sys.executable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
EXTRACTION_DIR = os.path.join(SCRIPTS_DIR, "extraction")
STEMMING_DIR = os.path.join(SCRIPTS_DIR, "stemming")
EVALUATION_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluation", "evaluate_keyphrases.py")

KEYBERT_SCRIPT = os.path.join(EXTRACTION_DIR, "keyword_extraction.py")
OLLAMA_SCRIPT = os.path.join(EXTRACTION_DIR, "ollama_extraction.py")
STEM_SCRIPT = os.path.join(STEMMING_DIR, "stem_extracted.py")


def run_standard_extraction():
    print("\nRunning Standard (KeyBERT) Extraction...")
    subprocess.run([PYTHON, KEYBERT_SCRIPT], check=True)
    subprocess.run([PYTHON, STEM_SCRIPT, "--model_option", "keybert"], check=True)

def run_ollama_extraction():
    print("\nRunning Ollama Extraction...")
    subprocess.run([PYTHON, OLLAMA_SCRIPT], check=True)
    subprocess.run([PYTHON, STEM_SCRIPT, "--model_option", "ollama"], check=True)

def run_evaluation():
    print("\nRunning Evaluation...")
    reference_dir = os.path.join("data", "keyphrases", "referenced")
    stemmed_reference_dir = os.path.join("data", "keyphrases", "stemmed", "referenced")
    split_ref_script = os.path.join("scripts", "evaluation", "split_reference.py")
    evaluate_script = os.path.join("scripts", "evaluation", "evaluate_keyphrases.py")

    # Check if the reference directory exists and is not empty
    if not os.path.exists(reference_dir) or not os.listdir(reference_dir):
        print("Reference files not found. Generating with split_reference.py...")
        subprocess.run([PYTHON, split_ref_script], check=True)

    subprocess.run([PYTHON, evaluate_script], check=True)

def display_menu():
    print("\n" + "=" * 50)
    print("Keyphrase Extraction Pipeline".center(50))
    print("=" * 50)
    print("1. Run both extractions and evaluate")
    print("2. Run standard extraction only")
    print("3. Run Ollama extraction only")
    print("4. Evaluate existing results only")
    print("5. Exit")

    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice in {"1", "2", "3", "4", "5"}:
            return int(choice)
        print("Invalid input. Please enter a number between 1 and 5.")

def main():
    option = display_menu()
    if option == 1:
        run_standard_extraction()
        run_ollama_extraction()
        run_evaluation()
    elif option == 2:
        run_standard_extraction()
    elif option == 3:
        run_ollama_extraction()
    elif option == 4:
        run_evaluation()
    elif option == 5:
        print("Exiting.")


if __name__ == "__main__":
    main()
