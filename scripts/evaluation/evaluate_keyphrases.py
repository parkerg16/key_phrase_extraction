import os
from collections import defaultdict
from typing import Dict, Set

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
KEYPHRASES_DIR = os.path.join(BASE_DIR, 'data', 'keyphrases')
STEMMED_DIR = os.path.join(KEYPHRASES_DIR, 'stemmed')
REFERENCE_DIR = os.path.join(KEYPHRASES_DIR, 'referenced')
STEMMED_REFERENCE_DIR = os.path.join(STEMMED_DIR, 'referenced')


def load_keyphrases(folder: str, suffix: str) -> Dict[int, Set[str]]:
    """Load keyphrases or references from a given folder"""
    phrases_by_chapter = defaultdict(set)
    if not os.path.exists(folder):
        print(f"Warning: {folder} not found")
        return phrases_by_chapter

    for filename in os.listdir(folder):
        if filename.endswith(suffix):
            try:
                chapter_num = int(filename.split("_")[1])
                path = os.path.join(folder, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    phrases_by_chapter[chapter_num] = {
                        line.strip().lower() for line in f if line.strip()
                    }
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
    return phrases_by_chapter


def compute_f1(reference: Set[str], extracted: Set[str]) -> tuple:
    """Calculate precision, recall, F1"""
    tp = len(reference & extracted)
    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(reference) if reference else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


def evaluate_model(model_name: str, model_path: str, ref_path: str, is_stemmed=False):
    print(f"\nEvaluating: {model_name}")
    extracted = load_keyphrases(model_path, "_keyphrases.txt")
    reference = load_keyphrases(ref_path, "_reference.txt")

    common_chapters = sorted(set(reference.keys()) & set(extracted.keys()))
    if not common_chapters:
        print("No matching chapters found to evaluate.")
        return

    total_f1 = 0
    for ch in common_chapters:
        precision, recall, f1 = compute_f1(reference[ch], extracted[ch])
        total_f1 += f1
        print(f"Chapter {ch}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

    avg_f1 = total_f1 / len(common_chapters)
    print(f"Average F1 Score for {model_name}: {avg_f1:.4f}")


def evaluate_existing_results():
    print("\n===== Evaluating Existing Keyphrase Results =====")

    models = {
        "KeyBERT": (os.path.join(KEYPHRASES_DIR, "keybert"), REFERENCE_DIR),
        "Ollama": (os.path.join(KEYPHRASES_DIR, "ollama"), REFERENCE_DIR),
        "KeyBERT (Stemmed)": (os.path.join(STEMMED_DIR, "keybert"), STEMMED_REFERENCE_DIR),
        "Ollama (Stemmed)": (os.path.join(STEMMED_DIR, "ollama"), STEMMED_REFERENCE_DIR),
    }

    for name, (model_path, ref_path) in models.items():
        evaluate_model(name, model_path, ref_path)


if __name__ == "__main__":
    evaluate_existing_results()
