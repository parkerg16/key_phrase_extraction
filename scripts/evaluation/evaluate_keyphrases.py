import os
import json
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
KEYPHRASES_DIR = os.path.join(BASE_DIR, 'data', 'keyphrases')
STEMMED_DIR = os.path.join(KEYPHRASES_DIR, 'stemmed')
REFERENCE_DIR = os.path.join(KEYPHRASES_DIR, 'referenced')
STEMMED_REFERENCE_DIR = os.path.join(STEMMED_DIR, 'referenced')
RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def evaluate_model(model_name: str, model_path: str, ref_path: str, is_stemmed=False) -> Dict:
    print(f"\nEvaluating: {model_name}")
    extracted = load_keyphrases(model_path, "_keyphrases.txt")
    reference = load_keyphrases(ref_path, "_reference.txt")

    common_chapters = sorted(set(reference.keys()) & set(extracted.keys()))
    if not common_chapters:
        print("No matching chapters found to evaluate.")
        return {"name": model_name, "chapters": {}, "avg_f1": 0, "avg_precision": 0, "avg_recall": 0}

    # Store results for each chapter
    chapter_results = {}
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    
    for ch in common_chapters:
        precision, recall, f1 = compute_f1(reference[ch], extracted[ch])
        chapter_results[ch] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        print(f"Chapter {ch}: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")

    # Calculate averages
    avg_f1 = total_f1 / len(common_chapters)
    avg_precision = total_precision / len(common_chapters)
    avg_recall = total_recall / len(common_chapters)
    
    print(f"Average F1 Score for {model_name}: {avg_f1:.4f}")
    
    # Return results as a dictionary
    return {
        "name": model_name,
        "chapters": chapter_results,
        "avg_f1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall
    }


def evaluate_existing_results():
    print("\n===== Evaluating Existing Keyphrase Results =====")

    models = {
        "KeyBERT": (os.path.join(KEYPHRASES_DIR, "keybert"), REFERENCE_DIR),
        "Ollama": (os.path.join(KEYPHRASES_DIR, "ollama"), REFERENCE_DIR),
        "TF-IDF": (os.path.join(KEYPHRASES_DIR, "tfidf"), REFERENCE_DIR),
        "TF-IDF+Ollama": (os.path.join(KEYPHRASES_DIR, "tfidf_ollama"), REFERENCE_DIR),
        "KeyBERT (Stemmed)": (os.path.join(STEMMED_DIR, "keybert"), STEMMED_REFERENCE_DIR),
        "Ollama (Stemmed)": (os.path.join(STEMMED_DIR, "ollama"), STEMMED_REFERENCE_DIR),
        "TF-IDF (Stemmed)": (os.path.join(STEMMED_DIR, "tfidf"), STEMMED_REFERENCE_DIR),
        "TF-IDF+Ollama (Stemmed)": (os.path.join(STEMMED_DIR, "tfidf_ollama"), STEMMED_REFERENCE_DIR),
    }

    # Collect results from all models
    all_results = []
    
    for name, (model_path, ref_path) in models.items():
        model_results = evaluate_model(name, model_path, ref_path)
        all_results.append(model_results)
    
    # Save results to a JSON file
    results_file = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nEvaluation results saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    evaluate_existing_results()
