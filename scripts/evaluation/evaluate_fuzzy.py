import os
import json
import argparse
from collections import defaultdict
from typing import Dict, Set, List, Tuple
from fuzzywuzzy import fuzz

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


def compute_fuzzy_f1(reference: Set[str], extracted: Set[str], threshold: int = 80) -> tuple:
    """
    Calculate precision, recall, F1 using fuzzy string matching.
    
    Args:
        reference: Set of reference keyphrases
        extracted: Set of extracted keyphrases
        threshold: Similarity threshold (0-100) for considering a match
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if not reference or not extracted:
        return 0, 0, 0
    
    # For each reference phrase, find its best match in extracted phrases
    ref_matches = []
    for ref_phrase in reference:
        best_score = 0
        best_match = None
        for ext_phrase in extracted:
            # Use token_sort_ratio to handle word order differences
            score = fuzz.token_sort_ratio(ref_phrase, ext_phrase)
            if score > best_score:
                best_score = score
                best_match = ext_phrase
        
        if best_score >= threshold:
            ref_matches.append((ref_phrase, best_match, best_score))
    
    # For each extracted phrase, find its best match in reference phrases
    ext_matches = []
    for ext_phrase in extracted:
        best_score = 0
        best_match = None
        for ref_phrase in reference:
            score = fuzz.token_sort_ratio(ext_phrase, ref_phrase)
            if score > best_score:
                best_score = score
                best_match = ref_phrase
        
        if best_score >= threshold:
            ext_matches.append((ext_phrase, best_match, best_score))
    
    # Calculate metrics
    tp_ref = len(ref_matches)  # True positives from reference perspective
    tp_ext = len(ext_matches)  # True positives from extraction perspective
    
    # Average the two perspectives for a more balanced metric
    tp = (tp_ref + tp_ext) / 2
    
    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(reference) if reference else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return precision, recall, f1


def evaluate_model_fuzzy(model_name: str, model_path: str, ref_path: str, threshold: int = 80) -> Dict:
    print(f"\nEvaluating: {model_name} (Fuzzy Matching, Threshold: {threshold})")
    extracted = load_keyphrases(model_path, "_keyphrases.txt")
    reference = load_keyphrases(ref_path, "_reference.txt")

    common_chapters = sorted(set(reference.keys()) & set(extracted.keys()))
    if not common_chapters:
        print("No matching chapters found to evaluate.")
        return {"name": model_name, "chapters": {}, "avg_f1": 0, "avg_precision": 0, "avg_recall": 0, "threshold": threshold}

    # Store results for each chapter
    chapter_results = {}
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    
    for ch in common_chapters:
        precision, recall, f1 = compute_fuzzy_f1(reference[ch], extracted[ch], threshold)
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
        "name": f"{model_name} (Fuzzy {threshold}%)",
        "chapters": chapter_results,
        "avg_f1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "threshold": threshold
    }


def evaluate_with_fuzzy_matching(threshold: int = 80):
    print(f"\n===== Evaluating Keyphrase Results with Fuzzy Matching (Threshold: {threshold}%) =====")

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
        # Skip models that don't exist yet
        if not os.path.exists(model_path):
            print(f"Skipping {name} - directory not found")
            continue
            
        model_results = evaluate_model_fuzzy(name, model_path, ref_path, threshold)
        all_results.append(model_results)
    
    # Save results to a JSON file
    results_file = os.path.join(RESULTS_DIR, f'fuzzy_evaluation_results_{threshold}.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nFuzzy evaluation results saved to {results_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate keyphrases with fuzzy matching")
    parser.add_argument("--threshold", type=int, default=80, 
                        help="Similarity threshold (0-100) for fuzzy matching")
    args = parser.parse_args()
    
    evaluate_with_fuzzy_matching(args.threshold)


if __name__ == "__main__":
    main()