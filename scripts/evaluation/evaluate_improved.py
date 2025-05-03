import os
import json
import argparse
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


def calculate_f1_scores(reference, extracted, method_name):
    """Calculate F1 scores for each chapter and overall using improved matching"""
    all_ref_phrases = set()
    all_ext_phrases = set()
    all_true_positives = 0
    
    # Results for each chapter
    print(f"\n{method_name} - F1 Scores by Chapter:")
    print("-" * 40)
    
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    chapter_count = 0
    
    # For fuzzy matching
    def normalize_phrase(phrase):
        """Normalize a phrase for fuzzy comparison"""
        return phrase.lower().strip()
    
    def find_best_match(phrase, reference_set):
        """Find if a phrase exists in the reference set (exact or close match)"""
        norm_phrase = normalize_phrase(phrase)
        
        # First try exact match
        for ref in reference_set:
            if normalize_phrase(ref) == norm_phrase:
                return True
                
        # Then try contained match (is the entire phrase contained in any reference)
        for ref in reference_set:
            if norm_phrase in normalize_phrase(ref) or normalize_phrase(ref) in norm_phrase:
                return True
                
        return False
    
    chapter_results = []
    skipped_chapters = []
    
    # Find all available chapters in both reference and extracted sets
    # Use numerical sorting for chapter numbers
    common_chapters = sorted(set(reference.keys()).intersection(set(extracted.keys())))
    missing_chapters = sorted(set(reference.keys()) - set(extracted.keys()))
    
    if missing_chapters:
        print(f"Note: Skipping chapters not found in extraction results: {', '.join(map(str, missing_chapters))}")
        skipped_chapters = missing_chapters
    
    # Process only chapters that exist in both sets
    for chapter in common_chapters:
        # Skip empty extracted chapters
        if not extracted[chapter]:
            print(f"Chapter {chapter}: No extracted keyphrases")
            chapter_results.append((chapter, 0, 0, 0))
            continue
            
        ref_phrases = reference[chapter]
        ext_phrases = extracted[chapter]
        
        # For overall calculations
        all_ref_phrases.update(ref_phrases)
        all_ext_phrases.update(ext_phrases)
        
        # Calculate true positives using both exact and fuzzy matching
        true_positives = 0
        for phrase in ext_phrases:
            if find_best_match(phrase, ref_phrases):
                true_positives += 1
                all_true_positives += 1
        
        # Calculate metrics
        precision = true_positives / len(ext_phrases) if ext_phrases else 0
        recall = true_positives / len(ref_phrases) if ref_phrases else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        chapter_results.append((chapter, f1, precision, recall))
        
        print(f"Chapter {chapter}: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
        
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        chapter_count += 1
    
    # Calculate average metrics
    avg_f1 = total_f1 / chapter_count if chapter_count > 0 else 0
    avg_precision = total_precision / chapter_count if chapter_count > 0 else 0
    avg_recall = total_recall / chapter_count if chapter_count > 0 else 0
    
    # Calculate overall metrics
    overall_precision = all_true_positives / len(all_ext_phrases) if all_ext_phrases else 0
    overall_recall = all_true_positives / len(all_ref_phrases) if all_ref_phrases else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("-" * 40)
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f} (Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f})")
    
    # Determine best and worst chapters
    if chapter_results:
        sorted_by_f1 = sorted(chapter_results, key=lambda x: x[1])
        worst_chapter = sorted_by_f1[0]
        best_chapter = sorted_by_f1[-1]
        
        print(f"\nBest performing chapter: Chapter {best_chapter[0]} (F1: {best_chapter[1]:.4f})")
        print(f"Worst performing chapter: Chapter {worst_chapter[0]} (F1: {worst_chapter[1]:.4f})")
    
    # Prepare result data for storing
    result_data = {
        "name": method_name,
        "chapters": {str(ch): {"f1": f1, "precision": p, "recall": r} for ch, f1, p, r in chapter_results},
        "avg_f1": avg_f1,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "overall_f1": overall_f1,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "best_chapter": int(best_chapter[0]) if chapter_results else None,
        "worst_chapter": int(worst_chapter[0]) if chapter_results else None
    }
    
    return result_data


def evaluate_model_improved(model_name: str, model_path: str, ref_path: str) -> Dict:
    """Evaluate a model using improved F1 calculation method"""
    print(f"\nEvaluating: {model_name} (Improved Method)")
    extracted = load_keyphrases(model_path, "_keyphrases.txt")
    reference = load_keyphrases(ref_path, "_reference.txt")

    if not extracted:
        print(f"No extracted keyphrases found for {model_name}")
        return {
            "name": model_name,
            "chapters": {},
            "avg_f1": 0,
            "avg_precision": 0,
            "avg_recall": 0,
            "overall_f1": 0,
            "overall_precision": 0,
            "overall_recall": 0
        }

    result_data = calculate_f1_scores(reference, extracted, model_name)
    return result_data


def evaluate_with_improved_method():
    """Run evaluation with improved method on all models"""
    print("\n===== Evaluating Keyphrase Results with Improved Method =====")

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
            
        model_results = evaluate_model_improved(name, model_path, ref_path)
        all_results.append(model_results)
    
    # Save results to a JSON file
    results_file = os.path.join(RESULTS_DIR, 'improved_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nImproved evaluation results saved to {results_file}")
    
    return all_results


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Evaluate keyphrases with improved method")
    parser.add_argument("--model", type=str, 
                        help="Specific model to evaluate (keybert, ollama, deepseek). If not provided, all will be evaluated.")
    parser.add_argument("--stemmed", action="store_true", 
                        help="Evaluate stemmed versions instead of regular versions")
    args = parser.parse_args()
    
    if args.model:
        # Evaluate a single model
        model_dir = args.model.lower()
        model_name = args.model.capitalize()
        
        if args.stemmed:
            model_path = os.path.join(STEMMED_DIR, model_dir)
            ref_path = STEMMED_REFERENCE_DIR
            model_name += " (Stemmed)"
        else:
            model_path = os.path.join(KEYPHRASES_DIR, model_dir)
            ref_path = REFERENCE_DIR
        
        # Check if the model exists
        if not os.path.exists(model_path):
            print(f"Error: Model directory not found: {model_path}")
            return
        
        # Evaluate the model
        result = evaluate_model_improved(model_name, model_path, ref_path)
        
        # Save the result
        results_file = os.path.join(RESULTS_DIR, f'improved_{model_dir}_{"stemmed" if args.stemmed else "regular"}_results.json')
        with open(results_file, 'w') as f:
            json.dump([result], f, indent=4)
        
        print(f"Results saved to {results_file}")
    else:
        # Evaluate all models
        evaluate_with_improved_method()


if __name__ == "__main__":
    main()