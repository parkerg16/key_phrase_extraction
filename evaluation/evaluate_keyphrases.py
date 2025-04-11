import os
import argparse
from collections import defaultdict
from fuzzywuzzy import fuzz


def load_index_by_chapter(index_path):
    chapter_keywords = defaultdict(set)
    current_chapter = None

    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("chapter"):
                current_chapter = line.split()[1]
            elif current_chapter:
                chapter_keywords[current_chapter].add(line.lower())
    return chapter_keywords


def load_extracted_keyphrases(test_keyphrase_dir):
    extracted = dict()
    for fname in os.listdir(test_keyphrase_dir):
        if not fname.endswith(".txt"):
            continue
        chapter_num = fname.split("_")[1]
        with open(os.path.join(test_keyphrase_dir, fname), 'r', encoding='utf-8') as f:
            phrases = [line.split(':')[0].strip().lower() for line in f if line.strip()]
        extracted[chapter_num] = phrases
    return extracted


def evaluate_fuzzy(index_terms, extracted_terms, threshold=80):
    matched = 0
    for extracted in extracted_terms:
        for index_term in index_terms:
            score = fuzz.partial_ratio(extracted, index_term)
            if score >= threshold:
                matched += 1
                break
    return matched, len(extracted_terms), len(index_terms)


def run_evaluation(index_path, extracted_dir):
    index_by_chapter = load_index_by_chapter(index_path)
    extracted_by_chapter = load_extracted_keyphrases(extracted_dir)

    results = {}
    for chapter_num in extracted_by_chapter:
        if chapter_num not in index_by_chapter:
            print(f"Warning: Chapter {chapter_num} missing in index.")
            continue
        matched, extracted_total, index_total = evaluate_fuzzy(
            index_by_chapter[chapter_num],
            extracted_by_chapter[chapter_num]
        )
        precision = matched / extracted_total if extracted_total else 0
        recall = matched / index_total if index_total else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        results[chapter_num] = {
            "matched": matched,
            "extracted_total": extracted_total,
            "index_total": index_total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    return results


def print_results(results):
    print("\nEvaluation Results by Chapter:")
    for chapter, metrics in sorted(results.items()):
        print(f"\nChapter {chapter}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate extracted keyphrases against index terms.")
    parser.add_argument('--index_path', type=str, default='data/index_by_chapter.txt')
    parser.add_argument('--extracted_dir', type=str, default='data/keyphrases/test')
    args = parser.parse_args()

    results = run_evaluation(args.index_path, args.extracted_dir)
    print_results(results)
