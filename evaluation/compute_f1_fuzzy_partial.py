import os
from fuzzywuzzy import fuzz

def compute_f1_fuzzy_partial(pred_file, gt_file, threshold=60):
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_lines = f.readlines()
        pred_phrases = [line.split(':')[0].strip().lower() for line in pred_lines]
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_phrases = [line.strip().lower() for line in f.readlines()]

    true_positive = 0
    matched_gt = set()

    for pred in pred_phrases:
        for gt in gt_phrases:
            if gt not in matched_gt:
                similarity = fuzz.partial_ratio(pred, gt)
                if similarity >= threshold:
                    true_positive += 1
                    matched_gt.add(gt)
                    break

    false_positive = len(pred_phrases) - true_positive
    false_negative = len(gt_phrases) - true_positive

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

if __name__ == "__main__":
    key_phrases_path = 'key_phrases'
    ground_truth_path = 'ground_truth'
    chapter_files = sorted([f for f in os.listdir(key_phrases_path) if f.endswith('_keyphrases.txt')])
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_chapters = 0
    for file in chapter_files:
        # Assuming file name format is 'chapter_X_keyphrases.txt'
        chapter_num = file.split('_')[1]
        pred_file = os.path.join(key_phrases_path, file)
        gt_file = os.path.join(ground_truth_path, f'chapter_{chapter_num}_ground_truth.txt')
        if os.path.exists(gt_file):
            precision, recall, f1 = compute_f1_fuzzy_partial(pred_file, gt_file, threshold=60)
            print(f"Chapter {chapter_num}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_chapters += 1
        else:
            print(f"Ground truth file not found for chapter {chapter_num}")
    if num_chapters > 0:
        avg_precision = total_precision / num_chapters
        avg_recall = total_recall / num_chapters
        avg_f1 = total_f1 / num_chapters
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
    else:
        print("No chapters with ground truth found.")