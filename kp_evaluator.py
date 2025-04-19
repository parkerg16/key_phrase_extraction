import os
import re
import string
from collections import defaultdict
from nltk.stem import PorterStemmer
from sklearn.metrics import precision_recall_fscore_support

class KPEvaluator:
    """
    Advanced keyphrase evaluator using techniques from KPEval and similar academic tools.
    This evaluator uses more sophisticated matching techniques beyond exact matching.
    """
    
    def __init__(self, stemming=True):
        """Initialize the evaluator with options"""
        self.stemmer = PorterStemmer() if stemming else None
        self.stemming = stemming
    
    def normalize_phrase(self, phrase):
        """
        Normalize a phrase by:
        - Converting to lowercase
        - Removing punctuation
        - Stemming if enabled
        """
        # Convert to lowercase
        phrase = phrase.lower()
        
        # Remove punctuation
        phrase = re.sub(f'[{re.escape(string.punctuation)}]', ' ', phrase)
        
        # Remove extra whitespace
        phrase = ' '.join(phrase.split())
        
        # Stem if enabled
        if self.stemming and self.stemmer:
            words = phrase.split()
            words = [self.stemmer.stem(word) for word in words]
            phrase = ' '.join(words)
            
        return phrase
    
    def _are_phrases_matching(self, gold_phrase, pred_phrase, match_type="exact"):
        """
        Check if two phrases match according to the specified match type.
        
        Args:
            gold_phrase (str): Gold standard phrase
            pred_phrase (str): Predicted phrase
            match_type (str): Type of matching ('exact', 'partial', 'contains')
            
        Returns:
            bool: True if phrases match, False otherwise
        """
        # Normalize both phrases
        gold_norm = self.normalize_phrase(gold_phrase)
        pred_norm = self.normalize_phrase(pred_phrase)
        
        if match_type == "exact":
            # Exact matching (after normalization)
            return gold_norm == pred_norm
        
        elif match_type == "partial":
            # Partial matching (one is substring of the other)
            return (gold_norm in pred_norm) or (pred_norm in gold_norm)
        
        elif match_type == "contains":
            # Check if the terms in gold phrase are in predicted phrase
            gold_terms = set(gold_norm.split())
            pred_terms = set(pred_norm.split())
            return len(gold_terms.intersection(pred_terms)) > 0
            
        else:
            # Default to exact matching
            return gold_norm == pred_norm
    
    def evaluate_chapter(self, gold_keyphrases, pred_keyphrases, match_type="exact"):
        """
        Evaluate keyphrases for a single chapter.
        
        Args:
            gold_keyphrases (list): List of gold standard keyphrases
            pred_keyphrases (list): List of predicted keyphrases
            match_type (str): Type of matching ('exact', 'partial', 'contains')
        
        Returns:
            dict: Dictionary with precision, recall, f1 scores
        """
        # Handle empty cases
        if not gold_keyphrases or not pred_keyphrases:
            return {
                'precision': 0.0, 
                'recall': 0.0, 
                'f1': 0.0,
                'true_positives': 0,
                'false_positives': len(pred_keyphrases) if pred_keyphrases else 0,
                'false_negatives': len(gold_keyphrases) if gold_keyphrases else 0
            }
        
        # Find matches (true positives)
        true_positives = 0
        matched_gold = set()
        matched_pred = set()
        
        # For each predicted keyphrase, check if it matches any gold keyphrase
        for i, pred in enumerate(pred_keyphrases):
            for j, gold in enumerate(gold_keyphrases):
                if j in matched_gold:
                    continue  # Skip already matched gold keyphrases
                    
                if self._are_phrases_matching(gold, pred, match_type):
                    true_positives += 1
                    matched_gold.add(j)
                    matched_pred.add(i)
                    break
        
        # Calculate precision, recall, F1
        precision = true_positives / len(pred_keyphrases) if pred_keyphrases else 0
        recall = true_positives / len(gold_keyphrases) if gold_keyphrases else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': len(pred_keyphrases) - true_positives,
            'false_negatives': len(gold_keyphrases) - true_positives
        }
    
    def evaluate_all_chapters(self, gold_keyphrases_by_chapter, pred_keyphrases_by_chapter, match_types=None):
        """
        Evaluate keyphrases for all chapters using multiple matching types.
        
        Args:
            gold_keyphrases_by_chapter (dict): Dictionary mapping chapter numbers to gold keyphrases
            pred_keyphrases_by_chapter (dict): Dictionary mapping chapter numbers to predicted keyphrases
            match_types (list): List of match types to use (default: ['exact', 'partial', 'contains'])
            
        Returns:
            dict: Dictionary with results for each match type and overall
        """
        if match_types is None:
            match_types = ['exact', 'partial', 'contains']
        
        # Get chapters present in both gold and predictions
        common_chapters = sorted(set(gold_keyphrases_by_chapter.keys()).intersection(
            set(pred_keyphrases_by_chapter.keys())))
        
        # Initialize results
        results = {}
        for match_type in match_types:
            results[match_type] = {
                'chapters': {},
                'macro_avg': {'precision': 0, 'recall': 0, 'f1': 0},
                'micro_avg': {'precision': 0, 'recall': 0, 'f1': 0}
            }
            
            # Accumulators for micro-averaging
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            # Evaluate each chapter
            for chapter in common_chapters:
                gold_kps = gold_keyphrases_by_chapter[chapter]
                pred_kps = pred_keyphrases_by_chapter[chapter]
                
                chapter_result = self.evaluate_chapter(gold_kps, pred_kps, match_type)
                results[match_type]['chapters'][chapter] = chapter_result
                
                # Accumulate for macro-averaging
                results[match_type]['macro_avg']['precision'] += chapter_result['precision']
                results[match_type]['macro_avg']['recall'] += chapter_result['recall']
                results[match_type]['macro_avg']['f1'] += chapter_result['f1']
                
                # Accumulate for micro-averaging
                total_tp += chapter_result['true_positives']
                total_fp += chapter_result['false_positives']
                total_fn += chapter_result['false_negatives']
            
            # Calculate macro averages (average of chapter metrics)
            num_chapters = len(common_chapters)
            if num_chapters > 0:
                results[match_type]['macro_avg']['precision'] /= num_chapters
                results[match_type]['macro_avg']['recall'] /= num_chapters
                results[match_type]['macro_avg']['f1'] /= num_chapters
            
            # Calculate micro averages (aggregate across all chapters)
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            
            results[match_type]['micro_avg']['precision'] = micro_precision
            results[match_type]['micro_avg']['recall'] = micro_recall
            results[match_type]['micro_avg']['f1'] = micro_f1
            
            # Best and worst chapters by F1 score
            if common_chapters:
                chapter_f1_scores = [(ch, results[match_type]['chapters'][ch]['f1']) for ch in common_chapters]
                best_chapter = max(chapter_f1_scores, key=lambda x: x[1])
                worst_chapter = min(chapter_f1_scores, key=lambda x: x[1])
                
                results[match_type]['best_chapter'] = {
                    'chapter': best_chapter[0],
                    'f1': best_chapter[1]
                }
                
                results[match_type]['worst_chapter'] = {
                    'chapter': worst_chapter[0],
                    'f1': worst_chapter[1]
                }
        
        return results
    
    @staticmethod
    def load_keyphrases_from_dir(directory):
        """
        Load keyphrases from files in a directory.
        
        Args:
            directory (str): Path to directory containing keyphrase files
            
        Returns:
            dict: Dictionary mapping chapter numbers to lists of keyphrases
        """
        keyphrases = defaultdict(list)
        
        if not os.path.exists(directory):
            return keyphrases
            
        for filename in os.listdir(directory):
            if filename.startswith("chapter_") and filename.endswith(".txt"):
                try:
                    # Extract chapter number
                    chapter_num = int(filename.split('_')[1])
                    
                    # Read keyphrases
                    filepath = os.path.join(directory, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        phrases = [line.strip() for line in f if line.strip()]
                        keyphrases[chapter_num] = phrases
                except (ValueError, IndexError):
                    continue
                
        return keyphrases

# Example usage
if __name__ == "__main__":
    # Example usage to test the evaluator
    reference_dir = "reference_keyphrases"
    extracted_dir = "key_phrases_ollama"
    
    evaluator = KPEvaluator(stemming=True)
    
    reference_keyphrases = evaluator.load_keyphrases_from_dir(reference_dir)
    extracted_keyphrases = evaluator.load_keyphrases_from_dir(extracted_dir)
    
    results = evaluator.evaluate_all_chapters(
        reference_keyphrases, 
        extracted_keyphrases, 
        match_types=['exact', 'partial', 'contains']
    )
    
    # Print summary
    print("\nKPEval Summary:")
    for match_type, result in results.items():
        print(f"\n{match_type.upper()} MATCHING:")
        print(f"Macro Avg F1: {result['macro_avg']['f1']:.4f}")
        print(f"Micro Avg F1: {result['micro_avg']['f1']:.4f}")