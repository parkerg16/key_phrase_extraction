import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from collections import defaultdict
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"
CHARTS_DIR = BASE_DIR / "data" / "charts"

# Create charts directory if it doesn't exist
os.makedirs(CHARTS_DIR, exist_ok=True)

def load_fuzzy_results(threshold=None):
    """Load fuzzy evaluation results from one or all threshold files"""
    results = []
    
    if threshold:
        # Load a specific threshold file
        results_file = RESULTS_DIR / f"fuzzy_evaluation_results_{threshold}.json"
        if not os.path.exists(results_file):
            print(f"Error: Results file not found at {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            results.append({"threshold": threshold, "data": json.load(f)})
    else:
        # Load all threshold files
        for t in [70, 80, 90]:
            results_file = RESULTS_DIR / f"fuzzy_evaluation_results_{t}.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results.append({"threshold": t, "data": json.load(f)})
    
    return results

def create_threshold_comparison_chart(all_results):
    """Create a bar chart comparing F1 scores across different thresholds"""
    plt.figure(figsize=(14, 8))
    
    # Group results by model and threshold
    thresholds = []
    model_f1_scores = defaultdict(dict)
    
    for result_set in all_results:
        threshold = result_set["threshold"]
        thresholds.append(threshold)
        
        for model_result in result_set["data"]:
            model_name = model_result["name"].split(" (Fuzzy")[0]  # Remove the fuzzy suffix
            model_f1_scores[model_name][threshold] = model_result["avg_f1"]
    
    # Sort thresholds and use fixed model order for consistency
    thresholds = sorted(thresholds)
    models = ["KeyBERT", "Ollama", "TF-IDF+Ollama"]
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.25  # Width of each bar
    
    # Create a grouped bar chart
    for i, threshold in enumerate(thresholds):
        f1_scores = [model_f1_scores[model].get(threshold, 0) for model in models]
        offset = (i - 1) * width  # Center the bars
        bars = plt.bar(x + offset, f1_scores, width, label=f'Threshold {threshold}%')
        
        # Add value labels on top of bars
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.01:  # Only show non-zero values
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add labels, title and legend
    plt.xlabel('Model')
    plt.ylabel('Average F1 Score')
    plt.title('F1 Scores by Model with Different Fuzzy Matching Thresholds')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    
    # Set y-axis limit with extra padding for labels
    max_score = max([model_f1_scores[model].get(t, 0) for model in models for t in thresholds])
    plt.ylim(0, max_score * 1.25)  # Add more headroom for labels
    
    # Use subplot adjustments instead of tight_layout to prevent warnings
    plt.gcf().subplots_adjust(right=0.95, left=0.1, top=0.95, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / "fuzzy_threshold_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Fuzzy threshold comparison chart saved to {output_path}")
    plt.close()

def create_model_comparison_chart(results_data, threshold):
    """Create a bar chart comparing models for a specific threshold"""
    plt.figure(figsize=(12, 6))
    
    # Extract model names and F1 scores
    model_names = [result["name"].split(" (Fuzzy")[0] for result in results_data]  # Remove the fuzzy suffix
    f1_scores = [result["avg_f1"] for result in results_data]
    
    # Separate regular and stemmed models
    regular_models = []
    regular_f1 = []
    stemmed_models = []
    stemmed_f1 = []
    
    for name, f1 in zip(model_names, f1_scores):
        if "(Stemmed)" in name:
            # Remove the "(Stemmed)" part for cleaner labels
            stemmed_models.append(name.replace(" (Stemmed)", ""))
            stemmed_f1.append(f1)
        else:
            regular_models.append(name)
            regular_f1.append(f1)
    
    # Make sure the lists are the same length by adding zeros for missing models
    max_len = max(len(regular_models), len(stemmed_models))
    regular_models += [''] * (max_len - len(regular_models))
    regular_f1 += [0] * (max_len - len(regular_f1))
    stemmed_models += [''] * (max_len - len(stemmed_models))
    stemmed_f1 += [0] * (max_len - len(stemmed_f1))
    
    # Create plot
    x = np.arange(max_len)
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, regular_f1, width, label='Original')
    plt.bar(x + width/2, stemmed_f1, width, label='Stemmed')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Average F1 Score')
    plt.title(f'F1 Scores by Model with Fuzzy Matching (Threshold: {threshold}%)')
    
    # Use only non-empty model names for tick labels
    tick_labels = [r if r else s for r, s in zip(regular_models, stemmed_models)]
    plt.xticks(x, tick_labels)
    
    # Add value labels
    for i, v in enumerate(regular_f1):
        if v > 0.01:
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    for i, v in enumerate(stemmed_f1):
        if v > 0.01:
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.legend()
    
    # Set y-axis limit with extra padding for labels
    max_value = max(max(regular_f1), max(stemmed_f1))
    plt.ylim(0, max_value * 1.25)  # Add extra headroom for labels
    
    # Use subplot adjustments instead of tight_layout to prevent warnings
    plt.gcf().subplots_adjust(right=0.9, left=0.1, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / f"fuzzy_models_threshold_{threshold}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Fuzzy model comparison chart (threshold {threshold}%) saved to {output_path}")
    plt.close()

def create_chapter_performance_chart(results_data, threshold):
    """Create a line chart showing F1 scores by chapter for each model with fuzzy matching"""
    plt.figure(figsize=(12, 7))
    
    # Line styles and colors
    colors = ['blue', 'green', 'orange']
    line_styles = ['-', '--']  # solid for original, dashed for stemmed
    
    # Plot F1 scores for each model across chapters
    for i, result in enumerate(results_data):
        model_name = result["name"].split(" (Fuzzy")[0]  # Remove the fuzzy suffix
        
        # Skip TF-IDF and DeepSeek (but keep TF-IDF+Ollama)
        if model_name == "TF-IDF" or model_name == "TF-IDF (Stemmed)" or "DeepSeek" in model_name:
            continue
            
        is_stemmed = "(Stemmed)" in model_name
        base_name = model_name.replace(" (Stemmed)", "")
        
        # Determine color based on base model name with explicit matching
        if base_name == "KeyBERT":
            color_idx = 0
        elif base_name == "Ollama":
            color_idx = 1
        elif "TF-IDF+Ollama" in base_name:
            color_idx = 2
        else:
            color_idx = 0  # Default fallback
        
        # Get chapter data
        chapters = sorted([int(ch) for ch in result["chapters"].keys()])
        if not chapters:
            continue
            
        f1_scores = [result["chapters"][str(ch)]["f1"] for ch in chapters]
        
        # Check if we have non-zero scores
        if max(f1_scores) > 0:
            # Plot line
            plt.plot(chapters, f1_scores, 
                    color=colors[color_idx], 
                    linestyle=line_styles[1] if is_stemmed else line_styles[0],
                    marker='o' if not is_stemmed else 's',
                    linewidth=2,
                    label=model_name)
    
    # Add labels and title
    plt.xlabel('Chapter Number')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score by Chapter with Fuzzy Matching (Threshold: {threshold}%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set chapter numbers as x-ticks
    all_chapters = set()
    for result in results_data:
        all_chapters.update([int(ch) for ch in result["chapters"].keys()])
    plt.xticks(sorted(all_chapters))
    
    # Add legend in a better position to prevent layout issues
    plt.legend(loc='upper right')
    # Use subplot adjustments instead of tight_layout to prevent warnings
    plt.gcf().subplots_adjust(right=0.9, left=0.1, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / f"fuzzy_chapters_threshold_{threshold}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Fuzzy chapter performance chart (threshold {threshold}%) saved to {output_path}")
    plt.close()

def create_precision_recall_chart(results_data, threshold):
    """Create a scatter plot of precision vs. recall for each model with fuzzy matching"""
    plt.figure(figsize=(8, 8))
    
    # Extract data
    model_names = [result["name"].split(" (Fuzzy")[0] for result in results_data]  # Remove the fuzzy suffix
    precisions = [result["avg_precision"] for result in results_data]
    recalls = [result["avg_recall"] for result in results_data]
    
    # Check if all values are zero or very close to zero
    all_near_zero = all(p < 0.01 for p in precisions) and all(r < 0.01 for r in recalls)
    
    if all_near_zero:
        # Create a message about zero values
        plt.text(0.5, 0.5, 
                f"All precision and recall values are near zero with threshold {threshold}%.\nTry a lower threshold for better results.", 
                ha='center', va='center', fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":10})
        
        # Add labels and title
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision vs. Recall by Model (Fuzzy Matching, Threshold: {threshold}%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits with some padding
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
    else:
        # Create markers for regular vs stemmed models
        markers = ['o', 's']  # circle for regular, square for stemmed
        colors = ['blue', 'green', 'orange']  # one color per base model type
        
        # Plot each point
        for i, (name, precision, recall) in enumerate(zip(model_names, precisions, recalls)):
            # Skip TF-IDF (but keep TF-IDF+Ollama)
            if name == "TF-IDF" or name == "TF-IDF (Stemmed)" or "DeepSeek" in name:
                continue
                
            is_stemmed = "(Stemmed)" in name
            base_name = name.replace(" (Stemmed)", "")
            
            # Determine color based on base model name with explicit matching
            if base_name == "KeyBERT":
                color_idx = 0
            elif base_name == "Ollama":
                color_idx = 1
            elif "TF-IDF+Ollama" in base_name:
                color_idx = 2
            else:
                color_idx = 0  # Default fallback
            
            # Plot the point
            plt.scatter(recall, precision, 
                       marker=markers[1] if is_stemmed else markers[0],
                       color=colors[color_idx],
                       s=100,
                       label=name)
            
            # Add label next to point
            plt.annotate(name, (recall, precision), 
                        textcoords="offset points", 
                        xytext=(5, 5), 
                        ha='left')
        
        # Add diagonal F1 score contours
        f1_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for f1 in f1_values:
            x = np.linspace(0.01, 1, 100)
            # Handle divide by zero and negative values
            with np.errstate(divide='ignore', invalid='ignore'):
                y = (f1 * x) / (2 * x - f1)
            # Filter valid values (positive and within plot range)
            valid_idx = np.isfinite(y) & (y >= 0) & (y <= 1)
            if np.any(valid_idx):  # Only plot if we have valid points
                plt.plot(x[valid_idx], y[valid_idx], 'k--', alpha=0.3)
                
                # Label the line
                mid_idx = len(x[valid_idx]) // 2
                if mid_idx > 0:
                    plt.text(x[valid_idx][mid_idx], y[valid_idx][mid_idx], f'F1={f1}', 
                            fontsize=8, ha='center', va='bottom')
        
        # Add labels and title
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision vs. Recall (Fuzzy Matching, Threshold: {threshold}%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits with more padding at the top
        plt.xlim(0, max(recalls) * 1.1 + 0.05)
        plt.ylim(0, max(precisions) * 1.2 + 0.1)  # Extra padding at the top for value labels
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Model types (colors)
        for i, model_type in enumerate(["KeyBERT", "Ollama", "TF-IDF+Ollama"]):
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=colors[i], markersize=10, label=model_type))
        
        # Processing types (markers)
        legend_elements.append(Line2D([0], [0], marker=markers[0], color='gray', markersize=10, label='Original'))
        legend_elements.append(Line2D([0], [0], marker=markers[1], color='gray', markersize=10, label='Stemmed'))
        
        plt.legend(handles=legend_elements, loc='lower right')
    
    # Use subplot adjustments instead of tight_layout to prevent warnings
    plt.gcf().subplots_adjust(right=0.9, left=0.1, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / f"fuzzy_precision_recall_{threshold}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Fuzzy precision-recall chart (threshold {threshold}%) saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize fuzzy evaluation results")
    parser.add_argument("--threshold", type=int, 
                        help="Specific threshold to visualize. If not provided, will visualize all thresholds.")
    args = parser.parse_args()
    
    if args.threshold:
        # Load and visualize a specific threshold
        results = load_fuzzy_results(args.threshold)
        if not results:
            return
            
        create_model_comparison_chart(results[0]["data"], results[0]["threshold"])
        create_precision_recall_chart(results[0]["data"], results[0]["threshold"])
        create_chapter_performance_chart(results[0]["data"], results[0]["threshold"])
    else:
        # Load and visualize all thresholds
        results = load_fuzzy_results()
        if not results:
            print("No fuzzy evaluation results found. Please run the fuzzy evaluation first.")
            return
            
        # Create the comparison chart for all thresholds
        create_threshold_comparison_chart(results)
        
        # Create individual charts for each threshold
        for result_set in results:
            create_model_comparison_chart(result_set["data"], result_set["threshold"])
            create_precision_recall_chart(result_set["data"], result_set["threshold"])
            create_chapter_performance_chart(result_set["data"], result_set["threshold"])
    
    print(f"All fuzzy evaluation charts saved to {CHARTS_DIR}")

if __name__ == "__main__":
    main()