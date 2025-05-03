import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "data" / "results"
CHARTS_DIR = BASE_DIR / "data" / "charts"

# Create charts directory if it doesn't exist
os.makedirs(CHARTS_DIR, exist_ok=True)

def load_results():
    """Load evaluation results from the JSON file"""
    results_file = RESULTS_DIR / "evaluation_results.json"
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please run the evaluation script first to generate results.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_f1_bar_chart(results):
    """Create a bar chart comparing average F1 scores across models"""
    # Extract model names and F1 scores
    model_names = [result["name"] for result in results]
    f1_scores = [result["avg_f1"] for result in results]
    
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
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Fix model ordering to ensure consistent presentation
    ordered_models = []
    ordered_regular_f1 = []
    ordered_stemmed_f1 = []
    
    # Define the expected order of models
    model_order = ["KeyBERT", "Ollama", "TF-IDF+Ollama"]
    
    # Create ordered lists based on the expected model order
    for model in model_order:
        if model in regular_models:
            idx = regular_models.index(model)
            ordered_models.append(model)
            ordered_regular_f1.append(regular_f1[idx])
            
            # Find the corresponding stemmed model
            stemmed_idx = stemmed_models.index(model) if model in stemmed_models else -1
            if stemmed_idx >= 0:
                ordered_stemmed_f1.append(stemmed_f1[stemmed_idx])
            else:
                ordered_stemmed_f1.append(0)  # No stemmed version
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(ordered_models))
    
    # Create grouped bars
    plt.bar(index, ordered_regular_f1, bar_width, label='Original')
    plt.bar(index + bar_width, ordered_stemmed_f1, bar_width, label='Stemmed')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Scores by Model')
    plt.xticks(index + bar_width/2, ordered_models)
    
    # Set y-axis limit to a reasonable value even if all scores are zero
    max_score = max(max(regular_f1), max(stemmed_f1))
    if max_score > 0:
        plt.ylim(0, max_score * 1.1)  # Add some headroom
    else:
        plt.ylim(0, 0.1)  # If all zeros, show a reasonable range
    
    # Add value labels on top of bars
    for i, v in enumerate(ordered_regular_f1):
        plt.text(i - 0.05, v + 0.01, f"{v:.3f}", ha='center')
    
    for i, v in enumerate(ordered_stemmed_f1):
        plt.text(i + bar_width - 0.05, v + 0.01, f"{v:.3f}", ha='center')
    
    # Add a note about low scores if needed
    if max_score < 0.01:
        plt.figtext(0.5, 0.01, 
                   "Note: All scores are near zero. This suggests a mismatch between\n"
                   "reference keyphrases and extracted keyphrases.",
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    output_path = CHARTS_DIR / "f1_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"F1 comparison chart saved to {output_path}")
    plt.close()

def create_precision_recall_chart(results):
    """Create a scatter plot of precision vs. recall for each model"""
    plt.figure(figsize=(8, 8))
    
    # Extract data
    model_names = [result["name"] for result in results]
    precisions = [result["avg_precision"] for result in results]
    recalls = [result["avg_recall"] for result in results]
    
    # Check if all values are zero
    all_zeros = all(p == 0 for p in precisions) and all(r == 0 for r in recalls)
    
    if all_zeros:
        # Create a message about zero values
        plt.text(0.5, 0.5, 
                "All precision and recall values are zero.\nReference and extracted keyphrases have no overlap.", 
                ha='center', va='center', fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":10})
        
        # Add labels and title
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs. Recall by Model (No Matching Keyphrases)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits with some padding
        max_recall = max(0.1, max(recalls) * 1.1 + 0.05) if any(r > 0 for r in recalls) else 1.0
        max_precision = max(0.1, max(precisions) * 1.1 + 0.05) if any(p > 0 for p in precisions) else 1.0
        plt.xlim(0, min(max_recall, 1.0))
        plt.ylim(0, min(max_precision, 1.0))
        
    else:
        # Create markers for regular vs stemmed models
        markers = ['o', 's']  # circle for regular, square for stemmed
        colors = ['blue', 'green', 'orange']  # one color per base model type
        
        # Plot each point
        for i, (name, precision, recall) in enumerate(zip(model_names, precisions, recalls)):
            # Skip TF-IDF (but keep TF-IDF+Ollama)
            if name == "TF-IDF" or name == "TF-IDF (Stemmed)":
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
        plt.title('Precision vs. Recall by Model')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits with some padding
        max_recall = max(0.1, max(recalls) * 1.1 + 0.05) if any(r > 0 for r in recalls) else 1.0
        max_precision = max(0.1, max(precisions) * 1.1 + 0.05) if any(p > 0 for p in precisions) else 1.0
        plt.xlim(0, min(max_recall, 1.0))
        plt.ylim(0, min(max_precision, 1.0))
        
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
    
    # Use adjustable subplot to avoid layout issues
    plt.gcf().subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / "precision_recall.png"
    plt.savefig(output_path, dpi=300)
    print(f"Precision-Recall chart saved to {output_path}")
    plt.close()

def create_chapter_performance_chart(results):
    """Create a line chart showing F1 scores by chapter for each model"""
    plt.figure(figsize=(12, 7))
    
    # Line styles and colors
    colors = ['blue', 'green', 'purple', 'orange']
    line_styles = ['-', '--']  # solid for original, dashed for stemmed
    
    # Check if we have any non-zero scores
    has_nonzero = False
    for result in results:
        for ch in result["chapters"]:
            if result["chapters"][ch]["f1"] > 0:
                has_nonzero = True
                break
        if has_nonzero:
            break
            
    if not has_nonzero:
        # Create a message about zero values
        plt.text(0.5, 0.5, 
                "All F1 scores are zero for all chapters and models.\nReference and extracted keyphrases have no overlap.", 
                ha='center', va='center', fontsize=14, 
                bbox={"facecolor":"orange", "alpha":0.2, "pad":10},
                transform=plt.gca().transAxes)
                
        # Add labels and title
        plt.xlabel('Chapter Number')
        plt.ylabel('F1 Score')
        plt.title('F1 Score by Chapter for Each Model (No Matching Keyphrases)')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        # Plot F1 scores for each model across chapters
        for i, result in enumerate(results):
            model_name = result["name"]
            
            # Skip TF-IDF (but keep TF-IDF+Ollama)
            if model_name == "TF-IDF" or model_name == "TF-IDF (Stemmed)":
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
        plt.title('F1 Score by Chapter for Each Model')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set chapter numbers as x-ticks
        all_chapters = set()
        for result in results:
            all_chapters.update([int(ch) for ch in result["chapters"].keys()])
        plt.xticks(sorted(all_chapters))
        
        # Add legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save chart
    output_path = CHARTS_DIR / "chapter_performance.png"
    plt.savefig(output_path, dpi=300)
    print(f"Chapter performance chart saved to {output_path}")
    plt.close()

def main():
    results = load_results()
    if not results:
        return
    
    print("Creating visualization charts...")
    create_f1_bar_chart(results)
    create_precision_recall_chart(results)
    create_chapter_performance_chart(results)
    print(f"All charts saved to {CHARTS_DIR}")

if __name__ == "__main__":
    main()