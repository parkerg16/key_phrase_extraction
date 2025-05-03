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
CHARTS_DIR = BASE_DIR / "data" / "charts" / "improved"

# Create charts directory if it doesn't exist
os.makedirs(CHARTS_DIR, exist_ok=True)

def load_improved_results():
    """Load improved evaluation results from JSON file"""
    results_file = RESULTS_DIR / "improved_evaluation_results.json"
    
    if not os.path.exists(results_file):
        print(f"Error: Results file not found at {results_file}")
        print("Please run the improved evaluation script first to generate results.")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def create_f1_bar_chart(results):
    """Create bar charts comparing different F1 metrics across models"""
    if not results:
        print("No results to visualize")
        return
        
    # Extract model names, avg F1, and overall F1
    model_names = [result["name"] for result in results]
    avg_f1_scores = [result["avg_f1"] for result in results]
    overall_f1_scores = [result["overall_f1"] for result in results]
    
    # Separate regular and stemmed models
    regular_models = []
    regular_avg_f1 = []
    regular_overall_f1 = []
    stemmed_models = []
    stemmed_avg_f1 = []
    stemmed_overall_f1 = []
    
    for name, avg_f1, overall_f1 in zip(model_names, avg_f1_scores, overall_f1_scores):
        if "(Stemmed)" in name:
            # Remove the "(Stemmed)" part for cleaner labels
            stemmed_models.append(name.replace(" (Stemmed)", ""))
            stemmed_avg_f1.append(avg_f1)
            stemmed_overall_f1.append(overall_f1)
        else:
            regular_models.append(name)
            regular_avg_f1.append(avg_f1)
            regular_overall_f1.append(overall_f1)
    
    # Create two separate charts: one for average F1 and one for overall F1
    
    # Print diagnostic info
    print(f"Regular models: {regular_models}, len={len(regular_models)}")
    print(f"Stemmed models: {stemmed_models}, len={len(stemmed_models)}")
    
    # 1. Average F1 Chart
    plt.figure(figsize=(12, 6))
    
    # Set up the bar chart with fixed model names to avoid confusion
    unique_models = ["KeyBERT", "Ollama", "TF-IDF+Ollama"]
    num_models = len(unique_models)
    bar_width = 0.35
    index = np.arange(num_models)
    
    # Group data by model name (just the base name without "Stemmed")
    original_by_model = {model: 0 for model in unique_models}
    stemmed_by_model = {model: 0 for model in unique_models}
    
    # Prepare the data for each model type
    for j, name in enumerate(model_names):
        base_name = name.replace(" (Stemmed)", "")
        
        # Use exact matching to avoid confusion between TF-IDF and TF-IDF+Ollama
        if base_name in unique_models:
            if "(Stemmed)" in name:
                stemmed_by_model[base_name] = avg_f1_scores[j]
            else:
                original_by_model[base_name] = avg_f1_scores[j]
    
    # Create arrays for plotting
    original_values = [original_by_model[model] for model in unique_models]
    stemmed_values = [stemmed_by_model[model] for model in unique_models]
    
    # Plot bars
    plt.bar(index - bar_width/2, original_values, bar_width, label='Original')
    plt.bar(index + bar_width/2, stemmed_values, bar_width, label='Stemmed')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Average F1 Score')
    plt.title('Average F1 Scores by Model (Improved Method)')
    plt.xticks(index, unique_models)
    
    # Set y-axis limit
    max_score = max(max(original_values), max(stemmed_values))
    plt.ylim(0, max_score * 1.15)  # Add some headroom
    
    # Add value labels on top of bars
    for i, v in enumerate(original_values):
        if v > 0:
            plt.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    for i, v in enumerate(stemmed_values):
        if v > 0:
            plt.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.legend()
    # Use adjustable subplot to avoid layout issues
    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / "average_f1_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Average F1 chart saved to {output_path}")
    plt.close()
    
    # 2. Overall F1 Chart
    plt.figure(figsize=(12, 6))
    
    # Group overall F1 scores by model name using fixed model names
    original_by_model_overall = {model: 0 for model in unique_models}
    stemmed_by_model_overall = {model: 0 for model in unique_models}
    
    # Prepare the data for each model type
    for j, name in enumerate(model_names):
        base_name = name.replace(" (Stemmed)", "")
        
        # Use exact matching to avoid confusion between TF-IDF and TF-IDF+Ollama
        if base_name in unique_models:
            if "(Stemmed)" in name:
                stemmed_by_model_overall[base_name] = overall_f1_scores[j]
            else:
                original_by_model_overall[base_name] = overall_f1_scores[j]
    
    # Create arrays for plotting
    original_overall_values = [original_by_model_overall[model] for model in unique_models]
    stemmed_overall_values = [stemmed_by_model_overall[model] for model in unique_models]
    
    # Create bars
    plt.bar(index - bar_width/2, original_overall_values, bar_width, label='Original')
    plt.bar(index + bar_width/2, stemmed_overall_values, bar_width, label='Stemmed')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Overall F1 Score')
    plt.title('Overall F1 Scores by Model (Improved Method)')
    plt.xticks(index, unique_models)
    
    # Set y-axis limit
    max_score = max(max(original_overall_values), max(stemmed_overall_values))
    plt.ylim(0, max_score * 1.15)  # Add some headroom
    
    # Add value labels on top of bars
    for i, v in enumerate(original_overall_values):
        if v > 0:
            plt.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    for i, v in enumerate(stemmed_overall_values):
        if v > 0:
            plt.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.legend()
    # Use adjustable subplot to avoid layout issues
    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / "overall_f1_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Overall F1 chart saved to {output_path}")
    plt.close()

def create_precision_recall_chart(results):
    """Create precision-recall scatter plot with the improved results"""
    plt.figure(figsize=(10, 8))
    
    # Extract model names and metrics
    model_names = [result["name"] for result in results]
    precisions = [result["overall_precision"] for result in results]
    recalls = [result["overall_recall"] for result in results]
    f1_scores = [result["overall_f1"] for result in results]
    
    # Create markers for regular vs stemmed models
    markers = ['o', 's']  # circle for regular, square for stemmed
    colors = ['blue', 'green', 'orange']  # one color per base model type
    
    # Plot each point
    for i, (name, precision, recall, f1) in enumerate(zip(model_names, precisions, recalls, f1_scores)):
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
        plt.annotate(f"{name}\nF1={f1:.3f}", (recall, precision), 
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
    plt.title('Precision vs. Recall (Improved Method)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with some padding and ensure we catch all points
    max_recall = max(0.1, max(recalls) * 1.1 + 0.05)
    max_precision = max(0.1, max(precisions) * 1.1 + 0.05)
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
    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / "precision_recall.png"
    plt.savefig(output_path, dpi=300)
    print(f"Precision-Recall chart saved to {output_path}")
    plt.close()

def create_chapter_performance_chart(results):
    """Create a line chart showing F1 scores by chapter for each model"""
    plt.figure(figsize=(14, 8))
    
    # Line styles and colors
    colors = ['blue', 'green', 'orange']
    line_styles = ['-', '--']  # solid for original, dashed for stemmed
    
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
        chapter_data = result["chapters"]
        chapters = sorted([int(ch) for ch in chapter_data.keys()])
        if not chapters:
            print(f"Skipping {model_name} - no chapter data")
            continue
            
        f1_scores = [chapter_data[str(ch)]["f1"] for ch in chapters]
        
        # Skip if not enough data points for a meaningful line
        if len(chapters) < 2:
            print(f"Skipping {model_name} - not enough chapters ({len(chapters)})")
            continue
            
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
    plt.title('F1 Score by Chapter (Improved Method)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set chapter numbers as x-ticks
    all_chapters = set()
    for result in results:
        all_chapters.update([int(ch) for ch in result["chapters"].keys()])
    plt.xticks(sorted(all_chapters))
    
    # Add legend in a better position
    plt.legend(loc='upper right')
    
    # Use adjustable subplot to avoid layout issues
    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Save chart
    output_path = CHARTS_DIR / "chapter_performance.png"
    plt.savefig(output_path, dpi=300)
    print(f"Chapter performance chart saved to {output_path}")
    plt.close()

def create_best_worst_chart(results):
    """Create a bar chart showing the best and worst performing chapters for each model"""
    # Extract model names and best/worst chapters
    model_names = []
    best_chapters = []
    worst_chapters = []
    
    for result in results:
        model_name = result["name"]
        
        # Skip TF-IDF (but keep TF-IDF+Ollama)
        if model_name == "TF-IDF" or model_name == "TF-IDF (Stemmed)":
            continue
            
        model_names.append(model_name)
        
        # Get best and worst chapters, defaulting to None if not available
        best_chapters.append(result.get("best_chapter", None))
        worst_chapters.append(result.get("worst_chapter", None))
    
    # Create a bar chart
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    # Replace None values with 0 for plotting (but we'll still skip labeling them)
    best_plot = [0 if v is None else v for v in best_chapters]
    worst_plot = [0 if v is None else v for v in worst_chapters]
    
    # Plot bars
    plt.bar(index - bar_width/2, best_plot, bar_width, label='Best Chapter')
    plt.bar(index + bar_width/2, worst_plot, bar_width, label='Worst Chapter')
    
    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Chapter Number')
    plt.title('Best and Worst Performing Chapters by Model')
    # Increase rotation angle and position for better label fit and visibility
    plt.xticks(index, model_names, rotation=45, ha='right', fontsize=10)
    # Add extra space for x-axis
    plt.subplots_adjust(bottom=0.35)
    
    # Add value labels on top of bars with minimal padding
    for i, v in enumerate(best_chapters):
        if v is not None:
            plt.text(i - bar_width/2, v + 0.6, f'Ch {v}', ha='center', fontsize=9)
    
    for i, v in enumerate(worst_chapters):
        if v is not None:
            plt.text(i + bar_width/2, v + 0.6, f'Ch {v}', ha='center', fontsize=9)
    
    plt.legend()
    
    # Determine the y-axis limit with moderate padding for labels
    max_chapter = max([v for v in best_chapters + worst_chapters if v is not None], default=1)
    plt.ylim(0, max_chapter * 1.15 + 1)  # Add moderate headroom for labels
    
    # Use subplot adjustments with significantly more bottom padding for x-axis labels
    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.35)
    
    # Save chart
    output_path = CHARTS_DIR / "best_worst_chapters.png"
    plt.savefig(output_path, dpi=300)
    print(f"Best/worst chapters chart saved to {output_path}")
    plt.close()

def create_metrics_table(results):
    """Create a table visualization showing the raw metrics (precision, recall, F1) for each model"""
    # Extract model names and metrics
    model_names = [result["name"] for result in results]
    precisions = [result["overall_precision"] for result in results]
    recalls = [result["overall_recall"] for result in results]
    f1_scores = [result["overall_f1"] for result in results]
    
    # Create a figure and axis
    fig, ax = plt.figure(figsize=(12, len(model_names) * 0.75 + 1)), plt.gca()
    
    # Hide axes
    ax.axis('off')
    ax.axis('tight')
    
    # Create table data
    table_data = []
    for name, prec, rec, f1 in zip(model_names, precisions, recalls, f1_scores):
        # Skip TF-IDF (but keep TF-IDF+Ollama)
        if name == "TF-IDF" or name == "TF-IDF (Stemmed)":
            continue
        table_data.append([name, f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
    
    # Define headers
    headers = ["Model", "Precision", "Recall", "F1 Score"]
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:
            # Alternate row colors for better readability
            if row % 2 == 0:
                cell.set_facecolor('#E6F0FF')
            else:
                cell.set_facecolor('#F2F7FF')
                
            # Highlight F1 score column
            if col == 3:  # F1 Score column
                cell._text.set_weight('bold')
    
    # Add a title
    plt.title('Model Performance Metrics (Improved Evaluation)', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save the figure
    output_path = CHARTS_DIR / "metrics_table.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics table saved to {output_path}")
    plt.close()

def main():
    results = load_improved_results()
    if not results:
        return
    
    print("Creating visualization charts for improved evaluation results...")
    create_f1_bar_chart(results)
    create_precision_recall_chart(results)
    create_chapter_performance_chart(results)
    create_best_worst_chart(results)
    create_metrics_table(results)
    print(f"All improved evaluation charts saved to {CHARTS_DIR}")

if __name__ == "__main__":
    main()