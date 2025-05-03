import os
import subprocess
import sys

PYTHON = sys.executable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
EXTRACTION_DIR = os.path.join(SCRIPTS_DIR, "extraction")
STEMMING_DIR = os.path.join(SCRIPTS_DIR, "stemming")
MAPPING_DIR = os.path.join(SCRIPTS_DIR, "mapping")
EVALUATION_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluation", "evaluate_keyphrases.py")
FUZZY_EVALUATION_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluation", "evaluate_fuzzy.py")
IMPROVED_EVALUATION_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluation", "evaluate_improved.py")
VISUALIZATION_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluation", "visualize_metrics.py")
FUZZY_VISUALIZATION_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluation", "visualize_fuzzy.py")
IMPROVED_VISUALIZATION_SCRIPT = os.path.join(SCRIPTS_DIR, "evaluation", "visualize_improved.py")

KEYBERT_SCRIPT = os.path.join(EXTRACTION_DIR, "keyword_extraction.py")
OLLAMA_SCRIPT = os.path.join(EXTRACTION_DIR, "ollama_extraction.py")
DEEPSEEK_SCRIPT = os.path.join(EXTRACTION_DIR, "deepseek_extraction.py")
TFIDF_SCRIPT = os.path.join(EXTRACTION_DIR, "tfidf_extraction.py")
TFIDF_OLLAMA_SCRIPT = os.path.join(EXTRACTION_DIR, "tfidf_ollama_extraction.py")
STEM_SCRIPT = os.path.join(STEMMING_DIR, "stem_extracted.py")
NLTK_DOWNLOAD_SCRIPT = os.path.join(STEMMING_DIR, "download_nltk_resources.py")

# Mapping scripts
PREPROCESS_MAPPING_SCRIPT = os.path.join(MAPPING_DIR, "preprocessing.py")
RELATIONSHIP_MAPPING_SCRIPT = os.path.join(MAPPING_DIR, "relationship_mapping.py")


def run_standard_extraction():
    print("\nRunning Standard (KeyBERT) Extraction...")
    subprocess.run([PYTHON, KEYBERT_SCRIPT], check=True)
    subprocess.run([PYTHON, STEM_SCRIPT, "--model_option", "keybert"], check=True)

def run_ollama_extraction():
    print("\nRunning Ollama Extraction...")
    subprocess.run([PYTHON, OLLAMA_SCRIPT], check=True)
    subprocess.run([PYTHON, STEM_SCRIPT, "--model_option", "ollama"], check=True)

def run_deepseek_extraction():
    print("\nRunning DeepSeek Extraction...")
    subprocess.run([PYTHON, DEEPSEEK_SCRIPT], check=True)
    subprocess.run([PYTHON, STEM_SCRIPT, "--model_option", "deepseek"], check=True)
    
def run_tfidf_extraction():
    print("\nRunning TF-IDF Extraction...")
    subprocess.run([PYTHON, TFIDF_SCRIPT], check=True)
    subprocess.run([PYTHON, STEM_SCRIPT, "--model_option", "tfidf"], check=True)
    
def run_tfidf_ollama_extraction():
    print("\nRunning TF-IDF + Ollama Hybrid Extraction...")
    subprocess.run([PYTHON, TFIDF_OLLAMA_SCRIPT], check=True)
    subprocess.run([PYTHON, STEM_SCRIPT, "--model_option", "tfidf_ollama"], check=True)

def run_evaluation():
    print("\nRunning Evaluation...")
    reference_dir = os.path.join("data", "keyphrases", "referenced")
    stemmed_reference_dir = os.path.join("data", "keyphrases", "stemmed", "referenced")
    split_ref_script = os.path.join("scripts", "evaluation", "split_reference.py")
    evaluate_script = os.path.join("scripts", "evaluation", "evaluate_keyphrases.py")

    # Check if the reference directory exists and is not empty
    if not os.path.exists(reference_dir) or not os.listdir(reference_dir):
        print("Reference files not found. Generating with split_reference.py...")
        subprocess.run([PYTHON, split_ref_script], check=True)

    subprocess.run([PYTHON, evaluate_script], check=True)
    
def run_fuzzy_evaluation():
    print("\nRunning Fuzzy Evaluation...")
    reference_dir = os.path.join("data", "keyphrases", "referenced")
    split_ref_script = os.path.join("scripts", "evaluation", "split_reference.py")

    # Check if the reference directory exists and is not empty
    if not os.path.exists(reference_dir) or not os.listdir(reference_dir):
        print("Reference files not found. Generating with split_reference.py...")
        subprocess.run([PYTHON, split_ref_script], check=True)

    # Run with three different thresholds
    print("Running with 70% threshold...")
    subprocess.run([PYTHON, FUZZY_EVALUATION_SCRIPT, "--threshold", "70"], check=True)
    
    print("Running with 80% threshold...")
    subprocess.run([PYTHON, FUZZY_EVALUATION_SCRIPT, "--threshold", "80"], check=True)
    
    print("Running with 90% threshold...")
    subprocess.run([PYTHON, FUZZY_EVALUATION_SCRIPT, "--threshold", "90"], check=True)
    
def run_improved_evaluation():
    print("\nRunning Improved Evaluation...")
    reference_dir = os.path.join("data", "keyphrases", "referenced")
    split_ref_script = os.path.join("scripts", "evaluation", "split_reference.py")

    # Check if the reference directory exists and is not empty
    if not os.path.exists(reference_dir) or not os.listdir(reference_dir):
        print("Reference files not found. Generating with split_reference.py...")
        subprocess.run([PYTHON, split_ref_script], check=True)

    # Run the improved evaluation script
    subprocess.run([PYTHON, IMPROVED_EVALUATION_SCRIPT], check=True)
    
def run_visualization():
    print("\nGenerating Visualization Charts...")
    subprocess.run([PYTHON, VISUALIZATION_SCRIPT], check=True)
    
def run_fuzzy_visualization():
    print("\nGenerating Fuzzy Visualization Charts...")
    subprocess.run([PYTHON, FUZZY_VISUALIZATION_SCRIPT], check=True)
    
def run_improved_visualization():
    print("\nGenerating Improved Visualization Charts...")
    subprocess.run([PYTHON, IMPROVED_VISUALIZATION_SCRIPT], check=True)

def download_nltk_resources():
    """Download required NLTK resources to fix tokenizer issues"""
    print("\nDownloading NLTK resources...")
    subprocess.run([PYTHON, NLTK_DOWNLOAD_SCRIPT], check=True)
    print("NLTK resources download completed.")

def run_keyphrase_preprocessing(model="keybert", stemmed=False):
    """Preprocess extracted keyphrases for mapping"""
    print(f"\nPreprocessing keyphrases for {model} (stemmed={stemmed})...")
    
    # Ensure sanitized directories exist
    sanitized_base = os.path.join("data", "keyphrases", "sanitized")
    stemmed_sanitized = os.path.join(sanitized_base, "stemmed")
    os.makedirs(sanitized_base, exist_ok=True)
    os.makedirs(stemmed_sanitized, exist_ok=True)
    
    # Create model-specific directories 
    model_sanitized = os.path.join(sanitized_base, model)
    stemmed_model_sanitized = os.path.join(stemmed_sanitized, model)
    os.makedirs(model_sanitized, exist_ok=True)
    os.makedirs(stemmed_model_sanitized, exist_ok=True)
    
    # Run preprocessing script
    cmd = [PYTHON, PREPROCESS_MAPPING_SCRIPT, "--model", model]
    if stemmed:
        cmd.append("--stemmed")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Preprocessing for {model} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing keyphrases: {e}")
        print("Please ensure you have run keyword extraction first.")
        return False
    
    return True
    
def open_image_file(file_path):
    """Open an image file with the default image viewer"""
    try:
        if sys.platform == "win32":
            os.startfile(file_path)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        else:  # Linux
            subprocess.run(["xdg-open", file_path], check=True)
        return True
    except Exception as e:
        print(f"Error opening file: {e}")
        return False

def run_relationship_mapping(model="keybert", stemmed=False, interactive=True, keyword=None, auto_open=True, show_labels=True):
    """Run relationship mapping to generate concept maps"""
    print(f"\nRunning relationship mapping for {model} (stemmed={stemmed})...")
    
    # Ensure preprocessing is done first
    if not run_keyphrase_preprocessing(model, stemmed):
        print("Failed to preprocess keyphrases. Mapping cannot continue.")
        return False
    
    # Check for required packages for relationship mapping
    try:
        import sentence_transformers
        import torch
        import torch_geometric
    except ImportError as e:
        missing_pkg = str(e).split("'")[1]
        print(f"Required package missing: {missing_pkg}")
        print(f"Please install it with: pip install {missing_pkg}")
        if "torch_geometric" in str(e):
            print("For torch_geometric, you may need to run: pip install torch-geometric")
        return False
    
    cmd = [PYTHON, RELATIONSHIP_MAPPING_SCRIPT, "--model", model]
    if stemmed:
        cmd.append("--stemmed")
    if show_labels:
        cmd.append("--show_labels")
    
    if interactive and not keyword:
        # For interactive mode without specified keyword, ask for a keyword here
        # rather than letting the subprocess ask, which may cause hanging
        user_keyword = input("Enter a keyword for generating the concept map: ").strip().lower()
        if not user_keyword:
            print("No keyword provided. Cannot continue.")
            return False
        
        # Set the max nodes and depth
        try:
            max_nodes = int(input("Max number of nodes to plot (e.g. 25): ").strip())
        except:
            max_nodes = 25
        try:
            max_depth = int(input("Max depth from keyword (e.g. 2): ").strip())
        except:
            max_depth = 2
            
        # Ask for relationship labels preference
        #show_labels = input("Show relationship labels on edges? (y/n): ").strip().lower() == 'y'
            
        # Pass all parameters as command-line arguments to avoid interactive input in subprocess
        cmd.extend(["--keyword", user_keyword, "--max_nodes", str(max_nodes), "--max_depth", str(max_depth)])
        interactive = False  # Switch to non-interactive mode now that we have all parameters
    elif keyword:
        cmd.extend(["--keyword", keyword])
    
    try:
        # Use capture_output only for non-interactive mode
        if not interactive:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Extract the output path from the script's output
            output_path = None
            for line in result.stdout.split('\n'):
                if "Concept map saved to:" in line:
                    output_path = line.split("Concept map saved to:")[1].strip()
                    break
                print(line)  # Print the output for visibility
        else:
            # For truly interactive mode (though we should never reach here now)
            subprocess.run(cmd, check=True)
            output_path = None
        
        print(f"Relationship mapping for {model} completed successfully.")
        
        # Automatically open the output file if available
        if auto_open and output_path and os.path.exists(output_path):
            print(f"Opening concept map: {output_path}")
            open_image_file(output_path)
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running relationship mapping: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print("STDOUT:", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("STDERR:", e.stderr)
        return False
    except KeyboardInterrupt:
        print("\nRelationship mapping interrupted by user.")
        return False
    
def create_evaluation_guide():
    """Create a helper guide explaining evaluation approaches"""
    guide_path = os.path.join(BASE_DIR, "EVALUATION_GUIDE.md")
    with open(guide_path, "w") as f:
        f.write("""# Keyphrase Extraction Evaluation Guide

## Why Fuzzy Matching Is Recommended

When evaluating keyphrase extraction, we use three different approaches:

1. **Standard Evaluation**: Requires exact matches between extracted and reference keyphrases
   - Very strict and results in artificially low scores
   - Doesn't account for variations in wording or phrasing

2. **Improved Evaluation**: Uses partial matching and considers contained phrases
   - Better than standard, but still fairly rigid
   - Can miss valid matches due to word order or minor variations

3. **Fuzzy Matching (RECOMMENDED)**: Uses string similarity with configurable thresholds
   - Most realistic evaluation of extraction quality
   - Allows for minor variations in terminology
   - Provides multiple thresholds (70%, 80%, 90%) to tune strictness

The fuzzy matching approach with a 70% threshold provides the most reasonable representation of actual model performance, as it accounts for the reality that many valid keyphrases may be expressed with slight variations.

## How to Evaluate Your Results

1. First run your extraction models (options 3, 5, or 6)
2. Run the fuzzy evaluation (option 8)
3. Generate the fuzzy visualization charts (option 11)
4. Review the charts in the `data/charts` directory

The charts will show you:
- Comparative F1 scores across different models
- Precision-Recall trade-offs
- Chapter-by-chapter performance

## Interpreting the Results

- **High Precision**: Model extracts mostly relevant keyphrases (quality)
- **High Recall**: Model finds most of the reference keyphrases (coverage)
- **High F1**: Good balance between precision and recall

Our hybrid TF-IDF+Ollama approach typically shows the highest overall performance, with strong precision scores.
""")
    print(f"Evaluation guide created at {guide_path}")

def display_menu():
    print("\n" + "=" * 60)
    print("Keyphrase Extraction Pipeline".center(60))
    print("=" * 60)
    print("EXTRACTION OPTIONS:")
    print("1. Run all extractions and evaluate")
    print("2. Run standard extraction only")
    print("3. Run Ollama extraction only")
    print("4. Run DeepSeek extraction only (deprecated)")
    print("5. Run TF-IDF extraction only")
    print("6. Run TF-IDF + Ollama hybrid extraction")
    
    print("\nEVALUATION OPTIONS:")
    print("7. Evaluate existing results only (standard)")
    print("8. Evaluate with fuzzy matching (70%, 80%, 90% thresholds) - RECOMMENDED")
    print("9. Evaluate with improved matching")
    
    print("\nVISUALIZATION OPTIONS:")
    print("10. Generate standard visualization charts (not recommended)")
    print("11. Generate fuzzy evaluation charts only - RECOMMENDED")
    print("12. Generate improved evaluation charts only")
    print("13. Run full pipeline with visualizations")
    
    print("\nMAPPING OPTIONS:")
    print("14. Create concept map from KeyBERT keyphrases (interactive)")
    print("15. Create concept map from Ollama keyphrases (interactive)")
    print("16. Create concept map from TF-IDF + Ollama hybrid keyphrases (interactive)")
    print("17. Create concept map from stemmed KeyBERT keyphrases (interactive)")
    print("18. Create concept map from stemmed Ollama keyphrases (interactive)")
    print("19. Create concept map from stemmed TF-IDF + Ollama hybrid keyphrases (interactive)")
    print("20. Create concept map with specific keyword (non-interactive)")
    
    print("\nUTILITIES:")
    print("21. Download NLTK resources (fix errors)")
    print("22. Generate evaluation guide (recommended)")
    print("23. Exit")

    while True:
        choice = input("\nEnter your choice (1-23): ").strip()
        if choice in {str(i) for i in range(1, 24)}:
            return int(choice)
        print("Invalid input. Please enter a number between 1 and 23.")

def main():
    option = display_menu()
    if option == 1:
        download_nltk_resources()  # Ensure resources are downloaded first
        run_standard_extraction()
        run_ollama_extraction()
        run_deepseek_extraction()
        run_tfidf_extraction()
        run_tfidf_ollama_extraction()
        run_evaluation()
    elif option == 2:
        download_nltk_resources()
        run_standard_extraction()
    elif option == 3:
        download_nltk_resources()
        run_ollama_extraction()
    elif option == 4:
        download_nltk_resources()
        run_deepseek_extraction()
    elif option == 5:
        download_nltk_resources()
        run_tfidf_extraction()
    elif option == 6:
        download_nltk_resources()
        run_tfidf_ollama_extraction()
    elif option == 7:
        download_nltk_resources()
        run_evaluation()
    elif option == 8:
        download_nltk_resources()
        run_fuzzy_evaluation()
        run_fuzzy_visualization()
    elif option == 9:
        download_nltk_resources()
        run_improved_evaluation()
        run_improved_visualization()
    elif option == 10:
        run_visualization()
    elif option == 11:
        run_fuzzy_visualization()
    elif option == 12:
        run_improved_visualization()
    elif option == 13:
        # Run full pipeline with visualizations
        download_nltk_resources()
        run_standard_extraction()
        run_tfidf_extraction()
        run_tfidf_ollama_extraction()
        run_evaluation()
        run_fuzzy_evaluation()
        run_improved_evaluation()
        run_visualization()
        run_fuzzy_visualization()
        run_improved_visualization()
    # Mapping options
    elif option == 14:
        download_nltk_resources()
        # Check if KeyBERT extraction has been run
        keyphrases_dir = os.path.join("data", "keyphrases", "keybert")
        if not os.path.exists(keyphrases_dir) or not os.listdir(keyphrases_dir):
            print("KeyBERT keyphrases not found. Running KeyBERT extraction first...")
            run_standard_extraction()
        run_relationship_mapping(model="keybert", stemmed=False, interactive=True)
    elif option == 15:
        download_nltk_resources()
        # Check if Ollama extraction has been run
        keyphrases_dir = os.path.join("data", "keyphrases", "ollama")
        if not os.path.exists(keyphrases_dir) or not os.listdir(keyphrases_dir):
            print("Ollama keyphrases not found. Running Ollama extraction first...")
            run_ollama_extraction()
        run_relationship_mapping(model="ollama", stemmed=False, interactive=True)
    elif option == 16:
        download_nltk_resources()
        # Check if TF-IDF + Ollama hybrid extraction has been run
        keyphrases_dir = os.path.join("data", "keyphrases", "tfidf_ollama")
        if not os.path.exists(keyphrases_dir) or not os.listdir(keyphrases_dir):
            print("TF-IDF + Ollama hybrid keyphrases not found. Running hybrid extraction first...")
            run_tfidf_ollama_extraction()
        run_relationship_mapping(model="tfidf_ollama", stemmed=False, interactive=True)
    elif option == 17:
        download_nltk_resources()
        # Check if stemmed KeyBERT extraction has been run
        keyphrases_dir = os.path.join("data", "keyphrases", "stemmed", "keybert")
        if not os.path.exists(keyphrases_dir) or not os.listdir(keyphrases_dir):
            print("Stemmed KeyBERT keyphrases not found. Running KeyBERT extraction and stemming first...")
            run_standard_extraction()
        run_relationship_mapping(model="keybert", stemmed=True, interactive=True)
    elif option == 18:
        download_nltk_resources()
        # Check if stemmed Ollama extraction has been run
        keyphrases_dir = os.path.join("data", "keyphrases", "stemmed", "ollama")
        if not os.path.exists(keyphrases_dir) or not os.listdir(keyphrases_dir):
            print("Stemmed Ollama keyphrases not found. Running Ollama extraction and stemming first...")
            run_ollama_extraction()
        run_relationship_mapping(model="ollama", stemmed=True, interactive=True)
    elif option == 19:
        download_nltk_resources()
        # Check if stemmed TF-IDF + Ollama hybrid extraction has been run
        keyphrases_dir = os.path.join("data", "keyphrases", "stemmed", "tfidf_ollama")
        if not os.path.exists(keyphrases_dir) or not os.listdir(keyphrases_dir):
            print("Stemmed TF-IDF + Ollama hybrid keyphrases not found. Running hybrid extraction and stemming first...")
            run_tfidf_ollama_extraction()
        run_relationship_mapping(model="tfidf_ollama", stemmed=True, interactive=True)
    # Non-interactive mapping
    elif option == 20:
        # Ask user for model, keyword, and stemming preference
        print("\nNon-interactive concept mapping with relationship types:")
        print("Available models: keybert, ollama, tfidf_ollama")
        model = input("Enter model name: ").strip().lower()
        if model not in ["keybert", "ollama", "tfidf_ollama"]:
            print(f"Invalid model: {model}. Using keybert as default.")
            model = "keybert"
            
        keyword = input("Enter keyword for concept map: ").strip()
        if not keyword:
            print("No keyword provided. Exiting.")
            return
            
        stemmed = input("Use stemmed keyphrases? (y/n): ").strip().lower() == 'y'
        show_labels = input("Show relationship labels on edges? (y/n): ").strip().lower() == 'y'
        
        try:
            max_nodes = int(input("Max number of nodes to plot (default=25): ").strip() or "25")
        except:
            max_nodes = 25
            
        try:
            max_depth = int(input("Max depth from keyword (default=2): ").strip() or "2")
        except:
            max_depth = 2
        
        # Check if extraction has been run
        keyphrases_dir = os.path.join("data", "keyphrases", "stemmed" if stemmed else "", model)
        if not os.path.exists(keyphrases_dir) or not os.listdir(keyphrases_dir):
            print(f"{model.title()} keyphrases not found. Running extraction first...")
            if model == "keybert":
                run_standard_extraction()
            elif model == "ollama":
                run_ollama_extraction()
            elif model == "tfidf_ollama":
                run_tfidf_ollama_extraction()
        
        # Run mapping with specified keyword and options
        cmd = [PYTHON, RELATIONSHIP_MAPPING_SCRIPT, "--model", model, "--keyword", keyword, 
               "--max_nodes", str(max_nodes), "--max_depth", str(max_depth)]
        
        if stemmed:
            cmd.append("--stemmed")
        if show_labels:
            cmd.append("--show_labels")
            
        print(f"Running relationship mapping with: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Print output
            for line in result.stdout.split('\n'):
                print(line)
                
            # Extract the output path
            output_path = None
            for line in result.stdout.split('\n'):
                if "Concept map saved to:" in line:
                    output_path = line.split("Concept map saved to:")[1].strip()
                    break
            
            # Open the generated file
            if output_path and os.path.exists(output_path):
                print(f"Opening concept map: {output_path}")
                open_image_file(output_path)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running relationship mapping: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
        
    # Utilities options
    elif option == 21:
        download_nltk_resources()
    elif option == 22:
        create_evaluation_guide()
    elif option == 23:
        print("Exiting.")


if __name__ == "__main__":
    main()
