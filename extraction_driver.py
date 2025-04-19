import os
import subprocess
import sys
from collections import defaultdict

# Get the path to the current Python interpreter
PYTHON_EXECUTABLE = sys.executable

# Set TERM environment variable if not set
if 'TERM' not in os.environ:
    os.environ['TERM'] = 'xterm-256color'

def run_extraction_scripts(run_standard=True, run_ollama=True, run_stemming=True):
    """
    Run keyphrase extraction scripts as specified
    
    Args:
        run_standard: Whether to run standard extraction
        run_ollama: Whether to run Ollama extraction
        run_stemming: Whether to create stemmed versions
    """
    success = True
    
    # Run standard extraction if requested
    if run_standard:
        print("Running standard keyword extraction...")
        try:
            subprocess.run([PYTHON_EXECUTABLE, "key_word_extraction.py"], check=True)
            print("Standard keyword extraction completed successfully.")
            
            # Create stemmed version if requested
            if run_stemming:
                print("Creating stemmed version of standard keyphrases...")
                try:
                    subprocess.run([PYTHON_EXECUTABLE, "stem_extracted.py", "key_phrases", "key_phrases_stemmed"], check=True)
                    print("Standard stemmed keyphrases created successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error creating stemmed standard keyphrases: {e}")
                    # Continue anyway
        except subprocess.CalledProcessError as e:
            print(f"Error running key_word_extraction.py: {e}")
            success = False
    
    # Run Ollama extraction if requested
    if run_ollama:
        print("\nRunning Ollama-based extraction...")
        try:
            subprocess.run([PYTHON_EXECUTABLE, "ollama_extraction.py"], check=True)
            print("Ollama-based extraction completed successfully.")
            
            # Create stemmed version if requested
            if run_stemming:
                print("Creating stemmed version of Ollama keyphrases...")
                try:
                    subprocess.run([PYTHON_EXECUTABLE, "stem_extracted.py", "key_phrases_ollama", "key_phrases_ollama_stemmed"], check=True)
                    print("Ollama stemmed keyphrases created successfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error creating stemmed Ollama keyphrases: {e}")
                    # Continue anyway
        except subprocess.CalledProcessError as e:
            print(f"Error running ollama_extraction.py: {e}")
            success = False
    
    return success

def load_reference_keyphrases(reference_dir, max_chapters=20):
    """
    Load reference keyphrases from individual files in reference_dir
    Each file should be named chapter_X_reference.txt
    """
    reference_keyphrases = defaultdict(set)
    
    # Check if we need to create the reference files first
    if not os.path.exists(reference_dir):
        print(f"Reference directory {reference_dir} does not exist. Creating it...")
        # Run the split_reference.py script
        try:
            subprocess.run([PYTHON_EXECUTABLE, "split_reference.py"], check=True)
            print("Reference keyphrases split successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error splitting reference keyphrases: {e}")
            # Fallback to reading the index file directly
            return load_reference_keyphrases_from_index("../index_by_chapter.txt")
    
    # Find all reference files in the folder
    if os.path.exists(reference_dir):
        # Pattern matching to find chapter numbers directly from filenames
        reference_files = []
        for filename in os.listdir(reference_dir):
            if filename.startswith("chapter_") and filename.endswith("_reference.txt"):
                try:
                    # Extract chapter number from filename (chapter_X_reference.txt)
                    chapter_num = int(filename.split('_')[1])
                    reference_files.append((chapter_num, os.path.join(reference_dir, filename)))
                except (IndexError, ValueError):
                    print(f"Warning: Could not determine chapter number from filename: {filename}")
                    continue
        
        # Sort files by chapter number
        reference_files.sort()  # Sorts by first element (chapter_num)
        
        # Load keyphrases from each file
        for chapter_num, filepath in reference_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    phrase = line.strip().lower()
                    if phrase:
                        reference_keyphrases[chapter_num].add(phrase)
            
            print(f"Loaded {len(reference_keyphrases[chapter_num])} reference keyphrases for Chapter {chapter_num}")
    
    return reference_keyphrases

def load_reference_keyphrases_from_index(index_file):
    """Load the reference keyphrases from the index file directly (fallback method)"""
    reference_keyphrases = defaultdict(set)
    current_chapter = None
    
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Chapter"):
                current_chapter = int(line.split()[1])
            else:
                # Add the keyphrase to the current chapter
                reference_keyphrases[current_chapter].add(line.strip().lower())
    
    return reference_keyphrases

def load_extracted_keyphrases(folder, max_chapters=20):
    """Load extracted keyphrases from the specified folder"""
    extracted_keyphrases = defaultdict(set)
    processed_chapters = []
    skipped_chapters = []
    
    # First, find all keyphrase files in the folder
    if os.path.exists(folder):
        # Pattern matching to find chapter numbers directly from filenames
        keyphrase_files = []
        for filename in os.listdir(folder):
            if filename.startswith("chapter_") and filename.endswith("_keyphrases.txt"):
                try:
                    # Extract chapter number from filename (chapter_X_keyphrases.txt)
                    chapter_num = int(filename.split('_')[1])
                    keyphrase_files.append((chapter_num, os.path.join(folder, filename)))
                except (IndexError, ValueError):
                    print(f"Warning: Could not determine chapter number from filename: {filename}")
                    continue
        
        # Sort files by chapter number
        keyphrase_files.sort()  # Sorts by first element (chapter_num)
        
        # Load keyphrases from each file
        for chapter_num, filepath in keyphrase_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    # Skip files that are empty or contain error markers
                    if not lines or (len(lines) == 1 and lines[0].startswith('#')):
                        print(f"Warning: Chapter {chapter_num} has no valid keyphrases or contains errors")
                        skipped_chapters.append(chapter_num)
                        continue
                        
                    # Process valid keyphrases
                    for line in lines:
                        phrase = line.strip().lower()
                        if phrase and not phrase.startswith('#'):
                            extracted_keyphrases[chapter_num].add(phrase)
                    
                processed_chapters.append(chapter_num)
                print(f"Loaded {len(extracted_keyphrases[chapter_num])} keyphrases for Chapter {chapter_num}")
            except Exception as e:
                print(f"Error loading keyphrases for Chapter {chapter_num}: {e}")
                skipped_chapters.append(chapter_num)
    
    if skipped_chapters:
        print(f"Skipped chapters due to errors or empty files: {', '.join(map(str, skipped_chapters))}")
    
    print(f"Successfully loaded keyphrases for {len(processed_chapters)} chapters")
    return extracted_keyphrases

def calculate_f1_scores(reference, extracted, method_name):
    """Calculate F1 scores for each chapter and overall"""
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
    
    return overall_f1, avg_f1

def display_menu():
    """Display simple console menu"""
    # Clear screen (safely)
    try:
        # Try to clear the screen, but don't fail if it doesn't work
        os.system('cls' if os.name == 'nt' else 'clear')
    except:
        # If clearing fails, just print newlines for spacing
        print("\n" * 5)
    
    # Show header
    print("=" * 50)
    print("Keyphrase Extraction Pipeline".center(50))
    print("=" * 50)
    print("Using pre-chunked files (ch1.txt - ch19.txt)".center(50))
    print("-" * 50)
    
    # Define menu options
    menu_options = [
        "Run both extraction scripts and evaluate",
        "Run standard extraction only",
        "Run Ollama extraction only",
        "Evaluate existing results (no extraction)",
        "Exit"
    ]
    
    # Display options
    for i, option in enumerate(menu_options, 1):
        print(f"{i}. {option}")
    
    # Get user choice
    while True:
        try:
            print("\nEnter your choice (1-5): ", end="")
            choice = int(input().strip())
            if 1 <= choice <= 5:
                # Return 0-based index to match previous implementation
                return choice - 1
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def run_pipeline(option):
    """Execute pipeline based on menu selection"""
    # File paths
    reference_dir = "reference_keyphrases"
    stemmed_reference_dir = "reference_keyphrases_stemmed"
    basic_keyphrases_folder = "key_phrases"
    basic_stemmed_folder = "key_phrases_stemmed"
    ollama_keyphrases_folder = "key_phrases_ollama"
    ollama_stemmed_folder = "key_phrases_ollama_stemmed"
    
    # Process option
    if option == 0:  # Run both extraction scripts and evaluate
        print("Running complete pipeline...\n")
        if not run_extraction_scripts(run_standard=True, run_ollama=True, run_stemming=True):
            print("⚠️ Some extraction scripts failed. Continuing with evaluation of available results.")
    elif option == 1:  # Run standard extraction only
        print("Running standard extraction only...\n")
        if not run_extraction_scripts(run_standard=True, run_ollama=False, run_stemming=True):
            print("⚠️ Standard extraction failed. Continuing with evaluation of available results.")
    elif option == 2:  # Run Ollama extraction only
        print("Running Ollama extraction only...\n")
        if not run_extraction_scripts(run_standard=False, run_ollama=True, run_stemming=True):
            print("⚠️ Ollama extraction failed. Continuing with evaluation of available results.")
    elif option == 3:  # Evaluate existing results
        print("Evaluating existing results...\n")
    elif option == 4:  # Exit
        print("Exiting program.")
        return
    
    # Skip evaluation if exiting
    if option == 4:
        return
    
    # First, ensure we have the reference keyphrases in individual files
    if option != 4 and (not os.path.exists(reference_dir) or len(os.listdir(reference_dir)) == 0):
        print(f"\nReference directory {reference_dir} doesn't exist or is empty.")
        print("Running split_reference.py to create reference files...")
        try:
            subprocess.run([PYTHON_EXECUTABLE, "split_reference.py"], check=True)
            print("Reference keyphrases split successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error splitting reference keyphrases: {e}")
            print("Proceeding with direct index file reading.")
    
    # Load reference keyphrases (both regular and stemmed)
    print("\nLoading reference keyphrases...")
    if os.path.exists(reference_dir) and len(os.listdir(reference_dir)) > 0:
        reference_keyphrases = load_reference_keyphrases(reference_dir)
    else:
        reference_keyphrases = load_reference_keyphrases_from_index("../index_by_chapter.txt")
    print(f"Loaded regular reference keyphrases for {len(reference_keyphrases)} chapters.")
    
    if os.path.exists(stemmed_reference_dir) and len(os.listdir(stemmed_reference_dir)) > 0:
        stemmed_reference_keyphrases = load_reference_keyphrases(stemmed_reference_dir)
        print(f"Loaded stemmed reference keyphrases for {len(stemmed_reference_keyphrases)} chapters.")
    else:
        print(f"Stemmed reference directory not found. Running split_reference.py to create it...")
        try:
            subprocess.run([PYTHON_EXECUTABLE, "split_reference.py"], check=True)
            stemmed_reference_keyphrases = load_reference_keyphrases(stemmed_reference_dir)
            print(f"Loaded stemmed reference keyphrases for {len(stemmed_reference_keyphrases)} chapters.")
        except:
            print("Could not create stemmed reference keyphrases. Using regular ones instead.")
            stemmed_reference_keyphrases = reference_keyphrases
    
    # Results storage
    results = {}
    
    # Evaluate standard extraction if the folder exists
    if os.path.exists(basic_keyphrases_folder) and (option in [0, 1, 3]):
        print("\n===== Evaluating Standard Extraction =====")
        # Regular evaluation
        print("\nRegular Evaluation:")
        std_keyphrases = load_extracted_keyphrases(basic_keyphrases_folder)
        std_overall_f1, std_avg_f1 = calculate_f1_scores(reference_keyphrases, std_keyphrases, "Standard Extraction")
        results["standard"] = {"overall_f1": std_overall_f1, "avg_f1": std_avg_f1}
        
        # Stemmed evaluation if available
        if os.path.exists(basic_stemmed_folder):
            print("\nStemmed Evaluation:")
            std_stemmed_keyphrases = load_extracted_keyphrases(basic_stemmed_folder)
            std_stemmed_overall_f1, std_stemmed_avg_f1 = calculate_f1_scores(
                stemmed_reference_keyphrases, std_stemmed_keyphrases, "Standard Extraction (Stemmed)"
            )
            results["standard_stemmed"] = {"overall_f1": std_stemmed_overall_f1, "avg_f1": std_stemmed_avg_f1}
        else:
            print(f"\nStemmed keyphrases for standard extraction not found. Creating them...")
            try:
                subprocess.run([PYTHON_EXECUTABLE, "stem_extracted.py", "key_phrases", "key_phrases_stemmed"], check=True)
                std_stemmed_keyphrases = load_extracted_keyphrases(basic_stemmed_folder)
                std_stemmed_overall_f1, std_stemmed_avg_f1 = calculate_f1_scores(
                    stemmed_reference_keyphrases, std_stemmed_keyphrases, "Standard Extraction (Stemmed)"
                )
                results["standard_stemmed"] = {"overall_f1": std_stemmed_overall_f1, "avg_f1": std_stemmed_avg_f1}
            except:
                print("Could not create stemmed keyphrases for standard extraction.")
                results["standard_stemmed"] = {"overall_f1": 0, "avg_f1": 0}
    else:
        results["standard"] = {"overall_f1": 0, "avg_f1": 0}
        results["standard_stemmed"] = {"overall_f1": 0, "avg_f1": 0}
    
    # Evaluate Ollama extraction if the folder exists
    if os.path.exists(ollama_keyphrases_folder) and (option in [0, 2, 3]):
        print("\n===== Evaluating Ollama Extraction =====")
        # Regular evaluation
        print("\nRegular Evaluation:")
        ollama_keyphrases = load_extracted_keyphrases(ollama_keyphrases_folder)
        ollama_overall_f1, ollama_avg_f1 = calculate_f1_scores(reference_keyphrases, ollama_keyphrases, "Ollama Extraction")
        results["ollama"] = {"overall_f1": ollama_overall_f1, "avg_f1": ollama_avg_f1}
        
        # Stemmed evaluation if available
        if os.path.exists(ollama_stemmed_folder):
            print("\nStemmed Evaluation:")
            ollama_stemmed_keyphrases = load_extracted_keyphrases(ollama_stemmed_folder)
            ollama_stemmed_overall_f1, ollama_stemmed_avg_f1 = calculate_f1_scores(
                stemmed_reference_keyphrases, ollama_stemmed_keyphrases, "Ollama Extraction (Stemmed)"
            )
            results["ollama_stemmed"] = {"overall_f1": ollama_stemmed_overall_f1, "avg_f1": ollama_stemmed_avg_f1}
        else:
            print(f"\nStemmed keyphrases for Ollama extraction not found. Creating them...")
            try:
                subprocess.run([PYTHON_EXECUTABLE, "stem_extracted.py", "key_phrases_ollama", "key_phrases_ollama_stemmed"], check=True)
                ollama_stemmed_keyphrases = load_extracted_keyphrases(ollama_stemmed_folder)
                ollama_stemmed_overall_f1, ollama_stemmed_avg_f1 = calculate_f1_scores(
                    stemmed_reference_keyphrases, ollama_stemmed_keyphrases, "Ollama Extraction (Stemmed)"
                )
                results["ollama_stemmed"] = {"overall_f1": ollama_stemmed_overall_f1, "avg_f1": ollama_stemmed_avg_f1}
            except:
                print("Could not create stemmed keyphrases for Ollama extraction.")
                results["ollama_stemmed"] = {"overall_f1": 0, "avg_f1": 0}
    else:
        results["ollama"] = {"overall_f1": 0, "avg_f1": 0}
        results["ollama_stemmed"] = {"overall_f1": 0, "avg_f1": 0}
    
    # Compare all methods
    if option in [0, 3] or (option in [1, 2] and os.path.exists(ollama_keyphrases_folder) and os.path.exists(basic_keyphrases_folder)):
        print("\n========== Comparison of All Methods ==========")
        print("-" * 60)
        print(f"{'Method':<25} {'Average F1':<15} {'Overall F1':<15}")
        print("-" * 60)
        print(f"{'Standard Extraction':<25} {results['standard']['avg_f1']:<15.4f} {results['standard']['overall_f1']:<15.4f}")
        print(f"{'Standard (Stemmed)':<25} {results['standard_stemmed']['avg_f1']:<15.4f} {results['standard_stemmed']['overall_f1']:<15.4f}")
        print(f"{'Ollama Extraction':<25} {results['ollama']['avg_f1']:<15.4f} {results['ollama']['overall_f1']:<15.4f}")
        print(f"{'Ollama (Stemmed)':<25} {results['ollama_stemmed']['avg_f1']:<15.4f} {results['ollama_stemmed']['overall_f1']:<15.4f}")
        print("-" * 60)
        
        # Find the best method based on average F1
        methods = [
            ("Standard", results['standard']['avg_f1']),
            ("Standard (Stemmed)", results['standard_stemmed']['avg_f1']),
            ("Ollama", results['ollama']['avg_f1']),
            ("Ollama (Stemmed)", results['ollama_stemmed']['avg_f1'])
        ]
        best_method = max(methods, key=lambda x: x[1])
        
        print(f"\nBest performing method: {best_method[0]} with Average F1 = {best_method[1]:.4f}")
        
        # Effect of stemming
        std_improvement = results['standard_stemmed']['avg_f1'] - results['standard']['avg_f1']
        ollama_improvement = results['ollama_stemmed']['avg_f1'] - results['ollama']['avg_f1']
        
        print("\nEffect of Stemming:")
        print(f"Standard: {'Improved' if std_improvement > 0 else 'Decreased'} by {abs(std_improvement):.4f}")
        print(f"Ollama: {'Improved' if ollama_improvement > 0 else 'Decreased'} by {abs(ollama_improvement):.4f}")

def main():
    """Main entry point"""
    try:
        # Display menu and get user choice
        option = display_menu()
        
        # Run the selected option
        run_pipeline(option)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()