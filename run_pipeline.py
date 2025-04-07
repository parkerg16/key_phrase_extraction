import subprocess
import os
import argparse
from pathlib import Path

def run_script(name, args=None):
    print(f"\n🚀 Running: {name}")
    cmd = ["python", name]
    if args:
        cmd += args
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.resolve(),
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("⚠️ Errors:", result.stderr)

def main(train_model=False, model_path=None):
    root = Path(__file__).parent.resolve()

    # Step 1: Extract and chunk book
    run_script("scripts/extract_and_chunk_book.py", [
        "--book_path", str(root / "data" / "raw" / "new_book.pdf"),
        "--output_dir", str(root / "data" / "chunks"),
        "--skip_first"
    ])

    # Step 2: Extract index terms
    run_script("scripts/extract_index_terms.py", [
        "--input", str(root / "data" / "extracted_text.txt"),
        "--output", str(root / "data" / "index_by_chapter.txt")
    ])

    # Step 3: Train model (optional)
    if train_model:
        run_script("training/training_pipeline.py")
        model_path = str(root / "models" / "keybert" / "my_finetuned_model")

    # Step 4: Train keyword extraction
    run_script("scripts/keyword_extraction_train.py", [
        "--input_dir", str(root / "data" / "processed_chunks"),
        "--output_dir", str(root / "data" / "keyphrases" / "train"),
        "--model", model_path or "distilroberta-base-msmarco-v2"
    ])

    # Step 5: Test keyword extraction
    run_script("scripts/keyword_extraction_test.py", [
        "--input_dir", str(root / "data" / "processed_chunks"),
        "--output_dir", str(root / "data" / "keyphrases" / "test"),
        "--model", model_path or "distilroberta-base-msmarco-v2"
    ])

    # Step 6: Evaluate test keyphrases
    run_script("evaluation/evaluate_keyphrases.py", [
        "--index_path", str(root / "data" / "index_by_chapter.txt"),
        "--extracted_dir", str(root / "data" / "keyphrases" / "test")
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full keyphrase extraction pipeline.")
    parser.add_argument('--train_model', action='store_true', help="Run training pipeline before extraction")
    parser.add_argument('--model_path', type=str, help="Path to fine-tuned model to use for extraction")
    args = parser.parse_args()
    main(train_model=args.train_model, model_path=args.model_path)