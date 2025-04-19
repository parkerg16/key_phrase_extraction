import os
import re
import argparse

def sanitize_line(line: str) -> str:
    """Remove numbering like '1. keyphrase' and extra whitespace"""
    line = line.strip()
    if not line:
        return ""
    if line.lower().startswith("here are"):
        return ""
    match = re.match(r"^\d+\.\s*(.+)", line)
    return match.group(1).strip() if match else line

def sanitize_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = [sanitize_line(line) for line in lines]
    cleaned = [line for line in cleaned if line]  # remove empty lines

    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(cleaned))

def process_model(model: str, stemmed: bool):
    base = os.path.join("data", "keyphrases", "stemmed" if stemmed else "", model)
    out_base = os.path.join("data", "keyphrases", "sanitized", "stemmed" if stemmed else "", model)
    os.makedirs(out_base, exist_ok=True)

    if not os.path.exists(base):
        print(f" Source directory not found: {base}")
        return

    for filename in os.listdir(base):
        if filename.endswith("_keyphrases.txt"):
            in_file = os.path.join(base, filename)
            out_file = os.path.join(out_base, filename)
            sanitize_file(in_file, out_file)
            print(f" Sanitized: {filename} → {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Sanitize keyphrase files for concept mapping.")
    parser.add_argument("--model", type=str, choices=["keybert", "ollama"], required=True)
    parser.add_argument("--stemmed", action="store_true", help="Use stemmed directory")
    args = parser.parse_args()

    process_model(args.model, args.stemmed)

if __name__ == "__main__":
    main()
