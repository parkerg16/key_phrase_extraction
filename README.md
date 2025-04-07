# Key Phrase Extraction & Concept Mapping Pipeline

A modular NLP pipeline that extracts keyphrases from a textbook, maps their relationships, and evaluates extraction quality using fuzzy metrics. Built using `pdfminer.six`, `KeyBERT`, `sentence-transformers`, and `fuzzywuzzy`.

---

## 🔍 Project Overview

This project takes a textbook PDF and transforms it into structured knowledge via:

- 📖 **PDF Text Extraction**
- ✂️ **Chapter Chunking**
- 🧼 **Text Preprocessing**
- 🧠 **Keyphrase Extraction (KeyBERT + Transformers)**
- ✅ **Evaluation using Index Terms**
- 🌐 **Graph-based Concept Mapping (via GCN)**

---

## 📂 Directory Structure

```plaintext
key_phrase_extraction/
├── data/
│   ├── raw/                     # PDF and raw sources
│   ├── chunks/                  # Chunked chapters
│   ├── keyphrases/             # Extracted keyphrases (train/test)
│   └── index_by_chapter.txt    # Index for evaluation
├── models/                     # Checkpoints for KeyBERT / GCN
├── sandbox/                    # Writing out test files
├── scraping/                   # Data scraping scripts
├── scripts/                    # Extraction & preprocessing scripts
├── evaluation/                 # Evaluation metrics and results
├── gcn_mapping/                # Concept map and relationship graphing
├── run_pipeline.py             # 🚀 Master script to run pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Features

### ✅ PDF Text Processing
- Extracts text using `pdfminer.six`
- Automatically chunks textbook by chapter markers (e.g. `\fChapter 1`)
- Preprocesses text (removes dates, emails, figure/table numbers, etc.)

### 💡 Keyphrase Extraction
- Uses `KeyBERT` with transformer backends (default: `distilroberta-base-msmarco-v2`)
- Supports extraction for both train and test sets (based on chapter splits)
- Saves top-ranked phrases per chapter

### 📊 Evaluation
- Parses `index_by_chapter.txt` to get ground truth per chapter
- Matches keyphrases against index terms using fuzzy partial ratio
- Calculates precision, recall, and F1 per chapter and overall

### 🧠 Relationship Mapping *(coming soon)*
- Constructs knowledge graphs of keyphrases using Graph Convolutional Networks
- Explores semantic relationships, concept overlaps, and cluster-based learning

---

## 🧪 Prerequisites

- Python 3.8+

```bash
pip install -r requirements.txt
```
Ensure `fuzzywuzzy[speedup]`, `sentence-transformers`, and `pdfminer.six` are included.

---

## 🚀 Running the Full Pipeline

The entire workflow can be executed from start to finish using:

```bash
python run_pipeline.py
```
This will:
1. Extract text from `data/raw/new_book.pdf`
2. Chunk into chapters → `data/chunks/`
3. Preprocess → `data/processed_chunks/`
4. Run KeyBERT on training chapters → `data/keyphrases/train/`
5. Run KeyBERT on test chapters → `data/keyphrases/test/`
6. Evaluate test output using fuzzy match → results printed per chapter

---

## 🧩 Script Reference (Individual Steps)

### 📖 1. Extract & Chunk Book
```bash
python scripts/extract_and_chunk_book.py --book_path data/raw/new_book.pdf --output_dir data/chunks --skip_first
```

### 🏷️ 2. Extract Index Terms
```bash
python scripts/extract_index_terms.py --input data/extracted_text.txt --output data/index_by_chapter.txt
```

### 🧠 3. Fine-Tune Sentence Transformer (optional)
```bash
python training/training_pipeline.py
```
This will save a fine-tuned model to: 
```
models/keybert/my_finetuned_model/
```

### 🧠 4. Keyword Extraction - Train Set
```bash
python scripts/keyword_extraction_test.py \
  --input_dir data/processed_chunks \
  --output_dir data/keyphrases/train \
  --model models/keybert/my_finetuned_model  # or any HuggingFace model
```

### 🧠 5. Keyword Extraction - Test Set
```bash
python scripts/keyword_extraction_test.py \
  --input_dir data/processed_chunks \
  --output_dir data/keyphrases/test \
  --model models/keybert/my_finetuned_model  # or any HuggingFace model
```

### 🧪 6. Evaluate Key phrases (F1, Precision, Recall)
```bash
python evaluation/evaluate_keyphrases.py \
  --index_path data/index_by_chapter.txt \
  --extracted_dir data/keyphrases/test
```
You can also evaluate the training set: 
```bash
python evaluation/evaluate_keyphrases.py \
  --index_path data/index_by_chapter.txt \
  --extracted_dir data/keyphrases/train
```

---

## 🔬 Future Work
- Integrate Wikipedia scraping to enrich keyphrase context
- Apply GCN to build visual concept maps
- Extend evaluation to compare against multiple annotators or sources

---

## 📘 Example Output

```bash
🚀 Running: extract_and_chunk_book.py
✔ PDF text extracted to extracted_text.txt
✔ Written Chapter 1 to data/chunks/chapter_1_chunk.txt
✔ Processed chapter_1_chunk.txt → data/processed_chunks/
...

🚀 Running: evaluate_keyphrases.py

Chapter 10:
  matched: 45
  extracted_total: 100
  index_total: 92
  precision: 0.4500
  recall: 0.4891
  f1_score: 0.4689
```

---

## 🧠 Credits
- Built on top of the amazing [`pdfminer.six`](https://github.com/pdfminer/pdfminer.six), [`KeyBERT`](https://github.com/MaartenGr/KeyBERT), and `sentence-transformers`

For any questions, feel free to open an issue or contribute!

---

Happy extracting 🧠💡
