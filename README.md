# Key Phrase Extraction & Concept Mapping Pipeline

A modular NLP pipeline for extracting keyphrases from textbooks, mapping their relationships, and evaluating extraction quality. Built using KeyBERT, sentence-transformers, LLMs (Llama 3), and Graph Convolutional Networks.

---

## Project Overview

This project takes a textbook and transforms it into structured knowledge via:

- **Text Extraction & Preprocessing**
- **Keyphrase Extraction** (multiple methods)
- **Fuzzy Evaluation Metrics**
- **Relationship Mapping & Concept Visualization**
- **Performance Analysis & Visualization**

---

## Directory Structure

```plaintext
key_phrase_extraction/
├── data/
│   ├── chapters/                # Chapter text files
│   ├── charts/                  # Visualization output
│   ├── concept_maps/            # Generated concept maps
│   ├── keyphrases/              # Extracted keyphrases with subfolders:
│   │   ├── keybert/             # KeyBERT extraction results
│   │   ├── ollama/              # LLM-based extraction results
│   │   ├── tfidf/               # TF-IDF statistical extraction
│   │   ├── tfidf_ollama/        # Hybrid extraction results
│   │   ├── stemmed/             # Stemmed versions of all above models
│   │   └── sanitized/           # Cleaned versions for mapping
│   ├── results/                 # Evaluation results (JSON)
│   └── index_by_chapter.txt     # Reference data for evaluation
├── scripts/
│   ├── evaluation/              # Evaluation scripts and visualization
│   ├── extraction/              # Extraction implementations
│   ├── mapping/                 # Relationship mapping & concept maps
│   ├── preprocessing/           # Text preprocessing utilities
│   ├── stemming/                # Stemming tools and utilities
│   └── training/                # Training pipeline for fine-tuning
├── sandbox/                     # Test implementations
├── extraction_driver.py         # Main interface with interactive menu
├── run_pipeline.py              # Pipeline execution script
├── requirements.txt
└── README.md
```

---

## Features

### Text Processing
- Extracts and preprocesses text from chapters
- Removes dates, emails, figure/table numbers, etc.
- Standardizes input for extraction models

### Keyphrase Extraction (Multiple Methods)
- **KeyBERT**: Transformer-based extraction using sentence embeddings
- **Ollama (LLM)**: Zero-shot extraction using Llama 3
- **TF-IDF**: Statistical term frequency-inverse document frequency
- **TF-IDF + Ollama Hybrid**: Statistical pre-filtering with LLM refinement

### Stemming & Normalization
- Stemming for normalizing terms
- Supports stemmed keyphrases for all extraction methods
- Improves matching performance in evaluation

### Multi-Method Evaluation
- **Standard Evaluation**: Exact match against textbook index
- **Fuzzy Matching**: Configurable thresholds (70%, 80%, 90%)
- **Improved Matching**: Enhanced partial matching system
- Calculates precision, recall, and F1 per chapter and overall

### Relationship Mapping & Concept Visualization
- Builds knowledge graphs of keyphrases using Graph Convolutional Networks
- Determines relationship types between concepts:
  - "is-a" and "type-of" relationships
  - "prerequisite-of" connections
  - "strongly-related-to" and "related-to" based on similarity
- Visual differentiation of relationship types with color and styles
- Interactive concept map generation from any extraction model

---

## Prerequisites

- Python 3.8+
- NLTK resources (downloadable via menu option)

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

The entire workflow can be executed through the interactive menu:

```bash
python extraction_driver.py
```

This provides options for:
1. Running different extraction models (KeyBERT, Ollama, TF-IDF, Hybrid)
2. Evaluating results with different matching methods
3. Generating visualizations and charts
4. Creating concept maps from any extraction model
5. Utilities for downloading resources

---

## Key Components

### Extraction Models

| Model | Description | Strengths |
|-------|-------------|-----------|
| KeyBERT | Transformer-based extraction | Good for technical terms |
| Ollama (LLM) | Llama 3-based zero-shot extraction | Better contextual understanding |
| TF-IDF | Statistical term frequency analysis | Fast, reliable baseline |
| TF-IDF + Ollama | Pre-filtering with LLM refinement | Best overall performance |

### Evaluation Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| Standard | Exact match evaluation | Baseline comparison |
| Fuzzy | String similarity with thresholds | Most realistic evaluation |
| Improved | Enhanced partial matching | Alternative to fuzzy |

### Concept Map Generation

Interactive creation of concept maps with relationship typing:
- Choose extraction model (including hybrid model)
- Select root keyword and visualization parameters
- Graph Convolutional Network (GCN) generates refined embeddings
- Relationship types are visually distinguished
- Maps can be saved and exported

---

## Example Results

The hybrid TF-IDF + Ollama model consistently outperforms other methods:

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| KeyBERT | 0.101 | 0.101 | 0.101 |
| KeyBERT (Stemmed) | 0.142 | 0.143 | 0.143 |
| Ollama | 0.233 | 0.234 | 0.233 |
| Ollama (Stemmed) | 0.294 | 0.247 | 0.297 |
| TF-IDF + Ollama | 0.678 | 0.236 | 0.351 |
| TF-IDF + Ollama (Stemmed) | 0.729 | 0.247 | 0.369 |

---

## Future Work
- Enhance LLM-assisted concept map refinement
- Apply more advanced GCN architectures
- Extend to additional textbook domains
- Improve visualization interactivity

---

## Credits
- Built using KeyBERT, sentence-transformers, Ollama, PyTorch Geometric
- Employs fuzzywuzzy for fuzzy matching evaluation