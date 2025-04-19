# From Text to Map: Automating Concept Extraction and Relationship Modeling

This project implements a pipeline for extracting key concepts from text and modeling their relationships using graph structures. The implementation supports the paper "From Text to Map: Automating Concept Extraction and Relationship Modeling".

## Features

- **PDF Extraction:** Automatically extracts text from PDF documents
- **Chapter Chunking:** Splits the extracted text into individual chapters
- **Keyphrase Extraction:** Identifies important concepts using KeyBERT and sentence transformers
- **Co-occurrence Analysis:** Models relationships between concepts using co-occurrence patterns
- **Graph Visualization:** Creates visual concept maps using NetworkX
- **Enhanced Embeddings:** Learns improved concept representations with Graph Convolutional Networks (GCN)

## Prerequisites

- Python 3.6 or higher

## Installation

```bash
pip install -r requirements.txt
```

## Pipeline

The project consists of multiple steps:

1. **Text Extraction**: Extract and chunk text from PDF documents
   ```bash
   python chunking.py
   ```

2. **Keyphrase Extraction**: Extract important keyphrases from each chunk
   ```bash
   python key_word_extraction.py
   ```

3. **Co-occurrence Analysis**: Build a graph of related concepts
   ```bash
   python co_occurrence.py
   ```

4. **GCN Modeling**: Learn improved embeddings with Graph Convolutional Networks
   ```bash
   python gcn_model.py
   ```

## Output

- Extracted text chunks in `book_chunks/`
- Extracted keyphrases in `key_phrases/`
- Co-occurrence graph visualization in `graph_data/keyphrase_graph.png`
- GCN embeddings visualization in `gcn_output/keyphrase_embeddings.png`
- Concept embeddings exported to `gcn_output/keyphrase_embeddings.json`

## Implementation Details

### Text Extraction (chunking.py)
- Uses pdfminer.six to extract text from PDF
- Splits text into chapters using regular expressions
- Provides visual feedback using Colorama

### Keyphrase Extraction (key_word_extraction.py)
- Uses KeyBERT with a fine-tuned sentence transformer model
- Extracts top keyphrases from each chapter
- Stores keyphrases with confidence scores

### Co-occurrence Analysis (co_occurrence.py)
- Builds a co-occurrence matrix of keyphrases
- Creates a NetworkX graph with keyphrases as nodes
- Edge weights represent co-occurrence frequency
- Visualizes the graph with node sizes based on centrality

### GCN Modeling (gcn_model.py)
- Implements a 3-layer Graph Convolutional Network
- Uses contrastive learning to capture concept relationships
- Visualizes embeddings using t-SNE and k-means clustering
- Exports embeddings for downstream applications

## References

- KeyBERT: Grootendorst, M. (2020). KeyBERT: Minimal keyword extraction with BERT
- Sentence-Transformers: Reimers et al. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
- GCN: Kipf and Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks