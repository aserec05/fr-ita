# French-Italian Multilingual NLP Pipeline

Cross-lingual word embeddings, Procrustes alignment, and multilingual NLP tasks for French and Italian.

**Author:** Tommaso Ceresa  
**Course:** Natural Language Processing (Dr. Andrea Mauri)  
**Period:** January - March 2025  
**Institution:** Université Claude Bernard Lyon 1

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Pre-trained Models](#pre-trained-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Output Files](#output-files)
- [Results Summary](#results-summary)
- [Technical Notes](#technical-notes)
- [References](#references)

---

## 🎯 Overview

This project implements a complete multilingual NLP pipeline for French and Italian:

1. **Embedding Generation**: 5 methods (one-hot, TF-IDF, Word2Vec, FastText, GloVe)
2. **Procrustes Alignment**: Cross-lingual mapping using MUSE dictionary
3. **Visualization**: t-SNE and PCA projections
4. **Analysis**: Synonym detection, cognate similarity, polysemy, OOV handling
5. **Language Identification**: Binary classification (99%+ accuracy)
6. **Sentiment Analysis**: Cross-lingual transfer experiments (French → Italian)

**Key Finding:** Pre-trained GloVe embeddings achieve 87.4% translation accuracy after Procrustes alignment, while corpus-trained embeddings fail due to insufficient training data (10K sentences).

---

## 📁 Project Structure
```
fr_ita/
├── data/
│   ├── embeddings/          # Generated embeddings (.pkl files)
│   ├── muse_dictionaries/   # Bilingual dictionaries
│   ├── sentiment/           # Sentiment datasets and results
│   ├── visualizations/      # t-SNE, PCA plots
│   └── fasttext_models/     # Pre-trained GloVe vectors (MANUAL DOWNLOAD)
├── fr_ita/
│   ├── generate_embeddings/ # Embedding generation scripts
│   │   ├── onehot.py
│   │   ├── tfidf.py
│   │   ├── word2vec.py
│   │   ├── fasttext.py
│   │   └── glove.py
│   ├── alignment/           # Procrustes alignment
│   │   ├── prepare_dictionary.py
│   │   ├── procrustes_alignment.py
│   │   └── analysis.py
│   ├── visualization/       # t-SNE, PCA, radar charts
│   │   ├── visualization.py
│   │   ├── viz_tsne.py
│   │   ├── viz_pca.py
│   │   ├── viz_alignment.py
│   │   └── viz_radar.py
│   ├── classifiers/         # Language ID classifier
│   │   └── train_classifier.py
│   ├── sentiment/           # Sentiment analysis
│   │   ├── prepare_sentiment.py
│   │   └── train_sentiment.py
│   ├── utils.py             # Helper functions
│   └── config.py            # Configuration
└── README.md
```

---

## 📊 Datasets

| Dataset | Purpose | Size | Auto-Download | Source |
|---------|---------|------|---------------|--------|
| **Tatoeba FR-IT** | Parallel corpus for embedding training | 10,000 sentence pairs | ✅ Yes | https://tatoeba.org |
| **MUSE Dictionary** | Procrustes alignment (FR-IT) | 5,000 word pairs | ✅ Yes | https://github.com/facebookresearch/MUSE |
| **Allocine** | French sentiment analysis | 160,000 movie reviews | ✅ Yes (HuggingFace) | https://huggingface.co/datasets/allocine |
| **Italian Sentiment** | Italian sentiment analysis | 3,033 Twitter samples | ✅ Yes (HuggingFace) | https://huggingface.co/datasets/theoracle/Italian.sentiment.analysis |

**Note:** All datasets are downloaded automatically when running the corresponding scripts. No manual download required.

---

## 🔧 Pre-trained Models

**⚠️ MANUAL DOWNLOAD REQUIRED**

GloVe embeddings must be downloaded manually and placed in `data/fasttext_models/`:

| Model | Language | Dimension | Vocabulary | Size | URL |
|-------|----------|-----------|------------|------|-----|
| **Common Crawl** | French | 300 | 2M words | ~6.8 GB | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz |
| **Common Crawl** | Italian | 300 | 2M words | ~6.2 GB | https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz |

### Download Instructions
```bash
# Create directory
mkdir -p data/fasttext_models
cd data/fasttext_models

# Download French GloVe (6.8 GB uncompressed)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz
gunzip cc.fr.300.vec.gz

# Download Italian GloVe (6.2 GB uncompressed)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz
gunzip cc.it.300.vec.gz

cd ../..
```

**Storage Requirements:**
- Tatoeba corpus: ~5 MB
- MUSE dictionary: ~2 MB
- Trained embeddings: ~200 MB
- **GloVe pre-trained: ~13 GB**
- Sentiment datasets: ~50 MB
- Visualizations: ~10 MB
- **Total: ~13.5 GB**

---

## 💻 Installation

### Requirements

- Python 3.8+
- Linux/macOS/Windows (WSL recommended for Windows)

### Dependencies
```bash
pip install numpy scipy scikit-learn nltk gensim matplotlib datasets pandas --break-system-packages
```

**Optional (for FastText):**
```bash
pip install fasttext --break-system-packages
```

**Note:** The `--break-system-packages` flag is required on some systems when using pip outside a virtual environment.

### NLTK Data

First-time setup requires downloading NLTK tokenizers:
```python
import nltk
nltk.download('punkt')
```

Or run any script—it will prompt for download automatically.

---

## 🚀 Quick Start

### Complete Pipeline (30-60 minutes)
```bash
# 1. Download GloVe embeddings (MANUAL - see above)
#    Skip this if you don't need GloVe

# 2. Generate all embeddings
python3 -m fr_ita.generate_embeddings.onehot
python3 -m fr_ita.generate_embeddings.tfidf
python3 -m fr_ita.generate_embeddings.word2vec
python3 -m fr_ita.generate_embeddings.fasttext
python3 -m fr_ita.generate_embeddings.glove  # Requires manual download

# 3. Prepare bilingual dictionary and align embeddings
python3 -m fr_ita.alignment.prepare_dictionary
python3 -m fr_ita.alignment.procrustes_alignment --all

# 4. Visualize embeddings
python3 -m fr_ita.visualization.visualization --all

# 5. Analyze embedding quality
python3 -m fr_ita.alignment.analysis --all

# 6. Train language identifier
python3 -m fr_ita.classifiers.train_classifier --all

# 7. Sentiment analysis with cross-lingual transfer
python3 -m fr_ita.sentiment.prepare_sentiment
python3 -m fr_ita.sentiment.train_sentiment --all --cross-lingual
```

---

## 📖 Detailed Usage

### 1. Embedding Generation

Generate word embeddings for French and Italian:
```bash
# One-hot encoding (baseline)
python3 -m fr_ita.generate_embeddings.onehot

# TF-IDF vectors
python3 -m fr_ita.generate_embeddings.tfidf

# Word2Vec (CBOW, 100d, trained on Tatoeba)
python3 -m fr_ita.generate_embeddings.word2vec

# FastText (100d, subword n-grams)
python3 -m fr_ita.generate_embeddings.fasttext

# GloVe (300d, pre-trained on Common Crawl)
# REQUIRES MANUAL DOWNLOAD FIRST
python3 -m fr_ita.generate_embeddings.glove
```

**Output:** `data/embeddings/{method}/french_{method}.pkl` and `italian_{method}.pkl`

**Parameters:**
- Word2Vec/FastText: `vector_size=100`, `window=5`, `min_count=2`, `epochs=10`
- TF-IDF: `max_features=5000`
- GloVe: Pre-trained 300d vectors, vocabulary filtered to Tatoeba corpus

---

### 2. Procrustes Alignment

Align French and Italian embeddings using orthogonal transformation:
```bash
# Download MUSE bilingual dictionary (auto-downloaded)
python3 -m fr_ita.alignment.prepare_dictionary

# Align specific embedding type
python3 -m fr_ita.alignment.procrustes_alignment --embedding glove

# Align all embedding types
python3 -m fr_ita.alignment.procrustes_alignment --all
```

**Output:** `data/embeddings/{method}/french_{method}_aligned.pkl`

**Evaluation Metrics:**
- Precision@1 (P@1): Translation accuracy using CSLS nearest neighbors
- Average cosine similarity of aligned pairs
- Iterative refinement: 5 iterations, top 75% confidence pairs

**Results:**
- GloVe: 87.4% P@1 (test set)
- TF-IDF: 76.8% P@1
- Word2Vec: 0% P@1 (failed due to small corpus)
- FastText: 0% P@1 (failed due to small corpus)

---

### 3. Visualization

Generate t-SNE and PCA visualizations:
```bash
# Visualize specific embedding
python3 -m fr_ita.visualization.visualization --embedding glove

# Visualize all embeddings + comparison charts
python3 -m fr_ita.visualization.visualization --all
```

**Output:** `data/visualizations/`
- `tsne_{method}.png`: t-SNE 2D projection
- `pca_{method}.png`: PCA 2D projection with cosine similarity color-coding
- `alignment_comparison.png`: Bar chart of Procrustes alignment quality
- `radar_comparison.png`: Multi-metric quality comparison

**Visualizations show:**
- Word2Vec/FastText: Strong language separation
- GloVe: Partial cross-lingual overlap
- PCA variance: FastText 82% (collapsed), GloVe 18% (distributed)

---

### 4. Embedding Quality Analysis

Analyze synonyms, cognates, polysemy, and OOV handling:
```bash
# Analyze specific embedding
python3 -m fr_ita.alignment.analysis --embedding glove

# Analyze all embeddings
python3 -m fr_ita.alignment.analysis --all
```

**Metrics:**
- **Synonyms/Antonyms**: Nearest neighbor coherence
- **Cognates**: Cross-lingual similarity (e.g., porte/porta, musique/musica)
- **Polysemy**: Neighbor variance for polysemous words
- **OOV**: Out-of-vocabulary rates

**Key Findings:**
- Word2Vec/FastText: All similarities ≈ 1.0 (collapsed semantic space)
- GloVe: Coherent neighbors (e.g., "heureux" → "malheureux", "ravi")
- OOV rates: Word2Vec 45%, GloVe 9%

---

### 5. Language Identification

Train binary classifier to distinguish French from Italian:
```bash
# Train on specific embedding
python3 -m fr_ita.classifiers.train_classifier --embedding glove

# Train on all embeddings
python3 -m fr_ita.classifiers.train_classifier --all
```

**Results:** 99-100% accuracy across all embedding types  
**Model:** Logistic regression with mean-pooled sentence vectors

---

### 6. Sentiment Analysis

Binary sentiment classification (positive/negative) with cross-lingual transfer:
```bash
# Step 1: Prepare datasets (auto-downloads Allocine + Italian Twitter)
python3 -m fr_ita.sentiment.prepare_sentiment

# Step 2: Train sentiment classifiers
# Monolingual only
python3 -m fr_ita.sentiment.train_sentiment --embedding glove

# With cross-lingual experiments
python3 -m fr_ita.sentiment.train_sentiment --embedding glove --cross-lingual

# All embeddings with cross-lingual
python3 -m fr_ita.sentiment.train_sentiment --all --cross-lingual
```

**Datasets:**
- French: Allocine movie reviews (5,000 samples, balanced)
- Italian: Twitter sentiment data (2,022 samples, balanced)

**Experiments:**
1. **Monolingual FR**: Train/test on French → 73% accuracy (GloVe)
2. **Monolingual IT**: Train/test on Italian → 69% accuracy (GloVe)
3. **Cross-lingual (raw)**: Train FR, test IT without alignment → 60% accuracy
4. **Cross-lingual (aligned)**: Train FR, test IT with Procrustes → 58% accuracy

**Key Finding:** For GloVe, Procrustes alignment does not improve cross-lingual transfer for sentiment tasks. Pre-trained embeddings already exhibit natural correspondences, and general translation dictionaries (MUSE) lack sentiment-specific vocabulary.

---

## 📦 Output Files

All outputs are saved to `data/`:
```
data/
├── embeddings/
│   ├── onehot/
│   │   ├── french_onehot.pkl
│   │   └── italian_onehot.pkl
│   ├── tfidf/
│   │   ├── french_tfidf.pkl
│   │   └── italian_tfidf.pkl
│   ├── word2vec/
│   │   ├── french_word2vec.pkl
│   │   ├── italian_word2vec.pkl
│   │   ├── french_word2vec_aligned.pkl  # After Procrustes
│   │   └── italian_word2vec_aligned.pkl
│   ├── fasttext/
│   │   └── ...
│   └── glove/
│       └── ...
├── muse_dictionaries/
│   └── fr-it.txt
├── sentiment/
│   ├── sentiment_data.pkl
│   └── sentiment_results.pkl
├── visualizations/
│   ├── tsne_*.png
│   ├── pca_*.png
│   ├── alignment_comparison.png
│   └── radar_comparison.png
└── classifier_results.pkl
```

---

## 📈 Results Summary

### Procrustes Alignment (Test P@1)

| Embedding | Accuracy | Notes |
|-----------|----------|-------|
| GloVe | **87.4%** | Pre-trained on 840B tokens |
| TF-IDF | 76.8% | Document-level features |
| Word2Vec | 0% | Failed (10K sentence corpus too small) |
| FastText | 0% | Failed (10K sentence corpus too small) |

### Language Identification

| Embedding | Accuracy |
|-----------|----------|
| All types | **99-100%** |

### Sentiment Analysis (GloVe)

| Experiment | Accuracy | F1 |
|------------|----------|-----|
| Monolingual FR | 73% | 0.73 |
| Monolingual IT | 69% | 0.69 |
| Cross-lingual (raw) | 60% | 0.58 |
| Cross-lingual (aligned) | 58% | 0.58 |

**Insight:** Procrustes does not improve sentiment transfer for pre-trained embeddings. MUSE dictionary lacks sentiment vocabulary.

---

## 🔧 Technical Notes

### Python Module Execution

All scripts use Python's module execution syntax:
```bash
python3 -m fr_ita.module.script
```

**Not:**
```bash
python3 fr_ita/module/script.py  # ❌ Won't work (import issues)
```

### File Paths

Scripts use paths relative to the project root:
- Input: `data/embeddings/{method}/`
- Output: `data/embeddings/{method}/`, `data/visualizations/`

### Memory Requirements

- Word2Vec/FastText training: ~500 MB RAM
- GloVe loading: ~2 GB RAM per language
- Visualization: ~1 GB RAM

### Known Issues

1. **Overleaf Compilation Timeout**: If using LaTeX on Overleaf free tier, reduce number of visualization figures (10 images may exceed 60s timeout)
2. **NLTK Download**: First run may require manual `nltk.download('punkt')`
3. **Windows Paths**: Use WSL or replace `/` with `\` in config.py if running on Windows CMD

---

## 📚 References

### Datasets & Dictionaries

- **Tatoeba**: https://tatoeba.org
- **MUSE**: Conneau et al. (2018) "Word Translation Without Parallel Data" - https://github.com/facebookresearch/MUSE
- **Allocine**: https://huggingface.co/datasets/allocine
- **Italian Sentiment**: https://huggingface.co/datasets/theoracle/Italian.sentiment.analysis

### Embedding Methods

- **Word2Vec**: Mikolov et al. (2013) "Efficient Estimation of Word Representations in Vector Space"
- **FastText**: Bojanowski et al. (2017) "Enriching Word Vectors with Subword Information"
- **GloVe**: Pennington et al. (2014) "GloVe: Global Vectors for Word Representation" - https://nlp.stanford.edu/projects/glove/

### Alignment

- **Procrustes**: Schönemann (1966) "A Generalized Solution of the Orthogonal Procrustes Problem"

---

## 👤 Author

**Tommaso Ceresa**  
Université Claude Bernard Lyon 1  
Natural Language Processing Course (Dr. Andrea Mauri)  
January - March 2025

---

## 📄 License

This project is for academic purposes as part of the NLP course curriculum.

---

## 🙏 Acknowledgments

- Dr. Andrea Mauri for the NLP course
- Facebook Research for MUSE bilingual dictionaries
- Stanford NLP Group for GloVe embeddings
- Tatoeba community for parallel sentence corpus