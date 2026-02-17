"""
Common utility functions for FR-IT NLP Assignment
Shared across generate_embeddings, alignment, visualization, classifier
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fr_ita.config import DATA_DIR, PROJECT_ROOT

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def load_tatoeba_tsv(max_sentences=10000):
    """
    Load Tatoeba FR-IT parallel corpus from TSV file

    Args:
        max_sentences (int): Maximum number of sentence pairs to load

    Returns:
        tuple: (french_sentences, italian_sentences, dataframe)
    """
    tsv_path = os.path.join(DATA_DIR, 'corpus_tatoeba.tsv')

    df = pd.read_csv(tsv_path, sep='\t', header=None,
                     names=['id_fr', 'sentence_fr', 'id_it', 'sentence_it'],
                     on_bad_lines='skip')

    df = df.dropna()

    if max_sentences:
        df = df.head(max_sentences)
        print(f"Using first {max_sentences} sentence pairs")

    french_sentences = df['sentence_fr'].tolist()
    italian_sentences = df['sentence_it'].tolist()

    return french_sentences, italian_sentences, df


def preprocess_text(corpus, language='french', remove_stopwords=False):
    """
    Preprocess text: lowercase, tokenize, clean

    Args:
        corpus (list): List of sentences
        language (str): Language for stopwords
        remove_stopwords (bool): Whether to remove stopwords

    Returns:
        list: List of tokenized sentences
    """
    processed = []

    for text in corpus:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)

        if remove_stopwords:
            try:
                stop_words = set(stopwords.words(language))
                tokens = [t for t in tokens if t not in stop_words]
            except:
                pass

        tokens = [t for t in tokens if len(t) > 0]
        processed.append(tokens)

    return processed


def save_embeddings(embeddings, filepath):
    """
    Save embeddings to disk

    Args:
        embeddings: Embeddings to save
        filepath (str): Relative or absolute path
    """
    if not os.path.isabs(filepath):
        filepath = os.path.join(PROJECT_ROOT, filepath)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings saved to {filepath}")


def load_embeddings(filepath):
    """
    Load embeddings from disk

    Args:
        filepath (str): Relative or absolute path

    Returns:
        Loaded embeddings
    """
    if not os.path.isabs(filepath):
        filepath = os.path.join(PROJECT_ROOT, filepath)

    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)

    print(f"Embeddings loaded from {filepath}")
    return embeddings


def load_bilingual_dictionary(lang1='fr', lang2='it', split='full'):
    """
    Load MUSE bilingual dictionary.

    Args:
        lang1 (str): Source language code
        lang2 (str): Target language code
        split (str): 'full'  -> fr-it.txt          (all pairs)
                     'train' -> fr-it.0-5000.txt    (first 5000, for alignment)
                     'test'  -> fr-it.5000-6500.txt (held-out 1500, for evaluation)

    Returns:
        dict: source_word (lowercased) -> target_word (lowercased)
    """
    split_suffix = {
        'full':  f'{lang1}-{lang2}.txt',
        'train': f'{lang1}-{lang2}.0-5000.txt',
        'test':  f'{lang1}-{lang2}.5000-6500.txt',
    }

    if split not in split_suffix:
        raise ValueError(f"split must be one of {list(split_suffix.keys())}")

    dict_file = os.path.join(DATA_DIR, 'muse_dictionaries', split_suffix[split])

    if not os.path.exists(dict_file):
        print(f"Dictionary not found: {dict_file}")
        print("Run alignment/prepare_dictionary.py first")
        return None

    bilingual_dict = {}
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # lowercase both sides to match preprocessed embeddings
                bilingual_dict[parts[0].lower()] = parts[1].lower()

    print(f"Loaded {len(bilingual_dict)} word pairs [{split}]")
    return bilingual_dict


def load_word_vectors(embedding_type):
    """
    Load French and Italian word vectors for a given embedding type

    Args:
        embedding_type (str): One of: tfidf, word2vec, fasttext, glove

    Returns:
        tuple: (fr_vectors, it_vectors)
    """
    fr_path = os.path.join('data', 'embeddings', embedding_type, f'french_{embedding_type}.pkl')
    it_path = os.path.join('data', 'embeddings', embedding_type, f'italian_{embedding_type}.pkl')

    fr_data = load_embeddings(fr_path)
    it_data = load_embeddings(it_path)

    fr_vectors = fr_data.get('word_vectors', fr_data.get('vectors'))
    it_vectors = it_data.get('word_vectors', it_data.get('vectors'))

    return fr_vectors, it_vectors


def create_output_dirs():
    """
    Create all required output directories
    """
    dirs = [
        os.path.join(DATA_DIR, 'embeddings', 'onehot'),
        os.path.join(DATA_DIR, 'embeddings', 'tfidf'),
        os.path.join(DATA_DIR, 'embeddings', 'word2vec'),
        os.path.join(DATA_DIR, 'embeddings', 'fasttext'),
        os.path.join(DATA_DIR, 'embeddings', 'glove'),
        os.path.join(DATA_DIR, 'muse_dictionaries'),
        os.path.join(DATA_DIR, 'visualizations'),
    ]

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            print(f"Created directory: {d}")

    print("Output directories created")


def get_vocabulary(corpus, min_freq=1):
    """
    Extract vocabulary from corpus

    Args:
        corpus (list): List of tokenized sentences
        min_freq (int): Minimum word frequency

    Returns:
        dict: word -> index mapping
    """
    word_counts = Counter()
    for sentence in corpus:
        word_counts.update(sentence)

    vocab = {word: idx for idx, (word, count)
             in enumerate(word_counts.items()) if count >= min_freq}

    return vocab