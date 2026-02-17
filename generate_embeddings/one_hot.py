import argparse
import pickle
import numpy as np
import os  
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec, FastText
from collections import Counter
from fr_ita.utils import load_tatoeba_tsv, preprocess_text, create_output_dirs, save_embeddings
from fr_ita.config import EMBEDDINGS_DIR  


def generate_onehot_encoding(corpus, vocab=None):
    """
    Generate one-hot encoding for a corpus
    
    Args:
        corpus (list): List of tokenized sentences
        vocab (dict): Optional vocabulary mapping word -> index
    
    Returns:
        tuple: (one_hot_vectors, vocabulary, word_to_index)
    """
    print("Generating ONE-HOT ENCODING")
    
    if vocab is None: # we build a vocab, if necessary
        all_words = [word for sentence in corpus for word in sentence]
        unique_words = sorted(set(all_words))
        vocab = {word: idx for idx, word in enumerate(unique_words)}
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # for each word in vocabulary, we create a vector
    one_hot_vectors = {}
    for word, idx in vocab.items():
        vector = np.zeros(vocab_size)
        vector[idx] = 1
        one_hot_vectors[word] = vector
    
    print(f"{len(one_hot_vectors)} one-hot vectors created.")
    
    return one_hot_vectors, vocab

if __name__ == "__main__":
    # Print debug info
    print(f"Embeddings will be saved to: {EMBEDDINGS_DIR}")
    
    create_output_dirs()  # This now creates directories in the project
    
    # preprocess
    french_corpus, italian_corpus, df = load_tatoeba_tsv(max_sentences=10000)
    french_processed = preprocess_text(french_corpus, language='french')
    italian_processed = preprocess_text(italian_corpus, language='italian')
    
    # generation
    fr_onehot, fr_vocab = generate_onehot_encoding(french_processed)
    it_onehot, it_vocab = generate_onehot_encoding(italian_processed)
    
    # Use paths relative to project root
    fr_path = os.path.join('data', 'embeddings', 'onehot', 'french_onehot.pkl')
    it_path = os.path.join('data', 'embeddings', 'onehot', 'italian_onehot.pkl')
    
    print(f"Saving French embeddings to: {fr_path}")
    print(f"Saving Italian embeddings to: {it_path}")
    
    save_embeddings({
        'vectors': fr_onehot,
        'vocab': fr_vocab
    }, fr_path)
    
    save_embeddings({
        'vectors': it_onehot,
        'vocab': it_vocab
    }, it_path)
    
    print(f"\nOne-hot encoding completed!")
    print(f"   French vocab size: {len(fr_vocab)}")
    print(f"   Italian vocab size: {len(it_vocab)}")