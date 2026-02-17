"""
Feature extraction for Language Identification (FR vs IT)
"""
import numpy as np


def sentence_to_vector(tokens, word_vectors, strategy='mean'):
    """
    Convert a tokenized sentence to a fixed-size vector.

    Args:
        tokens (list): Tokenized sentence
        word_vectors (dict): word -> vector mapping
        strategy (str): 'mean' or 'max'

    Returns:
        np.ndarray or None
    """
    vecs = [word_vectors[t] for t in tokens if t in word_vectors]

    if not vecs:
        return None

    vecs = np.array(vecs)

    if strategy == 'max':
        return vecs.max(axis=0)

    return vecs.mean(axis=0)


def build_features(fr_corpus, it_corpus, word_vectors, strategy='mean'):
    """
    Build feature matrix and labels from FR and IT corpora.

    Args:
        fr_corpus (list): Tokenized French sentences
        it_corpus (list): Tokenized Italian sentences
        word_vectors (dict): Combined word -> vector mapping
        strategy (str): 'mean' or 'max'

    Returns:
        tuple: (X, y)
    """
    X, y = [], []
    skipped = 0

    for tokens in fr_corpus:
        vec = sentence_to_vector(tokens, word_vectors, strategy)
        if vec is not None:
            X.append(vec)
            y.append(0)  # 0 = French
        else:
            skipped += 1

    for tokens in it_corpus:
        vec = sentence_to_vector(tokens, word_vectors, strategy)
        if vec is not None:
            X.append(vec)
            y.append(1)  # 1 = Italian
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} sentences (no vocabulary overlap)")

    return np.array(X), np.array(y)