"""
FastText Embeddings for French-Italian corpus.

Two modes:
  1. Train from scratch on Tatoeba corpus (fast, lower quality)
  2. Load pre-trained Wikipedia FastText vectors (recommended by MUSE, higher quality)
     Download first: python -m fr_ita.alignment.prepare_dictionary --wiki-fasttext
"""
import os
import argparse
import numpy as np
from gensim.models import FastText
from fr_ita.utils import load_tatoeba_tsv, preprocess_text, create_output_dirs, save_embeddings
from fr_ita.config import EMBEDDINGS_DIR, DATA_DIR


def train_fasttext(corpus_processed, vector_size=100, window=5,
                   min_count=2, epochs=10, sg=0):
    """
    Train FastText from scratch on a tokenized corpus.

    Args:
        corpus_processed (list): List of tokenized sentences
        vector_size (int): Embedding dimension
        window (int): Context window size
        min_count (int): Minimum word frequency
        epochs (int): Training epochs
        sg (int): 0=CBOW, 1=Skip-gram

    Returns:
        FastText model
    """
    print("Training FastText model...")
    model = FastText(
        sentences=corpus_processed,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,
        epochs=epochs,
        min_n=3,
        max_n=6,
    )
    print(f"  Vocabulary size: {len(model.wv)}")
    return model


def extract_word_vectors(model):
    """Extract word -> vector dict from a trained gensim FastText model."""
    return {word: model.wv[word] for word in model.wv.index_to_key}


def load_wiki_vec_file(path, max_vocab=200_000):
    """
    Load a Wikipedia FastText .vec file (text format).
    Words are lowercased to match our preprocessing pipeline.

    Args:
        path (str): Path to wiki.XX.vec file
        max_vocab (int): Max words to load

    Returns:
        dict: word -> np.ndarray (float32, 300d)
    """
    word_vectors = {}
    print(f"Loading Wikipedia FastText vectors: {path}")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip().split()
        if len(first_line) == 2 and first_line[0].isdigit():
            n_words, dim = int(first_line[0]), int(first_line[1])
            print(f"  {n_words} words, dim={dim} (loading up to {max_vocab})")
        else:
            word = first_line[0].lower()
            vec  = np.array(first_line[1:], dtype=np.float32)
            word_vectors[word] = vec

        for line in f:
            if len(word_vectors) >= max_vocab:
                break
            parts = line.rstrip().split(' ')
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                word_vectors[word] = vec
            except ValueError:
                continue

    print(f"  Loaded {len(word_vectors)} vectors")
    return word_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate FastText embeddings')
    parser.add_argument('--mode', choices=['scratch', 'wiki'], default='scratch',
                        help='"scratch": train on Tatoeba | "wiki": load Wikipedia pre-trained')
    parser.add_argument('--max-vocab', type=int, default=200_000,
                        help='Max vocab for Wikipedia mode (default: 200000)')
    args = parser.parse_args()

    print(f"Embeddings will be saved to: {EMBEDDINGS_DIR}")
    create_output_dirs()

    if args.mode == 'wiki':
        # ── Wikipedia pre-trained mode (recommended for MUSE alignment) ──
        print("\nFASTTEXT — Wikipedia pre-trained mode")
        print("=" * 60)

        fr_vec_path = os.path.join(DATA_DIR, 'fasttext_wiki', 'wiki.fr.vec')
        it_vec_path = os.path.join(DATA_DIR, 'fasttext_wiki', 'wiki.it.vec')

        if not os.path.exists(fr_vec_path) or not os.path.exists(it_vec_path):
            print("Wikipedia FastText files not found.")
            print("Run: python -m fr_ita.alignment.prepare_dictionary --wiki-fasttext")
            exit(1)

        fr_word_vectors = load_wiki_vec_file(fr_vec_path, max_vocab=args.max_vocab)
        it_word_vectors = load_wiki_vec_file(it_vec_path, max_vocab=args.max_vocab)

        embedding_key = 'fasttext_wiki'

    else:
        # ── Train from scratch on Tatoeba ──
        print("\nFASTTEXT — train from scratch on Tatoeba")
        print("=" * 60)
        print("Note: for better MUSE alignment results, use --mode wiki")

        french_corpus, italian_corpus, _ = load_tatoeba_tsv(max_sentences=10000)
        french_processed  = preprocess_text(french_corpus,  language='french')
        italian_processed = preprocess_text(italian_corpus, language='italian')

        print("\nFRENCH FASTTEXT")
        fr_model = train_fasttext(french_processed, vector_size=100, window=5,
                                  min_count=2, epochs=10, sg=0)
        fr_word_vectors = extract_word_vectors(fr_model)

        print("\nITALIAN FASTTEXT")
        it_model = train_fasttext(italian_processed, vector_size=100, window=5,
                                  min_count=2, epochs=10, sg=0)
        it_word_vectors = extract_word_vectors(it_model)

        # save gensim models
        fr_model.save(os.path.join(EMBEDDINGS_DIR, 'fasttext', 'french_fasttext.model'))
        it_model.save(os.path.join(EMBEDDINGS_DIR, 'fasttext', 'italian_fasttext.model'))

        embedding_key = 'fasttext'

    # ── save word vectors as .pkl ──
    fr_path = os.path.join('data', 'embeddings', 'fasttext', f'french_{embedding_key}.pkl')
    it_path = os.path.join('data', 'embeddings', 'fasttext', f'italian_{embedding_key}.pkl')

    sample_dim = next(iter(fr_word_vectors.values())).shape[0]

    save_embeddings({
        'word_vectors': fr_word_vectors,
        'vocab':        list(fr_word_vectors.keys()),
        'vector_size':  sample_dim,
        'source':       args.mode,
    }, fr_path)

    save_embeddings({
        'word_vectors': it_word_vectors,
        'vocab':        list(it_word_vectors.keys()),
        'vector_size':  sample_dim,
        'source':       args.mode,
    }, it_path)

    print(f"\nFastText embeddings saved!")
    print(f"  French  : {len(fr_word_vectors)} vectors  ({sample_dim}d)  -> {fr_path}")
    print(f"  Italian : {len(it_word_vectors)} vectors  ({sample_dim}d)  -> {it_path}")