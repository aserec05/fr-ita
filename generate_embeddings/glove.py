"""
GloVe-style Embeddings for French-Italian corpus
Load FastText .vec files and extract vocabulary-specific embeddings
"""
import os
import numpy as np
from fr_ita.utils import load_tatoeba_tsv, preprocess_text, create_output_dirs, save_embeddings
from fr_ita.config import EMBEDDINGS_DIR, DATA_DIR


def load_vec_embeddings(vec_file, vocab, max_vectors=None):
    """
    Load FastText .vec file and extract only vocabulary words
    
    Args:
        vec_file (str): Path to .vec file
        vocab (set): Vocabulary to extract
        max_vectors (int): Max vectors to read (None = all)
    
    Returns:
        dict: word -> vector mapping
    """
    print(f"Loading embeddings from {vec_file}")
    print(f"Target vocabulary size: {len(vocab)}")
    
    embeddings = {}
    found = 0
    
    with open(vec_file, 'r', encoding='utf-8') as f:
        # skip first line (metadata)
        n_words, dim = map(int, f.readline().split())
        print(f"File contains {n_words} words, dimension {dim}")
        
        for i, line in enumerate(f):
            if max_vectors and i >= max_vectors:
                break
            
            parts = line.rstrip().split(' ')
            word = parts[0]
            
            # only load if in vocabulary
            if word in vocab:
                vector = np.array(parts[1:], dtype='float32')
                embeddings[word] = vector
                found += 1
            
            # progress
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i+1} vectors, found {found} matches")
        
        print(f"Loaded {found} / {len(vocab)} words from vocabulary")
    
    return embeddings


def get_corpus_vocabulary(corpus_processed):
    """
    Args:
        corpus_processed (list): Tokenized sentences
    
    Returns:
        set: Unique words
    """
    vocab = set()
    for sentence in corpus_processed:
        vocab.update(sentence)
    return vocab


if __name__ == "__main__":
    print(f"Embeddings will be saved to: {EMBEDDINGS_DIR}")
    
    create_output_dirs()
    
    # preprocess
    french_corpus, italian_corpus, df = load_tatoeba_tsv(max_sentences=10000)
    french_processed = preprocess_text(french_corpus, language='french')
    italian_processed = preprocess_text(italian_corpus, language='italian')
    
    # get vocabularies
    fr_vocab = get_corpus_vocabulary(french_processed)
    it_vocab = get_corpus_vocabulary(italian_processed)
    
    # paths to .vec files
    models_dir = os.path.join(DATA_DIR, 'fasttext_models')
    fr_vec_file = os.path.join(models_dir, 'cc.fr.300.vec')
    it_vec_file = os.path.join(models_dir, 'cc.it.300.vec')
    
    # check files exist
    if not os.path.exists(fr_vec_file):
        print(f"French .vec file not found: {fr_vec_file}")
        print("Download from: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz")
        exit(1)
    
    if not os.path.exists(it_vec_file):
        print(f"Italian .vec file not found: {it_vec_file}")
        print("Download from: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.it.300.vec.gz")
        exit(1)
    
    # load embeddings (only for vocabulary words)
    print("\nFRENCH EMBEDDINGS")
    fr_word_vectors = load_vec_embeddings(fr_vec_file, fr_vocab)
    
    print("\nITALIAN EMBEDDINGS")
    it_word_vectors = load_vec_embeddings(it_vec_file, it_vocab)
    
    # save
    fr_path = os.path.join('data', 'embeddings', 'glove', 'french_glove.pkl')
    it_path = os.path.join('data', 'embeddings', 'glove', 'italian_glove.pkl')
    
    if fr_word_vectors:
        vector_size = len(next(iter(fr_word_vectors.values())))
        save_embeddings({
            'word_vectors': fr_word_vectors,
            'vocab': list(fr_word_vectors.keys()),
            'vector_size': vector_size
        }, fr_path)
        print(f"\nFrench saved: {len(fr_word_vectors)} words, dimension {vector_size}")
    
    if it_word_vectors:
        vector_size = len(next(iter(it_word_vectors.values())))
        save_embeddings({
            'word_vectors': it_word_vectors,
            'vocab': list(it_word_vectors.keys()),
            'vector_size': vector_size
        }, it_path)
        print(f"Italian saved: {len(it_word_vectors)} words, dimension {vector_size}")
    
    print(f"\nEmbeddings generation completed!")