"""
Word2Vec Embeddings for French-Italian corpus
"""
import os
from gensim.models import Word2Vec
from fr_ita.utils import load_tatoeba_tsv, preprocess_text, create_output_dirs, save_embeddings
from fr_ita.config import EMBEDDINGS_DIR


def train_word2vec(corpus_processed, vector_size=100, window=5, min_count=2, epochs=10, sg=0):
    """
    Args:
        corpus_processed (list): List of tokenized sentences
        vector_size (int): Dimensionality of word vectors
        window (int): Context window size
        min_count (int): Minimum word frequency
        epochs (int): Number of training epochs
        sg (int): 0=CBOW, 1=Skip-gram
    
    Returns:
        Word2Vec model
    """
    print("Training Word2Vec model...")
    
    model = Word2Vec(
        sentences=corpus_processed,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,
        epochs=epochs
    )
    
    print(f"Vocabulary size: {len(model.wv)}")
    
    return model


def extract_word_vectors(model):
    """
    Args:
        model: Trained Word2Vec model
    
    Returns:
        dict: word -> vector mapping
    """
    word_vectors = {}
    for word in model.wv.index_to_key:
        word_vectors[word] = model.wv[word]
    
    return word_vectors


if __name__ == "__main__":
    print(f"Embeddings will be saved to: {EMBEDDINGS_DIR}")
    
    create_output_dirs()
    
    # preprocess
    french_corpus, italian_corpus, df = load_tatoeba_tsv(max_sentences=10000)
    french_processed = preprocess_text(french_corpus, language='french')
    italian_processed = preprocess_text(italian_corpus, language='italian')
    
    # generation
    print("\nFRENCH WORD2VEC")
    fr_model = train_word2vec(french_processed, vector_size=100, window=5, min_count=2, epochs=10, sg=0)
    fr_word_vectors = extract_word_vectors(fr_model)
    
    print("\nITALIAN WORD2VEC")
    it_model = train_word2vec(italian_processed, vector_size=100, window=5, min_count=2, epochs=10, sg=0)
    it_word_vectors = extract_word_vectors(it_model)
    
    # save
    fr_model.save(os.path.join(EMBEDDINGS_DIR, 'word2vec', 'french_word2vec.model'))
    it_model.save(os.path.join(EMBEDDINGS_DIR, 'word2vec', 'italian_word2vec.model'))
    
    fr_path = os.path.join('data', 'embeddings', 'word2vec', 'french_word2vec.pkl')
    it_path = os.path.join('data', 'embeddings', 'word2vec', 'italian_word2vec.pkl')
    
    save_embeddings({
        'word_vectors': fr_word_vectors,
        'vocab': list(fr_model.wv.index_to_key),
        'vector_size': fr_model.wv.vector_size
    }, fr_path)
    
    save_embeddings({
        'word_vectors': it_word_vectors,
        'vocab': list(it_model.wv.index_to_key),
        'vector_size': it_model.wv.vector_size
    }, it_path)
    
    print(f"\nWord2Vec training completed!")
    print(f"   French vocabulary size: {len(fr_word_vectors)}")
    print(f"   Italian vocabulary size: {len(it_word_vectors)}")
