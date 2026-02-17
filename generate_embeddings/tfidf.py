"""
TF-IDF Vectorization for French-Italian corpus
"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from fr_ita.utils import load_tatoeba_tsv, preprocess_text, create_output_dirs, save_embeddings
from fr_ita.config import EMBEDDINGS_DIR


def generate_tfidf_vectors(corpus_processed, max_features=5000):
    """
    Args:
        corpus_processed (list): List of tokenized sentences
        max_features (int): Maximum vocabulary size
    
    Returns:
        tuple: (tfidf_matrix, vectorizer, feature_names, word_vectors)
    """
    print("Generating TF-IDF VECTORS")
    
    # back to strings for TfidfVectorizer
    corpus_strings = [' '.join(tokens) for tokens in corpus_processed]
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=False,
        token_pattern=r'\b\w+\b'
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus_strings)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(feature_names)}")
    
    # word vectors as mean TF-IDF across documents
    word_vectors = {}
    for idx, word in enumerate(feature_names):
        # Extract TF-IDF values for this word across all documents
        word_tfidf = tfidf_matrix[:, idx].toarray().flatten()
        # Use the full vector (not mean) for alignment
        word_vectors[word] = word_tfidf
    
    print(f"{len(word_vectors)} TF-IDF word vectors created.")
    print(f"Vector dimension: {len(word_tfidf)}")
    
    return tfidf_matrix, vectorizer, feature_names, word_vectors


if __name__ == "__main__":
    print(f"Embeddings will be saved to: {EMBEDDINGS_DIR}")
    
    create_output_dirs()
    
    # preprocess
    french_corpus, italian_corpus, df = load_tatoeba_tsv(max_sentences=10000)
    french_processed = preprocess_text(french_corpus, language='french')
    italian_processed = preprocess_text(italian_corpus, language='italian')
    
    # generation
    print("\nFRENCH TF-IDF")
    fr_tfidf_matrix, fr_vectorizer, fr_features, fr_word_vectors = generate_tfidf_vectors(french_processed)
    
    print("\nITALIAN TF-IDF")
    it_tfidf_matrix, it_vectorizer, it_features, it_word_vectors = generate_tfidf_vectors(italian_processed)
    
    # save
    fr_path = os.path.join('data', 'embeddings', 'tfidf', 'french_tfidf.pkl')
    it_path = os.path.join('data', 'embeddings', 'tfidf', 'italian_tfidf.pkl')
    
    save_embeddings({
        'matrix': fr_tfidf_matrix,
        'vectorizer': fr_vectorizer,
        'feature_names': fr_features,
        'word_vectors': fr_word_vectors
    }, fr_path)
    
    save_embeddings({
        'matrix': it_tfidf_matrix,
        'vectorizer': it_vectorizer,
        'feature_names': it_features,
        'word_vectors': it_word_vectors
    }, it_path)
    
    print(f"\nTF-IDF vectorization completed!")
    print(f"   French vocabulary size: {len(fr_features)}")
    print(f"   Italian vocabulary size: {len(it_features)}")