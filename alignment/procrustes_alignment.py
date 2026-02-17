"""
Procrustes Alignment
Align French and Italian embeddings using Procrustes method
"""
import os
import argparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
from fr_ita.utils import load_embeddings, save_embeddings, load_bilingual_dictionary
from fr_ita.config import EMBEDDINGS_DIR, DATA_DIR


def reduce_all_vectors(fr_vectors, it_vectors, max_dim=100):
    """
    Fit SVD on full corpus and reduce all vectors consistently

    Args:
        fr_vectors (dict): French word vectors
        it_vectors (dict): Italian word vectors
        max_dim (int): Target dimension

    Returns:
        tuple: (fr_vectors_reduced, it_vectors_reduced)
    """
    sample_dim = next(iter(fr_vectors.values())).shape[0]
    print(f"Reducing all vectors from {sample_dim} to {max_dim}d...")

    fr_words = list(fr_vectors.keys())
    it_words = list(it_vectors.keys())

    fr_array = np.array([fr_vectors[w] for w in fr_words])
    it_array = np.array([it_vectors[w] for w in it_words])

    # fit on combined corpus for consistency
    all_vectors = np.vstack([fr_array, it_array])
    svd = TruncatedSVD(n_components=max_dim, random_state=42)
    svd.fit(all_vectors)

    fr_reduced = svd.transform(fr_array)
    it_reduced = svd.transform(it_array)

    fr_vectors_reduced = {w: fr_reduced[i] for i, w in enumerate(fr_words)}
    it_vectors_reduced = {w: it_reduced[i] for i, w in enumerate(it_words)}

    print(f"Reduced: French={fr_reduced.shape}, Italian={it_reduced.shape}")

    return fr_vectors_reduced, it_vectors_reduced


def compute_alignment_matrix(source_vectors, target_vectors, bilingual_dict, max_pairs=2000):
    """
    Compute Procrustes alignment matrix

    Args:
        source_vectors (dict): Source word vectors
        target_vectors (dict): Target word vectors
        bilingual_dict (dict): Bilingual dictionary
        max_pairs (int): Maximum number of word pairs to use

    Returns:
        np.ndarray: Alignment matrix
    """
    print("Computing alignment matrix...")

    pairs = []
    for src_word, tgt_word in bilingual_dict.items():
        if src_word in source_vectors and tgt_word in target_vectors:
            pairs.append((src_word, tgt_word))
            if len(pairs) >= max_pairs:
                break

    print(f"Valid pairs for alignment: {len(pairs)}")

    if len(pairs) == 0:
        print("No valid pairs found")
        return None

    X = np.array([source_vectors[src] for src, _ in pairs])
    Y = np.array([target_vectors[tgt] for _, tgt in pairs])

    print(f"Matrix shapes: X={X.shape}, Y={Y.shape}")

    print("Computing SVD...")
    U, _, Vt = np.linalg.svd(Y.T @ X)
    W = U @ Vt

    print("Alignment matrix computed")

    return W


def align_embeddings(source_vectors, alignment_matrix):
    """
    Apply alignment matrix to source embeddings

    Args:
        source_vectors (dict): Source word vectors
        alignment_matrix (np.ndarray): Alignment matrix

    Returns:
        dict: Aligned word vectors
    """
    aligned_vectors = {}
    for word, vector in source_vectors.items():
        aligned_vectors[word] = vector @ alignment_matrix

    return aligned_vectors


def cosine_similarity(v1, v2):
    """
    Cosine similarity between two vectors
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def evaluate_alignment(aligned_source, target_vectors, bilingual_dict, topk=100):
    """
    Evaluate alignment quality using cosine similarity

    Args:
        aligned_source (dict): Aligned source vectors
        target_vectors (dict): Target vectors
        bilingual_dict (dict): Bilingual dictionary
        topk (int): Number of pairs to evaluate

    Returns:
        float: Average cosine similarity
    """
    print(f"\nEvaluating alignment on {topk} pairs...")

    similarities = []
    count = 0

    for src_word, tgt_word in bilingual_dict.items():
        if count >= topk:
            break

        if src_word in aligned_source and tgt_word in target_vectors:
            sim = cosine_similarity(aligned_source[src_word], target_vectors[tgt_word])
            similarities.append(sim)
            count += 1

            if count <= 10:
                print(f"  {src_word} -> {tgt_word}: {sim:.4f}")

    if not similarities:
        return 0.0

    avg_sim = np.mean(similarities)

    print(f"\nAverage cosine similarity: {avg_sim:.4f}")
    print(f"Min: {np.min(similarities):.4f}, Max: {np.max(similarities):.4f}")

    return avg_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align embeddings using Procrustes')
    parser.add_argument('--embedding', type=str, default='word2vec',
                        choices=['tfidf', 'word2vec', 'fasttext', 'glove'],
                        help='Embedding type to align')
    args = parser.parse_args()

    print("PROCRUSTES ALIGNMENT")
    print("="*60)

    # load dictionary via shared utils
    bilingual_dict = load_bilingual_dictionary('fr', 'it')
    if bilingual_dict is None:
        exit(1)

    embedding_type = args.embedding
    print(f"\nAligning {embedding_type} embeddings...")

    # load embeddings
    fr_path = os.path.join('data', 'embeddings', embedding_type, f'french_{embedding_type}.pkl')
    it_path = os.path.join('data', 'embeddings', embedding_type, f'italian_{embedding_type}.pkl')

    fr_data = load_embeddings(fr_path)
    it_data = load_embeddings(it_path)

    fr_vectors = fr_data.get('word_vectors', fr_data.get('vectors'))
    it_vectors = it_data.get('word_vectors', it_data.get('vectors'))

    if fr_vectors is None or it_vectors is None:
        print(f"Error: Cannot find vectors. Keys: {fr_data.keys()}")
        exit(1)

    print(f"French vectors: {len(fr_vectors)}")
    print(f"Italian vectors: {len(it_vectors)}")

    # check vector dimension
    sample_dim = next(iter(fr_vectors.values())).shape[0]
    print(f"Vector dimension: {sample_dim}")

    # for glove (pre-trained 300d), keep full dimension
    max_dim = 300 if embedding_type == 'glove' else 100

    if sample_dim > max_dim:
        fr_vectors, it_vectors = reduce_all_vectors(fr_vectors, it_vectors, max_dim=max_dim)

    # compute alignment matrix
    alignment_matrix = compute_alignment_matrix(fr_vectors, it_vectors, bilingual_dict)

    if alignment_matrix is None:
        print("Alignment failed")
        exit(1)

    # align
    fr_aligned = align_embeddings(fr_vectors, alignment_matrix)

    # evaluate
    avg_similarity = evaluate_alignment(fr_aligned, it_vectors, bilingual_dict, topk=100)

    # save
    aligned_path = os.path.join('data', 'embeddings', embedding_type, f'french_{embedding_type}_aligned.pkl')

    save_embeddings({
        'word_vectors': fr_aligned,
        'vocab': list(fr_aligned.keys()),
        'alignment_matrix': alignment_matrix,
        'avg_similarity': avg_similarity
    }, aligned_path)

    print(f"\nAlignment completed!")
    print(f"Saved to: {aligned_path}")