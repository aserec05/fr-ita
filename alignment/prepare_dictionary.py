"""
Procrustes Alignment — FIXED
Align French and Italian embeddings using supervised Procrustes.

Key fixes vs v1:
  - CRITICAL: corrected Procrustes formula (X.T @ Y, not Y.T @ X)
  - L2-normalization before alignment (required by MUSE)
  - Separate train / test splits (no train/test leak)
  - Iterative refinement (5 iterations)
  - CSLS nearest-neighbour search for P@1 (fixes hubness problem)
  - Diagnostic: evaluate on TRAIN pairs as sanity check
"""
import os
import argparse
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import TruncatedSVD
from fr_ita.utils import load_embeddings, save_embeddings, load_bilingual_dictionary
from fr_ita.config import EMBEDDINGS_DIR, DATA_DIR


# ── Vector utilities ────────────────────────────────────────────────────────────

def normalize_vectors(vectors):
    """
    L2-normalize all vectors.
    Required before Procrustes so that norms don't bias the rotation.
    """
    return {w: v / (np.linalg.norm(v) + 1e-9) for w, v in vectors.items()}


def mean_center(vectors):
    """
    Subtract the mean vector from all embeddings.
    Helps when the two language spaces have different centroids.
    """
    all_vecs = np.array(list(vectors.values()))
    mean     = all_vecs.mean(axis=0)
    return {w: v - mean for w, v in vectors.items()}


def reduce_all_vectors(fr_vectors, it_vectors, max_dim=100):
    """
    TruncatedSVD dimensionality reduction (only for high-dim embeddings like TF-IDF).
    Fitted on combined corpus for consistency.
    """
    sample_dim = next(iter(fr_vectors.values())).shape[0]
    print(f"Reducing vectors: {sample_dim}d → {max_dim}d...")

    fr_words = list(fr_vectors.keys())
    it_words = list(it_vectors.keys())
    fr_arr   = np.array([fr_vectors[w] for w in fr_words])
    it_arr   = np.array([it_vectors[w] for w in it_words])

    svd = TruncatedSVD(n_components=max_dim, random_state=42)
    svd.fit(np.vstack([fr_arr, it_arr]))

    fr_reduced = svd.transform(fr_arr)
    it_reduced = svd.transform(it_arr)

    print(f"  FR={fr_reduced.shape}, IT={it_reduced.shape}")
    return (
        {w: fr_reduced[i] for i, w in enumerate(fr_words)},
        {w: it_reduced[i] for i, w in enumerate(it_words)},
    )


def load_vec_file(path, max_vocab=200_000):
    """
    Load a .vec text file (fastText Wikipedia format).
    Words are lowercased to match our preprocessing pipeline.
    """
    word_vectors = {}
    print(f"Loading .vec file: {path}")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip().split()
        if len(first_line) == 2 and first_line[0].isdigit():
            print(f"  {first_line[0]} words, dim={first_line[1]}")
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
                word_vectors[word] = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue

    print(f"  Loaded {len(word_vectors)} vectors")
    return word_vectors


# ── Procrustes ──────────────────────────────────────────────────────────────────

def compute_procrustes_matrix(source_vectors, target_vectors, bilingual_dict, max_pairs=5000):
    """
    Compute orthogonal W such that source @ W ≈ target.

    CORRECT formula: SVD of X.T @ Y (source.T @ target), then W = U @ Vh
    Uses scipy.linalg.orthogonal_procrustes as reference implementation.

    Args:
        source_vectors (dict): Normalized source word vectors
        target_vectors (dict): Normalized target word vectors
        bilingual_dict (dict): TRAIN bilingual dictionary (anchor pairs)
        max_pairs (int): Max anchor pairs to use

    Returns:
        np.ndarray: (d, d) orthogonal alignment matrix
    """
    pairs = [
        (s, t) for s, t in bilingual_dict.items()
        if s in source_vectors and t in target_vectors
    ][:max_pairs]

    if len(pairs) == 0:
        raise ValueError("No valid anchor pairs found. Check vocabulary overlap.")

    print(f"  Anchor pairs: {len(pairs)} / {len(bilingual_dict)}")

    X = np.array([source_vectors[s] for s, _ in pairs], dtype=np.float64)  # (n, d)
    Y = np.array([target_vectors[t] for _, t in pairs], dtype=np.float64)  # (n, d)

    # scipy implementation is tested, correct, and numerically stable
    W, _ = orthogonal_procrustes(X, Y)   # W minimizes ||X @ W - Y||_F
    return W


def iterative_procrustes(source_vectors, target_vectors, bilingual_dict,
                         n_iter=5, max_pairs=5000):
    """
    Iterative Procrustes refinement.
    After each iteration, drop the lowest-similarity 25% anchor pairs
    to remove noisy translations from the anchor set.

    Args:
        source_vectors (dict): Normalized source vectors
        target_vectors (dict): Normalized target vectors
        bilingual_dict (dict): TRAIN bilingual dictionary
        n_iter (int): Number of refinement iterations
        max_pairs (int): Max anchor pairs

    Returns:
        np.ndarray: Final (d, d) alignment matrix
    """
    print(f"Iterative Procrustes ({n_iter} iterations)...")

    current_dict = bilingual_dict

    for i in range(n_iter):
        print(f"\n  Iteration {i + 1}/{n_iter}")
        W = compute_procrustes_matrix(source_vectors, target_vectors,
                                      current_dict, max_pairs)

        if i < n_iter - 1:
            # compute cosine sim for current anchors, drop bottom 25%
            pairs = [(s, t) for s, t in current_dict.items()
                     if s in source_vectors and t in target_vectors][:max_pairs]

            X_pairs = np.array([source_vectors[s] for s, _ in pairs], dtype=np.float64)
            Y_pairs = np.array([target_vectors[t] for _, t in pairs], dtype=np.float64)

            aligned = X_pairs @ W
            sims    = np.sum(aligned * Y_pairs, axis=1)  # cosine (already normalized)

            threshold    = np.percentile(sims, 25)
            current_dict = {s: t for (s, t), ok in zip(pairs, sims >= threshold) if ok}

            print(f"    Avg sim on anchors: {sims.mean():.4f}  "
                  f"→ keeping {len(current_dict)} pairs (dropped bottom 25%)")

    return W


def align_embeddings(source_vectors, alignment_matrix):
    """Apply alignment matrix to all source vectors."""
    return {word: vec @ alignment_matrix for word, vec in source_vectors.items()}


# ── CSLS nearest-neighbour search ───────────────────────────────────────────────

def csls_neighbors(src_vecs, tgt_mat, k_csls=10):
    """
    CSLS (Cross-domain Similarity Local Scaling) score:
      CSLS(x, y) = 2 cos(x, y) - r_T(x) - r_S(y)

    where r_T(x) = mean of top-k cosine similarities of x to its T-neighbours,
    and   r_S(y) = mean of top-k cosine similarities of y to its S-neighbours.

    CSLS reduces hubness (words that are NN of many query vectors) and
    dramatically improves word translation accuracy vs raw cosine sim.

    Args:
        src_vecs (np.ndarray): (n, d) aligned source query vectors
        tgt_mat  (np.ndarray): (V, d) target word matrix (normalized)
        k_csls   (int): number of neighbours for CSLS smoothing

    Returns:
        np.ndarray: (n, V) CSLS scores
    """
    # cosine similarities (vectors already normalized)
    cos = src_vecs @ tgt_mat.T   # (n, V)

    # r_T(x): mean of top-k target similarities for each source query
    top_src = np.sort(cos, axis=1)[:, -k_csls:]  # (n, k)
    r_T     = top_src.mean(axis=1, keepdims=True)  # (n, 1)

    # r_S(y): mean of top-k source similarities for each target word
    top_tgt = np.sort(cos, axis=0)[-k_csls:, :]   # (k, V)
    r_S     = top_tgt.mean(axis=0, keepdims=True)  # (1, V)

    return 2 * cos - r_T - r_S


# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate_alignment(aligned_source, target_vectors, eval_dict,
                       topk=None, label='eval', use_csls=True):
    """
    Evaluate alignment on a given dictionary split.

    Metrics:
      - Average cosine similarity (translation pair quality)
      - Precision@1 with CSLS nearest-neighbour search

    Args:
        aligned_source (dict): Aligned source vectors
        target_vectors (dict): Target vectors
        eval_dict (dict): Bilingual dictionary for evaluation
        topk (int|None): Max pairs; None = use all
        label (str): Label for printing
        use_csls (bool): Use CSLS for NN search (recommended)

    Returns:
        tuple: (avg_cosine_sim, precision_at_1)
    """
    pairs = [
        (s, t) for s, t in eval_dict.items()
        if s in aligned_source and t in target_vectors
    ]
    if topk:
        pairs = pairs[:topk]

    if not pairs:
        print(f"  [{label}] No valid pairs found")
        return 0.0, 0.0

    print(f"\n[{label}] Evaluating on {len(pairs)} pairs...")

    tgt_words = list(target_vectors.keys())
    tgt_mat   = np.array([target_vectors[w] for w in tgt_words], dtype=np.float64)
    tgt_index = {w: i for i, w in enumerate(tgt_words)}

    # ── cosine similarity (direct pairs) ──
    sims = np.array([
        np.dot(aligned_source[s], target_vectors[t])
        for s, t in pairs
    ])
    avg_sim = float(np.mean(sims))

    # ── Precision@1 ──
    src_mat = np.array([aligned_source[s] for s, _ in pairs], dtype=np.float64)

    if use_csls:
        scores = csls_neighbors(src_mat, tgt_mat, k_csls=10)  # (n, V)
    else:
        scores = src_mat @ tgt_mat.T  # (n, V) cosine sim

    pred_indices = np.argmax(scores, axis=1)
    hits = sum(
        tgt_words[pred_indices[i]] == pairs[i][1]
        for i in range(len(pairs))
    )
    p_at_1 = hits / len(pairs)

    print(f"  Avg cosine sim  : {avg_sim:.4f}")
    print(f"  Std             : {sims.std():.4f}  "
          f"[min={sims.min():.4f}, max={sims.max():.4f}]")
    nn_method = "CSLS" if use_csls else "cosine"
    print(f"  Precision@1 ({nn_method}): {p_at_1:.4f}  ({hits}/{len(pairs)})")

    # show examples
    print(f"\n  Sample (src → predicted  |  true):")
    for i in range(min(8, len(pairs))):
        s, t = pairs[i]
        pred = tgt_words[pred_indices[i]]
        ok   = "✓" if pred == t else "✗"
        print(f"    {s:18s} → {pred:18s}  [{ok} true: {t}]")

    return avg_sim, p_at_1


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align embeddings using Procrustes')
    parser.add_argument('--embedding', type=str, default='word2vec',
                        choices=['tfidf', 'word2vec', 'fasttext', 'glove'],
                        help='Embedding type to align')
    parser.add_argument('--wiki-fasttext', action='store_true',
                        help='Use pre-trained Wikipedia FastText .vec files')
    parser.add_argument('--n-iter', type=int, default=5,
                        help='Number of iterative Procrustes refinements (default: 5)')
    parser.add_argument('--max-vocab', type=int, default=200_000,
                        help='Max vocab when loading .vec files (default: 200000)')
    parser.add_argument('--center', action='store_true',
                        help='Apply mean-centering before alignment')
    args = parser.parse_args()

    print("PROCRUSTES ALIGNMENT (fixed)")
    print("=" * 60)

    # ── load TRAIN dict (for alignment) and TEST dict (for evaluation) ──
    train_dict = load_bilingual_dictionary('fr', 'it', split='train')
    test_dict  = load_bilingual_dictionary('fr', 'it', split='test')
    if train_dict is None or test_dict is None:
        print("Run prepare_dictionary.py first.")
        exit(1)

    embedding_type = args.embedding

    # ── load vectors ──
    if args.wiki_fasttext:
        print("\nUsing pre-trained Wikipedia FastText embeddings...")
        fr_path = os.path.join(DATA_DIR, 'fasttext_wiki', 'wiki.fr.vec')
        it_path = os.path.join(DATA_DIR, 'fasttext_wiki', 'wiki.it.vec')
        if not os.path.exists(fr_path) or not os.path.exists(it_path):
            print("Wikipedia FastText not found. Run: prepare_dictionary.py --wiki-fasttext")
            exit(1)
        fr_vectors = load_vec_file(fr_path, max_vocab=args.max_vocab)
        it_vectors = load_vec_file(it_path, max_vocab=args.max_vocab)
        embedding_type = 'fasttext_wiki'

    else:
        fr_path = os.path.join('data', 'embeddings', embedding_type,
                               f'french_{embedding_type}.pkl')
        it_path = os.path.join('data', 'embeddings', embedding_type,
                               f'italian_{embedding_type}.pkl')
        fr_data    = load_embeddings(fr_path)
        it_data    = load_embeddings(it_path)
        fr_vectors = fr_data.get('word_vectors', fr_data.get('vectors'))
        it_vectors = it_data.get('word_vectors', it_data.get('vectors'))
        if fr_vectors is None or it_vectors is None:
            print(f"Error: vectors not found. Keys: {fr_data.keys()}")
            exit(1)

    print(f"\nFrench vectors : {len(fr_vectors)}")
    print(f"Italian vectors: {len(it_vectors)}")

    sample_dim = next(iter(fr_vectors.values())).shape[0]
    print(f"Vector dim     : {sample_dim}")

    # ── optional dimensionality reduction (TF-IDF only) ──
    max_dim = 300 if embedding_type in ('glove', 'fasttext_wiki') else 100
    if sample_dim > max_dim:
        fr_vectors, it_vectors = reduce_all_vectors(fr_vectors, it_vectors, max_dim)

    # ── optional mean centering ──
    if args.center:
        print("Mean-centering vectors...")
        fr_vectors = mean_center(fr_vectors)
        it_vectors = mean_center(it_vectors)

    # ── L2 normalization ──
    print("L2-normalizing vectors...")
    fr_vectors = normalize_vectors(fr_vectors)
    it_vectors = normalize_vectors(it_vectors)

    # ── iterative Procrustes on TRAIN dict ──
    print(f"\nComputing alignment (n_iter={args.n_iter})...")
    alignment_matrix = iterative_procrustes(
        fr_vectors, it_vectors, train_dict,
        n_iter=args.n_iter
    )

    # ── apply alignment ──
    fr_aligned = align_embeddings(fr_vectors, alignment_matrix)

    # ── sanity check: evaluate on TRAIN pairs ──
    avg_train, p1_train = evaluate_alignment(
        fr_aligned, it_vectors, train_dict,
        topk=500, label='TRAIN (sanity check)', use_csls=True
    )

    # ── main evaluation on TEST dict ──
    avg_test, p1_test = evaluate_alignment(
        fr_aligned, it_vectors, test_dict,
        topk=None, label='TEST (held-out)', use_csls=True
    )

    # ── save ──
    aligned_path = os.path.join(
        'data', 'embeddings', embedding_type,
        f'french_{embedding_type}_aligned.pkl'
    )
    save_embeddings({
        'word_vectors':     fr_aligned,
        'vocab':            list(fr_aligned.keys()),
        'alignment_matrix': alignment_matrix,
        'avg_similarity':   avg_test,
        'precision_at_1':   p1_test,
    }, aligned_path)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS  [{embedding_type}]")
    print(f"  TRAIN avg cosine sim : {avg_train:.4f}  P@1: {p1_train:.4f}")
    print(f"  TEST  avg cosine sim : {avg_test:.4f}  P@1: {p1_test:.4f}")
    print(f"  Saved to             : {aligned_path}")