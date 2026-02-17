"""
visualization.py
Entry point — loads data and calls each visualization module.

Usage:
    python3 -m fr_ita.visualization.visualization --all
    python3 -m fr_ita.visualization.visualization --embedding word2vec
"""
import os
import argparse

from fr_ita.utils import load_word_vectors, load_bilingual_dictionary
from fr_ita.config import DATA_DIR

from .viz_utils import get_common_words
from .viz_tsne import plot_tsne
from .viz_pca import plot_pca
from .viz_alignment import plot_alignment_comparison
from .viz_radar import plot_radar_comparison


OUTPUT_DIR = os.path.join(DATA_DIR, 'visualizations')


def main():
    parser = argparse.ArgumentParser(description='Visualize word embeddings')
    parser.add_argument('--embedding', type=str, default='word2vec',
                        choices=['tfidf', 'word2vec', 'fasttext', 'glove'],
                        help='Embedding type to visualize (ignored when --all)')
    parser.add_argument('--all', action='store_true',
                        help='Run all embedding types + radar chart')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── load bilingual dictionary ──────────────────────────────────────────────
    bilingual_dict = load_bilingual_dictionary('fr', 'it')
    if bilingual_dict is None:
        exit(1)

    embedding_types = (
        ['word2vec', 'tfidf', 'fasttext', 'glove']
        if args.all else [args.embedding]
    )

    fr_vectors_all = {}
    it_vectors_all = {}

    # ── per-embedding t-SNE + PCA ──────────────────────────────────────────────
    for embedding_type in embedding_types:
        print(f"\n{'='*60}")
        print(f"  {embedding_type.upper()}")
        print(f"{'='*60}")

        try:
            fr_vectors, it_vectors = load_word_vectors(embedding_type)
            fr_vectors_all[embedding_type] = fr_vectors
            it_vectors_all[embedding_type] = it_vectors

            fr_words, it_words, fr_vecs, it_vecs = get_common_words(
                fr_vectors, it_vectors, bilingual_dict, n=50
            )
            print(f"  Word pairs for visualisation : {len(fr_words)}")

            plot_tsne(fr_words, it_words, fr_vecs, it_vecs,
                      embedding_type, OUTPUT_DIR)
            plot_pca(fr_words, it_words, fr_vecs, it_vecs,
                     embedding_type, OUTPUT_DIR)

        except Exception as e:
            print(f"  ⚠  Error for {embedding_type}: {e}")

    # ── alignment bar chart ────────────────────────────────────────────────────
    plot_alignment_comparison(OUTPUT_DIR)

    # ── radar chart (only when running --all with ≥2 types) ───────────────────
    if args.all and len(fr_vectors_all) >= 2:
        try:
            plot_radar_comparison(
                fr_vectors_all, it_vectors_all,
                bilingual_dict, OUTPUT_DIR
            )
        except Exception as e:
            print(f"  ⚠  Radar chart failed: {e}")

    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
