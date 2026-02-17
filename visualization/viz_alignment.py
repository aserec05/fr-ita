"""
viz_alignment.py
Bar chart comparing Procrustes alignment scores across embedding types.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from .viz_utils import EMB_COLORS
from fr_ita.utils import load_embeddings
from fr_ita.config import DATA_DIR


def plot_alignment_comparison(output_dir, save=True):
    """
    Polished bar chart comparing avg cosine similarity after Procrustes
    alignment for each embedding type. Includes a mean reference line
    and value labels above each bar.

    Args:
        output_dir (str): Directory to save the figure
        save (bool): Save figure to file
    """
    print("Plotting alignment comparison...")

    embedding_types = ['word2vec', 'tfidf', 'glove', 'fasttext']
    scores, labels  = [], []

    for emb_type in embedding_types:
        aligned_path = os.path.join(
            'data', 'embeddings', emb_type,
            f'french_{emb_type}_aligned.pkl'
        )
        full_path = os.path.join(DATA_DIR, '..', aligned_path)
        if os.path.exists(full_path):
            data = load_embeddings(full_path)
            scores.append(data.get('avg_similarity', 0))
            labels.append(emb_type)

    if not scores:
        print("  No aligned embeddings found — skipping comparison chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    colors = [EMB_COLORS.get(l, '#888') for l in labels]
    bars   = ax.bar(labels, scores, color=colors, alpha=0.88,
                    width=0.5, zorder=2,
                    edgecolor='white', linewidth=1.2)

    # value labels above bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f'{score:.4f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#333333')

    # mean reference line
    mean_score = np.mean(scores)
    ax.axhline(mean_score, color='#555', linewidth=1.0,
               linestyle='--', alpha=0.7, zorder=1,
               label=f'Mean = {mean_score:.4f}')

    ax.set_ylim(max(0, min(scores) - 0.15), min(1.0, max(scores) + 0.12))
    ax.set_xlabel('Embedding Type', fontsize=11, labelpad=8)
    ax.set_ylabel('Avg. Cosine Similarity  (aligned FR → IT)', fontsize=11, labelpad=8)
    ax.set_title(
        'Procrustes Alignment  ·  Cosine Similarity by Embedding Type',
        fontsize=14, fontweight='bold', pad=14, loc='left'
    )
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, 'alignment_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved → {path}")

    plt.close()
