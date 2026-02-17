"""
viz_tsne.py
t-SNE visualization of French and Italian word embeddings.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from .viz_utils import FR_COLOR, IT_COLOR, LINK_COLOR, ACCENT, style_axes, legend_handles


def plot_tsne(fr_words, it_words, fr_vecs, it_vecs, embedding_type,
              output_dir, save=True):
    """
    t-SNE visualization with highlighted top-5 closest translation pairs
    and an info box showing distance statistics.

    Args:
        fr_words (list): French words
        it_words (list): Italian words
        fr_vecs (np.ndarray): French vectors
        it_vecs (np.ndarray): Italian vectors
        embedding_type (str): Embedding type name
        output_dir (str): Directory to save the figure
        save (bool): Save figure to file
    """
    print(f"Computing t-SNE for {embedding_type}...")

    all_vecs   = np.vstack([fr_vecs, it_vecs])
    perplexity = min(30, len(fr_words) - 1)

    tsne   = TSNE(n_components=2, perplexity=perplexity,
                  random_state=42, max_iter=1500, init='pca')
    coords = tsne.fit_transform(all_vecs)

    n    = len(fr_words)
    fr_c = coords[:n]
    it_c = coords[n:]

    # highlight the 5 closest pairs in t-SNE space
    distances = np.linalg.norm(fr_c - it_c, axis=1)
    top5_idx  = np.argsort(distances)[:5]

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # ── links ──
    for i in range(n):
        lw    = 1.4 if i in top5_idx else 0.7
        alpha = 0.65 if i in top5_idx else 0.25
        color = ACCENT if i in top5_idx else LINK_COLOR
        ax.plot([fr_c[i, 0], it_c[i, 0]],
                [fr_c[i, 1], it_c[i, 1]],
                color=color, alpha=alpha, linewidth=lw, zorder=1)

    # ── scatter ──
    ax.scatter(fr_c[:, 0], fr_c[:, 1],
               c=FR_COLOR, s=70, zorder=3, alpha=0.88,
               edgecolors='white', linewidths=0.5)
    ax.scatter(it_c[:, 0], it_c[:, 1],
               c=IT_COLOR, s=70, zorder=3, alpha=0.88,
               edgecolors='white', linewidths=0.5)

    # ── word labels ──
    for i, (fw, iw) in enumerate(zip(fr_words, it_words)):
        fontsize = 8.5 if i in top5_idx else 6.5
        weight   = 'bold' if i in top5_idx else 'normal'
        ax.annotate(fw, fr_c[i], fontsize=fontsize, color=FR_COLOR,
                    fontweight=weight, xytext=(4, 4),
                    textcoords='offset points', zorder=4)
        ax.annotate(iw, it_c[i], fontsize=fontsize, color=IT_COLOR,
                    fontweight=weight, xytext=(4, 4),
                    textcoords='offset points', zorder=4)

    # ── info box ──
    info_txt = (
        f"Pairs visualised : {n}\n"
        f"Mean pair dist.  : {distances.mean():.2f}\n"
        f"Best match       : {fr_words[top5_idx[0]]} ↔ {it_words[top5_idx[0]]}"
    )
    ax.text(0.02, 0.97, info_txt, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    # ── legend ──
    top5_patch = Line2D([0], [0], color=ACCENT, linewidth=1.4,
                        label='Top-5 closest pairs')
    ax.legend(handles=legend_handles([top5_patch]),
              fontsize=9, framealpha=0.9, loc='lower right')

    style_axes(ax,
               f't-SNE  ·  {embedding_type.upper()} Embeddings  (FR → IT)',
               't-SNE Dimension 1', 't-SNE Dimension 2')

    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, f'tsne_{embedding_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved → {path}")

    plt.close()
