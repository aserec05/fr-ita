"""
viz_pca.py
PCA visualization with explained-variance inset and cosine-similarity colour coding.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from .viz_utils import FR_COLOR, IT_COLOR, EMB_COLORS, style_axes, legend_handles


def plot_pca(fr_words, it_words, fr_vecs, it_vecs, embedding_type,
             output_dir, save=True):
    """
    PCA visualization where links between translation pairs are coloured
    by cosine similarity (RdYlGn), with an explained-variance bar inset.

    Args:
        fr_words (list): French words
        it_words (list): Italian words
        fr_vecs (np.ndarray): French vectors
        it_vecs (np.ndarray): Italian vectors
        embedding_type (str): Embedding type name
        output_dir (str): Directory to save the figure
        save (bool): Save figure to file
    """
    print(f"Computing PCA for {embedding_type}...")

    all_vecs = np.vstack([fr_vecs, it_vecs])
    pca = PCA(n_components=10, random_state=42)
    pca.fit(all_vecs)
    coords    = pca.transform(all_vecs)[:, :2]
    explained = pca.explained_variance_ratio_

    n    = len(fr_words)
    fr_c = coords[:n]
    it_c = coords[n:]

    # per-pair cosine similarity for colour coding
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    sims      = np.array([cos_sim(fr_vecs[i], it_vecs[i]) for i in range(n)])
    norm_sims = (sims - sims.min()) / (sims.ptp() + 1e-9)
    cmap      = plt.cm.RdYlGn

    # ── layout ──
    fig = plt.figure(figsize=(16, 11))
    gs  = GridSpec(1, 1, figure=fig)
    ax  = fig.add_subplot(gs[0, 0])
    # small inset top-right
    ax_ins = ax.inset_axes([0.76, 0.72, 0.22, 0.25])

    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    # ── links coloured by cosine similarity ──
    for i in range(n):
        ax.plot([fr_c[i, 0], it_c[i, 0]],
                [fr_c[i, 1], it_c[i, 1]],
                color=cmap(norm_sims[i]), alpha=0.45,
                linewidth=0.9, zorder=1)

    # ── scatter ──
    ax.scatter(fr_c[:, 0], fr_c[:, 1],
               c=FR_COLOR, s=70, zorder=3, alpha=0.88,
               edgecolors='white', linewidths=0.5)
    ax.scatter(it_c[:, 0], it_c[:, 1],
               c=IT_COLOR, s=70, zorder=3, alpha=0.88,
               edgecolors='white', linewidths=0.5)

    # ── word labels ──
    for i, (fw, iw) in enumerate(zip(fr_words, it_words)):
        ax.annotate(fw, fr_c[i], fontsize=6.5, color=FR_COLOR,
                    xytext=(3, 3), textcoords='offset points', zorder=4)
        ax.annotate(iw, it_c[i], fontsize=6.5, color=IT_COLOR,
                    xytext=(3, 3), textcoords='offset points', zorder=4)

    # ── colorbar ──
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(sims.min(), sims.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.01, aspect=30)
    cbar.set_label('Cosine similarity (FR↔IT pair)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── inset: explained variance ──
    n_comp = min(10, len(explained))
    bars   = ax_ins.bar(range(1, n_comp + 1), explained[:n_comp] * 100,
                        color=EMB_COLORS.get(embedding_type, '#555'),
                        alpha=0.85, width=0.7)
    bars[0].set_color(FR_COLOR)
    bars[1].set_color(IT_COLOR)
    ax_ins.set_title('Explained var. (%)', fontsize=7, pad=4)
    ax_ins.set_xlabel('PC', fontsize=6)
    ax_ins.tick_params(axis='both', labelsize=6)
    ax_ins.set_facecolor('#F5F5F5')
    ax_ins.spines['top'].set_visible(False)
    ax_ins.spines['right'].set_visible(False)

    ax.legend(handles=legend_handles(), fontsize=9,
              framealpha=0.9, loc='lower left')

    style_axes(
        ax,
        f'PCA  ·  {embedding_type.upper()} Embeddings  (FR → IT)'
        f'\nPC1={explained[0]:.1%}   PC2={explained[1]:.1%}',
        f'PC1  ({explained[0]:.1%} variance)',
        f'PC2  ({explained[1]:.1%} variance)',
    )

    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, f'pca_{embedding_type}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved → {path}")

    plt.close()
