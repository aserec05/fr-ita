"""
viz_utils.py
Shared constants, colour palette, and helper functions for all visualizations.
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Matplotlib global style ────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Georgia', 'Times New Roman'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.6,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})

# ── Colour palette ─────────────────────────────────────────────────────────────
FR_COLOR   = '#1A6FBF'   # rich blue  → French
IT_COLOR   = '#C0392B'   # deep red   → Italian
LINK_COLOR = '#95A5A6'   # slate grey → translation links
ACCENT     = '#F39C12'   # amber      → highlights

EMB_COLORS = {
    'word2vec': '#2980B9',
    'tfidf':    '#27AE60',
    'glove':    '#8E44AD',
    'fasttext': '#E67E22',
}


def style_axes(ax, title, xlabel, ylabel):
    """Apply consistent axis styling."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14, loc='left')
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=8)
    ax.tick_params(axis='both', labelsize=8)


def legend_handles(extra=None):
    """Standard FR / IT legend handles."""
    handles = [
        mpatches.Patch(color=FR_COLOR, label='French'),
        mpatches.Patch(color=IT_COLOR, label='Italian'),
        Line2D([0], [0], color=LINK_COLOR, linewidth=0.9,
               linestyle='-', label='Translation pair'),
    ]
    if extra:
        handles += extra
    return handles


def get_common_words(fr_vectors, it_vectors, bilingual_dict, n=50):
    """
    Get translation pairs that exist in both vocabularies.

    Args:
        fr_vectors (dict): French word vectors
        it_vectors (dict): Italian word vectors
        bilingual_dict (dict): FR-IT bilingual dictionary
        n (int): Number of pairs to return

    Returns:
        tuple: (fr_words, it_words, fr_vecs, it_vecs)
    """
    fr_words, it_words, fr_vecs, it_vecs = [], [], [], []

    for fr_word, it_word in bilingual_dict.items():
        if fr_word in fr_vectors and it_word in it_vectors:
            fr_words.append(fr_word)
            it_words.append(it_word)
            fr_vecs.append(fr_vectors[fr_word])
            it_vecs.append(it_vectors[it_word])
            if len(fr_words) >= n:
                break

    import numpy as np
    return fr_words, it_words, np.array(fr_vecs), np.array(it_vecs)
