"""
viz_radar.py
Spider / radar chart comparing embedding quality across five metrics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity as cos_sim_matrix

from .viz_utils import EMB_COLORS


CATEGORIES = [
    'Avg cosine\nsimilarity',
    'FR vocab\ncoverage',
    'IT vocab\ncoverage',
    'Vector\nstability',
    'NN\nPrecision@1',
]


def _compute_metrics(fr_vecs, it_vecs, bilingual_dict):
    """
    Compute the five radar metrics for one embedding type.

    Returns:
        list of 5 floats in [0, 1], or None if not enough pairs.
    """
    pairs = [(fw, iw) for fw, iw in bilingual_dict.items()
             if fw in fr_vecs and iw in it_vecs][:200]

    if not pairs:
        return None

    fv = np.array([fr_vecs[fw] for fw, _ in pairs])
    iv = np.array([it_vecs[iw] for _, iw in pairs])

    # 1. average cosine similarity on aligned pairs
    sims    = np.array([
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        for a, b in zip(fv, iv)
    ])
    avg_sim = float(np.clip(np.mean(sims), 0, 1))

    # 2 & 3. vocabulary coverage vs. dictionary
    fr_cov = sum(1 for fw, _ in bilingual_dict.items()
                 if fw in fr_vecs) / max(len(bilingual_dict), 1)
    it_cov = sum(1 for _, iw in bilingual_dict.items()
                 if iw in it_vecs) / max(len(bilingual_dict), 1)

    # 4. vector stability = 1 - normalised std of norms
    all_norms = np.array([np.linalg.norm(v)
                          for v in list(fr_vecs.values())[:500]])
    stability = float(np.clip(
        1 - (all_norms.std() / (all_norms.mean() + 1e-9)), 0, 1))

    # 5. nearest-neighbour precision@1 (subset of 50 pairs)
    sub           = pairs[:50]
    sfv           = np.array([fr_vecs[fw] for fw, _ in sub])
    it_words_list = list(it_vecs.keys())[:2000]
    it_cands      = np.array([it_vecs[w] for w in it_words_list])
    sim_mat       = cos_sim_matrix(sfv, it_cands)
    nn_hits       = sum(
        it_words_list[int(np.argmax(sim_mat[i]))] == sub[i][1]
        for i in range(len(sub))
    )
    nn_prec = float(nn_hits) / len(sub)

    return [avg_sim, fr_cov, it_cov, stability, nn_prec]


def plot_radar_comparison(fr_vectors_all, it_vectors_all,
                          bilingual_dict, output_dir, save=True):
    """
    Spider chart comparing all embedding types on five quality metrics.

    Args:
        fr_vectors_all (dict): { emb_type: fr_word_vectors }
        it_vectors_all (dict): { emb_type: it_word_vectors }
        bilingual_dict (dict): FR-IT bilingual dictionary
        output_dir (str): Directory to save the figure
        save (bool): Save figure to file
    """
    print("Plotting radar comparison...")

    N      = len(CATEGORIES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFAFA')

    for emb_type, fr_vecs in fr_vectors_all.items():
        it_vecs = it_vectors_all[emb_type]
        metrics = _compute_metrics(fr_vecs, it_vecs, bilingual_dict)

        if metrics is None:
            print(f"  Skipping {emb_type}: no valid pairs found.")
            continue

        values = metrics + metrics[:1]   # close the polygon
        color  = EMB_COLORS.get(emb_type, '#888')
        ax.plot(angles, values, color=color, linewidth=2, label=emb_type)
        ax.fill(angles, values, color=color, alpha=0.10)

    # ── style ──
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color='#cccccc', linestyle='--', linewidth=0.6)
    ax.xaxis.grid(True, color='#cccccc', linestyle='--', linewidth=0.6)
    ax.spines['polar'].set_color('#cccccc')

    ax.set_title('Embedding Quality  ·  Multi-metric Radar Comparison',
                 fontsize=13, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08),
              fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save:
        path = os.path.join(output_dir, 'radar_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved → {path}")

    plt.close()
