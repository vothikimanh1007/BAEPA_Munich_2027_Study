import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ripser import ripser
from persim import plot_diagrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, OneHotEncoder, MinMaxScaler
from scipy.spatial.distance import pdist, squareform


def compute_multimodal_distance_matrix(
    df: pd.DataFrame,
    max_features: int = 350,
    n_components: int = 22,
    seed: int = 42,
) -> np.ndarray:
    """Construct L2-normalized multi-modal latent representations and Euclidean distance matrix."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_tfidf = tfidf.fit_transform(df["text_content"].fillna("")).toarray()

    d = min(n_components, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=d, random_state=seed)
    v_textual = svd.fit_transform(X_tfidf)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    v_material = ohe.fit_transform(df[["material"]].fillna("Unknown"))

    scaler = MinMaxScaler()
    v_spatio_chrono = scaler.fit_transform(df[["dating_year"]].fillna(850.0))

    X_modal = np.hstack([
        0.60 * v_textual,
        0.25 * v_material,
        0.15 * v_spatio_chrono,
    ])
    Z = normalize(X_modal, norm="l2")

    dist_mat = squareform(pdist(Z, metric="euclidean"))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1e-8, size=dist_mat.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)

    dist_mat = np.clip(dist_mat + noise, 0.0, None)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def compute_transmission_profile(intervals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate Betti curves across filtration grid scales."""
    counts = np.zeros(len(grid), dtype=int)
    for b, d in intervals:
        death = grid[-1] if np.isinf(d) else d
        counts += ((grid >= b) & (grid < death)).astype(int)
    return counts


def run_persistent_homology(base_data_dir: str = "../data", figures_dir: str = "../figures"):
    os.makedirs(figures_dir, exist_ok=True)
    processed_dir = os.path.join(base_data_dir, "processed_data")
    corpus_csv = os.path.join(processed_dir, "unified_buddhist_corpus_tripartite.csv")

    df = pd.read_csv(corpus_csv)
    total_count = len(df)
    dist_mat = compute_multimodal_distance_matrix(df)
    np.save(os.path.join(processed_dir, "pairwise_distance_matrix.npy"), dist_mat)

    ripser_res = ripser(dist_mat, distance_matrix=True, maxdim=1)
    dgms = ripser_res["dgms"]
    h0, h1 = dgms[0], dgms[1]

    pd.DataFrame(h0, columns=["birth", "death"]).to_csv(
        os.path.join(processed_dir, "persistence_h0.csv"), index=False
    )
    pd.DataFrame(h1, columns=["birth", "death"]).to_csv(
        os.path.join(processed_dir, "persistence_h1.csv"), index=False
    )

    print(f"[02_persistent_homology] Total H0 classes: {len(h0)} | Total H1 loops: {len(h1)}")
    print(f"[02_persistent_homology] Prominent H1 loops (death > 0.85): {np.sum(h1[:, 1] > 0.85)}")

    # Figure 1
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_axes([0.12, 0.16, 0.80, 0.75])
    plot_diagrams(dgms, ax=ax, show=False)
    ax.set_title("Cross-Cultural Persistence Profile of Buddhist Material Culture", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Cultural Proximity Scale (Birth)", fontsize=10)
    ax.set_ylabel("Persistence Scale (Death)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    f1_path = os.path.join(figures_dir, "Fig1_Persistence_Profile.png")
    plt.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2
    grid_scales = np.linspace(0.05, np.percentile(dist_mat, 96), 400)
    cluster_profile = compute_transmission_profile(h0, grid_scales)
    loop_profile = compute_transmission_profile(h1, grid_scales)

    fig = plt.figure(figsize=(9.5, 6.5))
    ax = fig.add_axes([0.10, 0.16, 0.82, 0.75])
    ax.plot(grid_scales, cluster_profile, label="Independent Monastic & Epigraphic Traditions ($H_0$)", color="#1f618d", lw=2.4)
    ax.plot(grid_scales, loop_profile, label="Inter-Regional Structural Circuits / Transmission Loops ($H_1$)", color="#c0392b", lw=2.4)
    ax.set_title(f"Multi-Scale Cultural Integration Profile (N={total_count})", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Cultural Distance Parameter ($\\epsilon$)", fontsize=10)
    ax.set_ylabel("Structural Count", fontsize=10)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#bdc3c7")
    ax.grid(True, linestyle="--", alpha=0.5)
    f2_path = os.path.join(figures_dir, "Fig2_Transmission_Profile.png")
    plt.savefig(f2_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[02_persistent_homology] Saved {f1_path} and {f2_path}")


if __name__ == "__main__":
    run_persistent_homology(base_data_dir="../data", figures_dir="../figures")
