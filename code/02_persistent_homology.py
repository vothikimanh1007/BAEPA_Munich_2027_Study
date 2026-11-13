"""
02_persistent_homology.py
-------------------------
Part of: Reconstructing Inter-Regional Transmission and Material Bridges in Asian Buddhist Epigraphy and Portable Icons
Repository: BAEPA_Munich_2027_Study
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import normalize, OneHotEncoder, StandardScaler
from scipy.spatial.distance import pdist, squareform


def compute_structured_distance_matrix(df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """Construct multi-modal structured feature representations (d=17) and Euclidean distance matrix."""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    B_icon = ohe.fit_transform(df[["iconography_class"]])
    B_mat = ohe.fit_transform(df[["material_clean"]])
    B_reg = ohe.fit_transform(df[["macro_region"]])

    scaler = StandardScaler()
    yr_scaled = scaler.fit_transform(df[["year_ce"]])
    dated_flag = df[["is_dated"]].astype(float).values
    B_time = np.hstack([yr_scaled, dated_flag])

    # Unit Frobenius norm scaling per block
    B_icon = B_icon / (np.sqrt(B_icon.shape[1]) + 1e-12)
    B_mat = B_mat / (np.sqrt(B_mat.shape[1]) + 1e-12)
    B_reg = B_reg / (np.sqrt(B_reg.shape[1]) + 1e-12)
    B_time = B_time / (np.sqrt(B_time.shape[1]) + 1e-12)

    X_struct = np.hstack([
        0.40 * B_icon,
        0.25 * B_mat,
        0.20 * B_reg,
        0.15 * B_time
    ])
    Z = normalize(X_struct, norm="l2")

    dist_mat = squareform(pdist(Z, metric="euclidean"))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1e-8, size=dist_mat.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)

    dist_mat = np.clip(dist_mat + noise, 0.0, None)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def run_persistent_homology(base_data_dir: str = "../data", figures_dir: str = "../figures"):
    os.makedirs(figures_dir, exist_ok=True)
    processed_dir = os.path.join(base_data_dir, "processed_data")
    corpus_csv = os.path.join(processed_dir, "unified_buddhist_corpus_tripartite.csv")

    df = pd.read_csv(corpus_csv)
    total_count = len(df)
    dist_mat = compute_structured_distance_matrix(df)
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

    # Figure 1: Persistence Profile
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

    print(f"[02_persistent_homology] Saved persistence diagram to {f1_path}")


if __name__ == "__main__":
    run_persistent_homology(base_data_dir="../data", figures_dir="../figures")
