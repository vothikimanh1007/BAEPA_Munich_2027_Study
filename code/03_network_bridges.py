import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def build_simplicial_skeleton(df: pd.DataFrame, dist_mat: np.ndarray, epsilon: float) -> nx.Graph:
    """Construct the 1-skeleton graph G(X, epsilon) from pairwise distance matrix."""
    n = len(df)
    G = nx.Graph()
    for i in range(n):
        G.add_node(
            i,
            artifact_id=df.iloc[i]["artifact_id"],
            partition=df.iloc[i]["corpus_partition"],
            source=df.iloc[i]["source"],
            title=df.iloc[i]["title"],
            material=df.iloc[i]["material"],
        )

    for i in range(n):
        for j in range(i + 1, n):
            if dist_mat[i, j] <= epsilon:
                G.add_edge(i, j, weight=float(dist_mat[i, j]))

    cb = nx.betweenness_centrality(G, weight="weight")
    nx.set_node_attributes(G, cb, "betweenness")
    return G


def extract_cross_corpus_bridges(df: pd.DataFrame, dist_mat: np.ndarray, delta: float = 0.05) -> pd.DataFrame:
    """Identify minimum inter-corpus distance D_min and top bridging artifacts at epsilon*."""
    n = len(df)
    cross_pairs = []
    min_dist = float("inf")

    for i in range(n):
        for j in range(i + 1, n):
            p_i = df.iloc[i]["corpus_partition"]
            p_j = df.iloc[j]["corpus_partition"]
            if (p_i == "DHARMA" and p_j in ["VA", "CMA"]) or (p_j == "DHARMA" and p_i in ["VA", "CMA"]):
                d = float(dist_mat[i, j])
                cross_pairs.append((d, i, j))
                if d < min_dist:
                    min_dist = d

    epsilon_star = min_dist + delta
    print(f"[03_network_bridges] Empirical D_min = {min_dist:.4f} | Horizon epsilon* = {epsilon_star:.4f}")

    G = build_simplicial_skeleton(df, dist_mat, epsilon_star)
    bridges = []
    for d, u, v in sorted(cross_pairs, key=lambda x: x[0]):
        if d <= epsilon_star:
            p_u = G.nodes[u]["partition"]
            d_node = u if p_u == "DHARMA" else v
            b_node = v if p_u == "DHARMA" else u
            bridges.append({
                "dharma_id": G.nodes[d_node]["artifact_id"],
                "dharma_title": G.nodes[d_node]["title"],
                "bronze_partition": G.nodes[b_node]["partition"],
                "bronze_id": G.nodes[b_node]["artifact_id"],
                "bronze_title": G.nodes[b_node]["title"],
                "distance_Dij": round(d, 4),
                "betweenness_CB": round(G.nodes[b_node]["betweenness"], 4),
                "epsilon_star": round(epsilon_star, 4),
            })

    df_bridges = pd.DataFrame(bridges)
    if not df_bridges.empty:
        df_bridges = df_bridges.sort_values(
            by=["betweenness_CB", "distance_Dij"], ascending=[False, True]
        ).reset_index(drop=True)
    return df_bridges


def generate_tripartite_network_figure(df: pd.DataFrame, dist_mat: np.ndarray, figures_dir: str):
    """Render Figure 3: Balanced 45-node tripartite simplicial network."""
    idx_sea = df[df["corpus_partition"] == "DHARMA"].index[:15]
    idx_va = df[df["corpus_partition"] == "VA"].index[:15]
    idx_cma = df[df["corpus_partition"] == "CMA"].index[:15]
    sel_indices = np.concatenate([idx_sea, idx_va, idx_cma])

    sample_df = df.iloc[sel_indices].reset_index(drop=True)
    sample_dist = dist_mat[np.ix_(sel_indices, sel_indices)]

    G = nx.Graph()
    color_map = {"DHARMA": "#2980b9", "VA": "#d35400", "CMA": "#27ae60"}

    for idx, row in sample_df.iterrows():
        node_id = f"[{idx+1}] {str(row['title'])[:11]}"
        G.add_node(node_id, color=color_map.get(row["corpus_partition"], "#7f8c8d"))

    for i in range(len(sample_df)):
        neighbors = np.argsort(sample_dist[i])[1:4]
        for nb in neighbors:
            node_i = f"[{i+1}] {str(sample_df.iloc[i]['title'])[:11]}"
            node_nb = f"[{nb+1}] {str(sample_df.iloc[nb]['title'])[:11]}"
            w = float(1.0 / (1.0 + sample_dist[i, nb]))
            G.add_edge(node_i, node_nb, weight=w)

    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    fig = plt.figure(figsize=(10.5, 9.0))
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.84])
    pos = nx.spring_layout(G, seed=42, k=0.45)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=550, alpha=0.92, edgecolors="#2c3e50", linewidths=0.6)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35, edge_color="#7f8c8d", width=1.3)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_family="sans-serif", font_weight="bold")

    ax.scatter([], [], c="#2980b9", label="Southeast Asian Epigraphy (DHARMA, n=15)", s=90, edgecolors="#2c3e50", linewidths=0.5)
    ax.scatter([], [], c="#d35400", label="Portable Devotional Bronzes (V&A London, n=15)", s=90, edgecolors="#2c3e50", linewidths=0.5)
    ax.scatter([], [], c="#27ae60", label="Himalayan/Gandharan Icons (Cleveland Museum, n=15)", s=90, edgecolors="#2c3e50", linewidths=0.5)
    ax.legend(scatterpoints=1, loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#bdc3c7", fontsize=9.5)
    ax.set_title("Simplicial Transmission Network: Epigraphic Traditions and Portable Material Culture", fontsize=11.5, fontweight="bold")
    ax.axis("off")

    f3_path = os.path.join(figures_dir, "Fig3_Simplicial_Network.png")
    plt.savefig(f3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[03_network_bridges] Saved {f3_path}")


def run_network_analysis(base_data_dir: str = "../data", figures_dir: str = "../figures"):
    os.makedirs(figures_dir, exist_ok=True)
    processed_dir = os.path.join(base_data_dir, "processed_data")

    df = pd.read_csv(os.path.join(processed_dir, "unified_buddhist_corpus_tripartite.csv"))
    dist_mat = np.load(os.path.join(processed_dir, "pairwise_distance_matrix.npy"))

    df_bridges = extract_cross_corpus_bridges(df, dist_mat, delta=0.05)
    bridges_csv = os.path.join(processed_dir, "cross_corpus_case_studies.csv")
    df_bridges.to_csv(bridges_csv, index=False)
    print(f"[03_network_bridges] Saved {len(df_bridges)} boundary edges to {bridges_csv}")

    generate_tripartite_network_figure(df, dist_mat, figures_dir)


if __name__ == "__main__":
    run_network_analysis(base_data_dir="../data", figures_dir="../figures")
