"""GRaSP (Greedy relaxation of Sparsest Permutation) 因果探索モジュール"""

import numpy as np
import pandas as pd
from causallearn.search.PermutationBased.GRaSP import grasp


def run_grasp(data, column_names, target, score_func="local_score_BIC",
              depth=3, n_bootstrap=100, threshold=0.50):
    """GRaSP アルゴリズム + Bootstrap を実行する。

    Returns:
        dict with keys: graph, edges_df, bootstrap_probs,
                        adjacent_to_target, columns
    """
    n = len(column_names)

    # 単一実行
    G = grasp(data, score_func=score_func, depth=depth,
              verbose=False, node_names=column_names)
    adj_matrix = G.graph

    # Bootstrap
    edge_counts = np.zeros((n, n))
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(data), size=len(data), replace=True)
        boot_data = data[idx]
        G_b = grasp(boot_data, score_func=score_func, depth=depth,
                    verbose=False, node_names=column_names)
        adj_b = G_b.graph
        for i in range(n):
            for j in range(n):
                if adj_b[i, j] != 0:
                    edge_counts[i, j] += 1

    probs = edge_counts / n_bootstrap
    probs_df = pd.DataFrame(probs, index=column_names, columns=column_names)

    # 閾値フィルタリング
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            skeleton_prob = max(probs[i, j], probs[j, i])
            if skeleton_prob < threshold:
                continue

            ei, ej = adj_matrix[i, j], adj_matrix[j, i]
            if ei == -1 and ej == 1:
                kind = "directed"
                src, dst = column_names[i], column_names[j]
            elif ei == 1 and ej == -1:
                kind = "directed"
                src, dst = column_names[j], column_names[i]
            elif ei == -1 and ej == -1:
                kind = "undirected"
                src, dst = column_names[i], column_names[j]
            else:
                kind = "undirected"
                src, dst = column_names[i], column_names[j]

            edges.append({
                "From": src, "To": dst, "type": kind,
                "probability": skeleton_prob,
            })

    edges_df = pd.DataFrame(edges) if edges else pd.DataFrame(
        columns=["From", "To", "type", "probability"]
    )

    # Y の隣接ノード
    adjacent_to_target = set()
    for _, e in edges_df.iterrows():
        if e["From"] == target:
            adjacent_to_target.add(e["To"])
        elif e["To"] == target:
            adjacent_to_target.add(e["From"])

    return {
        "graph": G,
        "edges_df": edges_df,
        "bootstrap_probs": probs_df,
        "adjacent_to_target": adjacent_to_target,
        "columns": column_names,
    }
