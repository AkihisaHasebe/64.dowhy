"""PC / FCI 因果探索モジュール"""

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci


def _extract_edges_from_graph(graph_matrix, column_names):
    """causal-learn の隣接行列からエッジ情報を抽出する。"""
    n = len(column_names)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            ei = graph_matrix[i, j]
            ej = graph_matrix[j, i]
            if ei == 0 and ej == 0:
                continue

            if ei == -1 and ej == 1:
                kind = "directed"
                src, dst = column_names[i], column_names[j]
            elif ei == 1 and ej == -1:
                kind = "directed"
                src, dst = column_names[j], column_names[i]
            elif ei == -1 and ej == -1:
                kind = "undirected"
                src, dst = column_names[i], column_names[j]
            elif ei == 1 and ej == 1:
                kind = "bidirected"
                src, dst = column_names[i], column_names[j]
            elif ei == 2 and ej == 1:
                # o-> (circle to arrow)
                kind = "circle_arrow"
                src, dst = column_names[j], column_names[i]
            elif ei == 1 and ej == 2:
                kind = "circle_arrow"
                src, dst = column_names[i], column_names[j]
            elif ei == 2 and ej == 2:
                kind = "circle_circle"
                src, dst = column_names[i], column_names[j]
            elif ei == 2 and ej == -1:
                kind = "directed"
                src, dst = column_names[i], column_names[j]
            elif ei == -1 and ej == 2:
                kind = "directed"
                src, dst = column_names[j], column_names[i]
            else:
                kind = f"other({ei},{ej})"
                src, dst = column_names[i], column_names[j]

            edges.append({"From": src, "To": dst, "type": kind})
    return edges


EDGE_TYPE_LABELS = {
    "directed": "有向 (->)",
    "undirected": "無向 (--)",
    "bidirected": "双方向 (<->)",
    "circle_arrow": "o->",
    "circle_circle": "o-o",
}


def run_pc(data, column_names, target, alpha=0.05, indep_test="fisherz",
           n_bootstrap=100, threshold=0.50):
    """PC アルゴリズム + Bootstrap を実行する。"""
    n = len(column_names)

    # 単一実行
    cg = pc(data=data, alpha=alpha, indep_test=indep_test, node_names=column_names)

    # Bootstrap
    edge_counts = np.zeros((n, n))
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(data), size=len(data), replace=True)
        boot_data = data[idx]
        cg_b = pc(data=boot_data, alpha=alpha, indep_test=indep_test,
                  node_names=column_names)
        adj_b = cg_b.G.graph
        for i in range(n):
            for j in range(n):
                if adj_b[i, j] != 0:
                    edge_counts[i, j] += 1

    probs = edge_counts / n_bootstrap
    probs_df = pd.DataFrame(probs, index=column_names, columns=column_names)

    # 閾値フィルタリング
    adj_matrix = cg.G.graph
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
                "From": src, "To": dst, "type": kind, "probability": skeleton_prob,
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
        "graph": cg,
        "edges_df": edges_df,
        "bootstrap_probs": probs_df,
        "adjacent_to_target": adjacent_to_target,
        "columns": column_names,
    }


def run_fci(data, column_names, target, alpha=0.05, indep_test="fisherz"):
    """FCI アルゴリズムを実行する。"""
    g, edges_result = fci(data, indep_test, alpha,
                          verbose=False, show_progress=False,
                          node_names=column_names)

    graph_matrix = g.graph
    edges_list = _extract_edges_from_graph(graph_matrix, column_names)
    edges_df = pd.DataFrame(edges_list) if edges_list else pd.DataFrame(
        columns=["From", "To", "type"]
    )

    # Y の隣接ノード
    adjacent_to_target = set()
    for _, e in edges_df.iterrows():
        if e["From"] == target:
            adjacent_to_target.add(e["To"])
        elif e["To"] == target:
            adjacent_to_target.add(e["From"])

    return {
        "graph": g,
        "edges_df": edges_df,
        "adjacent_to_target": adjacent_to_target,
        "columns": column_names,
    }
