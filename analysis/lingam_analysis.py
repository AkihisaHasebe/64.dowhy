"""LiNGAM 因果探索モジュール"""

import numpy as np
import pandas as pd
import lingam


def run_lingam(df, target, n_bootstrap=100, min_effect=0.01, threshold=0.50):
    """DirectLiNGAM + Bootstrap を実行する。

    Returns:
        dict with keys: adj_matrix, causal_order, bootstrap_probs,
                        significant_edges, effects_on_target, columns
    """
    columns = list(df.columns)
    data = df.values
    target_idx = columns.index(target)

    model = lingam.DirectLiNGAM()
    model.fit(data)

    adj_matrix = model.adjacency_matrix_
    causal_order = [columns[i] for i in model.causal_order_]

    # Y への直接因果効果
    effects_on_target = pd.Series(
        adj_matrix[target_idx, :], index=columns
    ).drop(target).sort_values(key=abs, ascending=False)

    # Y への総因果効果 (直接 + 間接パス経由)
    # 線形 SEM: x = Bx + e  →  x = (I-B)^{-1} e
    # 総因果効果 = [(I-B)^{-1}]_{target, j}  (j ≠ target)
    n_vars = len(columns)
    total_effect_matrix = np.linalg.inv(np.eye(n_vars) - adj_matrix)
    total_effects_on_target = pd.Series(
        total_effect_matrix[target_idx, :], index=columns
    ).drop(target).sort_values(key=abs, ascending=False)

    # 標準化: β_std = β_raw × (SD_x / SD_y)
    # 「X を 1SD 変化させた時に Y が何 SD 変化するか」
    sds = df.std()
    sd_target = sds[target]
    std_direct = pd.Series(
        {c: adj_matrix[target_idx, columns.index(c)] * sds[c] / sd_target
         for c in columns if c != target}
    ).sort_values(key=abs, ascending=False)
    std_total = pd.Series(
        {c: total_effect_matrix[target_idx, columns.index(c)] * sds[c] / sd_target
         for c in columns if c != target}
    ).sort_values(key=abs, ascending=False)

    # Bootstrap
    bs_result = model.bootstrap(data, n_sampling=n_bootstrap)
    probs = bs_result.get_probabilities(min_causal_effect=min_effect)
    probs_df = pd.DataFrame(probs, index=columns, columns=columns)

    # 有意なエッジ抽出
    n = len(columns)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and probs[i, j] >= threshold:
                edges.append({
                    "From": columns[j],
                    "To": columns[i],
                    "coefficient": adj_matrix[i, j],
                    "probability": probs[i, j],
                })

    edges_df = pd.DataFrame(edges) if edges else pd.DataFrame(
        columns=["From", "To", "coefficient", "probability"]
    )

    # Target への Bootstrap エッジ確率
    edge_probs_to_target = probs_df.loc[target].drop(target).sort_values(ascending=False)

    return {
        "adj_matrix": adj_matrix,
        "causal_order": causal_order,
        "bootstrap_probs": probs_df,
        "significant_edges": edges_df,
        "effects_on_target": effects_on_target,
        "total_effects_on_target": total_effects_on_target,
        "std_direct_effects": std_direct,
        "std_total_effects": std_total,
        "edge_probs_to_target": edge_probs_to_target,
        "columns": columns,
    }
