"""DoWhy を用いた因果効果推定モジュール"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Optional
import warnings

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    warnings.warn("DoWhy not installed. Causal effect estimation will be skipped.")


def estimate_causal_effects_with_dowhy(
    df: pd.DataFrame,
    target: str,
    features: list,
    consensus_graph: nx.DiGraph,
    method: str = "backdoor.linear_regression",
) -> Dict[str, Dict[str, float]]:
    """
    DoWhy を用いて各特徴量から目的変数への因果効果を推定する。

    Args:
        df: データフレーム
        target: 目的変数名
        features: 特徴量リスト
        consensus_graph: 統合因果グラフ (nx.DiGraph)
        method: DoWhy の推定手法 (デフォルト: backdoor.linear_regression)

    Returns:
        {feature: {"ate": ate_value, "stderr": stderr, "identified": bool}}
    """
    if not DOWHY_AVAILABLE:
        return {}

    results = {}

    # NetworkX グラフを GML 形式文字列に変換
    gml_graph = _networkx_to_gml_string(consensus_graph)

    for feature in features:
        try:
            # DoWhy モデル構築
            model = CausalModel(
                data=df,
                treatment=feature,
                outcome=target,
                graph=gml_graph,
            )

            # 識別 (Identification)
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True
            )

            # 推定 (Estimation)
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method,
            )

            # 結果を格納
            results[feature] = {
                "ate": estimate.value,
                "stderr": getattr(estimate, "stderr", None),
                "identified": ("Unidentified" not in str(identified_estimand)),
                "estimand": str(identified_estimand),
            }

        except Exception as e:
            # エラーが発生した場合は NaN で記録
            results[feature] = {
                "ate": np.nan,
                "stderr": np.nan,
                "identified": False,
                "error": str(e),
            }

    return results


def _networkx_to_gml_string(G: nx.DiGraph) -> str:
    """NetworkX DiGraph を GML 形式の文字列に変換する"""
    lines = ["graph [", "  directed 1"]

    # ノードを追加
    node_to_id = {node: i for i, node in enumerate(G.nodes())}
    for node, node_id in node_to_id.items():
        lines.append(f'  node [')
        lines.append(f'    id {node_id}')
        lines.append(f'    label "{node}"')
        lines.append(f'  ]')

    # エッジを追加 (無向エッジの重複を除く)
    added_edges = set()
    for u, v, data in G.edges(data=True):
        # 無向エッジは一方向のみ追加
        if data.get("undirected", False):
            edge_key = frozenset([u, v])
            if edge_key in added_edges:
                continue
            added_edges.add(edge_key)

        lines.append(f'  edge [')
        lines.append(f'    source {node_to_id[u]}')
        lines.append(f'    target {node_to_id[v]}')
        lines.append(f'  ]')

    lines.append("]")
    return "\n".join(lines)
