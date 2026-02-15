"""アンサンブル結果から統合因果グラフ (Consensus Graph) を構築するモジュール"""

import networkx as nx
import pandas as pd
from typing import Dict, Set, Tuple, Optional


def build_consensus_graph(
    session_state: dict,
    target: str,
    features: list,
    min_agreement: int = 2,
    directed_threshold: float = 0.6,
) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    各手法の結果から統合因果グラフを構築する。

    Args:
        session_state: Streamlit session_state (各手法の結果を含む)
        target: 目的変数名
        features: 特徴量リスト
        min_agreement: エッジを採用する最小手法数 (デフォルト: 2)
        directed_threshold: 方向付きエッジと判定する確率閾値 (デフォルト: 0.6)

    Returns:
        (consensus_graph, edge_support_df)
        - consensus_graph: 統合因果グラフ (nx.DiGraph)
        - edge_support_df: エッジごとのサポート情報 (DataFrame)
    """
    all_nodes = features + [target]

    # 各手法のエッジ候補を収集
    edge_votes = {}  # (from, to) -> {method_name: vote_info}

    # --- LiNGAM ---
    if "lingam_result" in session_state:
        res = session_state["lingam_result"]
        edges_df = res["significant_edges"]
        for _, row in edges_df.iterrows():
            edge = (row["From"], row["To"])
            if edge not in edge_votes:
                edge_votes[edge] = {}
            edge_votes[edge]["LiNGAM"] = {
                "directed": True,
                "probability": row["probability"],
                "coefficient": row.get("coefficient", 0),
            }

    # --- PC ---
    if "pc_result" in session_state:
        res = session_state["pc_result"]
        edges_df = res["edges_df"]
        for _, row in edges_df.iterrows():
            edge = (row["From"], row["To"])
            if edge not in edge_votes:
                edge_votes[edge] = {}
            is_directed = (row["type"] == "directed")
            edge_votes[edge]["PC"] = {
                "directed": is_directed,
                "probability": row.get("probability", 1.0),
                "coefficient": None,
            }
            # 無向エッジの場合は逆方向も追加
            if not is_directed:
                edge_rev = (row["To"], row["From"])
                if edge_rev not in edge_votes:
                    edge_votes[edge_rev] = {}
                edge_votes[edge_rev]["PC"] = {
                    "directed": False,
                    "probability": row.get("probability", 1.0),
                    "coefficient": None,
                }

    # --- FCI ---
    if "fci_result" in session_state:
        res = session_state["fci_result"]
        edges_df = res["edges_df"]
        for _, row in edges_df.iterrows():
            edge = (row["From"], row["To"])
            if edge not in edge_votes:
                edge_votes[edge] = {}
            is_directed = (row["type"] == "directed")
            edge_votes[edge]["FCI"] = {
                "directed": is_directed,
                "probability": 1.0,
                "coefficient": None,
            }
            # 非方向エッジの場合は逆方向も
            if not is_directed:
                edge_rev = (row["To"], row["From"])
                if edge_rev not in edge_votes:
                    edge_votes[edge_rev] = {}
                edge_votes[edge_rev]["FCI"] = {
                    "directed": False,
                    "probability": 1.0,
                    "coefficient": None,
                }

    # --- GRaSP ---
    if "grasp_result" in session_state:
        res = session_state["grasp_result"]
        edges_df = res["edges_df"]
        for _, row in edges_df.iterrows():
            edge = (row["From"], row["To"])
            if edge not in edge_votes:
                edge_votes[edge] = {}
            is_directed = (row["type"] == "directed")
            edge_votes[edge]["GRaSP"] = {
                "directed": is_directed,
                "probability": row.get("probability", 1.0),
                "coefficient": None,
            }
            if not is_directed:
                edge_rev = (row["To"], row["From"])
                if edge_rev not in edge_votes:
                    edge_votes[edge_rev] = {}
                edge_votes[edge_rev]["GRaSP"] = {
                    "directed": False,
                    "probability": row.get("probability", 1.0),
                    "coefficient": None,
                }

    # --- エッジの統合判定 ---
    consensus_edges = []

    for edge, votes in edge_votes.items():
        n_votes = len(votes)
        if n_votes < min_agreement:
            continue

        # 方向性の判定: 投票の過半数が directed なら directed
        directed_count = sum(1 for v in votes.values() if v["directed"])
        is_consensus_directed = (directed_count / n_votes) >= directed_threshold

        # 平均確率
        avg_prob = sum(v["probability"] for v in votes.values()) / n_votes

        # 係数 (LiNGAMがあれば使用)
        coef = None
        if "LiNGAM" in votes and votes["LiNGAM"]["coefficient"] is not None:
            coef = votes["LiNGAM"]["coefficient"]

        consensus_edges.append({
            "From": edge[0],
            "To": edge[1],
            "support": n_votes,
            "avg_probability": avg_prob,
            "directed": is_consensus_directed,
            "coefficient": coef,
            "methods": list(votes.keys()),
        })

    edge_support_df = pd.DataFrame(consensus_edges).sort_values(
        "support", ascending=False
    )

    # グラフ構築
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)

    for _, row in edge_support_df.iterrows():
        if row["directed"]:
            # 有向エッジ
            G.add_edge(
                row["From"], row["To"],
                support=row["support"],
                probability=row["avg_probability"],
                coefficient=row["coefficient"],
            )
        else:
            # 無向エッジは両方向に追加 (CPDAGの表現)
            G.add_edge(
                row["From"], row["To"],
                support=row["support"],
                probability=row["avg_probability"],
                coefficient=row["coefficient"],
                undirected=True,
            )
            G.add_edge(
                row["To"], row["From"],
                support=row["support"],
                probability=row["avg_probability"],
                coefficient=row["coefficient"],
                undirected=True,
            )

    return G, edge_support_df


def get_adjacent_to_target(consensus_graph: nx.DiGraph, target: str) -> Set[str]:
    """統合グラフで target に隣接するノードを取得"""
    adjacent = set()

    # target への入力エッジ
    adjacent.update(consensus_graph.predecessors(target))

    # target からの出力エッジ (無向エッジを含む)
    for succ in consensus_graph.successors(target):
        edge_data = consensus_graph.get_edge_data(target, succ)
        if edge_data and edge_data.get("undirected", False):
            adjacent.add(succ)

    return adjacent
