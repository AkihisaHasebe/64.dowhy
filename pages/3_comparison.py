"""Page 5: 全手法統合比較ダッシュボード"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.header("全手法統合比較")

if "df" not in st.session_state:
    st.warning("メインページでデータを読み込んでください。")
    st.stop()

target = st.session_state["target"]
features = st.session_state["features"]
gt = st.session_state.get("ground_truth")

# 結果の有無をチェック
has_lingam = "lingam_result" in st.session_state
has_pc = "pc_result" in st.session_state
has_fci = "fci_result" in st.session_state
has_grasp = "grasp_result" in st.session_state

if not any([has_lingam, has_pc, has_fci, has_grasp]):
    st.warning("分析実行ページで少なくとも1つの分析を実行してください。")
    st.stop()

# --- 統合比較テーブル ---
st.subheader(f"{target} への影響度 — 全手法比較")

comparison = pd.DataFrame(index=features)

if has_lingam:
    res = st.session_state["lingam_result"]
    comparison["LiNGAM直接効果"] = res["effects_on_target"].abs()
    comparison["LiNGAM総介入効果"] = res["total_effects_on_target"].abs()
    comparison["LiNGAM確率"] = res["edge_probs_to_target"]

if has_pc:
    res = st.session_state["pc_result"]
    pc_adj = res["adjacent_to_target"]
    # エッジ確率
    probs = res["bootstrap_probs"]
    target_col = probs.columns[probs.columns == target]
    if len(target_col) > 0:
        pc_probs_to_y = probs[target].drop(target, errors="ignore")
        comparison["PC確率"] = pc_probs_to_y
    comparison["PC隣接"] = [1 if f in pc_adj else 0 for f in features]

if has_fci:
    res = st.session_state["fci_result"]
    fci_adj = res["adjacent_to_target"]
    comparison["FCI隣接"] = [1 if f in fci_adj else 0 for f in features]

if has_grasp:
    res = st.session_state["grasp_result"]
    grasp_adj = res["adjacent_to_target"]
    grasp_probs = res["bootstrap_probs"]
    target_col = grasp_probs.columns[grasp_probs.columns == target]
    if len(target_col) > 0:
        grasp_probs_to_y = grasp_probs[target].drop(target, errors="ignore")
        comparison["GRaSP確率"] = grasp_probs_to_y
    comparison["GRaSP隣接"] = [1 if f in grasp_adj else 0 for f in features]

if gt:
    def _cat(f):
        if f in gt["direct_causes"]: return "直接原因"
        if f in gt["spurious"]: return "擬似相関"
        if f in gt["independent"]: return "独立ノイズ"
        if f in gt["upstream"]: return "上流変数"
        return "不明"
    comparison["カテゴリ"] = [_cat(f) for f in features]

# ランキング列を追加
for col in comparison.columns:
    if col in ["カテゴリ", "PC隣接", "FCI隣接", "GRaSP隣接"]:
        continue
    comparison[f"{col}_順位"] = comparison[col].rank(ascending=False).astype(int)

st.dataframe(
    comparison.sort_values(comparison.columns[0], ascending=False)
    if len(comparison.columns) > 0 else comparison,
    use_container_width=True, height=500,
)

# --- 各手法が「直接原因」と判定した変数の比較 ---
st.subheader(f"各手法が {target} の直接原因と判定した変数")

method_selections = {}

if has_lingam:
    lingam_eff = st.session_state["lingam_result"]["effects_on_target"]
    lingam_threshold = st.slider(
        "LiNGAM 因果効果閾値", 0.0, float(lingam_eff.abs().max()), 0.1,
        key="lingam_thresh"
    )
    method_selections["LiNGAM"] = set(
        lingam_eff[lingam_eff.abs() >= lingam_threshold].index
    )

if has_pc:
    method_selections["PC"] = st.session_state["pc_result"]["adjacent_to_target"]

if has_fci:
    method_selections["FCI"] = st.session_state["fci_result"]["adjacent_to_target"]

if has_grasp:
    method_selections["GRaSP"] = st.session_state["grasp_result"]["adjacent_to_target"]

# 統合グラフ (Consensus)
if "consensus_adjacent" in st.session_state:
    method_selections["統合 (2+手法)"] = st.session_state["consensus_adjacent"]

if method_selections:
    # UpSet 的なテーブル
    all_selected = set()
    for s in method_selections.values():
        all_selected |= s

    upset_data = []
    for feat in sorted(all_selected):
        row = {"変数": feat}
        for method, selected in method_selections.items():
            row[method] = "Yes" if feat in selected else ""
        count = sum(1 for s in method_selections.values() if feat in s)
        row["検出手法数"] = count
        if gt:
            row["カテゴリ"] = _cat(feat)
            row["真の直接原因"] = "Yes" if feat in gt["direct_causes"] else ""
        upset_data.append(row)

    upset_df = pd.DataFrame(upset_data).sort_values("検出手法数", ascending=False)
    st.dataframe(upset_df, use_container_width=True)

    # 検出手法数のバーチャート
    fig_upset = go.Figure()
    for feat_row in upset_df.itertuples():
        feat = feat_row.変数
        count = feat_row.検出手法数
        if gt:
            cat = _cat(feat)
            color_map = {
                "直接原因": "#2196F3", "擬似相関": "#F44336",
                "独立ノイズ": "#9E9E9E", "上流変数": "#FF9800", "不明": "#607D8B",
            }
            color = color_map.get(cat, "#607D8B")
        else:
            color = "#2196F3"
        fig_upset.add_trace(go.Bar(
            x=[feat], y=[count], marker_color=color, showlegend=False,
        ))
    fig_upset.update_layout(
        title="変数別 検出手法数",
        yaxis_title="検出した手法数",
        xaxis_tickangle=-45,
        height=400,
    )
    st.plotly_chart(fig_upset, use_container_width=True)

# --- 擬似相関判別能力の比較 ---
if gt:
    st.subheader("擬似相関判別能力の比較")

    spurious_vars = set(gt["spurious"])
    direct_vars = set(gt["direct_causes"])

    ability_data = []
    for method, selected in method_selections.items():
        spurious_excluded = spurious_vars - selected
        direct_detected = direct_vars & selected
        ability_data.append({
            "手法": method,
            "擬似相関の排除率": f"{len(spurious_excluded)}/{len(spurious_vars)} "
                              f"({len(spurious_excluded)/len(spurious_vars)*100:.0f}%)"
                              if spurious_vars else "N/A",
            "直接原因の検出率": f"{len(direct_detected)}/{len(direct_vars)} "
                              f"({len(direct_detected)/len(direct_vars)*100:.0f}%)"
                              if direct_vars else "N/A",
            "誤検出 (FP)": len(selected - direct_vars),
            "ノイズ混入": sorted(spurious_vars & selected) if spurious_vars & selected else "なし",
        })

    ability_df = pd.DataFrame(ability_data)
    st.dataframe(ability_df, use_container_width=True)

    # スケルトン比較 (LiNGAM, PC, GRaSP)
    if (has_lingam or has_pc or has_grasp) and gt.get("true_edges"):
        st.subheader("スケルトン比較 (真の DAG vs 推定)")
        true_edges = gt["true_edges"]
        true_skeleton = {frozenset(e) for e in true_edges}

        skeleton_comp = []
        if has_lingam:
            edges_df = st.session_state["lingam_result"]["significant_edges"]
            est_skel = {frozenset([r["From"], r["To"]]) for _, r in edges_df.iterrows()}
            correct = true_skeleton & est_skel
            prec = len(correct) / len(est_skel) if est_skel else 0
            rec = len(correct) / len(true_skeleton) if true_skeleton else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            skeleton_comp.append({
                "手法": "LiNGAM", "検出数": len(est_skel),
                "Precision": f"{prec:.4f}", "Recall": f"{rec:.4f}", "F1": f"{f1:.4f}",
            })

        if has_pc:
            edges_df = st.session_state["pc_result"]["edges_df"]
            est_skel = {frozenset([r["From"], r["To"]]) for _, r in edges_df.iterrows()}
            correct = true_skeleton & est_skel
            prec = len(correct) / len(est_skel) if est_skel else 0
            rec = len(correct) / len(true_skeleton) if true_skeleton else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            skeleton_comp.append({
                "手法": "PC", "検出数": len(est_skel),
                "Precision": f"{prec:.4f}", "Recall": f"{rec:.4f}", "F1": f"{f1:.4f}",
            })

        if has_grasp:
            edges_df = st.session_state["grasp_result"]["edges_df"]
            est_skel = {frozenset([r["From"], r["To"]]) for _, r in edges_df.iterrows()}
            correct = true_skeleton & est_skel
            prec = len(correct) / len(est_skel) if est_skel else 0
            rec = len(correct) / len(true_skeleton) if true_skeleton else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            skeleton_comp.append({
                "手法": "GRaSP", "検出数": len(est_skel),
                "Precision": f"{prec:.4f}", "Recall": f"{rec:.4f}", "F1": f"{f1:.4f}",
            })

        skeleton_df = pd.DataFrame(skeleton_comp)
        st.dataframe(skeleton_df, use_container_width=True)

        # F1 のバーチャート
        fig_f1 = px.bar(
            skeleton_df, x="手法", y="F1", text="F1",
            title="スケルトン F1 スコア比較",
            color="手法",
        )
        st.plotly_chart(fig_f1, use_container_width=True)

st.markdown("---")
st.markdown("""
### 手法の特性まとめ

| 観点 | LiNGAM | PC | FCI | GRaSP |
|:---|:---|:---|:---|:---|
| **問い** | 因果の方向と強度 | 因果スケルトン | 因果 + 潜在変数 | 因果スケルトン (順列ベース) |
| **擬似相関の扱い** | 区別可能 | 区別可能 | 区別可能 | 区別可能 |
| **方向の識別** | 可能 | 部分的 | 部分的 | 部分的 |
| **アプローチ** | ICA ベース | 制約ベース | 制約ベース (潜在変数対応) | 順列ベース (BIC) |
| **前提条件** | 線形 + 非ガウス | 忠実性 | 忠実性 | Adjacency faithfulness |
| **潜在変数** | 考慮なし | 考慮なし | 考慮あり | 考慮なし |
""")
