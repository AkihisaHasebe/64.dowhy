"""因果分析比較ダッシュボード — メインエントリーポイント"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="因果分析比較ダッシュボード",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("因果分析比較ダッシュボード")
st.markdown(
    "LiNGAM / PC / FCI / GES / GRaSP の結果を統合的に比較し、介入優先度を算出するツール"
)

# --- サイドバー: データ設定 ---
st.sidebar.header("データ設定")

# CSV アップロード or サンプルデータ選択
upload = st.sidebar.file_uploader("CSV をアップロード", type=["csv"])

base_dir = os.path.dirname(__file__)
sample_csvs = sorted([
    f for f in os.listdir(base_dir)
    if f.endswith(".csv") and os.path.isfile(os.path.join(base_dir, f))
])

data_filename = None

if upload is not None:
    df = pd.read_csv(upload)
    data_filename = upload.name
    st.sidebar.success(f"アップロード完了: {data_filename}")
elif sample_csvs:
    selected_csv = st.sidebar.selectbox(
        "サンプルデータ", sample_csvs, index=0,
    )
    df = pd.read_csv(os.path.join(base_dir, selected_csv))
    data_filename = selected_csv
    st.sidebar.info(f"{selected_csv} を使用中")
else:
    st.error("CSV ファイルが見つかりません。アップロードしてください。")
    st.stop()

st.session_state["df"] = df
st.session_state["data_filename"] = data_filename

# 目的変数の選択
columns = list(df.columns)

# 自動認識を試みる
from analysis.ground_truth import lookup_ground_truth, categorize_feature, CATEGORY_COLORS
auto_gt = lookup_ground_truth(data_filename, columns) if data_filename else None

if auto_gt:
    default_target_idx = columns.index(auto_gt["target"]) if auto_gt["target"] in columns else len(columns) - 1
else:
    default_target_idx = len(columns) - 1

target = st.sidebar.selectbox("目的変数", columns, index=default_target_idx)
st.session_state["target"] = target

features = [c for c in columns if c != target]
st.session_state["features"] = features

# --- 真の因果構造設定 (オプション) ---
st.sidebar.header("真の因果構造 (オプション)")

gt_mode = st.sidebar.radio(
    "因果構造の設定方法",
    ["自動認識", "手動設定", "なし (未知のデータ)"],
    index=0 if auto_gt else 2,
)

if gt_mode == "自動認識" and auto_gt:
    # レジストリから自動取得
    st.sidebar.success(f"レジストリから自動認識: {data_filename}")
    gt = auto_gt

    # 確認用に表示
    with st.sidebar.expander("認識された因果構造を確認"):
        st.write(f"**目的変数**: {gt['target']}")
        st.write(f"**直接原因**: {gt['direct_causes']}")
        st.write(f"**擬似相関**: {gt['spurious']}")
        st.write(f"**独立ノイズ**: {gt['independent']}")
        st.write(f"**上流変数**: {gt['upstream']}")
        st.write(f"**エッジ数**: {len(gt['true_edges'])}")

    st.session_state["ground_truth"] = gt

elif gt_mode == "手動設定":
    st.sidebar.info("各変数のカテゴリを手動で設定してください。")

    direct_causes = st.sidebar.multiselect("直接原因", features, default=[])
    spurious = st.sidebar.multiselect(
        "擬似相関", [f for f in features if f not in direct_causes], default=[],
    )
    remaining = [f for f in features if f not in direct_causes and f not in spurious]
    independent = st.sidebar.multiselect("独立ノイズ", remaining, default=[])
    upstream = [f for f in remaining if f not in independent]

    # 真のエッジ (手動入力 — オプション)
    true_edges_input = st.sidebar.text_area(
        "真のエッジ (1行1エッジ, 例: A -> B)",
        placeholder="Severity -> Outcome\nTreatment -> Outcome",
    )
    true_edges = set()
    if true_edges_input.strip():
        for line in true_edges_input.strip().split("\n"):
            parts = [p.strip() for p in line.replace("->", ",").replace("→", ",").split(",")]
            if len(parts) == 2 and parts[0] and parts[1]:
                true_edges.add((parts[0], parts[1]))

    st.session_state["ground_truth"] = {
        "target": target,
        "direct_causes": direct_causes,
        "spurious": spurious,
        "independent": independent,
        "upstream": upstream,
        "true_edges": true_edges if true_edges else None,
        "true_coefficients": {},
    }

else:
    st.session_state["ground_truth"] = None

# --- メインページ: データプレビュー ---
st.subheader("データプレビュー")
col1, col2, col3 = st.columns(3)
col1.metric("サンプル数", df.shape[0])
col2.metric("特徴量数", len(features))
col3.metric("目的変数", target)

st.dataframe(df.head(10), use_container_width=True)

# 因果構造の概要
gt = st.session_state.get("ground_truth")
if gt:
    st.subheader("因果構造の概要")
    cat_cols = st.columns(4)
    cat_cols[0].markdown(f"**直接原因** ({len(gt['direct_causes'])})")
    cat_cols[0].write(gt["direct_causes"] if gt["direct_causes"] else "なし")
    cat_cols[1].markdown(f"**擬似相関** ({len(gt['spurious'])})")
    cat_cols[1].write(gt["spurious"] if gt["spurious"] else "なし")
    cat_cols[2].markdown(f"**独立ノイズ** ({len(gt['independent'])})")
    cat_cols[2].write(gt["independent"] if gt["independent"] else "なし")
    cat_cols[3].markdown(f"**上流変数** ({len(gt['upstream'])})")
    cat_cols[3].write(gt["upstream"] if gt["upstream"] else "なし")

st.markdown("---")

# --- 分析フロー図 ---
st.subheader("分析の進め方")

st.graphviz_chart("""
digraph analysis_flow {
    graph [
        rankdir=TB
        bgcolor="transparent"
        fontname="Meiryo, sans-serif"
        pad="0.3"
        nodesep="0.4"
        ranksep="0.6"
    ]
    node [
        shape=box
        style="rounded,filled"
        fontname="Meiryo, sans-serif"
        fontsize="11"
        margin="0.15,0.08"
        penwidth="1.5"
    ]
    edge [
        fontname="Meiryo, sans-serif"
        fontsize="9"
        color="#666666"
        penwidth="1.2"
    ]

    // ---- Step 0 ----
    step0 [
        label="Step 0: データ準備\\n\\nCSV アップロード / 選択\\n目的変数の選択\\n真の因果構造の設定 (任意)"
        fillcolor="#E3F2FD"
        color="#1565C0"
    ]

    // ---- Step 1 ----
    step1 [
        label="Step 1: データ概要\\n\\n基本統計量\\n相関ヒートマップ\\n目的変数との相関係数"
        fillcolor="#E8F5E9"
        color="#2E7D32"
    ]

    // ---- Causal methods ----
    step2_lingam [
        label="LiNGAM\\n\\nDirectLiNGAM + Bootstrap\\n因果の方向と強度を推定\\nDAG の推定"
        fillcolor="#E8EAF6"
        color="#283593"
    ]

    step2_pc [
        label="PC + FCI\\n\\nPC: CPDAG + Bootstrap\\nFCI: PAG (潜在変数を許容)\\n条件付き独立性検定"
        fillcolor="#EDE7F6"
        color="#4527A0"
    ]

    step2_fges [
        label="FGES\\n\\nスコアベース最適化 (BIC) + Bootstrap\\nCPDAG の推定\\n(GES の高速版)"
        fillcolor="#E0F2F1"
        color="#00695C"
    ]

    step2_grasp [
        label="GRaSP\\n\\n順列ベース (BIC) + Bootstrap\\nCPDAG の推定"
        fillcolor="#E0F7FA"
        color="#00838F"
    ]

    // ---- Consensus + DoWhy ----
    consensus [
        label="統合因果グラフ構築\\n\\n2+ 手法が検出したエッジを統合\\nDoWhy による因果効果推定 (ATE)\\nBackdoor Adjustment"
        fillcolor="#FCE4EC"
        color="#C2185B"
    ]

    // ---- Integration ----
    step3 [
        label="Step 2: 介入優先度サマリー\\n\\n統合介入スコア = |効果量| × 因果確信度\\n因果確信度 = 全手法の平均確率\\n効果量 = DoWhy ATE > LiNGAM > OLS"
        fillcolor="#FFF3E0"
        color="#E65100"
    ]

    step4 [
        label="Step 3: 全手法統合比較\\n\\n手法間の変数選択比較\\n擬似相関の判別能力評価\\nスケルトン Precision / Recall / F1"
        fillcolor="#FFF3E0"
        color="#E65100"
    ]

    // ---- Conclusion ----
    conclusion [
        label="結論\\n\\n介入価値の高い変数を特定\\n因果エビデンスに基づく優先度付け\\n擬似相関変数を排除"
        fillcolor="#F3E5F5"
        color="#6A1B9A"
        penwidth="2"
    ]

    // ---- Edges ----
    step0 -> step1

    step1 -> step2_lingam [label="ICA ベース"]
    step1 -> step2_pc [label="制約ベース"]
    step1 -> step2_fges [label="スコアベース"]
    step1 -> step2_grasp [label="順列ベース"]

    step2_lingam -> consensus
    step2_pc -> consensus
    step2_fges -> consensus
    step2_grasp -> consensus

    consensus -> step3
    step3 -> step4
    step4 -> conclusion

    // ---- Layout hints ----
    { rank=same; step2_lingam; step2_pc; step2_fges; step2_grasp }
}
""", use_container_width=True)

st.caption(
    "5つの因果探索手法 (LiNGAM / PC / FCI / FGES / GRaSP) のアンサンブル結果から統合因果グラフを構築し、"
    "DoWhy による因果効果推定 (Backdoor Adjustment) を実行することで、"
    "因果関係の信頼度を高め、介入効果の大きい変数を特定します。"
)
