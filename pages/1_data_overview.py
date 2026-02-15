"""Page 1: データ概要・EDA"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.header("データ概要")

if "df" not in st.session_state:
    st.warning("メインページでデータを読み込んでください。")
    st.stop()

df = st.session_state["df"]
target = st.session_state["target"]
features = st.session_state["features"]
gt = st.session_state.get("ground_truth")

# --- 基本統計量 ---
st.subheader("基本統計量")
st.dataframe(df.describe().T, use_container_width=True)

# --- 相関ヒートマップ ---
st.subheader("相関ヒートマップ")
corr = df.corr()
fig_heatmap = px.imshow(
    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1, aspect="auto",
)
fig_heatmap.update_layout(height=600)
st.plotly_chart(fig_heatmap, use_container_width=True)

# --- Y との相関係数 ---
st.subheader(f"{target} との相関係数")
corr_with_target = df[features].corrwith(df[target]).sort_values(ascending=False)

if gt:
    def _get_color(feat):
        if feat in gt["direct_causes"]:
            return "#2196F3"
        if feat in gt["spurious"]:
            return "#F44336"
        if feat in gt["independent"]:
            return "#9E9E9E"
        if feat in gt["upstream"]:
            return "#FF9800"
        return "#607D8B"

    def _get_category(feat):
        if feat in gt["direct_causes"]:
            return "直接原因"
        if feat in gt["spurious"]:
            return "擬似相関"
        if feat in gt["independent"]:
            return "独立ノイズ"
        if feat in gt["upstream"]:
            return "上流変数"
        return "不明"

    colors = [_get_color(f) for f in corr_with_target.index]
    categories = [_get_category(f) for f in corr_with_target.index]

    fig_corr = go.Figure()
    for cat, color, label in [
        ("direct_causes", "#2196F3", "直接原因"),
        ("spurious", "#F44336", "擬似相関"),
        ("upstream", "#FF9800", "上流変数"),
        ("independent", "#9E9E9E", "独立ノイズ"),
    ]:
        mask = [f in gt.get(cat, []) for f in corr_with_target.index]
        vals = corr_with_target[mask]
        if len(vals) > 0:
            fig_corr.add_trace(go.Bar(
                x=vals.index.tolist(), y=vals.values,
                name=label, marker_color=color,
            ))
    fig_corr.update_layout(
        barmode="group", xaxis_tickangle=-45,
        title=f"各特徴量と {target} の相関係数",
        yaxis_title="相関係数",
    )
else:
    fig_corr = px.bar(
        x=corr_with_target.index, y=corr_with_target.values,
        labels={"x": "特徴量", "y": "相関係数"},
        title=f"各特徴量と {target} の相関係数",
    )

st.plotly_chart(fig_corr, use_container_width=True)

# --- 散布図マトリクス ---
st.subheader("散布図マトリクス")
selected_vars = st.multiselect(
    "表示する変数を選択", columns := [target] + features,
    default=[target] + features[:min(4, len(features))],
)
if len(selected_vars) >= 2:
    fig_scatter = px.scatter_matrix(
        df[selected_vars], dimensions=selected_vars, height=600,
    )
    fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("2つ以上の変数を選択してください。")
