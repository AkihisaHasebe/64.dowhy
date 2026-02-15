"""Page 2: 分析実行 — 手法選択 → 一括実行 → 結果表示"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
matplotlib.rcParams['font.family'] = 'Meiryo'

st.header("分析実行")

if "df" not in st.session_state:
    st.warning("メインページでデータを読み込んでください。")
    st.stop()

df = st.session_state["df"]
target = st.session_state["target"]
features = st.session_state["features"]
gt = st.session_state.get("ground_truth")


# ============================================================
# ユーティリティ
# ============================================================

def _cat(f):
    if not gt:
        return "不明"
    if f in gt["direct_causes"]:
        return "直接原因"
    if f in gt["spurious"]:
        return "擬似相関"
    if f in gt["independent"]:
        return "独立ノイズ"
    if f in gt["upstream"]:
        return "上流変数"
    return "不明"


def _color(f):
    cat = _cat(f)
    return {
        "直接原因": "#2196F3", "擬似相関": "#F44336",
        "独立ノイズ": "#9E9E9E", "上流変数": "#FF9800",
    }.get(cat, "#607D8B")


# ============================================================
# サイドバー: 手法選択 + パラメータ
# ============================================================

st.sidebar.subheader("実行する手法")
run_lingam = st.sidebar.checkbox("LiNGAM", value=True, key="sel_lingam")
run_pc = st.sidebar.checkbox("PC", value=True, key="sel_pc")
run_fci = st.sidebar.checkbox("FCI", value=True, key="sel_gfci")
run_grasp = st.sidebar.checkbox("GRaSP", value=True, key="sel_grasp")

# --- LiNGAM パラメータ ---
if run_lingam:
    st.sidebar.subheader("LiNGAM パラメータ")
    lingam_bootstrap = st.sidebar.slider("Bootstrap 回数", 10, 200, 100, 10,
                                          key="lingam_bs")
    lingam_min_effect = st.sidebar.number_input("min_causal_effect", 0.001, 0.1,
                                                 0.01, 0.005, key="lingam_me")
    lingam_threshold = st.sidebar.slider("エッジ採用確率閾値", 0.1, 0.9, 0.50, 0.05,
                                          key="lingam_th")

# --- PC / FCI 共通パラメータ ---
if run_pc or run_fci:
    st.sidebar.subheader("PC / FCI パラメータ")
    pcfci_alpha = st.sidebar.slider("有意水準 (alpha)", 0.01, 0.2, 0.05, 0.01,
                                     key="pcfci_a")
    pcfci_indep = st.sidebar.selectbox("独立性検定",
                                        ["fisherz", "chisq", "gsq", "kci"],
                                        key="pcfci_ind")

if run_pc:
    pc_bootstrap = st.sidebar.slider("Bootstrap 回数 (PC)", 10, 200, 100, 10,
                                      key="pc_bs")
    pc_threshold = st.sidebar.slider("エッジ採用確率閾値 (PC)", 0.1, 0.9, 0.50, 0.05,
                                      key="pc_th")

# --- GRaSP パラメータ ---
if run_grasp:
    st.sidebar.subheader("GRaSP パラメータ")
    grasp_score_func = st.sidebar.selectbox(
        "スコア関数", ["local_score_BIC", "local_score_BDeu"],
        key="grasp_score")
    grasp_depth = st.sidebar.slider("探索深度 (depth)", 1, 5, 3, 1,
                                     key="grasp_depth")
    grasp_bootstrap = st.sidebar.slider("Bootstrap 回数 (GRaSP)", 10, 200, 100, 10,
                                         key="grasp_bs")
    grasp_threshold = st.sidebar.slider("エッジ採用確率閾値 (GRaSP)", 0.1, 0.9, 0.50, 0.05,
                                         key="grasp_th")

# ============================================================
# 実行ボタン
# ============================================================

if not any([run_lingam, run_pc, run_fci, run_grasp]):
    st.info("サイドバーから実行する手法を1つ以上選択してください。")
    st.stop()

# ============================================================
# 実行時間推定関数
# ============================================================

def estimate_execution_time(df, methods_config):
    """
    データサイズとパラメータから実行時間を推定する

    Args:
        df: データフレーム
        methods_config: dict with method names and their parameters

    Returns:
        dict: {method: estimated_seconds, "total": total_seconds}
    """
    n_samples = len(df)
    n_features = len(df.columns)

    estimates = {}

    # 経験的係数 (2000サンプル, 20変数程度のデータでの実測値ベース)
    # 実際の環境やデータ特性により変動するため、あくまで目安

    for method, params in methods_config.items():
        if method == "LiNGAM":
            # O(n³) + Bootstrap
            # 基準: 2000サンプル, 100 bootstrap で約10-20秒
            base_time = 0.15 * (n_samples / 2000) ** 2  # ICA計算
            bootstrap_time = params.get("bootstrap", 100) * 0.12
            estimates["LiNGAM"] = base_time + bootstrap_time

        elif method == "PC":
            # O(n × p²) + Bootstrap
            # 基準: 2000サンプル, 20変数, 100 bootstrap で約30-50秒
            base_time = 0.3 * (n_samples / 2000) * (n_features / 20) ** 2
            bootstrap_time = params.get("bootstrap", 100) * 0.35
            estimates["PC"] = base_time + bootstrap_time

        elif method == "FCI":
            # O(n × p²) with greedy optimization, bootstrapなし
            # 基準: 2000サンプル, 20変数で約2-4秒 (条件付き独立性検定ベース)
            estimates["FCI"] = 3.0 * (n_samples / 2000) * (n_features / 20) ** 2

        elif method == "GRaSP":
            # O(p³ × depth) + Bootstrap
            # 基準: 20変数, depth=3, 100 bootstrap で約25-40秒
            depth = params.get("depth", 3)
            base_time = 0.6 * (n_features / 20) ** 3 * (depth / 3)
            bootstrap_time = params.get("bootstrap", 100) * 0.3
            estimates["GRaSP"] = base_time + bootstrap_time

    estimates["total"] = sum(estimates.values())
    return estimates

# ============================================================
# 実行時間推定の表示
# ============================================================

if any([run_lingam, run_pc, run_fci, run_grasp]):
    # 選択された手法とパラメータをまとめる
    methods_config = {}
    if run_lingam:
        methods_config["LiNGAM"] = {"bootstrap": lingam_bootstrap}
    if run_pc:
        methods_config["PC"] = {"bootstrap": pc_bootstrap}
    if run_fci:
        methods_config["FCI"] = {}
    if run_grasp:
        methods_config["GRaSP"] = {"bootstrap": grasp_bootstrap, "depth": grasp_depth}

    # 推定時間を計算
    time_estimates = estimate_execution_time(df, methods_config)
    total_estimate = time_estimates["total"]

    # 推定時間を表示
    if total_estimate < 60:
        time_str = f"約 {total_estimate:.0f} 秒"
    else:
        minutes = int(total_estimate // 60)
        seconds = int(total_estimate % 60)
        time_str = f"約 {minutes} 分 {seconds} 秒"

    # 警告レベルに応じた表示
    if total_estimate > 180:  # 3分以上
        st.warning(f"⚠️ 推定実行時間: {time_str} (データサイズや bootstrap 回数が大きいため時間がかかります)")
    elif total_estimate > 60:  # 1分以上
        st.info(f"⏱️ 推定実行時間: {time_str}")
    else:
        st.caption(f"⏱️ 推定実行時間: {time_str}")

    # 詳細を expander で表示
    with st.expander("📊 手法別の推定実行時間", expanded=False):
        est_df = pd.DataFrame([
            {"手法": method, "推定時間 (秒)": f"{est:.1f}",
             "割合 (%)": f"{est/total_estimate*100:.0f}"}
            for method, est in time_estimates.items() if method != "total"
        ])
        st.dataframe(est_df, use_container_width=True, hide_index=True)
        st.caption(
            "※ 推定時間は目安です。実際の実行時間は CPU 性能、メモリ、データ特性により変動します。"
        )

if st.button("分析を実行", type="primary", use_container_width=True):
    methods = []
    if run_lingam:
        methods.append("LiNGAM")
    if run_pc:
        methods.append("PC")
    if run_fci:
        methods.append("FCI")
    if run_grasp:
        methods.append("GRaSP")

    # 進捗表示用のコンテナ
    progress_container = st.container()
    with progress_container:
        progress = st.progress(0, text="準備中...")
        status_text = st.empty()

    total = len(methods)
    timing_results = {}
    overall_start = time.time()

    for i, method in enumerate(methods):
        method_start = time.time()
        start_time_str = datetime.now().strftime("%H:%M:%S")

        # 開始メッセージ
        progress.progress(i / total, text=f"{method} 実行中...")
        status_text.markdown(f"**{method}** を実行中... (開始: {start_time_str})")

        if method == "LiNGAM":
            from analysis.lingam_analysis import run_lingam as _run_lingam
            status_text.markdown(
                f"**LiNGAM** を実行中...\n"
                f"- DirectLiNGAM アルゴリズムで因果順序を推定中\n"
                f"- Bootstrap サンプリング: {lingam_bootstrap} 回\n"
                f"- 開始: {start_time_str}"
            )
            result = _run_lingam(df, target, lingam_bootstrap, lingam_min_effect,
                                 lingam_threshold)
            st.session_state["lingam_result"] = result
            elapsed = time.time() - method_start
            timing_results["LiNGAM"] = elapsed
            progress.progress((i + 0.5) / total,
                            text=f"{method} 完了 ({elapsed:.1f}秒)")
            status_text.markdown(
                f"✅ **LiNGAM** 完了 ({elapsed:.1f} 秒) — "
                f"因果順序: {len(result['causal_order'])} 変数"
            )

        elif method == "PC":
            from analysis.pc_fci_analysis import run_pc as _run_pc
            data = df.values
            column_names = list(df.columns)
            status_text.markdown(
                f"**PC** を実行中...\n"
                f"- 条件付き独立性検定 ({pcfci_indep}) で CPDAG を構築中\n"
                f"- Bootstrap サンプリング: {pc_bootstrap} 回\n"
                f"- 有意水準: {pcfci_alpha}\n"
                f"- 開始: {start_time_str}"
            )
            result = _run_pc(data, column_names, target, pcfci_alpha, pcfci_indep,
                             pc_bootstrap, pc_threshold)
            st.session_state["pc_result"] = result
            elapsed = time.time() - method_start
            timing_results["PC"] = elapsed
            progress.progress((i + 0.5) / total,
                            text=f"{method} 完了 ({elapsed:.1f}秒)")
            status_text.markdown(
                f"✅ **PC** 完了 ({elapsed:.1f} 秒) — "
                f"検出エッジ: {len(result['edges_df'])} 本"
            )

        elif method == "FCI":
            from analysis.pc_fci_analysis import run_fci as _run_fci
            data = df.values
            column_names = list(df.columns)
            status_text.markdown(
                f"**FCI** を実行中...\n"
                f"- 潜在交絡因子を考慮した PAG を構築中\n"
                f"- 独立性検定: {pcfci_indep}\n"
                f"- 有意水準: {pcfci_alpha}\n"
                f"- 開始: {start_time_str}"
            )
            result = _run_fci(data, column_names, target, pcfci_alpha, pcfci_indep)
            st.session_state["fci_result"] = result
            elapsed = time.time() - method_start
            timing_results["FCI"] = elapsed
            progress.progress((i + 0.5) / total,
                            text=f"{method} 完了 ({elapsed:.1f}秒)")
            status_text.markdown(
                f"✅ **FCI** 完了 ({elapsed:.1f} 秒) — "
                f"検出エッジ: {len(result['edges_df'])} 本"
            )

        elif method == "GRaSP":
            from analysis.grasp_analysis import run_grasp as _run_grasp
            data = df.values
            column_names = list(df.columns)
            status_text.markdown(
                f"**GRaSP** を実行中...\n"
                f"- 順列ベースの因果探索 (depth={grasp_depth})\n"
                f"- Bootstrap サンプリング: {grasp_bootstrap} 回\n"
                f"- スコア関数: {grasp_score_func}\n"
                f"- 開始: {start_time_str}"
            )
            result = _run_grasp(data, column_names, target, grasp_score_func,
                                depth=grasp_depth, n_bootstrap=grasp_bootstrap,
                                threshold=grasp_threshold)
            st.session_state["grasp_result"] = result
            elapsed = time.time() - method_start
            timing_results["GRaSP"] = elapsed
            progress.progress((i + 0.5) / total,
                            text=f"{method} 完了 ({elapsed:.1f}秒)")
            status_text.markdown(
                f"✅ **GRaSP** 完了 ({elapsed:.1f} 秒) — "
                f"検出エッジ: {len(result['edges_df'])} 本"
            )

    overall_elapsed = time.time() - overall_start

    # --- 統合グラフ構築 + DoWhy 因果効果推定 ---
    consensus_start = time.time()
    consensus_start_str = datetime.now().strftime("%H:%M:%S")
    progress.progress(0.95, text="統合因果グラフ構築 + DoWhy 因果効果推定中...")
    status_text.markdown(
        f"**統合因果グラフ構築** を実行中...\n"
        f"- 複数手法の結果を統合してコンセンサスグラフを構築\n"
        f"- DoWhy による因果効果推定 (Backdoor criterion)\n"
        f"- 開始: {consensus_start_str}"
    )
    try:
        from analysis.consensus_graph import build_consensus_graph, get_adjacent_to_target
        from analysis.dowhy_estimation import estimate_causal_effects_with_dowhy

        # 統合グラフ構築
        consensus_graph, edge_support_df = build_consensus_graph(
            st.session_state, target, features, min_agreement=2
        )
        st.session_state["consensus_graph"] = consensus_graph
        st.session_state["edge_support_df"] = edge_support_df
        st.session_state["consensus_adjacent"] = get_adjacent_to_target(
            consensus_graph, target
        )

        # DoWhy 因果効果推定
        dowhy_results = estimate_causal_effects_with_dowhy(
            df, target, features, consensus_graph
        )
        st.session_state["dowhy_results"] = dowhy_results

        consensus_elapsed = time.time() - consensus_start
        status_text.markdown(
            f"✅ **統合処理** 完了 ({consensus_elapsed:.1f} 秒) — "
            f"統合エッジ: {len(edge_support_df)} 本"
        )

    except Exception as e:
        st.session_state["consensus_graph"] = None
        st.session_state["dowhy_results"] = {}
        st.warning(f"統合グラフ or DoWhy 推定でエラー: {e}")

    overall_elapsed = time.time() - overall_start
    end_time_str = datetime.now().strftime("%H:%M:%S")
    progress.progress(1.0, text=f"完了 (合計: {overall_elapsed:.1f}秒)")
    status_text.markdown(
        f"🎉 **全ての分析が完了しました！**\n\n"
        f"- 実行手法: {', '.join(methods)}\n"
        f"- 総実行時間: {overall_elapsed:.1f} 秒\n"
        f"- 終了: {end_time_str}"
    )

    # 時間計測結果を保存
    st.session_state["timing_results"] = timing_results
    st.session_state["overall_time"] = overall_elapsed

    # 完了メッセージに時間を含める
    st.success(f"{', '.join(methods)} の実行が完了しました（合計 {overall_elapsed:.1f} 秒）")

    # 実行時間サマリー
    with st.expander("⏱️ 実行時間の詳細", expanded=False):
        timing_df = pd.DataFrame([
            {"手法": method, "実行時間 (秒)": f"{elapsed:.2f}",
             "割合 (%)": f"{elapsed/overall_elapsed*100:.1f}"}
            for method, elapsed in timing_results.items()
        ])
        st.dataframe(timing_df, use_container_width=True, hide_index=True)

        # 計算量の参考情報
        st.caption(
            f"**計算量の目安** (サンプル数: {len(df)}, 変数数: {len(df.columns)})\n"
            f"- LiNGAM: O(n³) - ICA反復 + Bootstrap\n"
            f"- PC/FCI: O(n × p²) - 条件付き独立性検定\n"
            f"- GRaSP: O(p³ × depth) - 順列探索 + Bootstrap\n"
            f"※ n=サンプル数, p=変数数"
        )

# ============================================================
# 結果がなければ停止
# ============================================================

has_lingam = "lingam_result" in st.session_state
has_pc = "pc_result" in st.session_state
has_fci = "fci_result" in st.session_state
has_grasp = "grasp_result" in st.session_state

if not any([has_lingam, has_pc, has_fci, has_grasp]):
    st.info("「分析を実行」ボタンを押してください。")
    st.stop()

# ============================================================
# 介入優先度サマリー (結果の冒頭に表示)
# ============================================================

st.markdown("---")
st.subheader(f"{target} への介入優先度サマリー")

# 実行時間情報の表示（既存結果の場合も表示）
if "timing_results" in st.session_state and st.session_state["timing_results"]:
    timing_info = st.session_state["timing_results"]
    overall_time = st.session_state.get("overall_time", sum(timing_info.values()))
    executed_methods = list(timing_info.keys())

    st.info(
        f"✓ 実行済み手法: {', '.join(executed_methods)} "
        f"（合計: {overall_time:.1f} 秒）"
    )

# ============================================================
# 効果量の推定 (OLS 標準化回帰係数 — 全手法で利用可能なベースライン)
# ============================================================
from sklearn.linear_model import LinearRegression

_X_all = df[features].values
_y_all = df[target].values
_lr = LinearRegression().fit(_X_all, _y_all)
_sds = df[features].std().values
_sd_y = df[target].std()
# 標準化回帰係数: β_std = β_raw × (SD_x / SD_y)
_ols_std_coefs = pd.Series(
    _lr.coef_ * _sds / _sd_y, index=features
)

# ============================================================
# 各変数の統合スコアを算出
# ============================================================

# DoWhy 結果の取得
dowhy_results = st.session_state.get("dowhy_results", {})
has_dowhy = len(dowhy_results) > 0

summary_rows = []
for f in features:
    row = {"変数": f}

    # --- 効果量 (標準化) ---
    # 優先順位: DoWhy ATE > LiNGAM 総介入効果 > OLS 回帰係数
    if has_dowhy and f in dowhy_results and not np.isnan(dowhy_results[f].get("ate", np.nan)):
        # DoWhy ATE を標準化 (SD単位に変換)
        ate = dowhy_results[f]["ate"]
        # ATE は通常「1単位増加に対する効果」なので、標準化するには SD で割る/掛ける
        # ここでは既に推定値が出ているので、そのまま使用 (後で標準化オプションを追加可能)
        row["効果量(std)"] = ate
        row["直接効果(std)"] = np.nan
        row["間接効果(std)"] = np.nan
        row["効果量ソース"] = "DoWhy"
        row["DoWhy識別"] = dowhy_results[f].get("identified", False)
    elif has_lingam:
        res_l = st.session_state["lingam_result"]
        std_t = res_l["std_total_effects"]
        std_d = res_l["std_direct_effects"]
        row["効果量(std)"] = std_t.get(f, 0)
        row["直接効果(std)"] = std_d.get(f, 0)
        row["間接効果(std)"] = std_t.get(f, 0) - std_d.get(f, 0)
        row["効果量ソース"] = "LiNGAM"
        row["DoWhy識別"] = np.nan
    else:
        row["効果量(std)"] = _ols_std_coefs.get(f, 0)
        row["直接効果(std)"] = np.nan
        row["間接効果(std)"] = np.nan
        row["効果量ソース"] = "OLS"
        row["DoWhy識別"] = np.nan
    row["|効果量(std)|"] = abs(row["効果量(std)"])

    # --- 各手法の因果確信度 (0-1) ---
    confidence_signals = []

    if has_lingam:
        prob = st.session_state["lingam_result"]["edge_probs_to_target"]
        row["LiNGAM確率"] = prob.get(f, 0)
        confidence_signals.append(prob.get(f, 0))
    else:
        row["LiNGAM確率"] = np.nan

    if has_pc:
        pc_probs = st.session_state["pc_result"]["bootstrap_probs"]
        # target 行 (target←f) または target 列 (f→target) の最大値
        pc_prob_f = max(
            pc_probs.loc[target].get(f, 0) if f in pc_probs.columns else 0,
            pc_probs[target].get(f, 0) if f in pc_probs.index else 0,
        )
        row["PC確率"] = pc_prob_f
        row["PC隣接"] = 1 if f in st.session_state["pc_result"]["adjacent_to_target"] else 0
        confidence_signals.append(pc_prob_f)
    else:
        row["PC確率"] = np.nan
        row["PC隣接"] = np.nan

    if has_fci:
        fci_adj = 1.0 if f in st.session_state["fci_result"]["adjacent_to_target"] else 0.0
        row["FCI隣接"] = int(fci_adj)
        confidence_signals.append(fci_adj)
    else:
        row["FCI隣接"] = np.nan

    if has_grasp:
        grasp_probs = st.session_state["grasp_result"]["bootstrap_probs"]
        grasp_prob_f = max(
            grasp_probs.loc[target].get(f, 0) if f in grasp_probs.columns else 0,
            grasp_probs[target].get(f, 0) if f in grasp_probs.index else 0,
        )
        row["GRaSP確率"] = grasp_prob_f
        row["GRaSP隣接"] = 1 if f in st.session_state["grasp_result"]["adjacent_to_target"] else 0
        confidence_signals.append(grasp_prob_f)
    else:
        row["GRaSP確率"] = np.nan
        row["GRaSP隣接"] = np.nan

    # --- 統合因果確信度 (全手法の平均) ---
    if confidence_signals:
        row["因果確信度"] = np.mean(confidence_signals)
    else:
        row["因果確信度"] = 0.0

    # --- 統合介入スコア = |効果量| × 因果確信度 ---
    row["統合介入スコア"] = row["|効果量(std)|"] * row["因果確信度"]

    # --- 因果エビデンス数 (判定用) ---
    causal_evidence = 0
    if has_lingam and row["LiNGAM確率"] > 0.5:
        causal_evidence += 1
    if has_pc and row.get("PC隣接", 0) == 1:
        causal_evidence += 1
    if has_fci and row.get("FCI隣接", 0) == 1:
        causal_evidence += 1
    if has_grasp and row.get("GRaSP隣接", 0) == 1:
        causal_evidence += 1
    row["因果エビデンス数"] = causal_evidence

    # --- 判定ロジック ---
    n_causal_methods = sum([has_lingam, has_pc, has_fci, has_grasp])
    if n_causal_methods > 0 and causal_evidence >= 2:
        row["判定"] = "直接原因 (高確信)"
    elif n_causal_methods > 0 and causal_evidence == 1:
        row["判定"] = "直接原因 (低確信)"
    else:
        row["判定"] = "影響なし"

    if gt:
        row["真のカテゴリ"] = _cat(f)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("変数")
summary_df = summary_df.sort_values("統合介入スコア", ascending=False)

# ============================================================
# 統合介入スコア バーチャート
# ============================================================

st.markdown(f"**{target} への統合介入スコア**")
st.caption(
    "統合介入スコア = |効果量 (標準化)| × 因果確信度。"
    "効果量は **DoWhy による因果効果推定 (統合グラフ使用)** > LiNGAM の総因果効果 > OLS 回帰係数 の優先順位で選択。"
    "因果確信度は LiNGAM / PC / FCI / GRaSP の平均確率 (0〜1)。"
    "「介入した時にどれだけ目的変数が変わるか」と"
    "「その因果関係がどれだけ信頼できるか」の両方を反映。"
)

fig_score = go.Figure()

for _, row in summary_df.iterrows():
    feat = row.name
    score = row["統合介入スコア"]
    judgment = row["判定"]

    if "高確信" in judgment:
        color = "#1565C0"
    elif "低確信" in judgment:
        color = "#64B5F6"
    else:
        color = "#E0E0E0"

    fig_score.add_trace(go.Bar(
        y=[feat], x=[score], orientation="h",
        marker_color=color, showlegend=False,
        text=f"{score:.3f} ({judgment})", textposition="outside",
        hovertemplate=(
            f"<b>{feat}</b><br>"
            f"|効果量|: {row['|効果量(std)|']:.3f} SD<br>"
            f"因果確信度: {row['因果確信度']:.2f}<br>"
            f"統合スコア: {score:.3f}<extra></extra>"
        ),
    ))

fig_score.update_layout(
    xaxis_title="統合介入スコア (|効果量| × 因果確信度)",
    yaxis=dict(autorange="reversed"),
    height=max(400, len(features) * 40),
    margin=dict(t=10, l=120, r=150),
)
st.plotly_chart(fig_score, use_container_width=True)

# ============================================================
# 内訳: 効果量 と 因果確信度 を並べて表示
# ============================================================

col_effect, col_conf = st.columns(2)

with col_effect:
    st.markdown(f"**効果量 |β| (標準化, SD 単位)**")
    fig_eff = go.Figure()
    for _, row in summary_df.iterrows():
        feat = row.name
        eff = row["効果量(std)"]
        src = row["効果量ソース"]
        judgment = row["判定"]
        if "高確信" in judgment:
            color = "#1565C0"
        elif "低確信" in judgment:
            color = "#64B5F6"
        else:
            color = "#E0E0E0"

        if has_lingam and not np.isnan(row.get("直接効果(std)", np.nan)):
            direct = row["直接効果(std)"]
            indirect = row["間接効果(std)"]
            fig_eff.add_trace(go.Bar(
                y=[feat], x=[direct], orientation="h",
                marker_color=color, marker_opacity=1.0,
                name="直接", showlegend=(feat == summary_df.index[0]),
                legendgroup="direct",
            ))
            fig_eff.add_trace(go.Bar(
                y=[feat], x=[indirect], orientation="h",
                marker_color=color, marker_opacity=0.4,
                name="間接", showlegend=(feat == summary_df.index[0]),
                legendgroup="indirect",
            ))
        else:
            fig_eff.add_trace(go.Bar(
                y=[feat], x=[eff], orientation="h",
                marker_color=color, showlegend=False,
                text=f"{eff:+.3f} ({src})", textposition="auto",
            ))
    fig_eff.update_layout(
        barmode="relative",
        xaxis_title="効果量 (SD単位)",
        yaxis=dict(autorange="reversed"),
        height=max(350, len(features) * 32),
        margin=dict(t=10, l=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig_eff.add_vline(x=0, line_color="gray", line_width=1)
    st.plotly_chart(fig_eff, use_container_width=True)

with col_conf:
    st.markdown(f"**因果確信度 (各手法の確率)**")
    conf_cols_data = []
    method_names = []
    method_colors = []
    if has_lingam:
        conf_cols_data.append("LiNGAM確率")
        method_names.append("LiNGAM")
        method_colors.append("#283593")
    if has_pc:
        conf_cols_data.append("PC確率")
        method_names.append("PC")
        method_colors.append("#4527A0")
    if has_fci:
        conf_cols_data.append("FCI隣接")
        method_names.append("FCI")
        method_colors.append("#6A1B9A")
    if has_grasp:
        conf_cols_data.append("GRaSP確率")
        method_names.append("GRaSP")
        method_colors.append("#00838F")

    fig_conf = go.Figure()
    for col_name, m_name, m_color in zip(conf_cols_data, method_names, method_colors):
        fig_conf.add_trace(go.Bar(
            y=summary_df.index, x=summary_df[col_name].fillna(0),
            name=m_name, marker_color=m_color, orientation="h",
        ))

    # 統合確信度を線で重ねる
    fig_conf.add_trace(go.Scatter(
        y=summary_df.index, x=summary_df["因果確信度"],
        mode="markers+lines", name="統合確信度",
        marker=dict(color="#FF6F00", size=8, symbol="diamond"),
        line=dict(color="#FF6F00", width=2),
    ))

    fig_conf.update_layout(
        barmode="group",
        xaxis_title="確率 / 確信度",
        xaxis=dict(range=[0, 1.05]),
        yaxis=dict(autorange="reversed"),
        height=max(350, len(features) * 32),
        margin=dict(t=10, l=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_conf, use_container_width=True)

# ============================================================
# サマリーテーブル
# ============================================================

st.markdown("**全変数の詳細**")
display_cols = ["判定", "統合介入スコア", "|効果量(std)|", "因果確信度",
                "因果エビデンス数"]
if has_lingam:
    display_cols += ["効果量(std)", "直接効果(std)", "間接効果(std)", "LiNGAM確率"]
else:
    display_cols.append("効果量(std)")
if has_pc:
    display_cols.append("PC確率")
if has_fci:
    display_cols.append("FCI隣接")
if has_grasp:
    display_cols.append("GRaSP確率")
if gt:
    display_cols.append("真のカテゴリ")

format_dict = {
    "統合介入スコア": "{:.3f}",
    "|効果量(std)|": "{:.3f}",
    "因果確信度": "{:.2f}",
    "効果量(std)": "{:+.3f}",
    "直接効果(std)": "{:+.3f}",
    "間接効果(std)": "{:+.3f}",
    "LiNGAM確率": "{:.2f}",
    "PC確率": "{:.2f}",
    "GRaSP確率": "{:.2f}",
}

st.dataframe(
    summary_df[display_cols].style.format(
        format_dict, na_rep="—"
    ).apply(
        lambda row: [
            "background-color: #E3F2FD" if "高確信" in str(row.get("判定", ""))
            else ""
            for _ in row
        ], axis=1
    ),
    use_container_width=True, height=min(600, 50 + len(features) * 35),
)

# ============================================================
# 統合因果グラフ + DoWhy 結果
# ============================================================

if "consensus_graph" in st.session_state and st.session_state["consensus_graph"] is not None:
    st.markdown("---")
    st.subheader("統合因果グラフ (Consensus Graph)")
    st.caption(
        "2つ以上の手法が検出したエッジのみを採用した統合因果グラフです。"
        "このグラフを用いて DoWhy により因果効果を推定します。"
    )

    with st.expander("📊 統合因果グラフの詳細", expanded=False):
        consensus_graph = st.session_state["consensus_graph"]
        edge_support_df = st.session_state["edge_support_df"]

        # エッジサポート情報
        st.markdown("**エッジサポート情報 (手法間の合意)**")
        st.dataframe(
            edge_support_df[[
                "From", "To", "support", "avg_probability", "directed", "methods"
            ]].rename(columns={
                "From": "起点", "To": "終点", "support": "手法数",
                "avg_probability": "平均確率", "directed": "有向", "methods": "検出手法"
            }),
            use_container_width=True,
        )

        # 統合グラフの可視化
        st.markdown("**統合因果グラフ (2+ 手法が合意したエッジのみ)**")
        if len(edge_support_df) > 0:
            fig_consensus, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(consensus_graph, seed=42, k=2)

            if gt:
                nc = [_color(n) if n != target else "#4CAF50" for n in consensus_graph.nodes()]
            else:
                nc = ["#4CAF50" if n == target else "#2196F3" for n in consensus_graph.nodes()]

            # エッジの色を support 数で変える
            edge_colors = []
            edge_widths = []
            for u, v, data in consensus_graph.edges(data=True):
                support = data.get("support", 1)
                edge_widths.append(1 + support * 0.5)
                if support >= 4:
                    edge_colors.append("#1565C0")  # 濃い青 (強い合意)
                elif support >= 3:
                    edge_colors.append("#42A5F5")  # 青
                else:
                    edge_colors.append("#90CAF9")  # 薄い青

            nx.draw(
                consensus_graph, pos, ax=ax, with_labels=True, node_color=nc,
                node_size=800, font_size=9, font_weight="bold",
                edge_color=edge_colors, width=edge_widths,
                arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1"
            )

            # サポート数をエッジラベルに表示
            edge_labels = {}
            for u, v, data in consensus_graph.edges(data=True):
                if not data.get("undirected", False):
                    support = data.get("support", "?")
                    edge_labels[(u, v)] = f"({support})"

            nx.draw_networkx_edge_labels(
                consensus_graph, pos, edge_labels, font_size=7, ax=ax
            )
            ax.set_title("統合因果グラフ (括弧内: サポート手法数)")
            st.pyplot(fig_consensus)
            plt.close(fig_consensus)
        else:
            st.warning("合意されたエッジが見つかりませんでした (全手法で異なる結果)。")

        # DoWhy 推定結果
        if has_dowhy:
            st.markdown("**DoWhy 因果効果推定結果**")
            dowhy_df_rows = []
            for feat, res in dowhy_results.items():
                dowhy_df_rows.append({
                    "変数": feat,
                    "ATE (平均因果効果)": res.get("ate", np.nan),
                    "標準誤差": res.get("stderr", np.nan),
                    "識別可能": "Yes" if res.get("identified", False) else "No",
                    "エラー": res.get("error", ""),
                })
            dowhy_df = pd.DataFrame(dowhy_df_rows).set_index("変数")
            dowhy_df = dowhy_df.sort_values("ATE (平均因果効果)", key=abs, ascending=False)

            st.dataframe(
                dowhy_df.style.format({
                    "ATE (平均因果効果)": "{:+.4f}",
                    "標準誤差": "{:.4f}",
                }, na_rep="—"),
                use_container_width=True,
            )

            st.caption(
                "**ATE (Average Treatment Effect)**: 変数を1単位増加させた時の目的変数への平均因果効果。"
                "Backdoor criterion に基づき、統合グラフから交絡因子を調整して推定。"
            )


# ============================================================
# レポートダウンロード
# ============================================================

st.markdown("---")
st.subheader("📥 分析レポートのダウンロード")

col1, col2 = st.columns(2)

with col1:
    # HTMLレポート生成
    from analysis.report_generator import generate_html_report

    html_report = generate_html_report(
        st.session_state,
        target,
        features,
        summary_df if 'summary_df' in locals() else None
    )

    st.download_button(
        label="📄 HTMLレポートをダウンロード",
        data=html_report,
        file_name=f"causal_analysis_report_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html",
        use_container_width=True,
    )
    st.caption("ブラウザで開けるHTML形式のレポート")

with col2:
    # CSVレポート生成（サマリーテーブル）
    if 'summary_df' in locals() and summary_df is not None:
        csv = summary_df.to_csv(index=True, encoding='utf-8-sig')
        st.download_button(
            label="📊 サマリーテーブル (CSV)",
            data=csv,
            file_name=f"intervention_summary_{target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Excel等で開けるCSV形式")


# ============================================================
# 各手法の詳細結果 (Expander)
# ============================================================

st.markdown("---")
st.subheader("各手法の詳細結果")

# ---- LiNGAM ----
if has_lingam:
    with st.expander("LiNGAM", expanded=False):
        res = st.session_state["lingam_result"]

        st.markdown("**推定因果順序**")
        st.write(" → ".join(res["causal_order"]))

        st.markdown(f"**{target} への推定因果効果 (標準化: 直接 vs 総介入効果)**")
        std_d = res["std_direct_effects"]
        std_t = res["std_total_effects"]

        effect_comp = pd.DataFrame({
            "直接効果 (std)": std_d,
            "総介入効果 (std)": std_t,
        }).sort_values("総介入効果 (std)", key=abs, ascending=True)

        fig_eff = go.Figure()
        fig_eff.add_trace(go.Bar(
            y=effect_comp.index, x=effect_comp["直接効果 (std)"],
            name="直接効果", marker_color="#1565C0", orientation="h",
        ))
        fig_eff.add_trace(go.Bar(
            y=effect_comp.index, x=effect_comp["総介入効果 (std)"],
            name="総介入効果 (直接+間接)", marker_color="#FF9800", orientation="h",
        ))
        fig_eff.update_layout(
            barmode="group",
            xaxis_title="標準化因果効果 (SD単位)",
            height=500, margin=dict(t=30),
        )
        fig_eff.add_vline(x=0, line_color="gray", line_width=1)
        st.plotly_chart(fig_eff, use_container_width=True)
        st.caption(
            "標準化済み: 「X を 1SD 変化させた時に Y が何 SD 変化するか」。"
            "変数間のスケール差を吸収し、介入効果の大きさを直接比較可能。"
        )

        st.markdown("**Bootstrap エッジ確率**")
        probs_df = res["bootstrap_probs"]
        fig_heat = px.imshow(
            probs_df, text_auto=".2f", color_continuous_scale="YlOrRd",
            zmin=0, zmax=1, aspect="auto",
        )
        fig_heat.update_layout(height=600, margin=dict(t=30))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("**推定 DAG**")
        edges_df = res["significant_edges"]
        if len(edges_df) > 0:
            G = nx.DiGraph()
            G.add_nodes_from(res["columns"])
            for _, row in edges_df.iterrows():
                G.add_edge(row["From"], row["To"], weight=abs(row["coefficient"]))

            fig_dag, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42, k=2)
            if gt:
                nc = [_color(n) if n != target else "#4CAF50" for n in G.nodes()]
            else:
                nc = ["#4CAF50" if n == target else "#2196F3" for n in G.nodes()]
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=nc,
                    node_size=800, font_size=9, font_weight="bold",
                    edge_color="#666", arrows=True, arrowsize=15,
                    connectionstyle="arc3,rad=0.1")
            edge_labels = {
                (r["From"], r["To"]): f"{r['coefficient']:.2f}"
                for _, r in edges_df.iterrows()
                if abs(r["coefficient"]) > 0.01
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
            ax.set_title("LiNGAM 推定 DAG")
            st.pyplot(fig_dag)
            plt.close(fig_dag)
        else:
            st.warning("有意なエッジが検出されませんでした。")

        st.markdown("**有意なエッジ一覧**")
        st.dataframe(edges_df, use_container_width=True)

        if gt and gt.get("true_edges"):
            st.markdown("**真の DAG との比較**")
            true_edges = gt["true_edges"]
            true_skeleton = {frozenset(e) for e in true_edges}
            est_skeleton = {frozenset([r["From"], r["To"]])
                            for _, r in edges_df.iterrows()}
            correct = true_skeleton & est_skeleton
            prec = len(correct) / len(est_skeleton) if est_skeleton else 0
            rec = len(correct) / len(true_skeleton) if true_skeleton else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{prec:.3f}")
            c2.metric("Recall", f"{rec:.3f}")
            c3.metric("F1", f"{f1:.3f}")

# ---- PC ----
if has_pc:
    with st.expander("PC アルゴリズム", expanded=False):
        pc_res = st.session_state["pc_result"]

        st.markdown(f"**{target} の隣接ノード**")
        st.write(sorted(pc_res["adjacent_to_target"]))

        st.markdown("**検出されたエッジ**")
        st.dataframe(pc_res["edges_df"], use_container_width=True)

        st.markdown("**推定 CPDAG**")
        pc_edges = pc_res["edges_df"]
        if len(pc_edges) > 0:
            G_pc = nx.DiGraph()
            G_pc.add_nodes_from(pc_res["columns"])
            for _, row in pc_edges.iterrows():
                if row["type"] == "directed":
                    G_pc.add_edge(row["From"], row["To"])
                else:
                    G_pc.add_edge(row["From"], row["To"])
                    G_pc.add_edge(row["To"], row["From"])

            fig_pc, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G_pc, seed=42, k=2)
            if gt:
                nc = [_color(n) if n != target else "#4CAF50" for n in G_pc.nodes()]
            else:
                nc = ["#4CAF50" if n == target else "#2196F3" for n in G_pc.nodes()]
            nx.draw(G_pc, pos, ax=ax, with_labels=True, node_color=nc,
                    node_size=800, font_size=9, font_weight="bold",
                    edge_color="#666", arrows=True, arrowsize=15,
                    connectionstyle="arc3,rad=0.1")
            ax.set_title("PC 推定 CPDAG")
            st.pyplot(fig_pc)
            plt.close(fig_pc)

        st.markdown("**Bootstrap エッジ確率**")
        probs_df = pc_res["bootstrap_probs"]
        fig_heat = px.imshow(
            probs_df, text_auto=".2f", color_continuous_scale="YlOrRd",
            zmin=0, zmax=1, aspect="auto",
        )
        fig_heat.update_layout(height=600, margin=dict(t=30))
        st.plotly_chart(fig_heat, use_container_width=True)

        if gt and gt.get("true_edges"):
            st.markdown("**真の DAG との比較**")
            true_edges = gt["true_edges"]
            true_skeleton = {frozenset(e) for e in true_edges}
            pc_skeleton = {frozenset([r["From"], r["To"]])
                           for _, r in pc_edges.iterrows()}
            correct = true_skeleton & pc_skeleton
            prec = len(correct) / len(pc_skeleton) if pc_skeleton else 0
            rec = len(correct) / len(true_skeleton) if true_skeleton else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{prec:.3f}")
            c2.metric("Recall", f"{rec:.3f}")
            c3.metric("F1", f"{f1:.3f}")

            true_direct = set(gt["direct_causes"])
            tp = true_direct & pc_res["adjacent_to_target"]
            fn = true_direct - pc_res["adjacent_to_target"]
            fp = pc_res["adjacent_to_target"] - true_direct
            st.markdown(f"**{target} の直接原因特定:**")
            st.write(f"- 正解 (TP): {sorted(tp)}")
            st.write(f"- 見逃し (FN): {sorted(fn)}")
            st.write(f"- 誤検出 (FP): {sorted(fp)}")

# ---- FCI ----
if has_fci:
    with st.expander("FCI アルゴリズム", expanded=False):
        gfci_res = st.session_state["fci_result"]

        st.markdown(f"**{target} の隣接ノード**")
        st.write(sorted(gfci_res["adjacent_to_target"]))

        st.markdown("**検出されたエッジ**")
        from analysis.pc_fci_analysis import EDGE_TYPE_LABELS
        gfci_display = gfci_res["edges_df"].copy()
        if len(gfci_display) > 0:
            gfci_display["種別"] = gfci_display["type"].map(
                lambda t: EDGE_TYPE_LABELS.get(t, t)
            )
        st.dataframe(gfci_display, use_container_width=True)

        if len(gfci_display) > 0:
            st.markdown("**エッジ種別の分布**")
            type_counts = gfci_display["種別"].value_counts()
            fig_types = px.pie(values=type_counts.values,
                               names=type_counts.index,
                               title="FCI エッジ種別")
            fig_types.update_layout(height=400, margin=dict(t=40))
            st.plotly_chart(fig_types, use_container_width=True)

        st.markdown("**推定 PAG**")
        gfci_edges = gfci_res["edges_df"]
        if len(gfci_edges) > 0:
            G_gfci = nx.DiGraph()
            G_gfci.add_nodes_from(gfci_res["columns"])
            edge_styles = {}
            for _, row in gfci_edges.iterrows():
                G_gfci.add_edge(row["From"], row["To"])
                edge_styles[(row["From"], row["To"])] = row["type"]

            fig_gfci, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G_gfci, seed=42, k=2)
            if gt:
                nc = [_color(n) if n != target else "#4CAF50" for n in G_gfci.nodes()]
            else:
                nc = ["#4CAF50" if n == target else "#2196F3" for n in G_gfci.nodes()]

            edge_colors = []
            for u, v in G_gfci.edges():
                etype = edge_styles.get((u, v), "directed")
                if etype == "bidirected":
                    edge_colors.append("#E91E63")
                elif etype in ("circle_arrow", "circle_circle"):
                    edge_colors.append("#9C27B0")
                else:
                    edge_colors.append("#666")

            nx.draw(G_gfci, pos, ax=ax, with_labels=True, node_color=nc,
                    node_size=800, font_size=9, font_weight="bold",
                    edge_color=edge_colors, arrows=True, arrowsize=15,
                    connectionstyle="arc3,rad=0.1")
            ax.set_title("FCI 推定 PAG")
            st.pyplot(fig_gfci)
            plt.close(fig_gfci)

        if gt and gt.get("true_edges"):
            st.markdown("**真の DAG との比較**")
            true_direct = set(gt["direct_causes"])
            tp = true_direct & gfci_res["adjacent_to_target"]
            fn = true_direct - gfci_res["adjacent_to_target"]
            fp = gfci_res["adjacent_to_target"] - true_direct
            st.markdown(f"**{target} の直接原因特定:**")
            st.write(f"- 正解 (TP): {sorted(tp)}")
            st.write(f"- 見逃し (FN): {sorted(fn)}")
            st.write(f"- 誤検出 (FP): {sorted(fp)}")

# ---- GRaSP ----
if has_grasp:
    with st.expander("GRaSP (Greedy relaxation of Sparsest Permutation)", expanded=False):
        grasp_res = st.session_state["grasp_result"]

        st.markdown(f"**{target} の隣接ノード**")
        st.write(sorted(grasp_res["adjacent_to_target"]))

        st.markdown("**検出されたエッジ**")
        st.dataframe(grasp_res["edges_df"], use_container_width=True)

        st.markdown("**推定 CPDAG**")
        grasp_edges = grasp_res["edges_df"]
        if len(grasp_edges) > 0:
            G_grasp = nx.DiGraph()
            G_grasp.add_nodes_from(grasp_res["columns"])
            for _, row in grasp_edges.iterrows():
                if row["type"] == "directed":
                    G_grasp.add_edge(row["From"], row["To"])
                else:
                    G_grasp.add_edge(row["From"], row["To"])
                    G_grasp.add_edge(row["To"], row["From"])

            fig_grasp, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G_grasp, seed=42, k=2)
            if gt:
                nc = [_color(n) if n != target else "#4CAF50" for n in G_grasp.nodes()]
            else:
                nc = ["#4CAF50" if n == target else "#2196F3" for n in G_grasp.nodes()]
            nx.draw(G_grasp, pos, ax=ax, with_labels=True, node_color=nc,
                    node_size=800, font_size=9, font_weight="bold",
                    edge_color="#666", arrows=True, arrowsize=15,
                    connectionstyle="arc3,rad=0.1")
            ax.set_title("GRaSP 推定 CPDAG")
            st.pyplot(fig_grasp)
            plt.close(fig_grasp)

        st.markdown("**Bootstrap エッジ確率**")
        grasp_probs_df = grasp_res["bootstrap_probs"]
        fig_heat = px.imshow(
            grasp_probs_df, text_auto=".2f", color_continuous_scale="YlOrRd",
            zmin=0, zmax=1, aspect="auto",
        )
        fig_heat.update_layout(height=600, margin=dict(t=30))
        st.plotly_chart(fig_heat, use_container_width=True)

        if gt and gt.get("true_edges"):
            st.markdown("**真の DAG との比較**")
            true_edges = gt["true_edges"]
            true_skeleton = {frozenset(e) for e in true_edges}
            grasp_skeleton = {frozenset([r["From"], r["To"]])
                              for _, r in grasp_edges.iterrows()}
            correct = true_skeleton & grasp_skeleton
            prec = len(correct) / len(grasp_skeleton) if grasp_skeleton else 0
            rec = len(correct) / len(true_skeleton) if true_skeleton else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{prec:.3f}")
            c2.metric("Recall", f"{rec:.3f}")
            c3.metric("F1", f"{f1:.3f}")

            true_direct = set(gt["direct_causes"])
            tp = true_direct & grasp_res["adjacent_to_target"]
            fn = true_direct - grasp_res["adjacent_to_target"]
            fp = grasp_res["adjacent_to_target"] - true_direct
            st.markdown(f"**{target} の直接原因特定:**")
            st.write(f"- 正解 (TP): {sorted(tp)}")
            st.write(f"- 見逃し (FN): {sorted(fn)}")
            st.write(f"- 誤検出 (FP): {sorted(fp)}")
