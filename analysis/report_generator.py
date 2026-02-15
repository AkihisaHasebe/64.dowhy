"""HTMLレポート生成モジュール"""

import pandas as pd
import numpy as np
from datetime import datetime


def generate_html_report(session_state, target, features, summary_df=None):
    """
    分析結果からHTMLレポートを生成する。

    Args:
        session_state: Streamlit session_state オブジェクト
            lingam_result, pc_result, fci_result, grasp_result を含む
        target: 目的変数名 (str)
        features: 特徴量のリスト (list)
        summary_df: 統合介入スコアのDataFrame (optional)

    Returns:
        str: HTML形式のレポート文字列
    """
    # 結果を取得
    lingam_result = session_state.get("lingam_result")
    pc_result = session_state.get("pc_result")
    fci_result = session_state.get("fci_result")
    grasp_result = session_state.get("grasp_result")

    # データファイル名を取得
    data_filename = session_state.get("data_filename", "不明")

    # 実行された手法のカウント
    executed_methods = []
    if lingam_result is not None:
        executed_methods.append("LiNGAM")
    if pc_result is not None:
        executed_methods.append("PC")
    if fci_result is not None:
        executed_methods.append("FCI")
    if grasp_result is not None:
        executed_methods.append("GRaSP")

    # 現在日時
    now = datetime.now()
    timestamp = now.strftime("%Y年%m月%d日 %H:%M:%S")

    # HTML構築開始
    html_parts = []

    # ヘッダー部分
    html_parts.append(f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>因果分析レポート - {data_filename}</title>
    <style>
        body {{
            font-family: Meiryo, 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #1565C0;
            border-bottom: 3px solid #1565C0;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 28px;
        }}
        h2 {{
            color: #283593;
            border-left: 5px solid #283593;
            padding-left: 15px;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 22px;
        }}
        h3 {{
            color: #4527A0;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 18px;
        }}
        .metadata {{
            background-color: #E3F2FD;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .metadata p {{
            margin: 5px 0;
            color: #1565C0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #1565C0;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #E3F2FD;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .method-badge {{
            display: inline-block;
            background-color: #4527A0;
            color: white;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 8px;
        }}
        .causal-order {{
            background-color: #F3E5F5;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
        }}
        .no-data {{
            color: #999;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #777;
            font-size: 12px;
        }}
        .number {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>因果分析レポート</h1>

        <div class="metadata">
            <p><strong>生成日時:</strong> {timestamp}</p>
            <p><strong>データファイル:</strong> {data_filename}</p>
            <p><strong>目的変数:</strong> {target}</p>
            <p><strong>特徴量数:</strong> {len(features)}</p>
            <p><strong>実行手法:</strong> {', '.join(executed_methods) if executed_methods else 'なし'}</p>
        </div>
""")

    # 統合介入スコアのサマリー
    if summary_df is not None and not summary_df.empty:
        html_parts.append("""
        <div class="section">
            <h2>統合介入スコア (Top 10)</h2>
            <p>統合介入スコア = |効果量| × 因果確信度</p>
""")

        # Top 10を抽出
        top10 = summary_df.head(10).copy()

        # テーブル作成
        html_parts.append("<table>")
        html_parts.append("<thead><tr>")
        for col in top10.columns:
            html_parts.append(f"<th>{col}</th>")
        html_parts.append("</tr></thead>")
        html_parts.append("<tbody>")

        for _, row in top10.iterrows():
            html_parts.append("<tr>")
            for col in top10.columns:
                val = row[col]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    html_parts.append(f'<td class="number">{val:.4f}</td>')
                else:
                    html_parts.append(f"<td>{val}</td>")
            html_parts.append("</tr>")

        html_parts.append("</tbody></table>")
        html_parts.append("</div>")

    # LiNGAM結果
    if lingam_result is not None:
        html_parts.append("""
        <div class="section">
            <h2><span class="method-badge">LiNGAM</span>DirectLiNGAM 分析結果</h2>
""")

        # 因果順序
        causal_order = lingam_result.get("causal_order", [])
        if causal_order:
            html_parts.append("<h3>因果順序</h3>")
            html_parts.append('<div class="causal-order">')
            html_parts.append(" → ".join(causal_order))
            html_parts.append("</div>")

        # 直接因果効果 Top 10
        effects = lingam_result.get("effects_on_target")
        if effects is not None and len(effects) > 0:
            html_parts.append(f"<h3>{target} への直接因果効果 (Top 10)</h3>")
            top_effects = effects.head(10)

            html_parts.append("<table>")
            html_parts.append("<thead><tr><th>変数</th><th>直接因果効果</th></tr></thead>")
            html_parts.append("<tbody>")
            for var, effect in top_effects.items():
                html_parts.append(f'<tr><td>{var}</td><td class="number">{effect:.6f}</td></tr>')
            html_parts.append("</tbody></table>")

        # Bootstrap エッジ確率 Top 10
        edge_probs = lingam_result.get("edge_probs_to_target")
        if edge_probs is not None and len(edge_probs) > 0:
            html_parts.append(f"<h3>{target} へのエッジ確率 (Bootstrap, Top 10)</h3>")
            top_probs = edge_probs.head(10)

            html_parts.append("<table>")
            html_parts.append("<thead><tr><th>変数</th><th>エッジ確率</th></tr></thead>")
            html_parts.append("<tbody>")
            for var, prob in top_probs.items():
                html_parts.append(f'<tr><td>{var}</td><td class="number">{prob:.4f}</td></tr>')
            html_parts.append("</tbody></table>")

        html_parts.append("</div>")

    # PC結果
    if pc_result is not None:
        html_parts.append("""
        <div class="section">
            <h2><span class="method-badge">PC</span>PC アルゴリズム分析結果</h2>
""")

        # 隣接ノード
        adjacent = pc_result.get("adjacent_to_target")
        if adjacent:
            html_parts.append(f"<h3>{target} の隣接ノード</h3>")
            html_parts.append("<p>" + ", ".join(sorted(adjacent)) + "</p>")

        # 検出エッジ Top 20
        edges_df = pc_result.get("edges_df")
        if edges_df is not None and not edges_df.empty:
            html_parts.append("<h3>検出されたエッジ (Top 20)</h3>")
            top_edges = edges_df.head(20)

            html_parts.append("<table>")
            html_parts.append("<thead><tr>")
            for col in top_edges.columns:
                html_parts.append(f"<th>{col}</th>")
            html_parts.append("</tr></thead>")
            html_parts.append("<tbody>")

            for _, row in top_edges.iterrows():
                html_parts.append("<tr>")
                for col in top_edges.columns:
                    val = row[col]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        html_parts.append(f'<td class="number">{val:.4f}</td>')
                    else:
                        html_parts.append(f"<td>{val}</td>")
                html_parts.append("</tr>")

            html_parts.append("</tbody></table>")

        html_parts.append("</div>")

    # FCI結果
    if fci_result is not None:
        html_parts.append("""
        <div class="section">
            <h2><span class="method-badge">FCI</span>FCI アルゴリズム分析結果</h2>
""")

        # 隣接ノード
        adjacent = fci_result.get("adjacent_to_target")
        if adjacent:
            html_parts.append(f"<h3>{target} の隣接ノード</h3>")
            html_parts.append("<p>" + ", ".join(sorted(adjacent)) + "</p>")

        # 検出エッジ Top 20
        edges_df = fci_result.get("edges_df")
        if edges_df is not None and not edges_df.empty:
            html_parts.append("<h3>検出されたエッジ (Top 20)</h3>")
            top_edges = edges_df.head(20)

            html_parts.append("<table>")
            html_parts.append("<thead><tr>")
            for col in top_edges.columns:
                html_parts.append(f"<th>{col}</th>")
            html_parts.append("</tr></thead>")
            html_parts.append("<tbody>")

            for _, row in top_edges.iterrows():
                html_parts.append("<tr>")
                for col in top_edges.columns:
                    val = row[col]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        html_parts.append(f'<td class="number">{val:.4f}</td>')
                    else:
                        html_parts.append(f"<td>{val}</td>")
                html_parts.append("</tr>")

            html_parts.append("</tbody></table>")

        html_parts.append("</div>")

    # GRaSP結果
    if grasp_result is not None:
        html_parts.append("""
        <div class="section">
            <h2><span class="method-badge">GRaSP</span>GRaSP アルゴリズム分析結果</h2>
""")

        # 隣接ノード
        adjacent = grasp_result.get("adjacent_to_target")
        if adjacent:
            html_parts.append(f"<h3>{target} の隣接ノード</h3>")
            html_parts.append("<p>" + ", ".join(sorted(adjacent)) + "</p>")

        # 検出エッジ Top 20
        edges_df = grasp_result.get("edges_df")
        if edges_df is not None and not edges_df.empty:
            html_parts.append("<h3>検出されたエッジ (Top 20)</h3>")
            top_edges = edges_df.head(20)

            html_parts.append("<table>")
            html_parts.append("<thead><tr>")
            for col in top_edges.columns:
                html_parts.append(f"<th>{col}</th>")
            html_parts.append("</tr></thead>")
            html_parts.append("<tbody>")

            for _, row in top_edges.iterrows():
                html_parts.append("<tr>")
                for col in top_edges.columns:
                    val = row[col]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        html_parts.append(f'<td class="number">{val:.4f}</td>')
                    else:
                        html_parts.append(f"<td>{val}</td>")
                html_parts.append("</tr>")

            html_parts.append("</tbody></table>")

        html_parts.append("</div>")

    # フッター
    html_parts.append("""
        <div class="footer">
            <p>Generated by DoWhy 因果分析比較ダッシュボード</p>
        </div>
    </div>
</body>
</html>
""")

    return "".join(html_parts)
