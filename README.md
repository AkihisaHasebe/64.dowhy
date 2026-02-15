# DoWhy 因果推論 PoC

因果推論手法のアンサンブル（LiNGAM, PC, FCI, GRaSP）による介入優先度スコアリングのStreamlitダッシュボード。

## セットアップ

### 前提条件
- Python 3.10

### 仮想環境の作成と依存パッケージのインストール

```bash
# 仮想環境の作成
python -m venv .venv

# 仮想環境の有効化
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# Linux / macOS
source .venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使い方

### Streamlit ダッシュボード
```bash
streamlit run app.py
```

### Jupyter Notebook
```bash
jupyter notebook
```
ノートブックは `notebook/` フォルダおよびプロジェクトルートにあります。

## プロジェクト構成

```
├── app.py                  # Streamlit メインページ
├── pages/
│   ├── 1_data_overview.py  # データ概要
│   ├── 2_run_analysis.py   # 分析実行・介入スコア算出
│   └── 3_comparison.py     # 手法間比較
├── analysis/
│   ├── lingam_analysis.py  # LiNGAM 分析
│   ├── pc_fci_analysis.py  # PC・FCI 分析
│   ├── grasp_analysis.py   # GRaSP 分析
│   ├── consensus_graph.py  # コンセンサスグラフ構築
│   └── dowhy_estimation.py # DoWhy 因果効果推定
├── dataset/
│   ├── generate_dataset.py       # 合成データ生成 (20変数)
│   ├── generate_dataset_valid.py # 検証用データ生成 (医療ドメイン)
│   ├── dataset.csv               # 分析用データセット
│   └── dataset_valid.csv         # 検証用データセット
├── notebook/               # Jupyter ノートブック
└── requirements.txt        # 依存パッケージ
```
