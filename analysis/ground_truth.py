"""真の因果構造レジストリ — 複数データセットに対応

各データセットの真の因果構造を登録し、データセット名から自動検索する。
未知のデータセットの場合はユーザーが手動で設定する。
"""


def _make_gt(target, direct_causes, spurious, independent, upstream,
             true_edges, true_coefficients=None):
    """ground truth dict を統一フォーマットで生成する。"""
    return {
        "target": target,
        "direct_causes": direct_causes,
        "spurious": spurious,
        "independent": independent,
        "upstream": upstream,
        "true_edges": true_edges,
        "true_coefficients": true_coefficients or {},
    }


# ====================================================================
# dataset.csv — 20 変数 (擬似相関チュートリアル)
# ====================================================================
DATASET_ORIGINAL = _make_gt(
    target="Y",
    direct_causes=["X", "M", "M2", "Z1", "Z2", "Z3", "D1"],
    spurious=["N1", "N4", "N5", "N7"],
    independent=["N2", "N3", "N6"],
    upstream=["W1", "W2", "IV", "IV2", "P1"],
    true_edges={
        ("W1", "Z1"),
        ("W2", "Z2"),
        ("Z1", "X"), ("Z1", "Y"), ("Z1", "N1"),
        ("Z2", "X"), ("Z2", "Y"),
        ("Z3", "M"), ("Z3", "Y"),
        ("IV", "X"), ("IV2", "X"),
        ("X", "M"), ("X", "M2"), ("X", "Y"), ("X", "N4"),
        ("M", "Y"), ("M", "N5"),
        ("M2", "Y"),
        ("P1", "M"),
        ("D1", "Y"), ("D1", "N7"),
    },
    true_coefficients={
        ("W1", "Z1"): 0.7, ("W2", "Z2"): 0.9,
        ("Z1", "X"): 0.6, ("Z1", "Y"): 0.4, ("Z1", "N1"): 0.8,
        ("Z2", "X"): 0.8, ("Z2", "Y"): 0.3,
        ("Z3", "M"): 0.5, ("Z3", "Y"): 0.5,
        ("IV", "X"): 1.2, ("IV2", "X"): 0.9,
        ("X", "M"): 0.7, ("X", "M2"): 0.6, ("X", "Y"): 0.5, ("X", "N4"): 0.7,
        ("M", "Y"): 0.9, ("M", "N5"): 0.6,
        ("M2", "Y"): 0.6, ("P1", "M"): 0.4,
        ("D1", "Y"): 0.7, ("D1", "N7"): 0.8,
    },
)

# ====================================================================
# dataset_valid.csv — 15 変数 (検証用: 医療ドメイン風)
# ====================================================================
DATASET_VALID = _make_gt(
    target="Outcome",
    direct_causes=["Treatment", "Severity", "Age", "DrugB"],
    spurious=["BP_reading", "Lab_marker", "Nurse_score"],
    independent=["Room_num", "Day_of_week"],
    upstream=["Genetics", "Lifestyle", "Region", "Insurance"],
    true_edges={
        # 上流 → 交絡因子
        ("Genetics", "Severity"), ("Genetics", "Age"),
        ("Lifestyle", "Severity"), ("Lifestyle", "Treatment"),
        ("Region", "Insurance"), ("Region", "Treatment"),
        # 交絡因子 → Treatment / Outcome
        ("Severity", "Treatment"), ("Severity", "Outcome"),
        ("Age", "Treatment"), ("Age", "Outcome"),
        # Treatment → Outcome (直接 + 媒介)
        ("Treatment", "DrugB"), ("Treatment", "Outcome"),
        # 媒介 → Outcome
        ("DrugB", "Outcome"),
        # 擬似相関変数 (Outcome の親の子)
        ("Severity", "BP_reading"),
        ("Treatment", "Lab_marker"),
        ("Age", "Nurse_score"),
        # Insurance は Outcome に直接影響しない (Region 経由で Treatment に影響)
        ("Insurance", "Treatment"),
    },
    true_coefficients={
        ("Genetics", "Severity"): 0.8, ("Genetics", "Age"): 0.3,
        ("Lifestyle", "Severity"): 0.6, ("Lifestyle", "Treatment"): 0.4,
        ("Region", "Insurance"): 0.7, ("Region", "Treatment"): 0.5,
        ("Severity", "Treatment"): 0.7, ("Severity", "Outcome"): 0.9,
        ("Age", "Treatment"): 0.3, ("Age", "Outcome"): 0.5,
        ("Treatment", "DrugB"): 0.8, ("Treatment", "Outcome"): 0.6,
        ("DrugB", "Outcome"): 0.7,
        ("Severity", "BP_reading"): 0.9,
        ("Treatment", "Lab_marker"): 0.8,
        ("Age", "Nurse_score"): 0.6,
        ("Insurance", "Treatment"): 0.3,
    },
)

# ====================================================================
# レジストリ: ファイル名 → 真の因果構造
# ====================================================================
REGISTRY = {
    "dataset.csv": DATASET_ORIGINAL,
    "dataset_valid.csv": DATASET_VALID,
}


def lookup_ground_truth(filename, columns):
    """ファイル名とカラム構成から真の因果構造を検索する。

    Returns:
        ground truth dict or None
    """
    # ファイル名の完全一致
    if filename in REGISTRY:
        gt = REGISTRY[filename]
        # カラムが一致するか確認
        required = set(gt["direct_causes"] + gt["spurious"]
                       + gt["independent"] + gt["upstream"] + [gt["target"]])
        if required.issubset(set(columns)):
            return gt

    # カラム構成でフォールバック検索
    col_set = set(columns)
    for gt in REGISTRY.values():
        required = set(gt["direct_causes"] + gt["spurious"]
                       + gt["independent"] + gt["upstream"] + [gt["target"]])
        if required == col_set:
            return gt

    return None


# カテゴリ関連のユーティリティ
CATEGORY_LABELS = {
    "direct_cause": "直接原因",
    "spurious": "擬似相関",
    "independent": "独立ノイズ",
    "upstream": "上流変数",
    "unknown": "不明",
}

CATEGORY_COLORS = {
    "直接原因": "#2196F3",
    "擬似相関": "#F44336",
    "独立ノイズ": "#9E9E9E",
    "上流変数": "#FF9800",
    "不明": "#607D8B",
}


def categorize_feature(feat, gt):
    """ground truth dict を使って特徴量をカテゴライズする。"""
    if gt is None:
        return "不明"
    if feat in gt["direct_causes"]:
        return "直接原因"
    if feat in gt["spurious"]:
        return "擬似相関"
    if feat in gt["independent"]:
        return "独立ノイズ"
    if feat in gt["upstream"]:
        return "上流変数"
    return "不明"
