"""検証用 合成データセット生成スクリプト

DAG構造 (17エッジ, 14変数):
    医療ドメイン風の因果構造。dataset.csv とは全く異なる設計。

    Genetics  -> Severity, Age (遺伝的素因が重症度・加齢スピードに影響)
    Lifestyle -> Severity, Treatment (生活習慣が重症度・治療選択に影響)
    Region    -> Insurance, Treatment (地域が保険・治療アクセスに影響)
    Insurance -> Treatment (保険種別が治療選択に影響)
    Severity  -> Treatment, Outcome, BP_reading
    Age       -> Treatment, Outcome, Nurse_score
    Treatment -> DrugB, Outcome, Lab_marker
    DrugB     -> Outcome

Outcome の直接原因: Treatment, Severity, Age, DrugB (4変数)
擬似相関変数: BP_reading (Severity経由), Lab_marker (Treatment経由),
              Nurse_score (Age経由)
独立ノイズ: Room_num, Day_of_week
上流変数: Genetics, Lifestyle, Region, Insurance

特徴:
  - dataset.csv と比較して変数数が少ない (14変数)
  - 交絡因子の構造が異なる (Severity, Age が直接原因かつ交絡因子)
  - 媒介変数が1つ (DrugB)
  - 操作変数が存在しない
  - binary 変数が多い (Insurance, Region, Day_of_week)
"""

import numpy as np
import pandas as pd

SEED = 123
N = 1500


def generate_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- 外生変数 (根ノード) ---
    Genetics = rng.normal(0, 1, n)
    Lifestyle = rng.normal(0.5, 1.2, n)  # やや右寄りの分布
    Region = rng.binomial(1, 0.4, n).astype(float)  # 地方=0, 都市=1

    # --- 中間変数 ---
    e_sev = rng.normal(0, 0.4, n)
    Severity = 0.8 * Genetics + 0.6 * Lifestyle + e_sev

    e_age = rng.normal(50, 8, n)  # ベースラインが 50歳付近
    Age = e_age + 0.3 * Genetics  # 遺伝的素因で少しシフト

    e_ins = rng.normal(0, 0.3, n)
    Insurance = (0.7 * Region + e_ins > 0.3).astype(float)  # 都市部で保険加入率高

    e_treat = rng.normal(0, 0.5, n)
    Treatment = (0.7 * Severity + 0.3 * Age / 50 + 0.5 * Region
                 + 0.4 * Lifestyle + 0.3 * Insurance + e_treat)

    e_drugb = rng.normal(0, 0.4, n)
    DrugB = 0.8 * Treatment + e_drugb  # 治療強度に応じた併用薬

    # --- 結果変数 (Outcome) ---
    e_out = rng.normal(0, 0.6, n)
    Outcome = (0.6 * Treatment + 0.9 * Severity + 0.5 * (Age / 50)
               + 0.7 * DrugB + e_out)

    # --- 擬似相関変数 (Outcome に因果効果なし) ---
    BP_reading = 0.9 * Severity + rng.normal(120, 5, n) / 50  # 重症度→血圧 (Y に直接影響なし)
    Lab_marker = 0.8 * Treatment + rng.normal(0, 0.3, n)       # 治療→検査値 (Y に直接影響なし)
    Nurse_score = 0.6 * Age / 50 + rng.normal(0, 0.3, n)       # 年齢→看護スコア (Y に直接影響なし)

    # --- 独立ノイズ変数 ---
    Room_num = rng.integers(100, 500, n).astype(float)  # 病室番号
    Day_of_week = rng.integers(0, 7, n).astype(float)   # 曜日

    df = pd.DataFrame({
        "Genetics": Genetics,
        "Lifestyle": Lifestyle,
        "Region": Region,
        "Insurance": Insurance,
        "Severity": Severity,
        "Age": Age,
        "Treatment": Treatment,
        "DrugB": DrugB,
        "BP_reading": BP_reading,
        "Lab_marker": Lab_marker,
        "Nurse_score": Nurse_score,
        "Room_num": Room_num,
        "Day_of_week": Day_of_week,
        "Outcome": Outcome,
    })
    return df


# --- 真の因果構造メタデータ ---
DIRECT_CAUSES_OF_OUTCOME = ["Treatment", "Severity", "Age", "DrugB"]
NOISE_FEATURES = ["BP_reading", "Lab_marker", "Nurse_score", "Room_num", "Day_of_week"]
SPURIOUS_CORR = ["BP_reading", "Lab_marker", "Nurse_score"]
INDEPENDENT_NOISE = ["Room_num", "Day_of_week"]
UPSTREAM = ["Genetics", "Lifestyle", "Region", "Insurance"]
ALL_FEATURES = [
    "Genetics", "Lifestyle", "Region", "Insurance",
    "Severity", "Age", "Treatment", "DrugB",
    "BP_reading", "Lab_marker", "Nurse_score",
    "Room_num", "Day_of_week",
]


if __name__ == "__main__":
    df = generate_data()
    output_path = "dataset_valid.csv"
    df.to_csv(output_path, index=False)
    print(f"検証用データセット生成完了: {output_path} ({len(df)} rows, {len(df.columns)} cols)")
    print(f"\n列一覧: {list(df.columns)}")
    print(f"Outcome の直接原因: {DIRECT_CAUSES_OF_OUTCOME}")
    print(f"擬似相関変数: {SPURIOUS_CORR}")
    print(f"独立ノイズ: {INDEPENDENT_NOISE}")
    print(f"上流変数: {UPSTREAM}")
    print(f"\n先頭5行:")
    print(df.head())
