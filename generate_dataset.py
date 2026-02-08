"""
因果グラフ定義に基づく PoC用 合成データセット生成スクリプト

DAG構造 (21エッジ, 20変数):
    W1 -> Z1
    W2 -> Z2
    Z1 -> X, Z1 -> Y, Z1 -> N1
    Z2 -> X, Z2 -> Y
    Z3 -> M, Z3 -> Y
    IV -> X, IV2 -> X
    X  -> M, X -> M2, X -> Y
    M  -> Y, M -> N5
    M2 -> Y
    P1 -> M
    D1 -> Y, D1 -> N7

目的:
    因果グラフで特定された直接原因のみの特徴量 vs 全特徴量 で
    Random Forest の回帰精度を比較し、因果分析による次元削減効果を確認する。
"""

import numpy as np
import pandas as pd

SEED = 42
N = 2000  # サンプル数


def generate_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """因果グラフに従って合成データを生成する。

    データ生成過程 (DGP):
        --- 外生変数 (根ノード) ---
        W1  ~ N(0, 1)                                  # Z1 の上流
        W2  ~ Bernoulli(0.5)                            # Z2 の上流
        IV  ~ Bernoulli(0.5)                            # 操作変数
        IV2 ~ Bernoulli(0.5)                            # 操作変数2
        Z3  ~ N(0, 1)                                   # 交絡因子 (M-Y間)
        P1  ~ N(0, 1)                                   # M の親 (X以外)
        D1  ~ N(0, 1)                                   # Y の直接原因

        --- 中間変数 ---
        Z1  = 0.7*W1 + e_z1                             # 交絡因子
        Z2  = 0.9*W2 + e_z2                              # 交絡因子
        X   = 0.6*Z1 + 0.8*Z2 + 1.2*IV + 0.9*IV2 + e_x # 処置変数
        M   = 0.7*X + 0.5*Z3 + 0.4*P1 + e_m            # 媒介変数
        M2  = 0.6*X + e_m2                              # 媒介変数2

        --- 結果変数 ---
        Y   = 0.5*X + 0.9*M + 0.6*M2 + 0.4*Z1 + 0.3*Z2 + 0.5*Z3 + 0.7*D1 + e_y

        --- ノイズ特徴量 (Y に因果効果なし) ---
        N1  = 0.8*Z1 + e_n1     # Z1 経由で擬似相関
        N4  = 0.7*X + e_n4      # X 経由で擬似相関
        N5  = 0.6*M + e_n5      # M 経由で擬似相関
        N7  = 0.8*D1 + e_n7     # D1 経由で擬似相関
        N2  ~ Uniform(0, 1)     # 独立ノイズ
        N3  ~ N(0, 1)           # 独立ノイズ
        N6  ~ N(0, 1)           # 独立ノイズ
    """
    rng = np.random.default_rng(seed)

    # --- 外生変数 (根ノード) ---
    W1 = rng.normal(0, 1, n)
    W2 = rng.binomial(1, 0.5, n).astype(float)
    IV = rng.binomial(1, 0.5, n).astype(float)
    IV2 = rng.binomial(1, 0.5, n).astype(float)
    Z3 = rng.normal(0, 1, n)
    P1 = rng.normal(0, 1, n)
    D1 = rng.normal(0, 1, n)

    # --- 交絡因子 ---
    e_z1 = rng.normal(0, 0.5, n)
    Z1 = 0.7 * W1 + e_z1

    e_z2 = rng.normal(0, 0.3, n)
    Z2 = 0.9 * W2 + e_z2  # W2 の影響 + ノイズ

    # --- 処置変数 (Treatment) ---
    e_x = rng.normal(0, 0.5, n)
    X = 0.6 * Z1 + 0.8 * Z2 + 1.2 * IV + 0.9 * IV2 + e_x

    # --- 媒介変数 (Mediator) ---
    e_m = rng.normal(0, 0.5, n)
    M = 0.7 * X + 0.5 * Z3 + 0.4 * P1 + e_m

    e_m2 = rng.normal(0, 0.5, n)
    M2 = 0.6 * X + e_m2

    # --- 結果変数 (Outcome) ---
    e_y = rng.normal(0, 0.5, n)
    Y = 0.5 * X + 0.9 * M + 0.6 * M2 + 0.4 * Z1 + 0.3 * Z2 + 0.5 * Z3 + 0.7 * D1 + e_y

    # --- ノイズ特徴量 (Y に因果効果なし) ---
    e_n1 = rng.normal(0, 0.5, n)
    N1 = 0.8 * Z1 + e_n1       # Z1 の子 → Z1 経由で Y と擬似相関

    e_n4 = rng.normal(0, 0.5, n)
    N4 = 0.7 * X + e_n4        # X の子 → X 経由で Y と擬似相関

    e_n5 = rng.normal(0, 0.5, n)
    N5 = 0.6 * M + e_n5        # M の子 → M 経由で Y と擬似相関

    e_n7 = rng.normal(0, 0.5, n)
    N7 = 0.8 * D1 + e_n7       # D1 の子 → D1 経由で Y と擬似相関

    N2 = rng.uniform(0, 1, n)   # 独立ノイズ
    N3 = rng.normal(0, 1, n)    # 独立ノイズ
    N6 = rng.normal(0, 1, n)    # 独立ノイズ

    df = pd.DataFrame({
        "W1": W1,
        "W2": W2,
        "Z1": Z1,
        "Z2": Z2,
        "Z3": Z3,
        "IV": IV,
        "IV2": IV2,
        "X": X,
        "M": M,
        "M2": M2,
        "P1": P1,
        "D1": D1,
        "N1": N1,
        "N2": N2,
        "N3": N3,
        "N4": N4,
        "N5": N5,
        "N6": N6,
        "N7": N7,
        "Y": Y,
    })
    return df


# --- 真の因果構造メタデータ ---
# Y の直接の親 (DAGのエッジで Y に直接入る変数)
DIRECT_CAUSES_OF_Y = ["X", "M", "M2", "Z1", "Z2", "Z3", "D1"]
# Y に因果効果を持たないノイズ変数
NOISE_FEATURES = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]
# 全特徴量 (Y を除く)
ALL_FEATURES = [
    "W1", "W2", "Z1", "Z2", "Z3", "IV", "IV2",
    "X", "M", "M2", "P1", "D1",
    "N1", "N2", "N3", "N4", "N5", "N6", "N7",
]


if __name__ == "__main__":
    df = generate_data()
    output_path = "dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"データセット生成完了: {output_path} ({len(df)} rows, {len(df.columns)} cols)")
    print(f"\n列一覧: {list(df.columns)}")
    print(f"Y の直接原因: {DIRECT_CAUSES_OF_Y}")
    print(f"ノイズ特徴量: {NOISE_FEATURES}")
    print(f"\n先頭5行:")
    print(df.head())
