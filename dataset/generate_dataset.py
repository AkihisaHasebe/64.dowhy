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

分布設計 (実務的リアリティ):
  - 外生変数: t分布 (重い裾)、Exponential (右裾)、混合正規 (二峰性)、
              LogNormal (右裾) などを導入
  - 誤差項: t分布、混合正規、Gamma、LogNormal、Exponential を使用
  - DAG構造と係数は変更なし
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N = 2000  # サンプル数


def generate_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """因果グラフに従って合成データを生成する。

    データ生成過程 (DGP):
        --- 外生変数 (根ノード) ---
        W1  ~ t(df=5)                                  # Z1 の上流 (裾が重い)
        W2  ~ Bernoulli(0.5)                            # Z2 の上流
        IV  ~ Bernoulli(0.5)                            # 操作変数
        IV2 ~ Bernoulli(0.5)                            # 操作変数2
        Z3  ~ Exponential(1) - 1                        # 交絡因子 (右裾)
        P1  ~ MixtureNormal(0.5*N(-1,0.5)+0.5*N(1,0.5))# M の親 (二峰性)
        D1  ~ LogNormal(0,0.5) - median                 # Y の直接原因 (右裾)

        --- 中間変数 ---
        Z1  = 0.7*W1 + e_z1       e_z1 ~ t(df=5)*0.5
        Z2  = 0.9*W2 + e_z2       e_z2 ~ N(0, 0.3)
        X   = 0.6*Z1 + 0.8*Z2 + 1.2*IV + 0.9*IV2 + e_x   e_x ~ MixNormal
        M   = 0.7*X + 0.5*Z3 + 0.4*P1 + e_m               e_m ~ Gamma-centered
        M2  = 0.6*X + e_m2        e_m2 ~ t(df=5)*0.5

        --- 結果変数 ---
        Y   = 0.5*X + 0.9*M + 0.6*M2 + 0.4*Z1 + 0.3*Z2 + 0.5*Z3 + 0.7*D1 + e_y
            e_y ~ t(df=4)*0.5

        --- ノイズ特徴量 (Y に因果効果なし) ---
        N1  = 0.8*Z1 + e_n1       e_n1 ~ LogNormal-centered
        N4  = 0.7*X + e_n4        e_n4 ~ Exponential-centered
        N5  = 0.6*M + e_n5        e_n5 ~ t(df=4)*0.5
        N7  = 0.8*D1 + e_n7       e_n7 ~ MixNormal
        N2  ~ Uniform(0, 1)                             # 独立ノイズ
        N3  ~ t(df=3)                                   # 独立ノイズ (裾が重い)
        N6  ~ Exponential(1) - 1                        # 独立ノイズ (右裾)
    """
    rng = np.random.default_rng(seed)

    # --- 外生変数 (根ノード) ---
    W1 = rng.standard_t(5, n)                             # t分布: 裾が重い上流変数
    W2 = rng.binomial(1, 0.5, n).astype(float)
    IV = rng.binomial(1, 0.5, n).astype(float)
    IV2 = rng.binomial(1, 0.5, n).astype(float)
    Z3 = rng.exponential(1, n) - 1                        # Exponential: 右裾の交絡因子 (中心化)
    # P1: 混合正規 (二峰性)
    p1_mask = rng.random(n) < 0.5
    P1 = np.where(p1_mask, rng.normal(-1, 0.5, n), rng.normal(1, 0.5, n))
    # D1: LogNormal (右裾の直接原因、中心化)
    d1_raw = rng.lognormal(0, 0.5, n)
    D1 = d1_raw - np.median(d1_raw)

    # --- 交絡因子 ---
    e_z1 = rng.standard_t(5, n) * 0.5                    # t分布: 裾が重い
    Z1 = 0.7 * W1 + e_z1

    e_z2 = rng.normal(0, 0.3, n)                          # 小さいノイズは正規のまま
    Z2 = 0.9 * W2 + e_z2

    # --- 処置変数 (Treatment) ---
    # 混合正規: 処置のばらつき (0.8*N(0,0.3) + 0.2*N(0,1.5))
    mix_mask_x = rng.random(n) < 0.8
    e_x = np.where(mix_mask_x, rng.normal(0, 0.3, n), rng.normal(0, 1.5, n))
    X = 0.6 * Z1 + 0.8 * Z2 + 1.2 * IV + 0.9 * IV2 + e_x

    # --- 媒介変数 (Mediator) ---
    # Gamma (中心化): 媒介変数の非対称な誤差
    e_m = rng.gamma(4, 0.125, n) - 0.5
    M = 0.7 * X + 0.5 * Z3 + 0.4 * P1 + e_m

    e_m2 = rng.standard_t(5, n) * 0.5                    # t分布
    M2 = 0.6 * X + e_m2

    # --- 結果変数 (Outcome) ---
    e_y = rng.standard_t(4, n) * 0.5                      # t分布: 外れ値あり
    Y = 0.5 * X + 0.9 * M + 0.6 * M2 + 0.4 * Z1 + 0.3 * Z2 + 0.5 * Z3 + 0.7 * D1 + e_y

    # --- ノイズ特徴量 (Y に因果効果なし) ---
    # LogNormal (中心化): 対数正規ノイズ
    n1_raw = rng.lognormal(-0.7, 0.5, n)
    e_n1 = (n1_raw - np.median(n1_raw)) * 0.5
    N1 = 0.8 * Z1 + e_n1

    # Exponential (中心化): 右裾
    e_n4 = rng.exponential(0.5, n) - 0.5
    N4 = 0.7 * X + e_n4

    e_n5 = rng.standard_t(4, n) * 0.5                    # t分布
    N5 = 0.6 * M + e_n5

    # 混合正規: 混合ノイズ
    mix_mask_n7 = rng.random(n) < 0.7
    e_n7 = np.where(mix_mask_n7, rng.normal(0, 0.3, n), rng.normal(0, 1.0, n))
    N7 = 0.8 * D1 + e_n7

    N2 = rng.uniform(0, 1, n)                             # 独立ノイズ (一様)
    N3 = rng.standard_t(3, n)                             # 独立ノイズ (裾が重い)
    N6 = rng.exponential(1, n) - 1                        # 独立ノイズ (右裾)

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
    output_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"データセット生成完了: {output_path} ({len(df)} rows, {len(df.columns)} cols)")
    print(f"\n列一覧: {list(df.columns)}")
    print(f"Y の直接原因: {DIRECT_CAUSES_OF_Y}")
    print(f"ノイズ特徴量: {NOISE_FEATURES}")
    print(f"\n先頭5行:")
    print(df.head())
