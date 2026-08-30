"""FLIP GB1 の前処理成果物が揃っているか・整合しているかを検証する。

ベンチマークを回す前に、これで全て OK が出ることを確認する。
特に埋め込みは CSV を作り直す前のものが残っている可能性があるので、
mutant の集合が CSV と一致するかを実際に突き合わせる。
"""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

SPLITS = ["one_vs_rest", "two_vs_rest", "three_vs_rest", "sampled", "low_vs_high"]
ok_all = True

for s in SPLITS:
    dms = f"GB1_FLIP_{s}"
    print(f"=== {dms} ===")
    csv = Path(f"data/substitutions_singles/{dms}.csv")
    df = pd.read_csv(csv)

    # 1) 列
    need = {"mutant", "mutations", "mutated_sequence", "DMS_score", f"fold_flip_{s}"}
    miss = need - set(df.columns)
    print(f"  列: {'OK' if not miss else 'NG 欠落=' + str(miss)}  行数={len(df)}")
    ok_all &= not miss

    # 2) 埋め込み（mutant 集合が CSV と一致するか）
    h5 = Path(f"data/embeddings/substitutions_singles/ESM2/{dms}.h5")
    if h5.exists():
        with h5py.File(h5, "r") as f:
            emb = f["embeddings"][:]
            muts = [x.decode() for x in f["mutants"][:]]
        same = set(muts) == set(df["mutant"])
        print(f"  埋め込み: shape={emb.shape} mutant集合一致={'OK' if same else 'NG'}"
              f" NaN={np.isnan(emb).any()}")
        ok_all &= same and not np.isnan(emb).any()
    else:
        print("  埋め込み: NG 未生成"); ok_all = False

    # 3) ゼロショット
    z = Path(f"data/zero_shot_fitness_predictions/ESM2/650M/{dms}.csv")
    if z.exists():
        dz = pd.read_csv(z)
        col = "esm2_t33_650M_UR50D"
        has = col in dz.columns
        nan = dz[col].isna().sum() if has else -1
        wt = dz.loc[dz["mutant"] == "WT", col].tolist() if has and "mutant" in dz else []
        print(f"  ゼロショット: 行数={len(dz)} 列={'OK' if has else 'NG'} NaN={nan}"
              f" WTスコア={wt}")
        ok_all &= has and nan == 0
    else:
        print("  ゼロショット: NG 未生成"); ok_all = False

    # 4) 条件付き分布
    p = Path(f"data/conditional_probs/ProteinMPNN/{dms}.npy")
    if p.exists():
        a = np.load(p)
        rs = a.sum(axis=1)
        print(f"  条件付き分布: shape={a.shape} 行和≈1={np.allclose(rs,1,atol=1e-3)}"
              f" NaN={np.isnan(a).any()}")
        ok_all &= not np.isnan(a).any()
    else:
        print("  条件付き分布: NG 未生成"); ok_all = False

    # 5) 座標
    c = Path(f"data/structures/coords/{dms}.npy")
    if c.exists():
        co = np.load(c)
        print(f"  座標: shape={co.shape} NaN={np.isnan(co).any()}")
        ok_all &= co.shape == (56, 3)
    else:
        print("  座標: NG 未生成"); ok_all = False

print()
print("総合:", "すべて OK。ベンチマークを回せる。" if ok_all else "NG あり。上を確認すること。")
