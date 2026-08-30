"""ProteinMPNN の生出力を Kermut の条件付きアミノ酸分布に変換する（GB1用）。

Kermut 本体の extract_ProteinMPNN_probs.py は
`assert wt_seq_from_toks == wt_sequence` で PDB 由来配列と WT 配列の完全一致を
要求する。GB1 では 1PGA と FLIP 配列が位置2(T vs Q)で1残基異なるため、
そのままでは通らない。許容ミスマッチ数を明示的に持たせて切り出した。

出力: (seq_len, 20) の確率行列を DMS_id ごとに保存。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ProteinMPNN のアルファベット（21番目の X を落として20次元にする）
PROTEINMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
TOK_TO_AA = {i: aa for i, aa in enumerate(PROTEINMPNN_ALPHABET)}


def main() -> None:
    parser = argparse.ArgumentParser(description="GB1 の ProteinMPNN 条件付き分布抽出")
    parser.add_argument("--ref", default="data/DMS_substitutions.csv")
    parser.add_argument(
        "--raw-dir", default="data/conditional_probs/raw_ProteinMPNN_outputs"
    )
    parser.add_argument("--out-dir", default="data/conditional_probs/ProteinMPNN")
    parser.add_argument(
        "--max-mismatch",
        type=int,
        default=1,
        help="WT配列との許容ミスマッチ数（1PGA は位置2のみ想定）",
    )
    args = parser.parse_args()

    df_ref = pd.read_csv(args.ref)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in df_ref.itertuples():
        uniprot_id = row.UniProt_ID
        wt_sequence = row.target_seq

        file_path = (
            Path(args.raw_dir)
            / uniprot_id
            / "proteinmpnn/conditional_probs_only"
            / f"{uniprot_id}.npz"
        )
        if not file_path.exists():
            raise FileNotFoundError(
                f"ProteinMPNN の出力が見つからない: {file_path}\n"
                "先に example_scripts/flip_gb1_preprocess.sh の手順3を実行すること。"
            )

        raw = np.load(file_path)
        # log_p は (num_seq, L, 21)。サンプル方向に平均してから確率に戻す
        log_p_mean = raw["log_p"].mean(axis=0)
        p_mean = np.exp(log_p_mean)[:, :20]

        wt_seq_from_toks = "".join(TOK_TO_AA[tok] for tok in raw["S"])

        if len(wt_seq_from_toks) != len(wt_sequence):
            raise ValueError(
                f"{row.DMS_id}: 長さ不一致 PDB={len(wt_seq_from_toks)} "
                f"WT={len(wt_sequence)}"
            )

        mismatches = [
            (i + 1, a, b)
            for i, (a, b) in enumerate(zip(wt_seq_from_toks, wt_sequence))
            if a != b
        ]
        if len(mismatches) > args.max_mismatch:
            raise ValueError(f"{row.DMS_id}: ミスマッチが多すぎる {mismatches}")

        out_path = out_dir / f"{row.DMS_id}.npy"
        np.save(out_path, p_mean)
        print(
            f"{row.DMS_id}: shape={p_mean.shape} "
            f"ミスマッチ={mismatches or 'なし'} -> {out_path}"
        )


if __name__ == "__main__":
    main()
