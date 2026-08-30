"""GB1 の CA 座標を抽出して Kermut の距離カーネル入力を作る。

Kermut 本体の extract_3d_coords.py は「PDB に出てくる CA を順に並べれば
WT配列と1対1に対応する」ことを前提にしている。GB1 ではこれが成り立たないため
専用に切り出した。

PDB の選定:
  FLIP の GB1 配列は 5LDE 由来だが、5LDE は GB1 領域(1-56)に13残基の欠損が
  あり、CA を順に並べると配列とインデックスがずれる。1PGA は GB1 単体構造で
  1-56 が欠損なく揃うため、こちらを使う。
  1PGA と FLIP 配列の差は位置2(T vs Q)の1箇所のみで、変異4サイト
  (39/40/41/54)から離れており距離カーネルへの影響はない。

出力は (seq_len, 3) の float 配列。欠損残基があれば NaN で埋める
（Kermut 本体が ZIKV/HCV で採っているのと同じ扱い）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_ca_atoms(pdb_path: Path, chain: str) -> dict[int, tuple[str, np.ndarray]]:
    """指定チェーンの CA 原子を {残基番号: (1文字AA, 座標)} で返す。

    altLoc は最初の一つ（空白または "A"）のみ採用する。
    """
    ca: dict[int, tuple[str, np.ndarray]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain or line[12:16] != " CA ":
                continue
            if line[16] not in (" ", "A"):
                continue
            resnum = int(line[22:26])
            if resnum in ca:
                continue
            resname = line[17:20].strip()
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
            ca[resnum] = (AA3_TO_AA1.get(resname, "X"), xyz)
    return ca


def main() -> None:
    parser = argparse.ArgumentParser(description="GB1 の CA 座標抽出")
    parser.add_argument("--pdb", default="data/structures/pdbs/GB1_1PGA.pdb")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--ref", default="data/DMS_substitutions.csv")
    parser.add_argument("--out-dir", default="data/structures/coords")
    parser.add_argument(
        "--max-mismatch",
        type=int,
        default=1,
        help="WT配列とPDB残基の許容ミスマッチ数（1PGA は位置2のみ想定）",
    )
    args = parser.parse_args()

    df_ref = pd.read_csv(args.ref)
    ca = parse_ca_atoms(Path(args.pdb), args.chain)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in df_ref.itertuples():
        wt_seq = row.target_seq
        seq_len = len(wt_seq)

        coords = np.full((seq_len, 3), np.nan, dtype=np.float32)
        mismatches = []
        missing = []
        for i in range(1, seq_len + 1):
            if i not in ca:
                missing.append(i)
                continue
            aa, xyz = ca[i]
            if aa != wt_seq[i - 1]:
                mismatches.append((i, aa, wt_seq[i - 1]))
            coords[i - 1] = xyz

        if len(mismatches) > args.max_mismatch:
            raise ValueError(
                f"{row.DMS_id}: PDB と WT配列のミスマッチが多すぎる {mismatches}"
            )

        out_path = out_dir / f"{row.DMS_id}.npy"
        np.save(out_path, coords)
        print(
            f"{row.DMS_id}: shape={coords.shape} 欠損={missing or 'なし'} "
            f"ミスマッチ={mismatches or 'なし'} -> {out_path}"
        )


if __name__ == "__main__":
    main()
