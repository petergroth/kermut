"""FLIP の GB1 データを Kermut(ProteinGym 形式) の入力に変換する。

FLIP (https://github.com/J-SNACKKB/FLIP) の GB1 landscape は Wu et al. 2016 の
4残基飽和変異（V39/D40/G41/V54, 20^4=160,000）で、`four_mutations_full_data.csv`
に Variants・Fitness・sequence と各 split の train/test 列が入っている。

Kermut 側が要求する形式との差を、ここで吸収する:

  1. 変異表記   FLIP は "ADGV" の4文字短縮記法 → ProteinGym の "V39A:D40E" 記法へ
  2. 配列       FLIP の sequence は 5LDE 由来の265残基融合タンパク。
                57残基目以降は全変異体で同一なので GB1 本体の先頭56残基だけ残す
  3. split      FLIP は単一 train/test（`set` 列）。Kermut は fold 列を回す CV 前提。
                test を fold 0、train を fold 1 に割り当て、ベンチ側で fold 0 のみ
                評価すれば FLIP の split をそのまま再現できる
  4. 対象行     FLIP の公式 split は keep=True の 8,733 行の上でのみ定義される

変異位置(1-indexed)は 39, 40, 41, 54。WT4残基は VDGV。WT配列上で検証済み。

使い方:
    uv run python3 scripts/flip/prepare_flip_gb1.py \
        --flip-csv <path>/four_mutations_full_data.csv \
        --splits one_vs_rest two_vs_rest three_vs_rest sampled low_vs_high
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# GB1 の変異4サイト（1-indexed）と、そこに入る WT 残基。
# Wu et al. 2016 が飽和変異を入れた4箇所。FLIP の Variants 列("ADGV"等)の
# 1文字目がpos39、2文字目がpos40、3文字目がpos41、4文字目がpos54に対応する。
POSITIONS_1IDX = [39, 40, 41, 54]
WT4 = "VDGV"

# GB1 本体の長さ。
#
# FLIP の sequence 列は 265 残基あるが、これは 5LDE 由来の融合タンパクで
#   1-56    : GB1 本体（変異が入るのはここだけ）
#   57-     : リンカー + Viral FLICE + His-tag
# という構成。57残基目以降は全変異体で完全に同一（main() 内で検証している）。
#
# 265 残基のまま Kermut に渡すと:
#   - 変異カーネルの one-hot が 265*20 次元になる
#   - 距離カーネルが 265x265 行列を作る
#   - ESM-2 埋め込みも 265 残基分を毎回計算する
# と、情報が増えないのに計算量だけ増える。よって GB1 本体だけ切り出す。
GB1_LEN = 56

# FLIP が提供する 5 つの公式 split
FLIP_SPLITS = ["one_vs_rest", "two_vs_rest", "three_vs_rest", "sampled", "low_vs_high"]

# ProteinGym 側で使う識別子。参照ファイル・埋め込み・座標のファイル名に使われる
DMS_ID = "GB1_FLIP"
UNIPROT_ID = "GB1_1PGA"


def variants_to_mutant(variants: str) -> str:
    """FLIP の4文字短縮記法を ProteinGym の mutant 記法へ変換する。

    "ADGV" -> "V39A"        (1文字目のみ WT と異なる)
    "AEGV" -> "V39A:D40E"
    "VDGV" -> ""            (WT。変異なし)
    """
    muts = [
        f"{wt}{pos}{mt}"
        for mt, wt, pos in zip(variants, WT4, POSITIONS_1IDX)
        if mt != wt
    ]
    return ":".join(muts)


def build_mutated_sequence(wt_seq: str, variants: str) -> str:
    """WT 配列の4サイトを置換して、変異配列（GB1 本体56残基）を作る。"""
    seq = list(wt_seq)
    for mt, pos in zip(variants, POSITIONS_1IDX):
        seq[pos - 1] = mt
    return "".join(seq)


def main() -> None:
    parser = argparse.ArgumentParser(description="FLIP GB1 -> Kermut 入力への変換")
    parser.add_argument(
        "--flip-csv",
        required=True,
        help="FLIP の four_mutations_full_data.csv へのパス",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=FLIP_SPLITS,
        choices=FLIP_SPLITS,
        help="変換対象の split（既定: 全5種）",
    )
    parser.add_argument(
        "--out-dir",
        default="data/substitutions_singles",
        help="アッセイCSVの出力先",
    )
    parser.add_argument(
        "--ref-out",
        default="data/DMS_substitutions.csv",
        help="参照ファイル（ProteinGym の DMS_substitutions.csv 相当）の出力先",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.flip_csv, low_memory=False)

    # --- keep=True で絞る（ここを間違えると FLIP と比較不能になる）-----------------
    # four_mutations_full_data.csv は全 149,361 行（20^4 の全組み合わせに近い）だが、
    # FLIP の公式 split 列（one_vs_rest 等）に train/test が入っているのは
    # keep=True の 8,733 行だけ。残りは全て空欄になっている。
    #
    # keep=False の行は元論文で読み取りカウントが少なく信頼できないものが中心で、
    # FLIP はそれらをベンチマークから除外している。
    #
    # ここで絞らないと「FLIP の split で評価した」と言えなくなる（他手法のスコアと
    # 比較できない）ので、必ず絞る。
    # 内訳: HD=0:1, HD=1:27, HD=2:396, HD=3:2566, HD=4:5743 = 8,733
    # -----------------------------------------------------------------------------
    df = df[df["keep"] == True].reset_index(drop=True)  # noqa: E712

    # WT 行（HD=0）から GB1 本体の WT 配列を取る
    wt_row = df[df["HD"] == 0].iloc[0]
    wt_seq_full = wt_row["sequence"]
    wt_seq = wt_seq_full[:GB1_LEN]

    # --- 前提の検証1: 4サイトの WT 残基が VDGV であること ------------------------
    # POSITIONS_1IDX が1つでもズレていると、変異記法が全部間違ったまま静かに通って
    # しまう（例: pos38 と pos39 を取り違えても実行時エラーは出ない）。
    # ここで落としておけば、そのミスは必ず検出できる。
    # ---------------------------------------------------------------------------
    observed = "".join(wt_seq[p - 1] for p in POSITIONS_1IDX)
    if observed != WT4:
        raise ValueError(f"WT4残基が不一致: expected {WT4}, got {observed}")

    # --- 前提の検証2: GB1 本体より後ろは全変異体で不変 ---------------------------
    # GB1_LEN で切り詰めてよい根拠そのもの。もし57残基目以降にも変異が入っていたら
    # 切り詰めは情報の欠落になるので、その場合は止める。
    # （現状の FLIP データでは nunique()==1、つまり全変異体で同一と確認済み）
    # ---------------------------------------------------------------------------
    if df["sequence"].str[GB1_LEN:].nunique() != 1:
        raise ValueError(f"{GB1_LEN}残基目以降が変異体間で異なる。切り詰めは不正。")

    df["mutant"] = df["Variants"].apply(variants_to_mutant)
    # Kermut 内でも列名が揺れている。ProteinGym の元CSVは両方持っているので気付きにくいが:
    #   src/data/extract_esm2_embeddings.py     -> df["mutant"]
    #   src/data/extract_esm2_zero_shots.py     -> df["mutations"]
    #   src/experiments/proteingym_benchmark.py -> df["mutant"]
    # よって両方の列を出しておく（中身は同一）。片方だけだと
    # ゼロショット計算が KeyError: 'mutations' で落ちる。
    #
    # WT 行は変異なしのため mutant が空文字になる。そのままだと CSV 経由で NaN に
    # なり、埋め込みのラベル照合やゼロショットの merge を壊すので明示的に名前を付ける。
    # "WT" への置換を先にやってから mutations にコピーすること（順序を逆にすると
    # mutations 側が空文字のまま残る）。
    df.loc[df["mutant"] == "", "mutant"] = "WT"
    df["mutations"] = df["mutant"]
    df["mutated_sequence"] = df["Variants"].apply(
        lambda v: build_mutated_sequence(wt_seq, v)
    )
    df["DMS_score"] = df["Fitness"]

    # --- 前提の検証3: 生成した変異配列と mutant 記法の相互整合 -------------------
    # mutant 記法("V39A")と mutated_sequence は別々に作っているので、
    # 両者が食い違っていないかを突き合わせる。具体的には各変異について
    #   WT配列[pos] == 記法のWT残基     （位置の対応が正しい）
    #   変異配列[pos] == 記法の変異後残基（置換が効いている）
    # を確認する。全件だと遅いので先頭200行のみ（変換ロジックは行に依らないので十分）。
    # ---------------------------------------------------------------------------
    for row in df.head(200).itertuples():
        if row.mutant == "WT":
            assert row.mutated_sequence == wt_seq, "WT行の配列がWT配列と一致しない"
            continue
        for mut in row.mutant.split(":"):
            wt_aa, pos, mt_aa = mut[0], int(mut[1:-1]), mut[-1]
            assert wt_seq[pos - 1] == wt_aa, f"WT不一致: {mut}"
            assert row.mutated_sequence[pos - 1] == mt_aa, f"変異不一致: {mut}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_rows = []
    for split in args.splits:
        # --- FLIP の train/test を Kermut の fold 列に写す ------------------------
        # Kermut のベンチは「fold 列を持つ CSV」を前提に、各 fold を順に test にする
        # CV ループを回す。FLIP は単一 train/test なので、そのままでは形が合わない。
        #
        # そこで:
        #       FLIP の test  -> fold 0
        #       FLIP の train -> fold 1
        # と写し、ベンチ側で fold 0 を test にする回だけ回す
        # （src/experiments/proteingym_benchmark.py の「変更点 2/2」参照）。
        #
        # test を 0 にしているのは、ベンチ側の分岐を
        #   「fold 0 だけ残す」という単純な条件で書けるようにするため。
        # 逆（train=0）にすると train を予測してしまうので、この対応は逆にしないこと。
        #
        # 列名を "fold_flip_<split>" にしているのは、ベンチ側が
        # startswith("fold_flip_") で FLIP 用の分岐に入るため。この命名は規約。
        # -------------------------------------------------------------------------
        fold_col = f"fold_flip_{split}"
        # split 列が空欄（= その split の対象外）の行は落とす
        sub = df[df[split].isin(["train", "test"])].copy()
        sub[fold_col] = (sub[split] == "train").astype(int)

        dms_id = f"{DMS_ID}_{split}"
        cols = ["mutant", "mutations", "mutated_sequence", "DMS_score", fold_col]
        sub[cols].to_csv(out_dir / f"{dms_id}.csv", index=False)

        n_test = int((sub[fold_col] == 0).sum())
        n_train = int((sub[fold_col] == 1).sum())
        print(f"{dms_id}: train={n_train} test={n_test} -> {out_dir / f'{dms_id}.csv'}")

        # 参照ファイル(DMS_substitutions.csv 相当)の1行。
        # Kermut は以下の用途でこの行を引く:
        #   DMS_id      : アッセイCSV・埋め込み・座標・条件付き分布のファイル名
        #   UniProt_ID  : PDB と ProteinMPNN 生出力のディレクトリ名
        #   target_seq  : WT配列。変異カーネルの基準およびPDBとの整合チェックに使う
        # 残りの列は prepare_datasets() のフィルタ条件で参照されるもの。
        ref_rows.append(
            {
                "DMS_id": dms_id,
                "UniProt_ID": UNIPROT_ID,
                "target_seq": wt_seq,
                "seq_len": len(wt_seq),
                "includes_multiple_mutants": True,
                "DMS_total_number_mutants": len(sub),
                "DMS_number_single_mutants": int((sub["HD"] == 1).sum()),
                "DMS_number_multiple_mutants": int((sub["HD"] > 1).sum()),
            }
        )

    ref_path = Path(args.ref_out)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ref_rows).to_csv(ref_path, index=False)
    print(f"\n参照ファイル: {ref_path}")
    print(f"WT配列({len(wt_seq)}残基): {wt_seq}")


if __name__ == "__main__":
    main()
