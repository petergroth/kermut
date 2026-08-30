#!/bin/bash
# FLIP GB1 を Kermut で回すための前処理一式。
#
# 事前準備（詳細は FLIP_GB1.md の5章）:
#
#   a) FLIP のデータ
#      https://github.com/J-SNACKKB/FLIP/tree/main/splits/gb1 から
#      four_mutations_full_data.csv を取得し data/flip/ に置く
#
#   b) GB1 の構造。ファイル名は必ず GB1_1PGA.pdb にすること（理由は手順3のコメント）
#      curl -o data/structures/pdbs/GB1_1PGA.pdb https://files.rcsb.org/download/1PGA.pdb
#
#   c) ProteinMPNN。HuggingFace ではなく GitHub から clone する。
#      モデル重みはリポジトリに同梱されているので、clone するだけで完結する。
#      （pip install も重みの別途DLも不要。依存は torch と numpy のみ）
#        git clone https://github.com/dauparas/ProteinMPNN.git ~/tools/ProteinMPNN
#        export PROTEINMPNN_DIR=~/tools/ProteinMPNN
#
#   d) ESM-2 650M の重みを models/ に置く（Kermut 本体の README 参照）
#        curl -o models/esm2_t33_650M_UR50D.pt \
#            https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
#
# 手順1〜3は CPU で回る（GB1 は56残基と小さいので ProteinMPNN も CPU で数秒）。
# 手順4の ESM-2 だけ GPU 環境（gpu-a03, conda sample_pytorch）が必要。
set -euo pipefail

FLIP_CSV="${FLIP_CSV:-data/flip/four_mutations_full_data.csv}"

# 1) FLIP -> ProteinGym 形式のアッセイCSV + 参照ファイル
uv run python3 scripts/flip/prepare_flip_gb1.py --flip-csv "$FLIP_CSV"

# 2) CA 座標（距離カーネル用）。1PGA を使う理由はスクリプト冒頭の docstring 参照
uv run python3 scripts/flip/extract_gb1_coords.py

# 3) ProteinMPNN の条件付きアミノ酸分布
#
#    DATASET は参照ファイルの UniProt_ID と一致させる必要がある。
#    さらに重要なのは、ProteinMPNN の出力 .npz のファイル名が
#    「入力PDBのファイル名」から決まること（protein_mpnn_utils.py の parse_PDB が
#     my_dict['name'] = biounit[(fi+1):-4] としている）。
#    つまり PDB を 1pga.pdb のまま置くと 1pga.npz が出力され、
#    手順3の変換スクリプトが GB1_1PGA.npz を探して失敗する。
#    → PDB のファイル名は GB1_1PGA.pdb でなければならない。
#
#    PROTEINMPNN_DIR は clone したリポジトリのルートを指すこと。
#    protein_mpnn_run.py は重みのパスを「スクリプト自身の位置」から相対解決するので
#    （file_path[:k] + '/vanilla_model_weights/'）、スクリプトだけコピーすると動かない。
#    使われる重みは --model_name の既定値 v_48_020。
protein_mpnn_dir=$PROTEINMPNN_DIR
DATASET=GB1_1PGA
path_to_PDB="data/structures/pdbs/${DATASET}.pdb"
output_dir="data/conditional_probs/raw_ProteinMPNN_outputs/${DATASET}/proteinmpnn"
mkdir -p "$output_dir"

# --conditional_probs_only 1 : 配列を設計させるのではなく、各残基の条件付き
#                              アミノ酸分布だけを出す。変異カーネルが使うのはこれ
# --num_seq_per_target 10    : デコード順を変えて10回計算する。変換スクリプト側で
#                              log_p.mean(axis=0) して平均を取る
# --seed 37                  : Kermut 本家の例に合わせた固定値（再現性のため）
python "$protein_mpnn_dir/protein_mpnn_run.py" \
    --pdb_path "$path_to_PDB" \
    --save_score 1 \
    --conditional_probs_only 1 \
    --num_seq_per_target 10 \
    --batch_size 1 \
    --out_folder "$output_dir" \
    --seed 37

uv run python3 scripts/flip/extract_gb1_conditional_probs.py

# 4) ESM-2 埋め込みとゼロショットスコア（GPU 必須）
#    8,733配列 × 5 split になるので、ローカルCPUでは走らせないこと
for SPLIT in one_vs_rest two_vs_rest three_vs_rest sampled low_vs_high; do
    uv run python3 src/data/extract_esm2_embeddings.py --dataset "GB1_FLIP_${SPLIT}"
    uv run python3 src/data/extract_esm2_zero_shots.py --dataset "GB1_FLIP_${SPLIT}"
done
