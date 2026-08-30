#!/bin/bash
# FLIP GB1 を Kermut で回すための前処理一式。
# 事前に FLIP の four_mutations_full_data.csv を取得しておくこと:
#   https://github.com/J-SNACKKB/FLIP/tree/main/splits/gb1
set -euo pipefail

FLIP_CSV="${FLIP_CSV:-data/flip/four_mutations_full_data.csv}"

# 1) FLIP -> ProteinGym 形式のアッセイCSV + 参照ファイル
uv run python3 scripts/flip/prepare_flip_gb1.py --flip-csv "$FLIP_CSV"

# 2) CA 座標（距離カーネル用）。1PGA を使う理由はスクリプト冒頭の docstring 参照
uv run python3 scripts/flip/extract_gb1_coords.py

# 3) ProteinMPNN の条件付きアミノ酸分布
#    参照ファイルの UniProt_ID と揃える必要がある
protein_mpnn_dir=$PROTEINMPNN_DIR
DATASET=GB1_1PGA
path_to_PDB="data/structures/pdbs/${DATASET}.pdb"
output_dir="data/conditional_probs/raw_ProteinMPNN_outputs/${DATASET}/proteinmpnn"
mkdir -p "$output_dir"

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
for SPLIT in one_vs_rest two_vs_rest three_vs_rest sampled low_vs_high; do
    uv run python3 src/data/extract_esm2_embeddings.py --dataset "GB1_FLIP_${SPLIT}"
    uv run python3 src/data/extract_esm2_zero_shots.py --dataset "GB1_FLIP_${SPLIT}"
done
