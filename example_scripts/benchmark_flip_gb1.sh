#!/bin/bash
# FLIP GB1 の5 split で Kermut を評価する。
# 前処理は example_scripts/flip_gb1_preprocess.sh を先に実行しておくこと。
set -euo pipefail

for SPLIT in one_vs_rest two_vs_rest three_vs_rest sampled low_vs_high; do
    python src/experiments/proteingym_benchmark.py \
        dataset="GB1_FLIP_${SPLIT}" \
        split_method="fold_flip_${SPLIT}" \
        gp=kermut
done
