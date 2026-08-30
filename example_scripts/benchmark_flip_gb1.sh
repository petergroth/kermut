#!/bin/bash
# FLIP GB1 の5 split で Kermut を評価する。
# 前処理は example_scripts/flip_gb1_preprocess.sh を先に実行しておくこと。
set -euo pipefail

# --- GPU の使い方（共有サーバー向け）-------------------------------------------
# gpu-a03 は4枚構成。既定では GPU 0 に集中しがちなので、空いている GPU を
# 明示的に選ぶ。nvidia-smi で Memory-Usage と GPU-Util が低いものを指定すること。
#
#   CUDA_VISIBLE_DEVICES=2 bash example_scripts/<このスクリプト>
#
# VRAM の上限（既定80%）も環境変数で変えられる:
#   KERMUT_GPU_MEM_FRACTION=0.8
#
# 注意: CUDA_VISIBLE_DEVICES で絞ると、プログラムからは選んだGPUが cuda:0 に見える。
# ------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export KERMUT_GPU_MEM_FRACTION="${KERMUT_GPU_MEM_FRACTION:-0.8}"

for SPLIT in one_vs_rest two_vs_rest three_vs_rest sampled low_vs_high; do
    python src/experiments/proteingym_benchmark.py \
        dataset="GB1_FLIP_${SPLIT}" \
        split_method="fold_flip_${SPLIT}" \
        gp=kermut
done
