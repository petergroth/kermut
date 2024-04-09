#!/bin/bash

# Majority of datasets (~205/217) can be evaluated on a single GPU (48GB VRAM)
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=all \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut \
    use_gpu=true \
    limit_mutants=false \
    limit_mem=true

# Remainder is evaluated on CPU
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=large \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut \
    use_gpu=false \
    limit_mutants=false \
    limit_mem=false
