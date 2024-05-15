#!/bin/bash

# Evaluates on 174 datasets 
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=ablation \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut_no_g,kermut_no_d,kermut_no_h,kermut_no_hp,kermut_no_m,kermut_no_p,kermut_constant_mean,kermut_no_m_constant_mean\
    use_gpu=true