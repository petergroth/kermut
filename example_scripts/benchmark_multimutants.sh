#!/bin/bash

# To use the fold_rand_multiples split, simply specify the split
python src/experiments/proteingym_benchmark.py -m \
    dataset=benchmark \
    split_method=fold_rand_multiples \
    gp=kermut


