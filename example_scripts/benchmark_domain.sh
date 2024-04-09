#!/bin/bash

# Train on singles, test on doubles:
python src/experiments/proteingym_domains.py -m \
    dataset=all \
    gp=kermut \
    split_method=domain \
    use_gpu=true  \
    limit_mem=true

