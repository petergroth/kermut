#!/bin/bash

# Train on singles, test on doubles:
python src/experiments/proteingym_domains.py -m \
    dataset=benchmark \
    gp=kermut \
    split_method=domain