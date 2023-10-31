#!/bin/bash
python src/experiments/oh_regression.py --multirun \
  experiment=regression_GFP,regression_BLAT_ECOLX,regression_PARD3_10 \
  encoding=oh_seq,oh_mut