#!/bin/bash

python src/experiments/additive_baseline.py --multirun \
  experiment=regression_AAV,regression_SPG1,regression_GFP,regression_PARD3_10 \
  ++experiment.n_train=100,1000,10000 \
  ++n_muts=2,100