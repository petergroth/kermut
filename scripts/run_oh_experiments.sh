#!/bin/bash

N_THREADS=64

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

#python src/experiments/oh_regression.py --multirun \
#  experiment=regression_GFP,regression_BLAT_ECOLX,regression_PARD3_10 \
#  ++experiment.n_train=100,1000 \
#  encoding=oh_seq,oh_mut,mean_prediction

python src/experiments/oh_regression.py --multirun \
  experiment=regression_AAV \
  ++experiment.n_train=100,1000 \
  encoding=oh_seq,oh_mut,mean_prediction