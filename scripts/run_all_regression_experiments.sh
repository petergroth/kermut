#!/bin/bash
N_TRAIN=1000
N_THREADS=32

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}

python src/experiments/oh_regression.py --multirun \
  experiment=regression_GFP,regression_BLAT_ECOLX,regression_PARD3_10 \
  ++experiment.n_train=${N_TRAIN} \
  ++n_threads=${N_THREADS} \
  encoding=oh_seq,oh_mut,mean_prediction

python src/experiments/gp_regression.py --multirun \
  experiment=regression_GFP,regression_PARD3_10,regression_BLAT_ECOLX \
  ++experiment.n_train=${N_TRAIN} \
  gp=oh_rbf,kermut