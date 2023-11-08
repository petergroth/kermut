#!/bin/bash
N_TRAIN=1000
N_THREADS=64

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

python src/experiments/gp_regression.py --multirun \
  experiment=regression_GFP,regression_PARD3_10,regression_BLAT_ECOLX \
  ++experiment.n_train=${N_TRAIN} \
  gp=kermut,oh_mut_lin,oh_mut_rbf,oh_seq_rbf,oh_seq_lin

