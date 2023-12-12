#!/bin/bash
N_THREADS=64

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=32

python src/experiments/oh_regression.py --multirun \
  experiment=regression_PARD3_10,regression_BLAT_ECOLX,regression_GFP,regression_SPG1,regression_AAV \
  ++experiment.n_train=100,1000 \
  encoding=oh_seq,oh_mut,mean_prediction

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

python src/experiments/gp_regression.py --multirun \
  experiment=regression_PARD3_10,regression_BLAT_ECOLX,regression_GFP,regression_SPG1,regression_AAV \
  ++experiment.n_train=100,1000 \
  gp=oh_mut_lin,oh_mut_rbf,oh_seq_lin,oh_seq_rbf

python src/experiments/kermut_regression.py --multirun \
  experiment=regression_PARD3_10,regression_BLAT_ECOLX,regression_GFP,regression_SPG1,regression_AAV \
  gp=kermutBH_oh,kermutBH \
  ++experiment.n_train=100,1000
