#!/bin/bash
#N_TRAIN=100
N_THREADS=64

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=1


python src/experiments/oh_regression.py --multirun \
  experiment=regression_PARD3_10 \
  ++experiment.n_train=100,1000 \
  encoding=oh_seq,oh_mut,mean_prediction

python src/experiments/gp_regression.py --multirun \
  experiment=regression_PARD3_10 \
  ++experiment.n_train=100,1000 \
  gp=kermut,oh_mut_lin,oh_mut_rbf,oh_seq_rbf,oh_seq_lin
