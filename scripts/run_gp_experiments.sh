#!/bin/bash
N_THREADS=64

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

#python src/experiments/gp_regression.py --multirun \
#  experiment=regression_GFP,regression_PARD3_10,regression_BLAT_ECOLX \
#  ++experiment.n_train=${N_TRAIN} \
#  gp=oh_mut_lin,oh_mut_rbf,oh_seq_lin,oh_seq_rbf

python src/experiments/gp_regression.py --multirun \
  experiment=regression_AAV \
  ++experiment.n_train=100,1000 \
  gp=oh_mut_lin,oh_mut_rbf,oh_seq_lin,oh_seq_rbf
