#!/bin/bash
N_THREADS=64

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}

#python src/experiments/kermut_regression.py --multirun \
#  experiment=regression_GFP,regression_PARD3_10,regression_BLAT_ECOLX,regression_AAV,regression_SPG1 \
#  gp=kermutB,kermutP,kermutD,kermutBD \
#  ++experiment.n_train=100,1000

#python src/experiments/kermut_regression.py --multirun \
#  experiment=regression_SPG1,regression_GFP,regression_PARD3_10,regression_BLAT_ECOLX,regression_AAV \
#  gp=kermutBHNorm \
#  ++experiment.n_train=100,1000
#
python src/experiments/kermut_regression.py --multirun \
  experiment=regression_SPG1,regression_PARD3_10,regression_BLAT_ECOLX,regression_GFP,regression_AAV \
  gp=kermutBH_oh \
  ++experiment.n_train=100,1000 \
  gp.optim.log_to_wandb=true

