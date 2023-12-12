#!/bin/bash
N_THREADS=64

export MKL_NUM_THREADS=${N_THREADS}
export NUMEXPR_NUM_THREADS=${N_THREADS}
export OMP_NUM_THREADS=${N_THREADS}
export OPENBLAS_NUM_THREADS=${N_THREADS}


#python src/experiments/evaluate_on_splits_kermut.py --multirun \
#  gp=kermutP,kermutB,kermutD,kermutBD \
#  dataset=BLAT_ECOLX \
#  split=pos,aa_mixed,aa_diff \
#  n_train=100,1000

#python src/experiments/evaluate_on_splits_kermut.py --multirun \
#  gp=kermutP,kermutB,kermutD,kermutBD \
#  dataset=PARD3_10,SPG1 \
#  n_train=100,1000

#python src/experiments/evaluate_on_splits_gp.py --multirun \
#  gp=oh_seq_rbf,oh_seq_lin,oh_mut_rbf,oh_mut_lin \
#  dataset=BLAT_ECOLX \
#  split=pos,aa_mixed,aa_diff \
#  n_train=100,1000

python src/experiments/evaluate_on_splits_gp.py --multirun \
  gp=oh_mut_rbf,oh_mut_lin \
  dataset=PARD3_10,SPG1 \
  n_train=100,1000
