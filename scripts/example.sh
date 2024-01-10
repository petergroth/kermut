#!/bin/bash


# Single dataset, random split
python src/experiments/proteingym_kermut_reg.py \
 gp=kermutBH_oh \ 
 dataset=BLAT_ECOLX_Stiffler_2015 \
 split_method=fold_random_5 

# Output file: (note: ProteinMPNN is used for AA distributions)
# results/ProteinGym/per_dataset/BLAT_ECOLX_Stiffler_2015/kermut_ProteinMPNN_fold_random_5.csv


# Single dataset, all splits
python src/experiments/proteingym_kermut_reg.py \
 --multirun \
 gp=kermutBH_oh \ 
 dataset=BLAT_ECOLX_Stiffler_2015 \
 split_method=fold_random_5,fold_contiguous_5,fold_modulo_5

# Output files:
# results/ProteinGym/per_dataset/BLAT_ECOLX_Stiffler_2015/kermut_ProteinMPNN_fold_random_5.csv
# results/ProteinGym/per_dataset/BLAT_ECOLX_Stiffler_2015/kermut_ProteinMPNN_fold_contiguous_5.csv
# results/ProteinGym/per_dataset/BLAT_ECOLX_Stiffler_2015/kermut_ProteinMPNN_fold_modulo_5.csv


# Single dataset, random split, ESM_IF1 as zero-shot mean function
python src/experiments/proteingym_kermut_reg.py \
 gp=kermutBH_oh \ 
 use_zero_shot=true \
 zero_shot_method=ESM_IF1 \
 dataset=BLAT_ECOLX_Stiffler_2015 \
 split_method=fold_random_5 

# Output file: (note: ProteinMPNN is used for AA distributions, zero-shot mean function is ESM_IF1)
# results/ProteinGym/per_dataset/BLAT_ECOLX_Stiffler_2015/kermut_ProteinMPNN_ESM_IF1_fold_random_5.csv


# Single dataset, random split, ESM_IF1 as zero-shot mean function and AA distributions
# GP parameters are logged (as columns in output file), progress bar is shown during optimization
python src/experiments/proteingym_kermut_reg.py \
 gp=kermutBH_oh \ 
 gp.conditional_probs_method=ESM_IF1 \
 use_zero_shot=true \
 zero_shot_method=ESM_IF1 \
 dataset=BLAT_ECOLX_Stiffler_2015 \
 split_method=fold_random_5 \
 log_params=true \
 progress_bar=true

# Output file: (note: ProteinMPNN is used for AA distributions, zero-shot mean function is ESM_IF1)
# results/ProteinGym/per_dataset/BLAT_ECOLX_Stiffler_2015/kermut_ESM_IF1_ESM_IF1_fold_random_5.csv


