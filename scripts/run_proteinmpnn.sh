#!/bin/bash

protein_mpnn_dir=""
path_to_PDB="data/raw/BLAT_ECOLX.pdb"

output_dir="data/interim/BLAT_ECOLX/proteinmpnn"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

python $protein_mpnn_dir/protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --save_score 1 \
        --save_probs 1 \
        --conditional_probs_only 1 \
        --num_seq_per_target 10 \
        --batch_size 1 \
        --out_folder $output_dir \
        --seed 37