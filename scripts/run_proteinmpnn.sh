#!/bin/bash

protein_mpnn_dir=$PROTEINMPNN_DIR
DATASET="PARD3_10"
path_to_PDB="data/raw/${DATASET}/${DATASET}.pdb"

output_dir="data/interim/${DATASET}/proteinmpnn"
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

python $protein_mpnn_dir/protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --save_score 1 \
        --conditional_probs_only 1 \
        --num_seq_per_target 10 \
        --batch_size 1 \
        --out_folder $output_dir \
        --seed 37
