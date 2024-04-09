#!/bin/bash

# Ensure following variables are defined for the assay at hand:
ESM_DIR=$path_to_esm
ASSAY=TCRG1_MOUSE_Tsuboyama_2023_1E0L
WT_SEQUENCE=GATAVSEWTEYKTADGKTYYYNNRTLESTWEKPQELK

MODEL_PATH=models/esm2_t33_650M_UR50D.pt
DMS_INPUT=data/substitutions_singles/${ASSAY}.csv
DMS_OUTPUT=data/zero_shot_fitness_predictions/ESM2/650M/${ASSAY}.csv

python ${ESM_DIR}/examples/variant-prediction/predict.py \
    --model-location $MODEL_PATH \
    --sequence $WT_SEQUENCE \
    --dms-input $DMS_INPUT \
    --mutation-col mutant \
    --dms-output $DMS_OUTPUT \
    --scoring-strategy masked-marginals \
    --offset-idx 1

# Zero-shot scores are as saved as "models/esm2_t33_650M_UR50D.pt" column. Replace.
sed -i -- 's/models\/esm2_t33_650M_UR50D.pt/esm2_t33_650M_UR50D/g' $DMS_OUTPUT