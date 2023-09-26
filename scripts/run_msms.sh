#!/bin/bash

MSMS_DIR=""
QUERY="BLAT_ECOLX"
PDB_DIR="data/raw/"
OUTPUT_DIR="data/interim/${QUERY}/"

# Convert PDB to xyzrn format
${MSMS_DIR}/pdb_to_xyzrn ${PDB_DIR}${QUERY}.pdb > ${OUTPUT_DIR}${QUERY}.xyzrn

# Run MSMS
${MSMS_DIR}/msms -if ${OUTPUT_DIR}${QUERY}.xyzrn -of ${OUTPUT_DIR}${QUERY} -no_header

# Clean MSMS output
python src/data/clean_msms_output.py --wt_key ${QUERY}
