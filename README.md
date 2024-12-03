# Kermut

This is the official code repository for the NeurIPS 2024 Spotlight paper  _Kermut: Composite kernel regression for protein variant effects_ ([preprint](https://arxiv.org/abs/2407.00002)).


## Overview
Kermut is a carefully constructed Gaussian process which obtains state-of-the-art performance for supervised variant effect prediction on ProteinGym's substitution benchmark while providing well-calibrated uncertainties.

## Reproducibility

The `main` branch has been rewritten and restructured for ease of use and clarity. Implementation differences might results in minor numerical differences to those of the paper. To reproduce the paper results, instead use the `reproduce` branch and consult the extensive README in that branch:
```bash
git clone -b reproduce git@github.com:petergroth/kermut.git
```
## Installation
```bash
git clone git@github.com:petergroth/kermut.git
cd kermut
conda env create --file environment.yaml
conda activate kermut_envs
pip install -e .
```

### Optional 
To run Kermut from scratch without precomputed resources, e.g., for a new dataset, the ProteinMPNN repository must be installed. Additionally, the ESM-2 650M parameter model must be saved locally: 
#### ProteinMPNN
Kermut leverages structure-conditioned amino acid distributions from [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187), which can has to installed from the [official repository](https://github.com/dauparas/ProteinMPNN). An environment variable pointing to the installation location can then be set for later use:

```bash
export PROTEINMPNN_DIR=<path-to-ProteinMPNN-installation>
```

#### ESM-2 models 
Kermut leverages protein sequence embeddings and zero-shot scores extracted from ESM-2 ([paper](https://www.science.org/doi/10.1126/science.ade2574), [repo](https://github.com/facebookresearch/esm)). We concretely use the 650M parameter model (`esm2_t33_650M_UR50D`). While the ESM repository is installed above /via the yml-file), the model weights should be downloaded separately and placed in the `models` directory:

```bash
curl -o models/esm2_t33_650M_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

curl -o models/esm2_t33_650M_UR50D-contact-regression.pt https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

## Data access
This section describes how to access the data that was used to generate the results. To reproduce _all_ results from scratch, follow all steps in this section and in the [Data preprocessing](#data-preprocessing) section. To reproduce the benchmark results using precomputed resources (ESM-2 embeddings, conditional amino-acid distributions, etc.) see the section on [precomputed resources](#precomputed-resources).

Kermut is evaluated on the ProteinGym benchmark ([paper](https://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html), [repo](https://github.com/OATML-Markslab/ProteinGym)).
For full details on downloading the relevant data, please see the ProteinGym [resources](https://github.com/OATML-Markslab/ProteinGym?tab=readme-ov-file#resources). In the following, commands are provided to extract the relevant data.

- __Reference file__: A [reference file](https://github.com/OATML-Markslab/ProteinGym/blob/main/reference_files/DMS_substitutions.csv) with details on all assays can be downloaded from the ProteinGym repo and should be saved as `data/DMS_substitutions.csv`

The file can be downloaded by running the following:
```bash
curl -o data/DMS_substitutions.csv https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv
```

- __Assay data__: All assays (with CV folds) can be downloaded and extracted to `data`. Run the following to download and extract all single-mutant assays. Assays will be placed in `data/cv_folds_singles_substitutions`:
```bash
# Download zip archive
curl -o cv_folds_singles_substitutions.zip https://marks.hms.harvard.edu/proteingym/cv_folds_singles_substitutions.zip
# Unpack and remove zip archive
unzip cv_folds_singles_substitutions.zip -d data
rm cv_folds_singles_substitutions.zip
```


- __PDBs__: All predicted structure files are downloaded and placed in `data/structures/pdbs`. PDBs are accessed via `Predicted 3D structures from inverse-folding models` in ProteinGym.

```bash
# Download zip archive
curl -o ProteinGym_AF2_structures.zip https://marks.hms.harvard.edu/proteingym/ProteinGym_AF2_structures.zip
# Unpack and remove zip archive
unzip ProteinGym_AF2_structures.zip -d data/structures/pdbs
rm ProteinGym_AF2_structures.zip
```

- __Zero-shot scores__: For the zero-shot mean function, precomputed scores can be downloaded and placed in `zero_shot_fitness_predictions`, where each zero-shot method has its own directory. The precomputed zero-shot scores from ProteinGym can be accessed via `Zero-shot DMS Model scores - Substitutions`. __NOTE__: The full zip archive with all scores takes up approximately 44GB of storage. Alternatively, the zero-shot scores for the 650M parameter ESM-2 model is included in the [precomputed resources](#precomputed-resources), which in total is only approximately 4GB.

```bash
# Download zip archive
curl -o zero_shot_substitutions_scores.zip https://marks.hms.harvard.edu/proteingym/zero_shot_substitutions_scores.zip
unzip zero_shot_substitutions_scores.zip -d data/zero_shot_fitness_predictions
# Unpack and remove zip archive
rm zero_shot_substitutions_scores.zip
```
## Precomputed resources
All outputs from the preprocessing procedure (i.e., precomputed ESM-2 embeddings, conditional amino acid distributions, processed coordinate files, and zero-shot scores from ESM-2) can be readily accessed via a zip-archive hosted by the Electronic Research Data Archive (ERDA) by the University of Copenhagen using the following [link](https://sid.erda.dk/sharelink/c2EWrbGSCV). The file takes up approximately 4GB. To download and extract the data, run the following:

```bash
# Download zip archive
curl -o kermut_data.zip https://sid.erda.dk/share_redirect/c2EWrbGSCV/kermut_data.zip
# Unpack and remove zip archive
unzip kermut_data.zip && rm kermut_data.zip
```
## Data preprocessing
### Sequence embeddings
After downloading and extracting the relevant data in the [Data access section](#data-access), ESM-2 embeddings can be generated via:

```bash
python -m kermut.cmdline.preprocess_data.extract_esm2_embeddings \
    dataset=all 
```

To generate embeddings for an individual dataset (e.g., `BLAT_ECOLX_Stiffler_2015`), run
```bash
python -m kermut.cmdline.preprocess_data.extract_esm2_embeddings \
    dataset=single \
    dataset.single.use_id=true \
    dataset.single.id=BLAT_ECOLX_Stiffler_2015
```
To generate embeddings via index (i.e., row index in `DMS_substitutions.csv`), run
```bash
python -m kermut.cmdline.preprocess_data.extract_esm2_embeddings \
    dataset=single \
    dataset.single.id=23
```

The embeddings are located in `data/embeddings/substitutions_singles/ESM2` (for the single-mutant assays).

### Structure-conditioned amino acid distributions

The structure-conditioned amino acid distributions for all residues and assays, can be computed with ProteinMPNN via

```
bash example_scripts/conditional_probabilities.sh
```
For a single dataset, see `example_scripts/conditional_probabilities_single.sh` or `example_scripts/conditional_probabilities_all.sh`. This generates per-assay directories in `data/conditional_probs/raw_ProteinMPNN_outputs`. After this, postprocessing for easier access is performed via
```bash
python -m kermut.cmdline.preprocess_data.extract_ProteinMPNN_probs \ 
    dataset=all
```
This generates per-assay `npy`-files in `data/conditional_probs/ProteinMPNN`.

### 3D coordinates
Lastly, the 3D coordinates can be extracted from each PDB file via
```bash
python -m kermut.cmdline.preprocess_data.extract_3d_coords \
    dataset=all
```
This saves `npy`-files for each assay in `data/structures/coords`. 
For single assays, use same inputs as for embeddings.

### Optional: Zero-shot scores
If not relying on pre-computed zero-shot scores from ProteinGym, they can be computed for ESM-2 via:
```bash
python -m kermut.cmdline.preprocess_data.extract_esm2_zero_shots \
    dataset=all
```
# Usage

The implementation of Kermut relies on [Hydra](https://hydra.cc/). 
Configuration files are found in `kermut/hydra_configs`.
Data paths are defined in `data/paths.yaml` and must match your setup. 

To evaluate Kermut on the full benchmark, run the following
```bash
python proteingym_benchmark.py --multirun \
    dataset=benchmark \
    cv_scheme=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    kernel=kermut  # Default
```

To evaluate an alternative kernel (e.g., `kermut_constant_mean` as defined in `kermut/hydra_configs/kernel/kermut_constant_mean.yaml` ) on DMS assay with index 9, run
```bash
python proteingym_benchmark.py --multirun \
    dataset=single \
    dataset.single.id=9 \
    cv_scheme=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    kernel=kermut_constant_mean
```

## Postprocessing
To compute Spearman correlation and MSE per assay and cv-scheme, run
```bash
python -m kermut.cmdline.process_results.merge_results \
    dataset=benchmark
```
This will compute results for all models stored in the `model_names.benchmark` list in `kermut/hydra_configs/postprocessing/default.yaml`.
For a single model, run
```bash
python -m kermut.cmdline.process_results.merge_results \
    dataset=benchmark \
    "model_names=[kermut]"
```