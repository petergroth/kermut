# Kermut

This is the official code repository for the paper _Protein Property Prediction with Uncertainties_ ([link]())


## Overview
Kermut is a carefully constructed Gaussian process which obtains state-of-the-art performance for protein property prediction on ProteinGym's supervised substitution benchmark while providing meaningful overall calibration.

## Installation

After cloning the repository, the environment can be installed via

```bash
conda env create -f environment.yml
conda activate kermut_env
conda develop .
```

### ProteinMPNN
Kermut leverages structure-conditioned amino acid distributions from [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187), which can has to installed from the [official repository](https://github.com/dauparas/ProteinMPNN). An environment variable pointing to the installation location can then be set for later use:

```bash
export PROTEINMPNN_DIR=<path-to-ProteinMPNN-installation>
```

### ESM-2
Kermut leverages protein sequence embeddings and zero-shot scores extracted from ESM-2 ([paper](https://www.science.org/doi/10.1126/science.ade2574), [repo](https://github.com/facebookresearch/esm)). We concretely use the 650M parameter model (`esm2_t33_650M_UR50D`). While the ESM repository is installed above, the model should be downloaded separately and placed in the `models` directory. 


### Data downloading

Kermut is evaluated on the ProteinGym benchmark ([paper](https://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html), [repo](https://github.com/OATML-Markslab/ProteinGym)).
For downloading the relevant data, please see the ProteinGym [resources](https://github.com/OATML-Markslab/ProteinGym?tab=readme-ov-file#resources). The following data is used:

- __Reference file__: A [reference file](https://github.com/OATML-Markslab/ProteinGym/blob/main/reference_files/DMS_substitutions.csv) with details on all assays can be downloaded from the ProteinGym repo and should be saved as `data/DMS_substitutions.csv`

- __Assay data__: All assays (with CV folds) can be downloaded and extracted to `data`. This results in two subdirectories: `data/substitutions_singles` and `data/substitutions_multiples`. The data can be accessed via `CV folds - Substitutions - <Singles,Multiples>` in ProteinGym. 

- __PDBs__: All predicted structure files are downloaded and placed in `data/structures/pdbs`. PDBs are accessed via `Predicted 3D structures from inverse-folding models` in ProteinGym.


- __Zero-shot scores__: For the zero-shot mean function, pre-computed scores can be downloaded and placed in `zero_shot_fitness_predictions`, where each zero-shot method has its own directory. For ESM-2, this includes an additional subdirectory: `ESM2/650M`. The data can be downloaded via `Zero-shot DMS Model scores - Substitutions` from ProteinGym. _Alternatively_, see below for computing zero-shot scores with ESM-2 locally.

- (Optional) __Baselines scores__: Results from ProteinNPT and the baseline models from ProteinGym can be accessed via `Supervised DMS Model performance - Substitutions`. The resulting csv-file can be placed in `results/baselines`.

### Data preprocessing
#### Sequence embeddings
After downloading and extracting the relevant data, ESM-2 embeddings can be generated via the `example_scripts/generate_embeddings.sh` script or simply via: 

```bash
python src/data/extract_esm2_embeddings.py \
    --dataset=all \
    --which=singles
```
The embeddings for individual assays can be generated by replacing `all` with the full assay name (e.g., `--dataset=BLAT_ECOLX_Stiffler_2015`).
For each assay, an `h5` file is generated which contains all embeddings.
The embeddings are located in `data/embeddings/substitutions_singles/ESM2` (for the single-mutant assays).
For the multi-mutant assays, replace `singles` with `multiples`.

__Note__: Embeddings for all 217 single mutation DMS assays requires approximately 1.7TBs of storage. Currently, the full per-AA embeddings are saved. For reduced storage and loading times, this can be altered such that only the mean-pooled embeddings are saved. 

#### Structure-conditioned amino acid distributions

The structure-conditioned amino acid distributions for all residues and assays, can be computed with ProteinMPNN via

```
bash example_scripts/conditional_probabilities.sh
```
For a single dataset, see `example_scripts/conditional_probabilities_single.sh`.

After the probability distributions have been computed, they are postprocessed for easier access via
```bash
python src/data/extract_ProteinMPNN_probs.py
```
This generates per-assay files in `data/conditional_probs/ProteinMPNN`.

#### 3D coordinates
Lastly, the 3D coordinates can be extracted from each PDB file via
```bash
python src/data/extract_3d_coords.py
```
This saves `npy`-files for each assay in `data/structures/coords`. 

#### Optional: Zero-shot scores
If not relying on pre-computed zero-shot scores from ProteinGym, they can be computed for ESM-2 via:
```bash
bash example_scripts/zero_shot_scores.sh
```
See the script for usage details. For multi-mutant datasets, the log-likelihood ratios are summed for each mutant.

__Note__: This requires a local installation of the ESM repository and additionally requires `biopython` to be installed in the environment. 

## Usage

### Reproduce the main results
To reproduce the results (assuming that preprocessing has been carried out for all 217 DMS assays), run the following script:
```bash
bash example_scripts/benchmark_all_datasets.sh
```
This will run the Kermut GP on all 217 assays using the predefined random, modulo, and contiguous splits for cross validation. 
Each assay and split configuration generates a csv-file with the per-variant predictive means and variances. 
The prediction files can be merged using
```bash
python src/process_results/merge_score_files.py
```
This also merges the results with all baseline models from the ProteinGym paper.

Lastly, the results are aggregated across proteins and functional categories to obtain the final scores using ProteinGym functionality:
```bash
python src/process_results/performance_DMS_supervised_benchmarks.py \
    --input_scoring_file results/merged_scores.csv \
    --output_performance_file_folder results/summary
```
The post-processing steps for the main and ablation results can be seen in `example_scripts/process_results.sh`.

### Run on single assay
Assuming that preprocessed data from ProteinGym is used, Kermut can be evaluated on any assay individually as follows:
```bash
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=ASSAY_NAME \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut \
    use_gpu=true
```
The codebase was developed using [Hydra](https://hydra.cc/). The ablation GPs/kernels can be accessed straightforwardly:
```bash
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=ASSAY_NAME \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut_constant_,kermut_no_g \
    use_gpu=true
```

### Running on new splits
Any dataset for which the structure follows the ProteinGym data (and has been processed as above) can be evaluated using new splits. 
This simply requires adding new columns to the assay file in `data/substitutions_singles`. The column should contain integer values for each variant indicating which fold it belongs to. The chosen GP will then be evaluated using CV (i.e., tested on all unique paritions while trained on the remaining):
```bash
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=ASSAY_NAME \
    split_method=new_split_col \
    gp=kermut \
    use_gpu=true
```

## Citations
TBD