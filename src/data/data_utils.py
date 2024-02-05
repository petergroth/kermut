import ast
from pathlib import Path
from typing import Tuple

import hydra
import h5py
import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from omegaconf import DictConfig
from scipy.io import loadmat

from src import AA_TO_IDX, AA3_TO_AA


def process_substitution_matrices():
    # Based on mGPfusion by Jokinen et al. (2018)
    output_path = Path("data", "interim", "blosum62.npy")
    matrix_path = Path("data", "raw", "subMats.mat")
    matrix_file = loadmat(str(matrix_path))["subMats"]
    names = [name.item() for name in matrix_file[:, 1]]
    descriptions = [description.item() for description in matrix_file[:, 2]]

    full_matrix = np.zeros((21, 20, 20))
    for i in range(21):
        full_matrix[i] = matrix_file[i, 0]

    substitution_dict = {name: matrix for name, matrix in zip(names, full_matrix)}
    substitution_matrix_name = "HENS920102"  # Corresponds to BLOSUM62
    substitution_matrix = substitution_dict[substitution_matrix_name]
    # Alphabetize
    idx = [AA_TO_IDX[aa] for aa in "ARNDCQEGHILKMFPSTWYV"]
    substitution_matrix = substitution_matrix[idx][:, idx]
    np.save(output_path, substitution_matrix)


def one_hot_encode_mutation(df: pd.DataFrame):
    """One-hot encoding mutations.

    Each position with a mutation is represented by a 20-dimensional vector, regardless of whether each
    mutation is actually observed.

    Args:
        df (pd.DataFrame): Dataset with list of mutations in the `mut2wt` column.

    Returns:
        np.ndarray: One-hot encoded mutations (shape: (n_samples, n_mutated_positions * 20)).
    """
    if isinstance(df["mut2wt"].iloc[0], str):
        df["mut2wt"] = df["mut2wt"].apply(ast.literal_eval)
    mutated_positions = df["mut2wt"].explode().str[1:-1].astype(int).unique()
    mutated_positions = np.sort(mutated_positions)
    one_hot = np.zeros((len(df), len(mutated_positions), 20))
    pos_to_idx = {pos: i for i, pos in enumerate(mutated_positions)}
    for i, mut2wt in enumerate(df["mut2wt"]):
        for mut in mut2wt:
            pos = int(mut[1:-1])
            aa = mut[-1]
            one_hot[i, pos_to_idx[pos], AA_TO_IDX[aa]] = 1.0
    one_hot = one_hot.reshape(len(df), 20 * len(mutated_positions))
    return one_hot


def one_hot_encode_sequence(df: pd.DataFrame, as_tensor: bool = False):
    """One-hot encoding sequences.

    Args:
        df (pd.DataFrame): Dataset with sequence string in the `seq` column.
        as_tensor (bool, optional): Whether to return a torch tensor. Defaults to False.

    Returns:
        np.ndarray: One-hot encoded mutations (shape: (n_samples, seq_len * 20)).
    """
    seq_len = len(df.iloc[0]["seq"])
    one_hot = np.zeros((len(df), seq_len, 20))
    for i, seq in enumerate(df["seq"]):
        for j, aa in enumerate(seq):
            one_hot[i, j, AA_TO_IDX[aa]] = 1.0
    one_hot = one_hot.reshape(len(df), 20 * seq_len)
    if as_tensor:
        return torch.tensor(one_hot).long()
    return one_hot


def get_coords_from_pdb(dataset: str, only_ca: bool = True, as_tensor: bool = False):
    """Get the coordinates of the atoms in the protein from the PDB file.

    Args:
        dataset (str): Name of the dataset.
        only_ca (bool, optional): Whether to only use the alpha carbon atoms. Defaults to True.
        as_tensor (bool, optional): Whether to return a torch tensor. Defaults to False.
    """
    wt_df = pd.read_csv("data/processed/wt_sequences.tsv", sep="\t")
    wt_sequence = wt_df.loc[wt_df["dataset"] == dataset, "seq"].item()

    pdb_path = Path("data/raw", dataset, f"{dataset}.pdb")
    parser = PDBParser()
    structure = parser.get_structure(dataset, pdb_path)
    model = structure[0]
    chain = model["A"]
    if only_ca:
        coords = np.array(
            [atom.get_coord() for atom in chain.get_atoms() if atom.get_name() == "CA"]
        )
    else:
        coords = np.array(
            [
                atom.get_coord()
                for atom in chain.get_atoms()
                if atom.get_name() in ["CA", "C", "N", "O"]
            ]
        )

    residues = [
        atom.get_parent().get_resname()
        for atom in chain.get_atoms()
        if atom.get_name() == "CA"
    ]
    residues = [AA3_TO_AA[res] for res in residues]

    if dataset == "PARD3_10":
        # PDB is missing 2 initial and 6 final residues. Add zeros
        coords_expanded = np.zeros((coords.shape[0] + 8, coords.shape[1]))
        coords_expanded[2:-6] = coords
        coords = coords_expanded

    elif dataset == "AAV":
        coords = coords[(424 - 80) : (424 - 80 + 28)]
        wt = residues[(424 - 80) : (424 - 80 + 28)]
        assert wt == list(wt_sequence)

    # Print shape of coords
    print(f"{dataset} shape: {coords.shape}")

    if as_tensor:
        coords = torch.tensor(coords)

    return coords


def load_esmif1_proteingym(wt: pd.Series):
    conditional_probs_path = Path(
        "data",
        "conditional_probs",
        wt["UniProt_ID"].item(),
        f"{wt['UniProt_ID'].item()}_ESM_IF1.npy",
    )
    conditional_probs = np.load(conditional_probs_path)
    return torch.tensor(conditional_probs).float()


def load_proteinmpnn_proteingym(wt: pd.Series):
    conditional_probs_path = Path(
        "data",
        "conditional_probs",
        wt["UniProt_ID"].item(),
        "proteinmpnn",
        "conditional_probs_only",
        f"{wt['UniProt_ID'].item()}.npz",
    )
    proteinmpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    proteinmpnn_tok_to_aa = {i: aa for i, aa in enumerate(proteinmpnn_alphabet)}

    raw_file = np.load(conditional_probs_path)
    log_p = raw_file["log_p"]
    wt_toks = raw_file["S"]

    # Load sequence from ProteinMPNN outputs
    wt_seq_from_toks = "".join([proteinmpnn_tok_to_aa[tok] for tok in wt_toks])
    assert wt_seq_from_toks == wt["target_seq"].item()

    # Process logits
    log_p_mean = log_p.mean(axis=0)
    p_mean = np.exp(log_p_mean)
    p_mean = p_mean[:, :20]  # "X" is included as 21st AA in ProteinMPNN alphabet

    p_mean = torch.tensor(p_mean).float()
    return p_mean


def load_zero_shot(dataset: str, zero_shot_method: str):
    zero_shot_dir = (
        Path("results/ProteinGym_baselines/zero_shot_substitution_scores")
        / zero_shot_method
    )
    zero_shot_col = zero_shot_name_to_col(zero_shot_method)

    if zero_shot_method == "TranceptEVE":
        zero_shot_dir = zero_shot_dir / "TranceptEVE_L"

    try:
        df_zero = pd.read_csv(zero_shot_dir / f"{dataset}.csv")
    except FileNotFoundError:
        if (
            "Rocklin" in dataset
        ):  # Edit was made to raw ProteinGym datafiles after download
            dataset_alt = dataset.replace("Rocklin", "Tsuboyama")
            df_zero = pd.read_csv(zero_shot_dir / f"{dataset_alt}.csv")
        else:
            raise FileNotFoundError

    return df_zero[["mutant", zero_shot_col]]


def zero_shot_name_to_col(key):
    return {
        "ProteinMPNN": "pmpnn_ll",
        "ESM_IF1": "esmif1_ll",
        "EVE": "evol_indices_ensemble",
        "TranceptEVE": "avg_score",
    }[key]


def load_embeddings(dataset: str, df: pd.DataFrame) -> torch.Tensor:
    emb_path = (
        Path("data/embeddings/substitutions_multiples/MSA_Transformer")
        / f"{dataset}.h5"
    )
    with h5py.File(emb_path, "r") as h5f:
        embeddings = torch.tensor(h5f["embeddings"][:]).float()
        mutants = [x.decode("utf-8") for x in h5f["mutants"][:]]
    embeddings = embeddings.mean(dim=1)
    # Keep entries that are in the dataset
    keep = [x in df["mutant"].tolist() for x in mutants]
    embeddings = embeddings[keep]
    mutants = np.array(mutants)[keep]
    # Ensures matching indices
    idx = [df["mutant"].tolist().index(x) for x in mutants]
    embeddings = embeddings[idx]
    return embeddings


def prepare_kwargs(wt_df: pd.DataFrame, cfg: DictConfig):
    # Preprocess data if necessary

    kwargs = {"use_zero_shot": cfg.gp.use_zero_shot}
    if cfg.gp.use_mutation_kernel:
        tokenizer = hydra.utils.instantiate(cfg.gp.mutation_kernel.tokenizer)
        wt_sequence = wt_df["target_seq"].item()
        wt_sequence = tokenizer(wt_sequence).squeeze()
        if cfg.gp.mutation_kernel.conditional_probs_method == "ProteinMPNN":
            conditional_probs = load_proteinmpnn_proteingym(wt_df)
        elif cfg.gp.mutation_kernel.conditional_probs_method == "ESM_IF1":
            conditional_probs = load_esmif1_proteingym(wt_df)
        else:
            raise NotImplementedError

        kwargs["wt_sequence"] = wt_sequence
        kwargs["conditional_probs"] = conditional_probs
        kwargs["km_cfg"] = cfg.gp.mutation_kernel
        kwargs["use_global_kernel"] = cfg.gp.use_global_kernel

        if cfg.gp.mutation_kernel.use_distances:
            coords = np.load(f"data/interim/coords/{wt_df['UniProt_ID'].item()}.npy")
            kwargs["coords"] = torch.tensor(coords)
    # elif cfg.use_sequences:  # TODO: FIX FOR ONEHOTs
    # tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer)
    else:
        tokenizer = None
    return kwargs, tokenizer


def get_model_name(cfg: DictConfig) -> str:
    if "custom_name" in cfg:
        model_name = cfg.custom_name
    # else:
        # model_name = (
            # f"kermut_{cfg.gp.conditional_probs_method}"  # e.g. kermut_ProteinMPNN
        # )
        # if use_zero_shot:
            # model_name = f"{model_name}_{zero_shot_method}"
    else:
        model_name = cfg.gp.name

    return model_name


def load_proteingym_dataset(dataset: str, multiples: bool = False) -> pd.DataFrame:
    if multiples:
        base_path = Path("data/processed/proteingym_cv_folds_multiples_substitutions")
    else:
        base_path = Path("data/processed/proteingym_cv_folds_singles_substitutions")
    df = pd.read_csv(base_path / f"{dataset}.csv")
    df["n_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
    return df.reset_index(drop=True)


def get_wt_df(dataset: str) -> pd.DataFrame:
    ref_path = Path("data/processed/DMS_substitutions.csv")
    df_ref = pd.read_csv(ref_path)
    return df_ref.loc[df_ref["DMS_id"] == dataset]
