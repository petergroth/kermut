import ast
from pathlib import Path
from typing import Tuple

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

    # Save as torch tensor
    # blosum_matrix = torch.tensor(substitution_matrix)
    # torch.save(blosum_matrix.clone(), output_path)
    # blosum_matrix_loaded = torch.load(output_path)
    # assert torch.allclose(blosum_matrix, blosum_matrix_loaded)
    # Save substitution matrix
    np.save(output_path, substitution_matrix)


def load_split_regression_data(
    cfg: DictConfig, i: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Subsamples n_samples data points"""
    dataset = cfg.dataset
    assay_path = Path("data/processed", f"{dataset}.tsv")
    # Filter data
    df = pd.read_csv(assay_path, sep="\t")
    df = df.sample(
        n=min(cfg.n_total, len(df)),
        random_state=cfg.sample_seed,
    )

    if dataset == "BLAT_ECOLX":
        if cfg.split == "pos":
            all_positions = df["pos"].unique()
            n_train_pos = int(cfg.n_train / len(df) * len(all_positions))
            np.random.seed(cfg.seeds[i])
            train_positions = np.random.choice(
                all_positions, size=n_train_pos, replace=False
            )
            df_train = df[df["pos"].isin(train_positions)].reset_index(drop=True)
            df_test = df[~df["pos"].isin(train_positions)].reset_index(drop=True)
        elif cfg.split == "aa_mixed":
            # Similar AAs in different splits
            aa_train = ["C", "S", "T", "D", "E", "H", "M", "I", "W"]
            aa_test = ["A", "G", "P", "Q", "N", "R", "K", "L", "V", "Y", "F"]
            df_train = df[df["aa"].isin(aa_train)].reset_index(drop=True)
            df_test = df[df["aa"].isin(aa_test)].reset_index(drop=True)
            df_train = df_train.sample(n=cfg.n_train, random_state=cfg.seeds[i])
        elif cfg.split == "aa_diff":
            # Similar AAs in same splits. Challenging.
            aa_train = ["C", "S", "T", "A", "G", "P", "D", "E", "Q", "N"]
            aa_test = ["H", "R", "K", "M", "I", "L", "V", "W", "U", "F", "Y"]
            df_train = df[df["aa"].isin(aa_train)].reset_index(drop=True)
            df_test = df[df["aa"].isin(aa_test)].reset_index(drop=True)
            df_train = df_train.sample(
                n=min(cfg.n_train, len(df_train)), random_state=cfg.seeds[i]
            )
        else:
            raise ValueError(f"Unknown split: {cfg.split}")

    elif dataset == "SPG1":
        df_train = df[df["n_muts"] == 1]
        df_train = df_train.sample(
            n=min(cfg.n_train, len(df_train)), random_state=cfg.seeds[i]
        ).reset_index(drop=True)

        df_test = df[df["n_muts"] == 2].reset_index(drop=True)

    elif dataset == "PARD3_10":
        df_train = df[df["n_muts"] <= 3]
        df_train = df_train.sample(
            n=min(cfg.n_train, len(df_train)), random_state=cfg.seeds[i]
        ).reset_index(drop=True)

        df_test = df[df["n_muts"] > 3].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown split: {cfg.split}")

    return df_train, df_test


def load_sampled_regression_data(cfg: DictConfig) -> pd.DataFrame:
    """Subsamples n_samples data points"""
    dataset = cfg.experiment.dataset
    assay_path = Path("data/processed", f"{dataset}.tsv")
    # Filter data
    df = pd.read_csv(assay_path, sep="\t")
    if cfg.experiment.filter_mutations:
        df = df[df["n_muts"] <= cfg.experiment.max_mutations]
    df = df.sample(
        n=min(cfg.experiment.n_total, len(df)),
        random_state=cfg.experiment.sample_seed,
    )
    df = df.reset_index(drop=True)
    return df


def load_regression_data(cfg: DictConfig) -> pd.DataFrame:
    """Subsamples n_samples data points"""
    dataset = cfg.experiment.dataset
    assay_path = Path("data/processed", f"{dataset}.tsv")
    # Filter data
    df = pd.read_csv(assay_path, sep="\t")
    return df


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


if __name__ == "__main__":
    process_substitution_matrices()
