import ast
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from src import AA_TO_IDX
from src.experiments.investigate_correlations import load_protein_mpnn_outputs


def process_substitution_matrices():
    # Based on mGPfusion by Jokinen et al. (2018)
    output_path = Path("data", "interim", "substitution_matrices.pkl")
    matrix_path = Path("data", "raw", "subMats.mat")
    matrix_file = loadmat(str(matrix_path))["subMats"]
    names = [name.item() for name in matrix_file[:, 1]]
    descriptions = [description.item() for description in matrix_file[:, 2]]

    full_matrix = np.zeros((21, 20, 20))
    for i in range(21):
        full_matrix[i] = matrix_file[i, 0]

    substitution_dict = {name: matrix for name, matrix in zip(names, full_matrix)}
    # Save
    with open(output_path, "wb") as f:
        pickle.dump(substitution_dict, f)


def load_conditional_probs(dataset: str, method: str = "ProteinMPNN"):
    if method == "ProteinMPNN":
        conditional_probs_path = Path(
            "data",
            "interim",
            dataset,
            "proteinmpnn",
            "conditional_probs_only",
            f"{dataset}.npz",
        )
        if dataset == "GFP":
            drop_index = [0]
        else:
            drop_index = None
        conditional_probs = load_protein_mpnn_outputs(
            conditional_probs_path, as_tensor=True, drop_index=drop_index
        )
    elif method == "esm2":
        conditional_probs_path = Path(
            "data", "interim", dataset, "esm2_masked_probs.pt"
        )
        conditional_probs = torch.load(conditional_probs_path)
    else:
        raise ValueError(f"Unknown method: {method}")

    return conditional_probs


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
    if cfg.experiment.filter_mutations:
        df = df[df["n_muts"] <= cfg.experiment.max_mutations]
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
