"""Utility functions for data processing and kernel computation."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from src import AA_TO_IDX


def get_coords_from_pdb(dataset: str, only_ca: bool = True):
    """Get the coordinates of the atoms in the protein from the PDB file.

    Args:
        dataset (str): Name of the dataset.
        only_ca (bool, optional): Whether to only use the alpha carbon atoms. Defaults to True.
    """

    pdb_path = Path("data", "raw", f"{dataset}.pdb")
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

    return coords


def get_jenson_shannon_div(p: np.array, indices: np.array = None):
    """Compute the Jenson-Shannon divergence between all pairs in p

    Args:
        p (np.array): Array of probabilities. Shape (n, m) where n is the number of samples and m is the number of
        classes.
        indices (np.array, optional): Indices to compute the Jenson-Shannon divergence for. Defaults to None.

    Returns:
        np.array: Jenson-Shannon divergence between all pairs in p. Shape (n, n).
    """

    js_div = np.zeros((p.shape[0], p.shape[0]))
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            js_div[i, j] = 1 / 2 * np.sum(
                p[i] * (np.log(p[i]) - np.log(p[[i, j]].mean(axis=0)))
            ) + 1 / 2 * np.sum(p[j] * (np.log(p[j]) - np.log(p[[i, j]].mean(axis=0))))

    if indices is not None:
        js_div = apply_index(js_div, indices)
    return js_div


def load_blosum_matrix():
    """Load the BLOSUM62 substitution matrix.

    Returns:
        np.array: BLOSUM62 substitution matrix in alphabetical order.
    """
    substitution_matrix_path = Path("data", "interim", "substitution_matrices.pkl")

    # Load substitution matrix
    substitution_matrix_name = "HENS920102"  # Corresponds to BLOSUM62
    with open(substitution_matrix_path, "rb") as f:
        substitution_matrix = pickle.load(f)[substitution_matrix_name]
    # Alphabetize
    alphabetical_indexing = [AA_TO_IDX[aa] for aa in "ARNDCQEGHILKMFPSTWYV"]

    return apply_index(substitution_matrix, alphabetical_indexing)


def get_euclidean_distance(dataset: str, indices: np.array = None):
    """Compute the Euclidean distance between all pairs of amino acids in the protein.

    Args:
        dataset (str): Name of the dataset. Used to load coordinates from the PDB file.
        indices (np.array, optional): Indices to compute the Euclidean distance for. Defaults to None.

    Returns:
        np.array: Euclidean distance between all pairs of amino acids in the protein.
    """
    coords = get_coords_from_pdb(dataset, only_ca=True)
    euclidean_matrix = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(coords.shape[0]):
        for j in range(coords.shape[0]):
            euclidean_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    if indices is not None:
        return apply_index(euclidean_matrix, indices)

    return euclidean_matrix


def get_probabilities(p: np.array, indices: np.array, aa_indices: np.array):
    """Get the probabilities for each amino acid at each position.

    Args:
        p (np.array): Array of probabilities. Shape (n, m) where n is the number of samples and m is the number of
        classes.
        indices (np.array): Indices to compute the probabilities for. Shape (n_df).
        aa_indices (np.array): Amino acid indices to compute the probabilities for. Shape (n_df).

    Returns:
        tuple: Tuple of probabilities for each amino acid at each position. Shape ((n_df, n_df), (n_df, n_df)).
    """
    p_component = p[indices, aa_indices]  # Shape (n_df)
    p_x_component = np.repeat(
        p_component[:, np.newaxis], p_component.shape[0], axis=1
    )  # Shape (n_df, n_df)
    p_y_component = p_x_component.T  # Shape (n_df, n_df)

    return p_x_component, p_y_component


def apply_index(a: np.array, idx: np.array):
    """Apply the given index to the given array.

    Args:
        a (np.array): Array to apply the index to. Shape (n, m).
        idx (np.array): Index to apply. Shape (n_df).

    Returns:
        np.array: Array with the index applied. Shape (n_df, n_df).
    """
    return a[idx][:, idx]


def get_substitution_matrix(indices: np.array):
    """Get the substitution matrix for the given indices.

    Args:
        indices (np.array): Indices to compute the substitution matrix for. Shape (n_df).

    Returns:
        np.array: Substitution matrix for the given indices. Shape (n_df, n_df).
    """

    substitution_matrix = load_blosum_matrix()
    return apply_index(substitution_matrix, indices)


def get_fitness_matrix(
    df_assay: pd.DataFrame, target_key: str = "delta_fitness", absolute: bool = True
):
    """Get the fitness matrix for the given assay dataframe.

    Args:
        df_assay (pd.DataFrame): Assay dataframe.
        target_key (str): Key of the target column in the assay dataframe.
        absolute (bool): Whether to take the absolute value of the fitness deltas.

    Returns:
        np.array: Fitness matrix for the given assay dataframe. Shape (n_df, n_df).
    """

    # Process target values
    fitness = df_assay[target_key].values
    fitness_component = np.repeat(fitness[:, np.newaxis], fitness.shape[0], axis=1)
    if absolute:
        return abs(fitness_component - fitness_component.T)
    else:
        return fitness_component - fitness_component.T
