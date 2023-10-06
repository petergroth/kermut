"""Utility functions for data processing and kernel computation."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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


def js_divergence_pairwise(p: torch.tensor, q: torch.tensor):
    """Compute pairwise Jensen-Shannon divergence between two distributions.

    Args:
        p (torch.tensor): Shape (n, n_classes)
        q (torch.tensor): Shape (n, n_classes)

    Returns:
        torch.tensor: Shape (n, 1)
    """
    assert p.shape == q.shape
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def js_divergence(p: torch.tensor, q: torch.tensor):
    """Compute Jensen-Shannon divergence between all possible pairs of inputs. Generates symmetric matrix.

    Args:
        p (torch.tensor): Shape (n, n_classes)
        q (torch.tensor): Shape (n, n_classes)

    Returns:
        torch.tensor: Shape (n, n)

    """
    batch_size = p.shape[0]
    # Compute only the lower triangular elements if p == q
    if torch.allclose(p, q):
        tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=-1)
        m = 0.5 * (p[tril_i] + q[tril_j])
        kl_p_m = kl_divergence(p[tril_i], m)
        kl_q_m = kl_divergence(q[tril_j], m)
        js_tril = 0.5 * (kl_p_m + kl_q_m)
        # Build full matrix
        out = torch.zeros((batch_size, batch_size))
        out[tril_i, tril_j] = js_tril.squeeze()
        out[tril_j, tril_i] = js_tril.squeeze()
    else:
        mesh_i, mesh_j = torch.meshgrid(
            torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
        )
        mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
        m = 0.5 * (p[mesh_i] + q[mesh_j])
        kl_p_m = kl_divergence(p[mesh_i], m)
        kl_q_m = kl_divergence(q[mesh_j], m)
        out = 0.5 * (kl_p_m + kl_q_m)
        out = out.reshape(batch_size, batch_size)
    return out


def kl_divergence(p: torch.tensor, q: torch.tensor):
    """Compute KL divergence between two distributions.

    Args:
        p (torch.tensor): Shape (n, n_classes)
        q (torch.tensor): Shape (n, n_classes)

    Returns:
        torch.tensor: Shape (n, 1)
    """
    assert p.shape == q.shape
    return torch.sum(p * (p / q).log(), dim=1, keepdim=True)


def hellinger_distance(p: torch.tensor, q: torch.tensor):
    """Compute Hellinger distance between input distributions:

    HD(p, q) = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))

    Args:
        x1 (torch.Tensor): Shape (n, 20)
        x2 (torch.Tensor): Shape (n, 20)

    Returns:
        torch.Tensor: Shape (n, n)
    """
    batch_size = p.shape[0]
    # Compute only the lower triangular elements if p == q
    if torch.allclose(p, q):
        tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=-1)
        hellinger_tril = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[tril_i]) - torch.sqrt(q[tril_j])) ** 2, dim=1)
        )
        hellinger_matrix = torch.zeros((batch_size, batch_size))
        hellinger_matrix[tril_i, tril_j] = hellinger_tril
        hellinger_matrix[tril_j, tril_i] = hellinger_tril
    else:
        mesh_i, mesh_j = torch.meshgrid(
            torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
        )
        mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
        hellinger = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[mesh_i]) - torch.sqrt(q[mesh_j])) ** 2, dim=1)
        )
        hellinger_matrix = hellinger.reshape(batch_size, batch_size)
    return hellinger_matrix


def get_px1x2(
        x1: torch.Tensor,
        x2: torch.Tensor,
        **kwargs,
):
    batch_size = x1.shape[0]
    # Compute only the lower triangular elements if x1 == x2.
    if torch.allclose(x1, x2):
        tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=0)
        p_x1_tril = x1[tril_i]
        p_x2_tril = x2[tril_j]
        p_x1 = p_x1_tril[torch.arange(tril_i.numel()), kwargs["idx_1"][tril_i]]
        p_x2 = p_x2_tril[torch.arange(tril_j.numel()), kwargs["idx_1"][tril_j]]
        # Build full matrix
        p_x1x2 = torch.zeros((batch_size, batch_size))
        p_x1x2[tril_i, tril_j] = p_x1 * p_x2
        p_x1x2[tril_j, tril_i] = p_x1 * p_x2
    else:
        mesh_i, mesh_j = torch.meshgrid(
            torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
        )
        mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
        p_x1 = x1[mesh_i][torch.arange(mesh_i.numel()), kwargs["idx_1"]][mesh_i]
        p_x2 = x2[mesh_j][torch.arange(mesh_j.numel()), kwargs["idx_2"]][mesh_j]
        p_x1x2 = p_x1 * p_x2
        p_x1x2 = p_x1x2.reshape(batch_size, batch_size)
    return p_x1x2
