"""Utility functions for data processing and kernel computation."""
import pickle
from pathlib import Path

import numpy as np
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


def compute_jenson_shannon_div(p: np.array):
    """Compute the Jenson-Shannon divergence between all pairs in p

    Args:
        p (np.array): Array of probabilities. Shape (n, m) where n is the number of samples and m is the number of
        classes.

    Returns:
        np.array: Jenson-Shannon divergence between all pairs in p. Shape (n, n).
    """

    js_div = np.zeros((p.shape[0], p.shape[0]))
    for i in range(p.shape[0]):
        for j in range(p.shape[0]):
            js_div[i, j] = 1 / 2 * np.sum(
                p[i] * (np.log(p[i]) - np.log(p[[i, j]].mean(axis=0)))
            ) + 1 / 2 * np.sum(p[j] * (np.log(p[j]) - np.log(p[[i, j]].mean(axis=0))))
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
    substitution_matrix = substitution_matrix[alphabetical_indexing][
        :, alphabetical_indexing
    ]
    return substitution_matrix


def compute_euclidean_distance(dataset: str):
    """Compute the Euclidean distance between all pairs of amino acids in the protein.

    Args:
        dataset (str): Name of the dataset. Used to load coordinates from the PDB file.

    Returns:
        np.array: Euclidean distance between all pairs of amino acids in the protein.
    """
    coords = get_coords_from_pdb(dataset, only_ca=True)
    euclidean_matrix = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(coords.shape[0]):
        for j in range(coords.shape[0]):
            euclidean_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return euclidean_matrix
