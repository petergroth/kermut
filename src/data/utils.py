import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

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
