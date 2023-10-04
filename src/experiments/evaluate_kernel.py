from pathlib import Path

import gpytorch
import pandas as pd
import torch

from src import AA_TO_IDX
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.gp import ExactGPModelKermut

if __name__ == "__main__":
    dataset = "BLAT_ECOLX"
    conditional_probs_path = Path(
        "data",
        "interim",
        dataset,
        "proteinmpnn",
        "conditional_probs_only",
        f"{dataset}.npz",
    )
    assay_path = Path("data", "processed", f"{dataset}.tsv")

    # Load data
    p_mean = load_protein_mpnn_outputs(conditional_probs_path)  # Shape (n_pos, 20)
    df_assay = pd.read_csv(assay_path, sep="\t")
    df_assay["wt"] = df_assay["mut2wt"].str[0]
    df_assay["aa"] = df_assay["mut2wt"].str[-1]

    # Sequence and AA indices
    indices = df_assay["pos"].values - 1
    aa_indices = df_assay["aa"].apply(lambda x: AA_TO_IDX[x]).values

    i_aa_idx = aa_indices
    x = p_mean[indices]
    y = df_assay["delta_fitness"].values

    # To tensors
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    i_aa_idx = torch.tensor(i_aa_idx, dtype=torch.long)

    # Setup model and training parameters
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModelKermut(x, y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    output = model(x, **{"x1_idx": i_aa_idx, "x2_idx": i_aa_idx})
    loss = -mll(output, y)
    print(f"Loss: {loss.item():.3f}")
