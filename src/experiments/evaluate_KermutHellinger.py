from pathlib import Path

import gpytorch
import pandas as pd
import torch

from src import AA_TO_IDX
from src.model.utils import load_protein_mpnn_outputs
from src.model.gp import train_gp, ExactGPModelKermut

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
    p_mean = load_protein_mpnn_outputs(conditional_probs_path)
    df_assay = pd.read_csv(assay_path, sep="\t")

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
    kernel_params = {"learnable_hellinger": True}
    model = ExactGPModelKermut(x, y, likelihood, **kernel_params)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    kwargs = {"idx_1": i_aa_idx}

    output = model(x, **kwargs)
    loss = -mll(output, y)
    print(f"Loss: {loss.item():.3f}")

    training_iter = 100

    # Train model
    model_fitted = train_gp(
        model=model, likelihood=likelihood, x=x, y=y, max_iter=training_iter, **kwargs
    )
