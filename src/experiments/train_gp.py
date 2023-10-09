from pathlib import Path

import gpytorch
import pandas as pd
import torch
import wandb
import torch.nn.functional as F
from tqdm import tqdm

from src import AA_TO_IDX
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.gp import ExactGPModelKermutHellinger

if __name__ == "__main__":
    # LOAD DATA
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

    # Prepare inputs
    indices = df_assay["pos"].values - 1
    aa_indices = df_assay["aa"].apply(lambda x: AA_TO_IDX[x]).values

    i_aa_idx = aa_indices
    x = p_mean[indices]
    y = df_assay["delta_fitness"].values
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    i_aa_idx = torch.tensor(i_aa_idx, dtype=torch.long)

    # Setup model and training parameters
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel_params = {
        "learnable_hellinger": True,
        "p_B": 1.0,
        "p_Q": 1.0,
        "learnable_transform": True,
        "theta": 1.0,
        "gamma": 1.0,
    }

    seed = 42
    torch.manual_seed(seed)

    model = ExactGPModelKermutHellinger(x, y, likelihood, **kernel_params)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    kwargs = {"idx_1": i_aa_idx}

    training_iter = 150

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    run = wandb.init(
        # Set the project where this run will be logged
        project="kermut",
        # Track hyperparameters and run metadata
        config={
            **kernel_params,
            "training_iter": training_iter,
            "dataset": dataset,
            "seed": seed,
        },
    )

    for i in tqdm(range(training_iter)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x, **kwargs)
        # Calc loss and backprop gradients
        loss = -mll(output, y)

        # Log loss
        wandb.log({"negative_marginal_ll": loss.item()})
        # Log params
        wandb.log(
            {
                "theta": model.covar_module.hellinger_fn(
                    model.covar_module.theta
                ).item(),
                "gamma": model.covar_module.hellinger_fn(
                    model.covar_module.gamma
                ).item(),
                "p_B": model.covar_module.transform_fn(model.covar_module.p_B).item(),
                "p_Q": model.covar_module.transform_fn(model.covar_module.p_Q).item(),
                "noise": model.likelihood.noise.item(),
            }
        )
        loss.backward()
        optimizer.step()
