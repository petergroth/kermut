import ast
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src import COLORS
from src.model.utils import get_fitness_matrix
from src.model.gp import ExactGPModelRBF, train_gp

sns.set_style("dark")


def evaluate_RBF():
    sns.set_style("dark")
    ####################
    # Load data
    ####################

    # Define paths
    dataset = "BLAT_ECOLX"
    EVE_path = Path("data", "interim", dataset, "BLAT_ECOLX_EVE_samples.csv")
    assay_path = Path("data", "processed", f"{dataset}.tsv")

    # Load data and merge data
    df_assay = pd.read_csv(assay_path, sep="\t")
    df_eve = pd.read_csv(EVE_path, sep=",")
    df_assay = pd.merge(
        left=df_assay,
        right=df_eve[["mutations", "mean_encoder"]],
        left_on="mut2wt",
        right_on="mutations",
        how="inner",
    ).drop(columns=["mutations"])

    # Extract latent EVE representations
    z = np.array(df_assay["mean_encoder"].apply(ast.literal_eval).tolist())
    y = df_assay["delta_fitness"].values

    # Convert to tensors
    z = torch.tensor(z)
    y = torch.tensor(y)

    # Setup model and training parameters
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModelRBF(z, y, likelihood)
    training_iter = 200

    # Train model
    model = train_gp(
        model=model, likelihood=likelihood, x=z, y=y, max_iter=training_iter
    )

    # Evaluate covariance module between all datapoints
    with torch.no_grad():
        kernel_matrix = model.covar_module(z).detach().numpy()

    # Visualize kernel matrix
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(kernel_matrix, ax=ax, square=True)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    y_matrix = get_fitness_matrix(df_assay, absolute=True)

    # # Extract lower triangle only
    tril_mask = np.tril_indices_from(kernel_matrix)

    # Apply masks
    masked_distance_matrix = kernel_matrix[tril_mask]
    masked_fitness = y_matrix[tril_mask]

    # Subsample
    idx = np.random.choice(
        np.arange(masked_fitness.size),
        size=min(200000, masked_fitness.size),
        replace=False,
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        x=masked_distance_matrix.flatten()[idx],
        y=masked_fitness.flatten()[idx],
        ax=ax,
        color=COLORS[4],
        alpha=0.5,
        linewidth=0,
    )
    plt.title(
        f"Optimized RBF kernel between all pairs. Length scale = {model.covar_module.lengthscale.item():.3f}"
    )
    ax.set_ylabel("abs(y-y')")
    ax.set_xlabel(f"RBF(x, x')")
    plt.tight_layout()
    plt.savefig("figures/fitness_vs_kernel_EVE_RBF_optimized.png")

    plt.show()


if __name__ == "__main__":
    evaluate_RBF()
