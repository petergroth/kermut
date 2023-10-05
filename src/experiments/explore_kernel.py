from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src import AA_TO_IDX, COLORS
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.utils import (
    get_fitness_matrix,
)
from src.model.kernel import KermutKernel, KermutRBFKernel

if __name__ == "__main__":
    sample = False
    heatmap = True

    # Define paths
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
    df_assay["aa"] = df_assay["mut2wt"].str[-1]

    # Sequence and AA indices
    indices = df_assay["pos"].values - 1
    aa_indices = df_assay["aa"].apply(lambda x: AA_TO_IDX[x]).values

    # Subsample
    if sample:
        n = 1000
        np.random.seed(42)
        sample_idx_i = np.random.choice(np.arange(len(indices)), size=n, replace=False)
        sample_idx_j = np.random.choice(np.arange(len(indices)), size=n, replace=False)
    else:
        # Use all data
        sample_idx_i = np.arange(len(indices))
        sample_idx_j = np.arange(len(indices))

    i_idx = indices[sample_idx_i]
    j_idx = indices[sample_idx_j]
    i_aa_idx = aa_indices[sample_idx_i]
    j_aa_idx = aa_indices[sample_idx_j]
    x_i = p_mean[i_idx]
    x_j = p_mean[j_idx]

    # To tensors
    x_i = torch.tensor(x_i)
    x_j = torch.tensor(x_j)
    i_aa_idx = torch.tensor(i_aa_idx, dtype=torch.long)
    j_aa_idx = torch.tensor(j_aa_idx, dtype=torch.long)

    # Create kernel

    # js_exponent = 5.3942
    # p_exponent = -0.4051
    # kernel_params = {"js_exponent": js_exponent, "p_exponent": p_exponent}
    # kernel = KermutKernel(**kernel_params)

    kernel = KermutRBFKernel()

    kernel_samples = (
        kernel(x_i, x_j, **{"idx_1": i_aa_idx, "idx_2": j_aa_idx}).detach().numpy()
    )

    if heatmap:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            kernel_samples, ax=ax, cmap="flare", square=True, cbar_kws={"shrink": 0.8}
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Pairwise kernel values")
        plt.tight_layout()
        # plt.savefig()
        plt.show()

    # Visualize kernel against fitness
    y = get_fitness_matrix(df_assay, absolute=True)
    y_ij = y[sample_idx_i][:, sample_idx_j]

    # If full dataset, avoid diagonal and double counting
    if y.shape == y_ij.shape:
        tril_mask = np.tril_indices_from(y_ij, k=-1)
        y_ij = y_ij[tril_mask]
        kernel_samples = kernel_samples[tril_mask]

    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        x=kernel_samples.flatten(), y=y_ij.flatten(), color=COLORS[4], ax=ax
    )
    ax.set_ylabel("abs(y-y')")
    ax.set_xlabel("k(x, x')")
    plt.title(f"Kernel vs delta fitness (using Jensen-Shannon divergence)")
    # ax.set_xlim([0 - 0.05, 1 + 0.05])
    # ax.set_ylim([y_ij.min() - 0.25, y_ij.max() + 0.25])
    plt.tight_layout()
    # plt.savefig("figures/fitness_vs_kernel_scatter_optimized.png")
    plt.show()
