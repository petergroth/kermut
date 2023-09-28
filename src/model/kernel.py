from pathlib import Path

import pandas as pd
import numpy as np
import blosum as bl
from src import AA_TO_IDX
from src.experiments.investigate_correlations import load_protein_mpnn_outputs

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


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

    p_mean = load_protein_mpnn_outputs(conditional_probs_path)

    df_assay = pd.read_csv(assay_path, sep="\t")
    df_assay["wt"] = df_assay["mut2wt"].str[0]
    df_assay["aa"] = df_assay["mut2wt"].str[-1]

    # Inverse probabilities
    p_inv = 1 / p_mean

    # Compute KL between all pairs
    kl_divergence = np.zeros((p_mean.shape[0], p_mean.shape[0]))
    for i in range(p_mean.shape[0]):
        for j in range(p_mean.shape[0]):
            kl_divergence[i, j] = np.sum(p_mean[i] * np.log(p_mean[i] / p_mean[j]))

    # Unpack BLOSUM matrix
    BLOSUM = bl.BLOSUM(62)
    BLOSUM_TRUNCATED = np.zeros((len(AA_TO_IDX), len(AA_TO_IDX)))
    for i, aa_i in enumerate(AA_TO_IDX):
        for j, aa_j in enumerate(AA_TO_IDX):
            BLOSUM_TRUNCATED[i, j] = BLOSUM[aa_i][aa_j]

    # Get zero-index position for all assays
    indices = df_assay["pos"].values - 1
    # Get AA index for all assays
    aa_indices = df_assay["aa"].apply(lambda x: AA_TO_IDX[x]).values
    # Extract KL
    kl_component = kl_divergence[indices][:, indices]
    # Extract inverse probabilities
    p_component = p_mean[indices, aa_indices]
    p_x_component = np.repeat(p_component[:, np.newaxis], p_component.shape[0], axis=1)
    p_y_component = p_x_component.T
    # Extract BLOSUM scores
    blossum_component = BLOSUM_TRUNCATED[aa_indices][:, aa_indices]

    distance_matrix = (
        kl_component
        + 1 / (p_x_component)
        + 1 / (p_y_component)
        + 1 / (blossum_component + 1e-4)
    )

    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        distance_matrix, ax=ax, cmap="flare", square=True, cbar_kws={"shrink": 0.5}
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("figures/distance_matrix.pdf")
    plt.show()

    fitness = df_assay["delta_fitness"].values
    fitness_component = np.repeat(fitness[:, np.newaxis], fitness.shape[0], axis=1)
    fitness = abs(fitness_component - fitness_component.T)

    # Draw 10000 samples
    idx = np.random.choice(np.arange(len(fitness.flatten())), size=10000, replace=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=distance_matrix.flatten()[idx], y=fitness.flatten()[idx], ax=ax)
    ax.set_xlabel("Distance")
    ax.set_ylabel("|y-y'|")
    plt.tight_layout()
    plt.savefig("figures/fitness_vs_distance_scatter_sample.pdf")
    plt.show()

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.heatmap(
        data=kl_component,
        ax=ax[0, 0],
        cmap="flare",
        square=True,
    )
    sns.heatmap(
        data=1 / p_x_component,
        ax=ax[0, 1],
        cmap="flare",
        square=True,
    )
    sns.heatmap(
        data=1 / p_y_component,
        ax=ax[1, 0],
        cmap="flare",
        square=True,
    )
    sns.heatmap(
        data=1 / blossum_component,
        ax=ax[1, 1],
        cmap="flare",
        square=True,
    )
    ax[0, 0].set_title("KL")
    ax[0, 1].set_title("1/p(x)")
    ax[1, 0].set_title("1/p(y)")
    ax[1, 1].set_title("1/BLOSUM")

    # Remove ticks
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig("figures/components_heatmap.pdf")
    plt.show()
