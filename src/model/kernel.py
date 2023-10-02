from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import AA_TO_IDX, COLORS
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.utils import (
    get_jenson_shannon_div,
    get_euclidean_distance,
    get_probabilities,
    get_substitution_matrix,
    get_fitness_matrix,
)

if __name__ == "__main__":
    ####################
    # Load data
    ####################

    threshold_by_distance = False
    absolute_deltas = True

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
    df_assay["wt"] = df_assay["mut2wt"].str[0]
    df_assay["aa"] = df_assay["mut2wt"].str[-1]

    ####################
    # Compute quantities
    ####################

    # Sequence and AA indices
    indices = df_assay["pos"].values - 1
    aa_indices = df_assay["aa"].apply(lambda x: AA_TO_IDX[x]).values

    # Compute components of the kernel
    js_divergence = get_jenson_shannon_div(p_mean, indices)
    p_x_component, p_y_component = get_probabilities(p_mean, indices, aa_indices)
    substitution_component = get_substitution_matrix(aa_indices)
    euclidean_matrix = get_euclidean_distance(dataset, indices)
    # Process target values
    fitness = get_fitness_matrix(df_assay, absolute=absolute_deltas)

    ####################
    # Create kernel
    ####################

    DIV_COMP = js_divergence.max() - js_divergence
    SUB_COMP = substitution_component
    P_X_COMP = p_x_component
    P_Y_COMP = p_y_component
    DIST_COMP = euclidean_matrix

    distance_matrix = 1 - (DIV_COMP * P_X_COMP * P_Y_COMP)

    ####################
    # Show distance matrix
    ####################

    sns.set_style("darkgrid")
    print("Generating distance matrix plot...")
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        distance_matrix, ax=ax, cmap="flare", square=True, cbar_kws={"shrink": 0.8}
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Distance matrix")
    plt.tight_layout()
    plt.savefig("figures/distance_matrix.png")
    plt.show()

    ####################
    # Show components
    ####################

    print("Generating components plot...")
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    sns.heatmap(
        data=DIV_COMP,
        ax=ax[0, 0],
        cmap="flare",
        square=True,
    )
    sns.heatmap(
        data=P_X_COMP,
        ax=ax[0, 1],
        cmap="flare",
        square=True,
    )
    sns.heatmap(
        data=P_Y_COMP,
        ax=ax[1, 0],
        cmap="flare",
        square=True,
    )
    sns.heatmap(
        data=SUB_COMP,
        ax=ax[1, 1],
        cmap="flare",
        square=True,
    )
    ax[0, 0].set_title("Divergence component")
    ax[0, 1].set_title("p(x) component")
    ax[1, 0].set_title("p(y) component")
    ax[1, 1].set_title("Substitution matrix component")

    for i in range(2):
        for j in range(2):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.savefig("figures/components_heatmap.png")
    plt.show()

    ####################
    # Show scatter plot
    ####################

    # Extract lower triangle only
    tril_mask = np.tril_indices_from(distance_matrix)
    # Keep only pairs with distance < 5
    distance_mask = (euclidean_matrix < 5)[tril_mask]
    # Actually use all pairs [OPTIONAL]
    if not threshold_by_distance:
        distance_mask = np.ones_like(distance_mask, dtype=bool)

    # Apply masks
    masked_distance_matrix = distance_matrix[tril_mask][distance_mask]
    masked_fitness = fitness[tril_mask][distance_mask]

    # Subsample
    idx = np.random.choice(
        np.arange(masked_fitness.size),
        size=min(200000, masked_fitness.size),
        replace=False,
    )

    print("Generating fitness vs distance scatter plot...")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        x=masked_distance_matrix.flatten()[idx],
        y=masked_fitness.flatten()[idx],
        ax=ax,
        color=COLORS[4],
        alpha=0.5,
        linewidth=0,
    )
    ax.set_xlabel("Distance")
    ax.set_ylabel("abs(y-y')")
    if threshold_by_distance:
        plt.title(
            f"Fitness vs distance (all pairs < 5). N = {min(200000, masked_fitness.size)}"
        )
        plt.tight_layout()
        plt.savefig("figures/fitness_vs_distance_scatter_sample_distance_threshold.png")

    else:
        plt.title(
            f"Fitness vs distance (all pairs). N = {min(200000, masked_fitness.size)}"
        )
        plt.tight_layout()
        plt.savefig("figures/fitness_vs_distance_scatter_sample_all.png")

    plt.show()
