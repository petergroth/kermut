from pathlib import Path

from gpytorch.kernels import Kernel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src import AA_TO_IDX, COLORS
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.utils import (
    get_jenson_shannon_div,
    get_euclidean_distance,
    get_probabilities,
    get_substitution_matrix,
    get_fitness_matrix,
    apply_index,
)


def manual_kernel():
    """Manual approach of processing data and computing kernel. Ignore."""

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

    DIV_COMP = 1 - js_divergence / np.log(2)
    # SUB_COMP = substitution_component
    P_X_COMP = p_x_component
    P_Y_COMP = p_y_component
    # DIST_COMP = euclidean_matrix

    distance_matrix = DIV_COMP * P_X_COMP * P_Y_COMP


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
    """Compute Jensen-Shannon divergence between all possible pairs of inputs.

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


class KermutKernel(Kernel):
    """Custom kernel

    The kernel is the product of the following components:
    - Jensen-Shannon divergence between the conditional probabilities of x and x'
    - p(x[i]) and p(x[j]') (the probability of x and x' respectively)

    k(x, x') = (1-JS(x, x')) * p(x[i]) * p(x[j]')

    x and x' are probability distributions over the 20 amino acids.
    x[i] and x[j]' are the probabilities of the amino acids at position i and j respectively, where i and j are the
    indices of the amino acids in the variants being investigated.
    """

    def __init__(self):
        super(KermutKernel, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs):
        """Compute kernel.

        Args:
            x1 (torch.Tensor): Shape (n, 20)
            x2 (torch.Tensor): Shape (n, 20)
            **kwargs: x1_idx and x2_idx, the indices of the amino acids in the variants being investigated.

        Returns:
            torch.Tensor: Shape (n, n)
        """
        js = js_divergence(x1, x2)
        batch_size = x1.shape[0]

        # Compute only the lower triangular elements if x1 == x2. Include the diagonal for now.
        if torch.allclose(x1, x2):
            tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=0)
            p_x1_tril = x1[tril_i]
            p_x2_tril = x2[tril_j]
            # Get probability of the indexed amino acids at all (n*(n-1)/2) elements. Shape (n*(n-1)/2, 1)
            p_x1 = p_x1_tril[torch.arange(tril_i.numel()), kwargs["x1_idx"][tril_i]]
            p_x2 = p_x2_tril[torch.arange(tril_j.numel()), kwargs["x2_idx"][tril_j]]
            # Build full matrix
            out = torch.zeros((batch_size, batch_size))
            out[tril_i, tril_j] = p_x1 * p_x2
            out[tril_j, tril_i] = p_x1 * p_x2
        else:
            mesh_i, mesh_j = torch.meshgrid(
                torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
            )
            mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
            p_x1 = x1[mesh_i][torch.arange(mesh_i.numel()), kwargs["x1_idx"][mesh_i]]
            p_x2 = x2[mesh_j][torch.arange(mesh_j.numel()), kwargs["x2_idx"][mesh_j]]
            out = p_x1 * p_x2
            out = out.reshape(batch_size, batch_size)

        # Max value of JS divergence is ln(2).
        return (1 - (js / torch.log(torch.Tensor(2)))) * out


if __name__ == "__main__":
    # manual_kernel()

    # Define paths
    sample = False
    heatmap = True
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
    kernel = KermutKernel()

    kernel_samples = (
        kernel(x_i, x_j, **{"x1_idx": i_aa_idx, "x2_idx": j_aa_idx}).detach().numpy()
    )

    if heatmap:
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(
            kernel_samples, ax=ax, cmap="flare", square=True, cbar_kws={"shrink": 0.8}
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Kernel samples")
        plt.tight_layout()
        plt.savefig("figures/kernel_matrix.png")
        plt.show()

    # Visualize kernel against fitness
    y = get_fitness_matrix(df_assay, absolute=True)
    y_ij = y[sample_idx_i][:, sample_idx_j]

    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(
        x=kernel_samples.flatten(), y=y_ij.flatten(), color=COLORS[4], ax=ax
    )
    ax.set_ylabel("abs(y-y')")
    ax.set_xlabel("k(x, x')")
    plt.title(f"Kernel vs delta fitness (using Jensen-Shannon divergence)")
    plt.tight_layout()
    plt.savefig("figures/fitness_vs_kernel_scatter.png")
    plt.show()
