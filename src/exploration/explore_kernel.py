from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src import AA_TO_IDX, COLORS
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.utils import get_fitness_matrix, get_mutation_pair_matrix
from src.model.kernel import (
    KermutJSKernel,
    KermutJSD_RBFKernel,
    KermutHellingerKernel,
    KermutHellingerKernelMulti,
)
from src.model.utils import Tokenizer
from src import GFP_WT, BLAT_ECOLX_WT

if __name__ == "__main__":
    # Setup parameters
    dataset = "GFP"
    max_mutations = 2
    n_samples = "all"
    seed = 42
    how = "prod"

    if dataset == "GFP":
        wt_sequence = GFP_WT
    else:
        wt_sequence = BLAT_ECOLX_WT

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
    conditional_probs = load_protein_mpnn_outputs(
        conditional_probs_path,
        as_tensor=True,
        drop_index=[0] if dataset == "GFP" else None,
    )

    # Filter data
    wt_df = pd.read_csv("data/processed/wt_sequences.tsv", sep="\t")
    wt_sequence = wt_df.loc[wt_df["dataset"] == dataset, "seq"].item()
    df = pd.read_csv(assay_path, sep="\t")
    df = df[df["n_muts"] <= max_mutations]
    if n_samples != "all":
        # Sample from df
        np.random.seed(seed)
        df = df.sample(n=n_samples)

    # PREPARE MODEL
    tokenizer = Tokenizer()
    sequences = df["seq"].tolist()
    tokens = tokenizer(sequences)
    wt_sequence = tokenizer([wt_sequence])[0]
    kernel_kwargs = {
        "wt_sequence": wt_sequence,
        "conditional_probs": conditional_probs,
        "p_B": 15.0,
        "p_Q": 5.0,
        "theta": 1.0,
        "gamma": 1.0,
        "learnable_transform": False,
        "learnable_hellinger": False,
    }

    kernel = KermutHellingerKernelMulti(**kernel_kwargs)
    kernel_samples = kernel(tokens).detach().numpy()

    y = get_fitness_matrix(df, how=how)
    tril_mask = np.tril_indices_from(y, k=-1)
    y = y[tril_mask]
    kernel_samples = kernel_samples[tril_mask]

    # Collect in dataframe
    mutation_pairs = get_mutation_pair_matrix(df)[tril_mask]
    df_kernel = pd.DataFrame(
        {
            "kernel": kernel_samples.flatten(),
            "y_how": y.flatten(),
            "mutation_pair": mutation_pairs.flatten(),
        }
    )
    df_kernel.to_csv(f"data/interim/{dataset}/kernel_dataframe.csv", index=False)

    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(x=kernel_samples.flatten(), y=y.flatten(), color=COLORS[4], ax=ax)
    ax.set_ylabel(f"{how}")
    ax.set_xlabel("k(x, x')")
    plt.title(f"Kernel entries vs fitness for {dataset}")
    plt.tight_layout()
    # plt.savefig(f"figures/{dataset}_fitness_vs_kernel_scatter_Hellinger.png")
    plt.show()
