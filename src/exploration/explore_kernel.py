from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src import AA_TO_IDX, COLORS
from src.experiments.investigate_correlations import load_protein_mpnn_outputs
from src.model.utils import get_fitness_matrix
from src.model.kernel import (
    KermutJSKernel,
    KermutJSD_RBFKernel,
    KermutHellingerKernel,
    KermutHellingerKernelMulti,
)
from src.model.utils import Tokenizer
from src import GFP_WT, BLAT_ECOLX_WT

if __name__ == "__main__":
    # PREPARE DATA
    # dataset = "BLAT_ECOLX"
    dataset = "GFP"
    np.random.seed(42)
    if dataset == "GFP":
        wt_sequence = GFP_WT
        n_samples = 2000
    else:
        wt_sequence = BLAT_ECOLX_WT
        n_samples = "all"

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
    conditional_prob = load_protein_mpnn_outputs(
        conditional_probs_path,
        as_tensor=True,
        drop_index=[0] if dataset == "GFP" else None,
    )
    df = pd.read_csv(assay_path, sep="\t")
    if dataset == "GFP":
        df = df.iloc[1:].reset_index(drop=True)

    df = df[df["n_muts"] < 3]
    if n_samples != "all":
        # Sample from df
        df = df.sample(n=n_samples)
    print(f"Number of sequences: {len(df)}")

    df = df[df["n_muts"] == 2]

    # PREPARE MODEL
    tokenizer = Tokenizer()
    sequences = df["seq"].tolist()
    tokens = tokenizer(sequences)
    wt_sequence = tokenizer([wt_sequence])[0]
    # kernel_kwargs = {"p_B": 5.06, "p_Q": 5.06, "theta": 10.52, "gamma": 1.18}
    kernel_kwargs = {
        "p_B": 15.0,
        "p_Q": 5.0,
        "theta": 1.0,
        "gamma": 1.0,
        "learnable_transform": False,
        "learnable_hellinger": False,
    }

    # COMPUTE KERNEL
    kernel = KermutHellingerKernelMulti(
        conditional_prob=conditional_prob, wt_sequence=wt_sequence, **kernel_kwargs
    )
    kernel_samples = kernel(tokens).detach().numpy()

    y = get_fitness_matrix(df, absolute=True)
    tril_mask = np.tril_indices_from(y, k=-1)
    y = y[tril_mask]
    kernel_samples = kernel_samples[tril_mask]

    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(x=kernel_samples.flatten(), y=y.flatten(), color=COLORS[4], ax=ax)
    ax.set_ylabel("abs(y-y')")
    ax.set_xlabel("k(x, x')")
    plt.title(f"Kernel entries vs delta fitness for {dataset}")
    plt.tight_layout()
    plt.savefig(f"figures/{dataset}_fitness_vs_kernel_scatter_Hellinger.png")
    plt.show()
