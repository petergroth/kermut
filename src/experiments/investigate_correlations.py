from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser

from src import BLAT_ECOLX_WT, ALPHABET


def load_protein_mpnn_outputs(conditional_probs_path: Path):
    raw_file = np.load(conditional_probs_path)
    log_p = raw_file["log_p"]
    log_p_mean = log_p.mean(axis=0)
    p_mean = np.exp(log_p_mean)
    p_mean = p_mean[:, :20]  # "X" is included as 21st AA in ProteinMPNN alphabet
    return p_mean


def show_example(
        idx: int, p_mean: np.array, df_assay: pd.DataFrame, df_kl: pd.DataFrame
):
    row = df_kl.iloc[idx]

    # Visualize
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey="col")

    # Show AA probabilities in left column
    sns.barplot(x=list(ALPHABET), y=p_mean[row["pos_i"] - 1], ax=ax[0, 0])
    sns.barplot(x=list(ALPHABET), y=p_mean[row["pos_j"] - 1], ax=ax[1, 0])
    # Show assay data per AA in right column

    # Add WT to assay data for comparison
    df_i = df_assay[df_assay["pos"] == row["pos_i"]]
    df_i = pd.concat(
        [
            df_i,
            pd.Series({"aa": df_i["wt_aa"].iloc[0], "delta_fitness": 0}).to_frame().T,
        ]
    )
    df_j = df_assay[df_assay["pos"] == row["pos_j"]]
    df_j = pd.concat(
        [
            df_j,
            pd.Series({"aa": df_j["wt_aa"].iloc[0], "delta_fitness": 0}).to_frame().T,
        ]
    )

    sns.barplot(data=df_i.sort_values(by="aa"), x="aa", y="delta_fitness", ax=ax[0, 1])
    sns.barplot(data=df_j.sort_values(by="aa"), x="aa", y="delta_fitness", ax=ax[1, 1])
    ax[0, 0].set_title(f"Position: {row['pos_i']}. WT: {row['aa_i']}")
    ax[1, 0].set_title(f"Position: {row['pos_j']}. WT: {row['aa_j']}")
    ax[0, 1].set_title(f"Position: {row['pos_i']}. WT: {row['aa_i']}")
    ax[1, 1].set_title(f"Position: {row['pos_j']}. WT: {row['aa_j']}")
    ax[0, 1].set_xlabel("")
    ax[1, 1].set_xlabel("")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = "BLAT_ECOLX"
    # Set paths
    data_path = Path("data", "processed", f"{dataset}.tsv")
    surface_path = Path("data", "processed", f"{dataset}_surface.csv")
    conditional_probs_path = Path(
        "data",
        "interim",
        dataset,
        "proteinmpnn",
        "conditional_probs_only",
        f"{dataset}.npz",
    )
    pdb_path = Path("data", "raw", f"{dataset}.pdb")

    # Load assay and surface data
    df_assay = pd.read_csv(data_path, sep="\t")
    df_surface = pd.read_csv(surface_path)
    wt_seq = BLAT_ECOLX_WT

    # Load PDB file using BioPython
    parser = PDBParser()
    structure = parser.get_structure(dataset, pdb_path)
    model = structure[0]
    chain = model["A"]
    coords = np.array(
        [atom.get_coord() for atom in chain.get_atoms() if atom.get_name() == "CA"]
    )
    df_coords = pd.DataFrame(coords, columns=["x", "y", "z"])
    df_coords["pos"] = df_coords.index + 1

    # Compute pairwise distances
    distance_matrix = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(coords.shape[0]):
        for j in range(coords.shape[0]):
            distance_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])

    # Load and process conditional probabilities
    p_mean = load_protein_mpnn_outputs(conditional_probs_path)

    # Compute entropy of each position
    entropy = np.zeros(p_mean.shape[0])
    for i in range(p_mean.shape[0]):
        entropy[i] = -np.sum(p_mean[i] * np.log(p_mean[i]))

    # Get indices of distance_matrix where value < 5 and > 0
    idx_i, idx_j = np.where(distance_matrix < 5)
    idx_i, idx_j = idx_i[idx_i != idx_j], idx_j[idx_i != idx_j]

    # For all pairs, compute KL divergence
    kl_divergence = np.zeros(idx_i.shape[0])
    for i in range(idx_i.shape[0]):
        kl_divergence[i] = np.sum(
            p_mean[idx_i[i]] * np.log(p_mean[idx_i[i]] / p_mean[idx_j[i]])
        )

    # Collect in DataFrame
    wt_seq_arr = np.array(list(wt_seq))
    df_kl = pd.DataFrame(
        {
            "pos_i": idx_i + 1,
            "pos_j": idx_j + 1,
            "kl_divergence": kl_divergence,
            "entropy_i": entropy[idx_i],
            "entropy_j": entropy[idx_j],
            "distance": distance_matrix[idx_i, idx_j],
            "aa_i": wt_seq_arr[idx_i],
            "aa_j": wt_seq_arr[idx_j],
        }
    )

    # Filter out positions with no assay data
    df_kl = df_kl[
        (df_kl["pos_i"].isin(df_assay["pos"])) & (df_kl["pos_j"].isin(df_assay["pos"]))
    ]

    # Filter out positions with high KL divergence and low entropy
    kl_median = df_kl["kl_divergence"].median()
    entropy_median = df_kl["entropy_i"].median()
    df_kl = df_kl[
        (df_kl["kl_divergence"] < kl_median)
        & (df_kl["entropy_i"] > entropy_median)
        & (df_kl["entropy_j"] > entropy_median)
    ]

    # Sort by KL and inspect top pair
    df_kl = df_kl.sort_values(by="kl_divergence", ascending=True)

    # Show example
    show_example(0, p_mean, df_assay, df_kl)
