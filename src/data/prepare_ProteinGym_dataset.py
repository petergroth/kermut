from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS


def main(dataset: str):
    path_in = Path("data", "raw", dataset, f"{dataset}.csv")
    path_out = Path("data", "processed", f"{dataset}.tsv")
    path_reference = Path("data", "raw", "DMS_substitutions.csv")

    # Load and process
    df_ref = pd.read_csv(path_reference)
    df = pd.read_csv(path_in)

    wt_row = df_ref.loc[df_ref["DMS_id"] == dataset]
    # Get WT sequence.
    wt_sequence = wt_row["target_seq"].item()
    single_mutations = not wt_row["includes_multiple_mutants"].item()

    # Apply mutations
    for i, row in df.iterrows():
        df.at[i, "key"] = f"seq_id_{i}"

    df["n_muts"] = 1
    df["mut2wt"] = df["mut2wt"].apply(lambda x: [x])

    # Visualize target distribution
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x="delta_fitness", ax=ax, color=COLORS[4])
    ax.axvline(0, color="black", linestyle="--")
    plt.suptitle(f"{dataset} fitness distribution", fontsize=20)
    plt.tight_layout()
    plt.savefig(Path("figures/data_distributions", f"{dataset}.png"), dpi=300)
    plt.show()

    df.to_csv(path_out, sep="\t", index=False)


if __name__ == "__main__":
    dataset = "GFP_AEQVI_Sarkisyan_2016"
    # dataset = "BLAT_ECOLX_Stiffler_2015"
    main(dataset=dataset)
