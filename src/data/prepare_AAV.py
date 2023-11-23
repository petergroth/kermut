import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS


def main():
    # Data source: ProteinGym.
    # Original source: Sinai et al. 2021, Generative AAV capsid diversification by latent interpolation.
    dataset = "AAV"
    raw_path = Path("data", "raw", dataset, "CAPSD_AAV2S_Sinai_substitutions_2021.csv")
    processed_path = Path("data", "processed", f"{dataset}.tsv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    wt_sequence = list("DEEEIRTTNPVATEQYGSVSTNLQRGNR")

    df = pd.read_csv(raw_path)
    df = df[["mutant", "sequence", "num_mutations", "partition", "viral_selection"]]
    # Remove stop codons
    wt_df = df[df["partition"] == "wild_type"]
    df = df[~df["partition"].isin(["stop", "wild_type"])]
    df = df.rename(columns={"sequence": "seq", "num_mutations": "n_muts"})

    offset = 560
    df["mut2wt"] = df["mutant"].str.split(":")
    assert (df["mut2wt"].str.len() == df["n_muts"]).all()
    df["mut2wt"] = df["mut2wt"].apply(
        lambda mutations: [
            mut[0] + str(int(mut[1:-1]) - offset) + mut[-1] for mut in mutations
        ]
    )

    df["delta_fitness"] = df["viral_selection"] - wt_df["viral_selection"].item()

    # Create sequence column
    df["key"] = ""
    for i, row in df.iterrows():
        mutated_sequence = wt_sequence.copy()
        for mutation in row["mut2wt"]:
            wt, pos, mut = mutation[0], int(mutation[1:-1]), mutation[-1]
            assert mutated_sequence[pos - 1] == wt
            mutated_sequence[pos - 1] = mut
        df.at[i, "key"] = f"seq_id_{i}"

    # Filter
    df = df[
        [
            "key",
            "delta_fitness",
            "mut2wt",
            "n_muts",
            "seq",
            "partition",
            "viral_selection",
        ]
    ]

    # Visualize target distribution
    sns.set_style("dark")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=df, x="delta_fitness", ax=ax, color=COLORS[4])
    ax.axvline(0, color="black", linestyle="--")

    plt.suptitle(f"{dataset} fitness distribution", fontsize=20)
    plt.tight_layout()
    plt.savefig(Path("figures/data_distributions", f"{dataset}.png"), dpi=300)
    plt.show()

    # Save to file
    df.to_csv(processed_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
