import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS


def main():
    # Data source: Olson et al. 2014: A Comprehensive Biophysical Description of Pairwise Epistasis throughout an Entire Protein Domain
    # Fitness value of interest is "score" column (identical to lnW column), which is the log of the relative fitness of the mutant.
    # Values close to zero have identical fitness to the wildtype.
    # Mutations are all in the IgG binding domain of GB1, "Protein G domain B1"
    # Column W = Selection Count / Input Count
    # Mut fitness column = W_mut / W_wt

    dataset = "SPG1"
    raw_path = Path("data", "raw", dataset, "SPG1_STRSG_Olson_2014.csv")
    processed_path = Path("data", "processed", f"{dataset}.tsv")
    double_samples = 10000

    wt_sequence = list("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE")
    # Template used includes T2Q mutation (see Olson et al. 2014, supplementaries))
    wt_sequence[1] = "Q"

    df = pd.read_csv(raw_path)
    df = df[["mutant", "lnW"]]

    # mutant column has offset of 226, so subtract 226 from all positions
    df["mut2wt"] = df["mutant"].apply(lambda x: x.split(":"))
    df["mut2wt"] = df["mut2wt"].apply(
        lambda mutations: [
            mut[0] + str(int(mut[1:-1]) - 226) + mut[-1] for mut in mutations
        ]
    )
    df["n_muts"] = df["mut2wt"].apply(len)
    wt_fitness = np.log(3041819 / 1759616)  # From supplementary table 2

    # Work in log space; subtract wildtype
    df["delta_fitness"] = df["lnW"] - wt_fitness

    # Keep all singles, subsample doubles
    df_1 = df[df["n_muts"] == 1]
    df_2 = df[df["n_muts"] == 2].sample(n=double_samples, random_state=42)

    # Approximate additive mutations
    threshold = 0.1
    df_1["mut2wt"] = df_1["mut2wt"].apply(lambda x: x[0])
    df_1 = df_1.set_index("mut2wt")
    df_2["additive"] = False
    for i, row in df_2.iterrows():
        mut1, mut2 = row["mut2wt"]
        fitness_sum = df_1.loc[mut1, "delta_fitness"] + df_1.loc[mut2, "delta_fitness"]
        df_2.at[i, "fitness_sum"] = fitness_sum
        if (fitness_sum - threshold) < row["delta_fitness"] < (fitness_sum + threshold):
            df_2.at[i, "additive"] = True

    df = pd.concat([df[df["n_muts"] == 1], df_2])

    # Create sequence column
    df["seq"] = ""
    df["key"] = ""
    for i, row in df.iterrows():
        mutated_sequence = wt_sequence.copy()
        for mutation in row["mut2wt"]:
            wt, pos, mut = mutation[0], int(mutation[1:-1]), mutation[-1]
            assert mutated_sequence[pos - 1] == wt
            mutated_sequence[pos - 1] = mut
        df.at[i, "key"] = f"seq_id_{i}"
        df.at[i, "seq"] = "".join(mutated_sequence)

    # Filter
    df = df[
        [
            "key",
            "delta_fitness",
            "mut2wt",
            "n_muts",
            "fitness_sum",
            "additive",
            "lnW",
            "seq",
        ]
    ]

    # Visualize target distribution
    sns.set_style("dark")
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex="col")

    sns.histplot(df["delta_fitness"], ax=ax[0], color=COLORS[4])
    ax[0].axvline(0, color="black", linestyle="--")
    ax[0].set_title("Target distribution")
    # ax[0].set_xlabel("ln(W)-ln(W_wt)")

    sns.histplot(df.loc[df["n_muts"] == 1, "delta_fitness"], ax=ax[1], color=COLORS[0])
    ax[1].axvline(0, color="black", linestyle="--")
    ax[1].set_title(f"Single mutants ({len(df_1)} samples)")
    # ax[1].set_xlabel("ln(W)-ln(W_wt)")

    sns.histplot(df.loc[df["n_muts"] == 2, "delta_fitness"], ax=ax[2], color=COLORS[1])
    ax[2].axvline(0, color="black", linestyle="--")
    ax[2].set_title(f"Double mutants ({double_samples} samples)")
    ax[2].set_xlabel("ln(W)-ln(W_wt)")

    plt.suptitle("SPG1 fitness distribution", fontsize=20)
    plt.tight_layout()
    plt.savefig(Path("figures/data_distributions", "SPG1.png"), dpi=300)
    plt.show()

    # Save to file
    df.to_csv(processed_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
