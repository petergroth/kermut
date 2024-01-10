from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS


def main():
    # Source:  Evolvability as a Function of Purifying Selection in TEM-1 beta-Lactamase.
    # In ProteinGym: Stiffler et al. 2015
    dataset = "BLAT_ECOLX"
    wt_sequence = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    wt_sequence = list(wt_sequence)
    path_in = Path("data", "raw", dataset, "BLAT_ECOLX_Ranganathan2015.csv")
    path_out = Path("data", "processed", "BLAT_ECOLX.tsv")

    # Load and process
    df = pd.read_csv(path_in)
    df = df[["mutant", "2500"]].rename(
        columns={"mutant": "mut2wt", "2500": "delta_fitness"}
    )
    df["pos"] = df["mut2wt"].str[1:-1].astype(int)
    df["aa"] = df["mut2wt"].str[-1]
    df["wt"] = df["mut2wt"].str[0]
    df["seq"] = ""
    df["key"] = ""

    # Apply mutations
    for i, row in df.iterrows():
        mutated_sequence = wt_sequence.copy()
        assert mutated_sequence[row["pos"] - 1] == row["wt"]
        mutated_sequence[row["pos"] - 1] = row["aa"]
        df.at[i, "key"] = f"seq_id_{i}"
        df.at[i, "seq"] = "".join(mutated_sequence)

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
    main()
