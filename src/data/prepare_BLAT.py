from pathlib import Path
import pandas as pd


def main():
    path_in = Path("data", "raw", "BLAT_ECOLX_Ranganathan2015.csv")
    path_out = Path("data", "processed", "BLAT_ECOLX.tsv")

    # Load and process
    df = pd.read_csv(path_in)
    df = df[["mutant", "2500"]].rename(
        columns={"mutant": "mut2wt", "2500": "delta_fitness"}
    )
    df["pos"] = df["mut2wt"].str[1:-1].astype(int)

    # Save
    df.to_csv(path_out, sep="\t", index=False)


if __name__ == "__main__":
    main()
