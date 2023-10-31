import pandas as pd
from pathlib import Path


def main():
    dataset = "PARD3_10"
    target_key = "mean_lrr_norm"
    raw_path = Path("data", "raw", dataset, "df_at_10pos.csv")
    processed_path = Path("data", "processed", f"{dataset}.tsv")

    wt_sequence = "MANVEKMSVAVTPQQAAVMREAVEAGEYATASEIVREAVRDWLAKRELRHDDIRRLRQLWDEGKASGRPEPVDFDALRKEARQKLTEVPPNGR"
    wt_sequence = list(wt_sequence)

    df = pd.read_csv(raw_path, index_col=0)[
        ["Unnamed: 0", "stop", "wt", "mean_lrr", "mean_lrr_norm"]
    ]
    # Rename unnamed column
    df = df.rename(columns={"Unnamed: 0": "mut2wt"})

    wt_score = df.loc[df["wt"], [target_key]].mean()

    # Remove stop codons and wts, pool duplicates.
    df = df[~df["stop"] & ~df["wt"]][["mut2wt", target_key]]
    df = df.groupby(["mut2wt"], as_index=False).mean()
    df["delta_fitness"] = df[target_key] - wt_score.item()

    # Create sequence column
    df["seq"] = ""
    df["key"] = ""
    df["mut2wt"] = df["mut2wt"].apply(lambda x: x.split(":"))
    df["clean_mut2wt"] = ""
    for i, row in df.iterrows():
        mutated_sequence = wt_sequence.copy()
        mut_list = []
        for mutation in row["mut2wt"]:
            wt, pos, mut = mutation[0], int(mutation[1:-1]), mutation[-1]
            # Mutation are zero-indexed
            if wt == mut:
                continue
            assert mutated_sequence[pos] == wt
            mutated_sequence[pos] = mut
            # Change from 0-indexed to 1-indexed
            mut_list.append(wt + str(pos + 1) + mut)
        df.at[i, "clean_mut2wt"] = mut_list
        df.at[i, "key"] = f"seq_id_{i}"
        df.at[i, "seq"] = "".join(mutated_sequence)

    df["n_muts"] = df["clean_mut2wt"].apply(len)
    df["mut2wt"] = df["clean_mut2wt"]
    df = df[["key", "n_muts", "mut2wt", target_key, "delta_fitness", "seq"]]

    # # Save to file
    df.to_csv(processed_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
