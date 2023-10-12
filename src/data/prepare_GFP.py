import pandas as pd
from pathlib import Path
import numpy as np


def main():
    dataset = "GFP"
    raw_path = Path("data", "raw", dataset, "amino_acid_genotypes_to_brightness.tsv")
    processed_path = Path("data", "processed", f"{dataset}.tsv")

    # Define wildtype sequence (https://www.uniprot.org/uniprotkb/P42212/entry), remove M1, and apply L64F mutation
    wt_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    wt_sequence = list(wt_sequence)
    wt_sequence[63] = "L"
    wt_sequence = wt_sequence[1:]

    df = pd.read_csv(raw_path, sep="\t")
    df = df.rename(columns={"aaMutations": "mut2wt", "medianBrightness": "brightness"})[
        ["mut2wt", "brightness"]
    ]
    # Extract wildtype entry
    wt_df = pd.Series(
        data={
            "key": "wt",
            "mut2wt": np.nan,
            "brightness": df.iloc[0]["brightness"],
            "n_muts": 0,
            "seq": "".join(wt_sequence),
        }
    )
    df = df.iloc[1:]

    # Remove premature stop codons (where the mutation is a "*")
    df = df[~df["mut2wt"].str.contains("\*")]

    # Process mutations
    df["mut2wt"] = df["mut2wt"].apply(lambda x: x.split(":"))
    df["mut2wt"] = df["mut2wt"].apply(
        lambda mutations: [mutation[1:] for mutation in mutations]
    )
    df["n_muts"] = df["mut2wt"].apply(len)

    # Change from 0-indexed to 1-indexed
    df["initial_mut2wt"] = df["mut2wt"]
    df["mut2wt"] = df["mut2wt"].apply(
        lambda mutations: [
            mutation[0] + str(int(mutation[1:-1]) + 1) + mutation[-1]
            for mutation in mutations
        ]
    )

    # Create sequence column
    df["seq"] = ""
    df["key"] = ""
    for i, row in df.iterrows():
        if row["n_muts"] == 0:
            continue
        mutated_sequence = wt_sequence.copy()
        for mutation in row["mut2wt"]:
            wt, pos, mut = mutation[0], int(mutation[1:-1]), mutation[-1]
            assert mutated_sequence[pos - 1] == wt
            mutated_sequence[pos - 1] = mut
        df.at[i, "key"] = f"seq_id_{i}"
        df.at[i, "seq"] = "".join(mutated_sequence)

    # Add wt series to df
    df = pd.concat([wt_df.to_frame().T, df])

    # Save to file
    df.to_csv(processed_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
