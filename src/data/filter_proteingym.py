import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    ref_path = Path("data/processed/DMS_substitutions.csv")
    out_path = Path("data/processed/DMS_substitutions_reduced.csv")
    id_file = Path("data/interim/DMS_substitutions_reduced_UniProt_ID.csv")
    df_ref = pd.read_csv(ref_path)

    n_max = 6000
    df_ref = df_ref[df_ref["DMS_total_number_mutants"] <= n_max]
    df_ref.to_csv(out_path, index=False)

    # Get unique UniProt_ID values
    uniprot_ids = df_ref["UniProt_ID"].unique()
    df_uniprot_ids = pd.DataFrame(uniprot_ids, columns=["UniProt_ID"])
    df_uniprot_ids.to_csv(id_file, index=False)
