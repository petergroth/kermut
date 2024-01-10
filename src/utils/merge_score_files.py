import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    out_path = Path("results/ProteinGym", "merged_scores.csv")
    results_path = Path("results/ProteinGym")
    glob_path = results_path.glob("*")
    df_results = pd.DataFrame(
        columns=["fold_variable_name", "MSE", "Spearman", "assay_id", "model_name"]
    )

    compile_all = False

    for dir_path in glob_path:
        if dir_path.is_dir():
            for file_path in dir_path.glob("*.csv"):
                if compile_all:
                    df = pd.read_csv(file_path)
                    df_results = pd.concat([df_results, df])
                elif file_path.stem in [
                    "kermutBH_oh_fold_contiguous_5",
                    "kermutBH_oh_fold_modulo_5",
                    "kermutBH_oh_fold_random_5",
                    "kermutBH_oh_fold_contiguous_5_ESM_IF1",
                    "kermutBH_oh_fold_modulo_5_ESM_IF1",
                    "kermutBH_oh_fold_random_5_ESM_IF1",
                ]:
                    df = pd.read_csv(file_path)
                    df_results = pd.concat([df_results, df])
    df_avg = df_results.groupby(
        ["fold_variable_name", "assay_id", "model_name"], as_index=False
    ).mean()
    df_avg = df_avg.drop(columns=["fold"])
    df_avg.to_csv(out_path, index=False)
