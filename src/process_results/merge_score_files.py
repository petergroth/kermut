import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    out_path = Path("results/ProteinGym", "merged_scores.csv")
    results_path = Path("results/ProteinGym/per_dataset")
    glob_path = results_path.glob("*")
    df_results = pd.DataFrame(
        columns=["fold_variable_name", "MSE", "Spearman", "assay_id", "model_name"]
    )
    df_ref = pd.read_csv("data/processed/DMS_substitutions_reduced.csv")
    methods = ["fold_random_5", "fold_contiguous_5", "fold_modulo_5"]
    new_names = {
        "kermut_constant_mean": "Kermut (constant mean)",  # Main model, constant mean
        "kermut_no_blosum": "kermut (no distance)",  # Main model, no distance
        "kermut_no_rbf": "kermut (no RBF)",  # Main model, no RBF
        "MSAT_RBF_constant_mean": "MSAT",  # Only RBF kernel, constant mean
        "kermut_distance_no_blosum": "Kermut",  # Main model
    }
    models = list(new_names.keys())

    models_methods = [f"{model}_{method}" for model in models for method in methods]

    for dataset in df_ref["DMS_id"].unique():
        for model_method in models_methods:
            df = pd.read_csv(results_path / dataset / f"{model_method}.csv")
            df_results = pd.concat([df_results, df])

    df_avg = df_results.groupby(
        ["fold_variable_name", "assay_id", "model_name"], as_index=False
    ).mean()

    df_avg["model_name"] = df_avg["model_name"].map(new_names)
    df_avg = df_avg[["fold_variable_name", "MSE", "Spearman", "assay_id", "model_name"]]
    # df_avg["assay_id"].value_counts().unique()
    df_avg.to_csv(out_path, index=False)
