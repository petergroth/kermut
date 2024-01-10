from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")
from src import COLORS


def plot_score_vs_num_mutants():
    baseline_path = Path(
        "results/ProteinGym_baselines/DMS_supervised_substitutions_scores.csv"
    )
    kermut_path = Path("results/ProteinGym/merged_scores.csv")
    ref_path = Path("data/raw/DMS_substitutions.csv")

    df_baseline = pd.read_csv(baseline_path)
    df_kermut = pd.read_csv(kermut_path)
    df_ref = pd.read_csv(ref_path)
    cols = ["model_name", "assay_id", "fold_variable_name", "Spearman"]

    df_baseline = df_baseline.rename(columns={"Spearman_fitness": "Spearman"})[cols]
    df_kermut = df_kermut[cols]
    df_ref = df_ref[
        ["DMS_id", "seq_len", "includes_multiple_mutants", "DMS_total_number_mutants"]
    ]
    method = "fold_random_5"

    df_baseline = df_baseline[df_baseline["fold_variable_name"] == method]
    df_kermut = df_kermut[df_kermut["fold_variable_name"] == method]

    # Concatenate baseline and kermut
    df = pd.concat([df_baseline, df_kermut]).reset_index(drop=True)
    df = pd.merge(left=df, right=df_ref, left_on="assay_id", right_on="DMS_id")

    models_of_interest = [
        "kermutBH_oh",
        "ProteinNPT",
        # "Embeddings - Augmented - Tranception",
    ]

    df = df[df["DMS_total_number_mutants"] <= 6000]

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    sns.lineplot(
        data=df[df["model_name"].isin(models_of_interest)],
        y="Spearman",
        x="DMS_total_number_mutants",
        hue="model_name",
        # style="includes_multiple_mutants",
        ax=ax,
        palette=COLORS,
        alpha=0.8,
    )
    sns.scatterplot(
        data=df[df["model_name"].isin(models_of_interest)],
        y="Spearman",
        x="DMS_total_number_mutants",
        hue="model_name",
        # style="includes_multiple_mutants",
        ax=ax,
        palette=COLORS,
        alpha=0.8,
        legend=False
    )
    ax.set_xlim(0, 2000)
    plt.tight_layout()
    plt.savefig("figures/ProteinGym/score_vs_num_mutants.png", dpi=125)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(
        data=df[df["model_name"].isin(models_of_interest)],
        y="Spearman",
        x="seq_len",
        hue="model_name",
        ax=ax,
        palette=COLORS,
        alpha=0.8,
    )
    ax.set_xlim(0, 1000)
    plt.tight_layout()
    # plt.show()


def plot_average_score_order_by_zero_shot():
    res_path = Path("results/ProteinGym")

    # Our scores
    df_ker = pd.read_csv(res_path / "merged_scores.csv").drop(columns=["MSE"])
    df_ker = df_ker.groupby(["assay_id", "model_name"], as_index=False).mean(
        numeric_only=True
    )
    df_1 = df_ker[df_ker["model_name"] == "kermutBH_oh"]
    df_2 = df_ker[df_ker["model_name"] == "kermutBH_oh_ESM_IF1"]

    # Reference scores
    df_ref = pd.read_csv(
        "ProteinGym_outputs/Spearman/DMS_substitutions_Spearman_DMS_level.csv"
    )
    df_ref = df_ref[["DMS_id", "ProteinNPT"]].rename(
        columns={"DMS_id": "assay_id", "ProteinNPT": "Spearman"}
    )
    df_ref["model_name"] = "ProteinNPT"
    # Concatenate
    df = pd.concat([df_1, df_2, df_ref], axis=0).reset_index(drop=True)

    # Zero-shot scores
    df_zero_shot = pd.read_csv(
        "results/ProteinGym_baselines/DMS_substitutions_Spearman_DMS_level_zero_shot.csv"
    )
    df_zero_shot = df_zero_shot[["DMS ID", "TranceptEVE L"]].sort_values(
        by="TranceptEVE L", ascending=False
    )

    # Replace all Rocklin with Tsuboyama in assay_id col in df
    df["assay_id"] = df["assay_id"].str.replace("Rocklin", "Tsuboyama")
    df_zero_shot = df_zero_shot.rename(columns={"DMS ID": "assay_id"})
    df = pd.merge(
        left=df, right=df_zero_shot, left_on="assay_id", right_on="assay_id", how="left"
    )
    df = df.sort_values(by="TranceptEVE L", ascending=False)

    # df = df[df["model_name"].isin(["kermutBH_oh_ESM_IF1", "ProteinNPT"])]
    # df = df[df["model_name"].isin(["kermutBH_oh_ESM_IF1", "kermutBH_oh"])]

    fig, ax = plt.subplots(figsize=(25, 6))
    sns.lineplot(
        data=df,
        y="TranceptEVE L",
        x="assay_id",
        ax=ax,
        color="black",
        label="TranceptEVE L",
        alpha=0.5,
    )
    sns.lineplot(
        data=df, y="Spearman", x="assay_id", ax=ax, hue="model_name", palette=COLORS
    )
    sns.scatterplot(
        data=df,
        y="Spearman",
        x="assay_id",
        ax=ax,
        hue="model_name",
        palette=COLORS,
        legend=False,
    )
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-90)
    plt.tight_layout()

    plt.savefig("figures/ProteinGym/average_scores_order_by_evo_score.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    # plot_average_score_order_by_zero_shot()
    plot_score_vs_num_mutants()
