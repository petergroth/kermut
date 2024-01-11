from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import COLORS
from src.data.data_utils import load_zero_shot, zero_shot_name_to_col

sns.set_style("darkgrid")


def plot_score_vs_num_mutants():
    score_path = Path(
        "results/ProteinGym/summary/Spearman/DMS_substitutions_Spearman_DMS_level.csv"
    )
    ref_path = Path("data/processed/DMS_substitutions.csv")

    df_score = pd.read_csv(score_path)
    
    
    df_tmp = pd.DataFrame()
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    for method in methods:
        df_ = pd.read_csv(f"results/ProteinGym/summary/Spearman/DMS_substitutions_Spearman_DMS_level_{method}.csv")
        df_["fold_variable_name"] = method
        df_tmp = pd.concat([df_tmp, df_])
        
        
    df_ref = pd.read_csv(ref_path)[["DMS_id", "DMS_total_number_mutants"]]

    df = pd.merge(left=df_score, right=df_ref, on="DMS_id", how="left")
    df = df[df["DMS_total_number_mutants"] <= 6000]
    models_of_interest = [
        "ProteinNPT",
        # "kermutBH_oh",
        # "kermutBH_oh_ESM_IF1",
        "kermut_ProteinMPNN_TranceptEVE",
        "Tranception Embeddings",
        "MSA Transformer Embeddings",
    ]

    df = pd.melt(
        df,
        id_vars=["DMS_id", "DMS_total_number_mutants"],
        value_vars=models_of_interest,
        var_name="model_name",
        value_name="Spearman",
    )

    df["dataset_size"] = (df["DMS_total_number_mutants"] // 250 + 1) * 250

    df_g = df.groupby(["model_name", "dataset_size"], as_index=False).mean(
        numeric_only=True
    )

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    sns.scatterplot(
        data=df_g[df_g["model_name"].isin(models_of_interest)],
        y="Spearman",
        x="dataset_size",
        hue="model_name",
        ax=ax,
        palette=COLORS,
        alpha=0.8,
        legend=False,
    )
    sns.lineplot(
        data=df_g[df_g["model_name"].isin(models_of_interest)],
        y="Spearman",
        x="dataset_size",
        hue="model_name",
        ax=ax,
        palette=COLORS,
        alpha=0.8,
    )
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
        "results/ProteinGym_baselines/supervised_substitution_scores/DMS_substitutions_Spearman_DMS_level_zero_shot.csv"
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


def plot_zero_shot_vs_fitness():
    datasets = [
        "NPC1_HUMAN_Erwood_2022_RPE1",
        "DNJA1_HUMAN_Rocklin_2023_2LO1",
        "BLAT_ECOLX_Stiffler_2015",
        "A0A1I9GEU1_NEIME_Kennouche_2019",
        "NUSG_MYCTU_Rocklin_2023_2MI6",
        "PA_I34A1_Wu_2015",
        "EPHB2_HUMAN_Rocklin_2023_1F0M",
        "VKOR1_HUMAN_Chiasson_2020_activity",
        "TCRG1_MOUSE_Rocklin_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
        "R1AB_SARS2_Flynn_2022",
        "PTEN_HUMAN_Matreyek_2021",
    ]
    datasets = sorted(datasets)

    zero_shot_method = "ProteinMPNN"

    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    for i, dataset in enumerate(datasets):
        df_zero = load_zero_shot(dataset, zero_shot_method)
        zero_shot_col = zero_shot_name_to_col(zero_shot_method)
        df = pd.read_csv(f"data/processed/ProteinGym_substitutions_DMS/{dataset}.csv")
        df = pd.merge(left=df, right=df_zero, left_on="mutant", right_on="mutant")

        sns.scatterplot(
            data=df,
            x="DMS_score",
            y=zero_shot_col,
            ax=ax.flatten()[i],
            palette=COLORS,
        )
        ax.flatten()[i].set_title(f"{dataset}")
    plt.suptitle(f"Zero-shot vs DMS score ({zero_shot_method})", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        f"figures/ProteinGym/zero_shot_vs_fitness/{zero_shot_method}_examples.png",
        dpi=125,
    )


def compare_select_datasets():
    datasets = [
        "NPC1_HUMAN_Erwood_2022_RPE1",
        "DNJA1_HUMAN_Rocklin_2023_2LO1",
        "BLAT_ECOLX_Stiffler_2015",
        "A0A1I9GEU1_NEIME_Kennouche_2019",
        "NUSG_MYCTU_Rocklin_2023_2MI6",
        "PA_I34A1_Wu_2015",
        "EPHB2_HUMAN_Rocklin_2023_1F0M",
        "VKOR1_HUMAN_Chiasson_2020_activity",
        "TCRG1_MOUSE_Rocklin_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
        "R1AB_SARS2_Flynn_2022",
    ]
    datasets = sorted(datasets)

    model_name_1 = "kermut_ProteinMPNN_ESM_IF1_unconstrained"
    model_name_2 = "kermutBH_oh"
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]

    df = pd.DataFrame()
    for dataset in datasets:
        for method in methods:
            df_1 = pd.read_csv(
                f"results/ProteinGym/per_dataset/{dataset}/{model_name_1}_{method}.csv"
            )
            df_2 = pd.read_csv(
                f"results/ProteinGym/per_dataset/{dataset}/{model_name_2}_{method}_ESM_IF1.csv"
            )
            df = pd.concat([df, df_1, df_2])

    # df_avg = df[["Spearman", "model_name", "fold_variable_name"]].groupby(["model_name", "fold_variable_name"], as_index=False).mean()
    # Barplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="fold_variable_name",
        y="Spearman",
        hue="model_name",
        ax=ax,
        palette=COLORS,
    )
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Comparison over {len(datasets)} datasets. Constrained vs. unconstrained"
    )
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig("figures/ProteinGym/compare_zero_shot_scale.png", dpi=125)


if __name__ == "__main__":
    # plot_average_score_order_by_zero_shot()
    plot_score_vs_num_mutants()
    # plot_zero_shot_vs_fitness()
    # compare_select_datasets()
