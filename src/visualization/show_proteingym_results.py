import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS


if __name__ == "__main__":
    sns.set_style("darkgrid")

    methods = ["random", "modulo", "contiguous"]
    # methods = ["modulo", "contiguous"]
    DMS_plot = False

    if DMS_plot:
        for method in methods:
            # ref_score_path = Path(
            #     "results/ProteinGym_baselines",
            #     f"DMS_substitutions_Spearman_DMS_level_fold_{method}_5.csv",
            # )
            score_path = Path(
                "results/ProteinGym/summary/Spearman",
                f"DMS_substitutions_Spearman_DMS_level_fold_{method}_5.csv",
            )
            # df_ref = pd.read_csv(ref_score_path)
            df = pd.read_csv(score_path)

            # Merge on DMS_id
            # df_merge = pd.merge(df, df_ref, on="DMS_id", how="left")
            # Sort by DMS_id
            df = df.sort_values(by="DMS_id", ascending=True)

            # Melt
            df_melt = pd.melt(
                df,
                id_vars=["DMS_id"],
                var_name="model",
                value_name="Spearman",
            )

            df_melt["ours"] = "Baseline"
            df_melt.loc[df_melt["model"] == "kermutBH_oh", "ours"] = "kermutBH_oh"
            df_melt.loc[
                df_melt["model"] == "kermutBH_oh_ESM_IF1", "ours"
            ] = "kermutBH_oh_zeroshot"

            # Sort by ours
            df_melt = df_melt.sort_values(by=["DMS_id", "ours"], ascending=True)

            for DMS_id in df_melt["DMS_id"].unique():
                DMS = df_melt[df_melt["DMS_id"] == DMS_id]
                if DMS.loc[DMS["Spearman"].idxmax(), "model"] == "kermutBH_oh":
                    df_melt.loc[df_melt["DMS_id"] == DMS_id, "DMS_id"] = DMS_id + "*"

            sns.relplot(
                # data=df_melt[df_melt["DMS_id"] == "A0A2Z5U3Z0_9INFA_Wu_2014"],
                data=df_melt,
                x="Spearman",
                y="DMS_id",
                hue="model",
                style="ours",
                legend=False,
                s=100,
                alpha=0.8,
                palette=COLORS,
                height=25,
                aspect=0.3,
            )
            plt.xlim(0, 1)
            plt.title(f"Spearman correlation (split = {method})")
            plt.tight_layout()
            plt.savefig(
                f"figures/ProteinGym/Spearman_comparison_plot_{method}.png", dpi=300
            )
            plt.show()

    # Load summarized results
    summary_path = Path(
        "results/ProteinGym/summary/Spearman/Summary_performance_DMS_substitutions_Spearman.csv"
    )
    df = pd.read_csv(summary_path)
    df["ours"] = False
    df.loc[df["Model_name"] == "kermutBH_oh", "ours"] = True
    df.loc[df["Model_name"] == "kermutBH_oh_ESM_IF1", "ours"] = True
    df.loc[df["Model_name"] == "kermut_ProteinMPNN_TranceptEVE", "ours"] = True
    # Load non-summarized results
    per_dms_path = Path(
        "results/ProteinGym/summary/Spearman/DMS_substitutions_Spearman_DMS_level.csv",
    )
    df_per_dms = pd.read_csv(per_dms_path)
    df_per_dms = (
        df_per_dms.mean(numeric_only=True)
        .to_frame("Spearman_global_average")
        .reset_index(names="Model_name")
    )
    df = pd.merge(df, df_per_dms, on="Model_name", how="left")

    # Repeat for each split
    split_path = Path("results/ProteinGym/summary/Spearman")
    for method in methods:
        df_split = pd.read_csv(
            split_path / f"DMS_substitutions_Spearman_DMS_level_fold_{method}_5.csv"
        )
        df_per_dms_split = (
            df_split.mean(numeric_only=True)
            .to_frame(f"Spearman_fold_{method}_5_global_average")
            .reset_index(names="Model_name")
        )
        df = pd.merge(df, df_per_dms_split, on="Model_name", how="left")

    # Sort df by Spearman_global_average
    df = df.sort_values(by="Spearman_global_average", ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex="row", sharey="row")
    sns.barplot(
        data=df,
        x="Model_name",
        y="Average_Spearman",
        ax=ax[0],
        hue="ours",
        palette=COLORS,
    )
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")
    ax[0].set_ylabel("Average Spearman correlation")
    ax[0].set_title("Average Spearman correlation (per function category)")
    ax[0].set_ylim(0, 1)

    sns.barplot(
        data=df,
        x="Model_name",
        y="Spearman_global_average",
        ax=ax[1],
        hue="ours",
        palette=COLORS,
    )
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right")
    ax[1].set_title("Average Spearman correlation (global)")
    ax[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("figures/ProteinGym/Spearman_comparison_average.png", dpi=125)
    plt.show()

    #########################

    fig, ax = plt.subplots(3, 2, figsize=(12, 12), sharey="all", sharex="all")
    for i, method in enumerate(methods):
        sns.barplot(
            data=df,
            x="Model_name",
            y=f"Average_Spearman_fold_{method}_5",
            ax=ax[i, 0],
            hue="ours",
            palette=COLORS,
            legend=False,
        )
        ax[i, 0].set_ylabel("Average Spearman correlation")
        ax[i, 0].set_title(f"Split = {method}, per function")
        ax[i, 0].set_ylim(0, 1)
        sns.barplot(
            data=df,
            x="Model_name",
            y=f"Spearman_fold_{method}_5_global_average",
            ax=ax[i, 1],
            hue="ours",
            palette=COLORS,
            legend=False,
        )
        ax[i, 1].set_title(f"Split = {method}, global")
        ax[i, 1].set_ylim(0, 1)

    ax[-1, 0].set_xticklabels(ax[-1, 0].get_xticklabels(), rotation=45, ha="right")
    ax[-1, 1].set_xticklabels(ax[-1, 1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("figures/ProteinGym/Spearman_comparison_average_split.png", dpi=125)
