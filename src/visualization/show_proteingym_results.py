import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS

if __name__ == "__main__":
    sns.set_style("darkgrid")

    methods = ["random", "modulo", "contiguous"]
    # methods = ["random"]

    for method in methods:
        # ref_score_path = Path(
        #     "results/ProteinGym_baselines",
        #     f"DMS_substitutions_Spearman_DMS_level_fold_{method}_5.csv",
        # )
        score_path = Path(
            "ProteinGym_outputs/Spearman",
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

    score_path = Path(
        "ProteinGym_outputs/Spearman/Summary_performance_DMS_substitutions_Spearman.csv"
    )
    df = pd.read_csv(score_path)
    df["ours"] = False
    df.loc[df["Model_name"] == "kermutBH_oh", "ours"] = True

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=df, x="Model_name", y="Average_Spearman", ax=ax, hue="ours", palette=COLORS
    )
    # rotate x labels
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Average Spearman correlation")
    ax.set_title("Average Spearman correlation (all splits)")
    plt.tight_layout()
    plt.savefig("figures/ProteinGym/Spearman_comparison_average.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots(3, 1, figsize=(9, 14), sharey="col", sharex="col")
    for i, method in enumerate(["random", "modulo", "contiguous"]):
        sns.barplot(
            data=df,
            x="Model_name",
            y=f"Average_Spearman_fold_{method}_5",
            ax=ax[i],
            hue="ours",
            palette=COLORS,
            legend=False,
        )
        ax[i].set_ylabel("Average Spearman correlation")
        ax[i].set_title(f"Average Spearman correlation (split = {method})")
        # rotate x labels
    plt.setp(ax[-1].get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("figures/ProteinGym/Spearman_comparison_average_split.png", dpi=300)
    plt.show()
