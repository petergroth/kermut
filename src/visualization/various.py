# from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path

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
        df_ = pd.read_csv(
            f"results/ProteinGym/summary/Spearman/DMS_substitutions_Spearman_DMS_level_{method}.csv"
        )
        df_["fold_variable_name"] = method
        df_tmp = pd.concat([df_tmp, df_])

    df_ref = pd.read_csv(ref_path)[["DMS_id", "DMS_total_number_mutants"]]

    df = pd.merge(left=df_score, right=df_ref, on="DMS_id", how="left")
    df = df[df["DMS_total_number_mutants"] <= 6000]
    models_of_interest = [
        "ProteinNPT",
        # "kermutBH_oh",
        # "kermutBH_oh_ESM_IF1",
        "kermut_ProteinMPNN_TranceptEVE_MSAT",
        # "Tranception Embeddings",
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


def analyze_multiples():
    datasets = [
        "EPHB2_HUMAN_Rocklin_2023_1F0M",
        "FECA_ECOLI_Rocklin_2023_2D1U",
        "GCN4_YEAST_Staller_2018",
        "HECD1_HUMAN_Rocklin_2023_3DKM",
        "ISDH_STAAW_Rocklin_2023_2LHR",
        "MAFG_MOUSE_Rocklin_2023_1K1V",
        "MBD11_ARATH_Rocklin_2023_6ACV",
        "MYO3_YEAST_Rocklin_2023_2BTT",
        "NKX31_HUMAN_Rocklin_2023_2L9R",
        "NUSA_ECOLI_Rocklin_2023_1WCL",
        "NUSG_MYCTU_Rocklin_2023_2MI6",
        "OBSCN_HUMAN_Rocklin_2023_1V1C",
        "ODP2_GEOSE_Rocklin_2023_1W4G",
        "PIN1_HUMAN_Rocklin_2023_1I6C",
        "PITX2_HUMAN_Rocklin_2023_2L7M",
        "POLG_PESV_Rocklin_2023_2MXD",
        "PR40A_HUMAN_Rocklin_2023_1UZC",
        "PSAE_SYNP2_Rocklin_2023_1PSE",
        "RAD_ANTMA_Rocklin_2023_2CJJ",
        "RBP1_HUMAN_Rocklin_2023_2KWH",
        "RCD1_ARATH_Rocklin_2023_5OAO",
        "RCRO_LAMBD_Rocklin_2023_1ORC",
        "RD23A_HUMAN_Rocklin_2023_1IFY",
        "RFAH_ECOLI_Rocklin_2023_2LCL",
        "RL20_AQUAE_Rocklin_2023_1GYZ",
        "RPC1_BP434_Rocklin_2023_1R69",
        "SAV1_MOUSE_Rocklin_2023_2YSB",
        "SDA_BACSU_Rocklin_2023_1PV0",
        "SPA_STAAU_Rocklin_2023_1LP1",
        "SPG2_STRSG_Rocklin_2023_5UBS",
        "SPTN1_CHICK_Rocklin_2023_1TUD",
        "SR43C_ARATH_Rocklin_2023_2N88",
        "SRBS1_HUMAN_Rocklin_2023_2O2W",
        "TCRG1_MOUSE_Rocklin_2023_1E0L",
        "THO1_YEAST_Rocklin_2023_2WQG",
        "TNKS2_HUMAN_Rocklin_2023_5JRT",
        "UBE4B_HUMAN_Rocklin_2023_3L1X",
        "UBR5_HUMAN_Rocklin_2023_1I2T",
        "VILI_CHICK_Rocklin_2023_1YU5",
        "YAIA_ECOLI_Rocklin_2023_2KVT",
        "YNZC_BACSU_Rocklin_2023_2JVD",
    ]

    model_names = [
        "kermut_ProteinMPNN_TranceptEVE_MSAT",
        # "kermut_no_blosum",
        "MSAT_RBF",
        "OH_RBF",
    ]
    fig_dir = Path("figures", "ProteinGym", "multiples")

    for i in range(len(datasets)):
        # for i in range(1):
        dataset = datasets[i]

        fig, ax = plt.subplots(
            len(model_names),
            2,
            figsize=(6, len(model_names) * 2.5),
            sharex="col",
            # sharey="col",
        )

        for j, model_name in enumerate(model_names):
            # Load predictions
            df_pred = pd.read_csv(
                Path("results/ProteinGym/predictions_multiples")
                / dataset
                / f"{model_name}.csv"
            )
            df_res = pd.read_csv(
                Path(
                    "results/ProteinGym/per_dataset_multiples",
                    dataset,
                    f"{model_name}.csv",
                )
            )

            sns.violinplot(
                data=df_pred,
                x="n_mutations",
                y="y_var",
                # color=COLORS[i],
                ax=ax[j, 0],
            )
            sns.barplot(
                data=df_res,
                x="n_mutations",
                y="Spearman",
                # color=COLORS[i],
                ax=ax[j, 1],
            )
            ax[j, 0].set_xlabel("Number of mutations")
            ax[j, 0].set_ylabel(f"{model_name}\nPredictive variance")
            ax[j, 1].set_xlabel("Number of mutations")
            ax[j, 1].set_ylabel("Spearman")
            ax[j, 1].set_ylim(0, 1)

        plt.suptitle(f"{dataset}")
        plt.tight_layout()
        out_dir = fig_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"variance_comparison.png", dpi=125)
        plt.close()


def domain_comparison_per_dataset():
    df_ref = pd.read_csv("data/processed/DMS_substitutions_reduced.csv")
    datasets = df_ref[df_ref["includes_multiple_mutants"]]["DMS_id"].unique()

    model_names = ["MSAT_RBF", "kermut_distance_no_blosum"]
    fig_dir = Path("figures", "ProteinGym", "domains")

    # Settings: random, modulo, contiguous, multiples (2M)

    for i in range(len(datasets)):
        # for i in range(1):
        dataset = datasets[i]

        fig, ax = plt.subplots(
            len(model_names),
            1,
            figsize=(6, len(model_names) * 2.5),
            sharex="all",
            sharey="all",
        )

        for j, model_name in enumerate(model_names):
            df = pd.DataFrame()
            for method in ["random", "modulo", "contiguous"]:
                df_ = pd.read_csv(
                    Path(
                        "results/ProteinGym/predictions",
                        dataset,
                        f"{model_name}_fold_{method}_5.csv",
                    )
                )
                df_["domain"] = f"1M ({method})"
                df = pd.concat([df, df_])
            # Multiples
            df_ = pd.read_csv(
                Path("results/ProteinGym/predictions_multiples")
                / dataset
                / f"{model_name}.csv"
            )
            df_ = df_[df_["n_mutations"] == 2]
            df_["domain"] = "2M"
            df = pd.concat([df, df_]).reset_index(drop=True)
            df["n_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))

            df["fold"] = df["fold"].fillna("multiples")
            df_metric = df.groupby(["domain", "fold"], as_index=False).apply(
                lambda x: spearmanr(x["y"].values, x["y_pred"].values)[0]
            )
            df_metric = df_metric.rename(columns={None: "Spearman"})[
                ["domain", "Spearman"]
            ]
            df_metric = df_metric.groupby("domain", as_index=False).mean(
                numeric_only=True
            )

            sns.violinplot(
                data=df, x="domain", y="y_var", hue="domain", palette=COLORS, ax=ax[j]
            )
            # Add text box annotation with Spearman correlations
            s = "Spearman - Domain"
            for _, row in df_metric.iterrows():
                s += f"\n{row['Spearman']:.2f} - {row['domain']}"

            ax[j].text(
                0.05,
                0.95,
                s,
                transform=ax[j].transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

            ax[j].set_title(f"{model_name}")
            ax[j].set_xlabel("Domain")
            ax[j].set_ylabel(f"Predictive variance")

        plt.suptitle(f"{dataset}")
        plt.tight_layout()
        out_dir = fig_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / f"domain_variance.png", dpi=125)
        plt.close()


def domain_comparison():
    plt.rcParams["font.family"] = "serif"

    df_ref = pd.read_csv("data/processed/DMS_substitutions_reduced.csv")
    df_ref = df_ref[df_ref["includes_multiple_mutants"]]

    # model_names = ["kermut_distance_no_blosum"]
    model_names = ["kermut_distance_no_blosum"]
    fig_dir = Path("figures", "ProteinGym", "domains")

    df = pd.DataFrame()
    for i, model_name in enumerate(model_names):
        for dataset in df_ref["DMS_id"].unique():
            # df = pd.DataFrame()
            for method in ["random", "modulo", "contiguous"]:
                df_ = pd.read_csv(
                    Path(
                        "results/ProteinGym/predictions",
                        dataset,
                        f"{model_name}_fold_{method}_5.csv",
                    )
                )
                df_["domain"] = f"1M ({method})"
                df_["dataset"] = dataset
                df = pd.concat([df, df_])

            # Multiples
            df_ = pd.read_csv(
                Path("results/ProteinGym/predictions_multiples")
                / dataset
                / f"{model_name}.csv"
            )
            df_ = df_[df_["n_mutations"] == 2]
            df_["domain"] = "2M"
            df_["dataset"] = dataset
            df = pd.concat([df, df_])

    df["n_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
    df["fold"] = df["fold"].fillna("multiples")
    # df_metric = df.groupby(["domain", "fold"], as_index=False).apply(
    # lambda x: spearmanr(x["y"].values, x["y_pred"].values)[0]
    # )
    # df_metric = df_metric.rename(columns={None: "Spearman"})[["domain", "Spearman"]]
    # df_metric = df_metric.groupby("domain", as_index=False).mean(numeric_only=True)

    df_avg = df.groupby(["dataset", "domain"], as_index=False).mean(numeric_only=True)

    # df_avg["domain"] = df_avg["domain"].map(mapping)

    # Sort by domain in manual order
    df_avg["domain"] = pd.Categorical(
        df_avg["domain"], ["1M (random)", "1M (modulo)", "1M (contiguous)", "2M"]
    )
    df_avg = df_avg.sort_values("domain")

    fig, ax = plt.subplots(
        **{"figsize": (6.5, 4), "sharex": "all", "sharey": "all"},
    )
    font_kwargs = {
        "fontsize": 16,
    }
    sns.violinplot(
        data=df_avg,
        x="domain",
        y="y_var",
        hue="domain",
        palette=COLORS,
        ax=ax,
        saturation=1,
        hue_order=["1M (random)", "1M (modulo)", "1M (contiguous)", "2M"],
    )
    ax.set_xlabel("")
    ax.set_ylabel(r"Predictive variance $\hat{\sigma}^2$", **font_kwargs)

    mapping = {
        "1M (random)": r"1M$\rightarrow$1M" + "\n(random)",
        "1M (modulo)": r"1M$\rightarrow$1M" + " \n(modulo)",
        "1M (contiguous)": r"1M$\rightarrow$1M" + "\n(contiguous)",
        "2M": r"1M$\rightarrow$2M",
    }
    # Replace xticklabels with mapping
    ax.set_xticklabels([mapping[x.get_text()] for x in ax.get_xticklabels()])
    # Increase font size of xticklabels
    for xtick in ax.get_xticklabels():
        xtick.set_fontsize(14)
    # for y
    for ytick in ax.get_yticklabels():
        ytick.set_fontsize(14)

    # plt.suptitle(f"{dataset}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{model_name}_domain_variance_summary.png")
    plt.savefig(fig_dir / f"{model_name}_domain_variance_summary.pdf")
    plt.close()


if __name__ == "__main__":
    # plot_average_score_order_by_zero_shot()
    # plot_score_vs_num_mutants()
    # plot_zero_shot_vs_fitness()
    # compare_select_datasets()
    # analyze_multiples()
    domain_comparison()
    # domain_comparison_per_dataset()
