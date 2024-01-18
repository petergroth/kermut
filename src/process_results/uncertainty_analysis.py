import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS
from scipy import stats

sns.set_style("whitegrid")


def plot_y_vs_y_pred(
    datasets: list,
    folds: list,
    methods: list,
    model_name: str,
    fig_dir: Path,
    prediction_dir: Path,
):
    fig, ax = plt.subplots(
        len(methods),
        5,
        figsize=(12, len(methods) * 2.5),
        sharex="all",
        sharey="all",
    )

    for i, method in enumerate(methods):
        prediction_path = prediction_dir / dataset / f"{model_name}_{method}.csv"
        df = pd.read_csv(prediction_path)
        folds = sorted(df["fold"].unique())
        for j, fold in enumerate(folds):
            df_fold = df.loc[df["fold"] == fold]
            sns.scatterplot(
                data=df_fold,
                x="y_pred",
                y="y",
                color=COLORS[j],
                ax=ax[i, j],
            )
            # ax[i, j].set_title(f"Method: {method} (fold {fold})")
            if i == 0:
                ax[i, j].set_title(f"Test fold {fold}")
            if i == (len(methods) - 1):
                ax[i, j].set_xlabel("Predicted score")
            if j == 0:
                ax[i, j].set_ylabel(f"{method}", fontsize=12)
            # ax[i, j].set_aspect("equal", "box")

    plt.suptitle(f"{model_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_dir / f"pred_vs_true_{model_name}.png", dpi=125)


def plot_reliability_diagram(
    datasets: list,
    folds: list,
    methods: list,
    model_name: str,
    fig_dir: Path,
    prediction_dir: Path,
):
    number_quantiles = 10

    fig, ax = plt.subplots(
        2, len(methods), figsize=(6.5, 5), sharex="all", sharey="all"
    )

    for i, method in enumerate(methods):
        df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")
        df_combined = pd.DataFrame(
            columns=["percentile", "confidence", "fold", "ECE", "Sharpness"]
        )
        for j, fold in enumerate(folds):
            df_fold = df.loc[df["fold"] == fold]

            true = df_fold["y"].values
            preds = df_fold["y_pred"].values
            uncertainties = df_fold["y_var"].values

            perc = np.arange(0, 1.1, 1 / number_quantiles)
            count_arr = np.vstack(
                [
                    np.abs(true - preds)
                    <= stats.norm.interval(
                        q, loc=np.zeros(len(preds)), scale=np.sqrt(uncertainties)
                    )[1]
                    for q in perc
                ]
            )
            count = np.mean(count_arr, axis=1)
            ECE = np.mean(np.abs(count - perc))
            Sharpness = np.std(uncertainties, ddof=1) / np.mean(uncertainties)
            df_fold_ = pd.DataFrame(
                {
                    "percentile": perc,
                    "confidence": count,
                    "fold": fold,
                    "ECE": ECE,
                    "Sharpness": Sharpness,
                }
            )
            df_combined = pd.concat([df_combined, df_fold_])

        df_combined = df_combined.reset_index(drop=True)
        sns.scatterplot(
            data=df_combined,
            x="percentile",
            y="confidence",
            hue="fold",
            palette=COLORS,
            ax=ax[0, i],
            legend=False,
        )
        sns.lineplot(
            data=df_combined,
            x="percentile",
            y="confidence",
            hue="fold",
            palette=COLORS,
            ax=ax[0, i],
            legend=False,
        )

        sns.scatterplot(
            data=df_combined.groupby(["percentile"], as_index=False).agg(
                {"confidence": "mean"}
            ),
            x="percentile",
            y="confidence",
            # hue="fold",
            color=COLORS[i],
            ax=ax[1, i],
            legend=False,
        )
        sns.lineplot(
            data=df_combined,
            x="percentile",
            y="confidence",
            # hue="fold",
            color=COLORS[i],
            ax=ax[1, i],
            legend=False,
        )

        # Add box with EVE and sharpness value in top left corner of ax
        ax[1, i].text(
            0.95,
            0.05,
            f"ECE: {df_combined['ECE'].mean():.2f} (std: {df_combined['ECE'].std():.2f})\nSharpness: {df_combined['Sharpness'].mean():.2f} (std: {df_combined['Sharpness'].std():.2f})",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[1, i].transAxes,
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.5, pad=2),
        )

        ax[0, i].set_ylabel("Confidence")
        ax[0, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[0, i].set_yticks(np.arange(0, 1.1, 0.2))
        ax[0, i].set_title(f"{method}")

        ax[1, i].set_xlabel("Percentile")
        ax[1, i].set_ylabel("Confidence")
        ax[1, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[1, i].set_yticks(np.arange(0, 1.1, 0.2))

        # Add dotted black line from (0,0) to (1,1)
        ax[0, i].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[1, i].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[0, i].set_aspect("equal", "box")
        ax[1, i].set_aspect("equal", "box")

    plt.suptitle(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"rel_diag_{model_name}.png", dpi=125)


def plot_variance_histogram(
    datasets: list,
    folds: list,
    methods: list,
    model_name: str,
    fig_dir: Path,
    prediction_dir: Path,
):
    fig, ax = plt.subplots(2, 3, figsize=(3 * 2.5, 6), sharex="all", sharey="row")

    for i, method in enumerate(methods):
        df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")
        sns.histplot(
            data=df,
            x="y_var",
            hue="fold",
            palette=COLORS,
            ax=ax[0, i],
            bins=20,
            kde=True,
        )
        ax[0, i].set_title(f"{method}")
        ax[0, i].set_xlabel("Uncertainty")
        ax[0, i].set_ylabel("Count")

        sns.histplot(
            data=df,
            x="y_var",
            palette=COLORS[i],
            ax=ax[1, i],
            bins=20,
            kde=True,
        )
        ax[1, i].set_title(f"{method}")
        ax[1, i].set_xlabel("Variance")
        ax[1, i].set_ylabel("Count")

    plt.suptitle(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"var_histogram_{model_name}.png", dpi=125)


def plot_calibration_curve_fractions():
    df = pd.DataFrame()
    for i, method in enumerate(methods):
        df_ = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")
        df_[f"method"] = method
        df = pd.concat([df, df_])

    df = df.reset_index(drop=True)

    fractions = np.arange(0.1, 1.05, 0.1)
    df_calib = pd.DataFrame(columns=["fold", "method", "fraction", "Spearman", "MSE"])

    for fraction in fractions:
        for method in methods:
            for fold in folds:
                df_ = df[(df["method"] == method) & (df["fold"] == fold)]
                # Sort by y_var and keep top frac %
                df_ = df_.sort_values(["y_var"], ascending=True).head(
                    int(len(df_) * fraction)
                )
                spearman = stats.spearmanr(df_["y"], df_["y_pred"])[0]
                mse = np.mean((df_["y"] - df_["y_pred"]) ** 2)
                df_ = pd.DataFrame(
                    {
                        "fold": fold,
                        "method": method,
                        "fraction": fraction,
                        "Spearman": spearman,
                        "MSE": mse,
                    },
                    index=[0],
                )
                df_calib = pd.concat([df_calib, df_])

    # Mean over folds
    fig, ax = plt.subplots(
        2,
        len(methods),
        figsize=(3 * 2.5, 5),
        sharex="all",  # sharey="row"
    )
    df_calib = df_calib.reset_index(drop=True)

    for i, method in enumerate(methods):
        df_method = df_calib[df_calib["method"] == method]
        sns.scatterplot(
            data=df_method,
            x="fraction",
            y="Spearman",
            hue="fold",
            palette=COLORS,
            legend=False,
            ax=ax[0, i],
        )
        sns.lineplot(
            data=df_method,
            x="fraction",
            y="Spearman",
            hue="fold",
            palette=COLORS,
            legend=False,
            ax=ax[0, i],
        )
        sns.scatterplot(
            data=df_method,
            x="fraction",
            y="MSE",
            hue="fold",
            palette=COLORS,
            legend=False,
            ax=ax[1, i],
        )
        sns.lineplot(
            data=df_method,
            x="fraction",
            y="MSE",
            hue="fold",
            palette=COLORS,
            legend=False,
            ax=ax[1, i],
        )
        ax[0, i].set_title(f"{method}")
        ax[0, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[1, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[1, i].set_xlabel("Fraction")

    plt.suptitle(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"calibration_curve_{model_name}.png", dpi=125)


def plot_error_vs_variance(
    datasets, folds, methods, model_name, fig_dir, prediction_dir
):
    # Load data
    df = pd.DataFrame()
    for i, method in enumerate(methods):
        df_ = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")
        df_[f"method"] = method
        df = pd.concat([df, df_])

    df["abs_error"] = np.abs(df["y"] - df["y_pred"])

    fig, ax = plt.subplots(1, 3, figsize=(7, 3), sharey="all")

    for i, method in enumerate(methods):
        df_method = df[df["method"] == method]
        sns.scatterplot(
            data=df_method.sample(frac=1.0, random_state=42),
            x="y_var",
            y="abs_error",
            hue="fold",
            palette=COLORS,
            legend=False,
            ax=ax[i],
            s=16,
            alpha=0.7,
        )
        # Compute Spearmanr
        spearman = stats.spearmanr(df_method["y_var"], df_method["abs_error"])[0]
        pearson = stats.pearsonr(df_method["y_var"], df_method["abs_error"])[0]

        ax[i].text(
            0.95,
            0.05,
            f"Spearman: {spearman:.2f}\nPearson: {pearson:.2f}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[i].transAxes,
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.5, pad=2),
        )

        ax[i].set_title(f"{method}")
        ax[i].set_xlabel("Variance")
        ax[i].set_ylabel("Absolute error")

    plt.suptitle(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"error_vs_variance_{model_name}.png", dpi=125)


if __name__ == "__main__":
    datasets = [
        "BLAT_ECOLX_Stiffler_2015",
        "NPC1_HUMAN_Erwood_2022_RPE1",
        "DNJA1_HUMAN_Rocklin_2023_2LO1",
        "A0A1I9GEU1_NEIME_Kennouche_2019",
        "NUSG_MYCTU_Rocklin_2023_2MI6",
        "PA_I34A1_Wu_2015",
        "EPHB2_HUMAN_Rocklin_2023_1F0M",
        "VKOR1_HUMAN_Chiasson_2020_activity",
        "TCRG1_MOUSE_Rocklin_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
        # "R1AB_SARS2_Flynn_2022",
    ]
    for dataset in datasets:
        prediction_dir = Path("results/ProteinGym/predictions")
        fig_dir = Path("figures/ProteinGym/predictions") / dataset
        fig_dir.mkdir(parents=True, exist_ok=True)
        methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
        model_name = "kermut_ProteinMPNN_TranceptEVE_MSAT"
        folds = [0, 1, 2, 3, 4]

        ############################
        # Plot predictions vs true #
        ############################

        # plot_y_vs_y_pred(datasets, folds, methods, model_name, fig_dir, prediction_dir)

        ###########################
        # Compute scores          #
        ###########################

        # plot_reliability_diagram(
        # datasets, folds, methods, model_name, fig_dir, prediction_dir
        # )

        ###########################
        # Calibration curve       #
        ###########################

        # plot_calibration_curve_fractions(
        # datasets, folds, methods, model_name, fig_dir, prediction_dir
        # )

        ###########################
        # Variance histogram      #
        ###########################

        # plot_variance_histogram(
        # datasets, folds, methods, model_name, fig_dir, prediction_dir
        # )

        ###########################
        # Error vs variance       #
        ###########################

        plot_error_vs_variance(
            datasets, folds, methods, model_name, fig_dir, prediction_dir
        )
