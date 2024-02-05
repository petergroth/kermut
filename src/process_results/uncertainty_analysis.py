import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src import COLORS
from scipy import stats
import warnings
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

sns.set_style("whitegrid")


def plot_y_vs_y_pred(
    dataset: list,
    folds: list,
    methods: list,
    model_name: str,
    fig_dir: Path,
    prediction_dir: Path,
):
    if len(methods) > 1:
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
            if "train" in df:
                df = df[df["train"] == False]
            folds = sorted(df["fold"].unique())
            for j, fold in enumerate(folds):
                df_fold = df.loc[df["fold"] == fold]
                y_err = 2 * np.sqrt(df_fold["y_var"].values)
                # sns.scatterplot(
                # data=df_fold,
                # x="y_pred",
                # y="y",
                # color=COLORS[j],
                # ax=ax[i, j],
                # )
                ax[i, j].errorbar(
                    y=df_fold["y_pred"],
                    x=df_fold["y"],
                    fmt="o",
                    yerr=y_err,
                    ecolor=COLORS[i],
                    markerfacecolor=COLORS[i],
                    markeredgecolor="white",
                    capsize=2.5,
                )
                # Add dotted grey diagonal line
                y_min = min(df["y"].min(), df["y_pred"].min())
                y_max = max(df["y"].max(), df["y_pred"].max())

                ax[i, j].plot(
                    [y_min, y_max],
                    [y_min, y_max],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                    zorder=2.5,
                )
                if i == 0:
                    ax[i, j].set_title(f"Test fold {fold}")
                # if i == (len(methods) - 1):
                if j == 0:
                    ax[i, j].set_ylabel(f"{method}")

        plt.suptitle(f"{dataset} - {model_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_dir / f"pred_vs_true_{model_name}.png", dpi=125)
        plt.close()

    elif len(methods) == 1:
        fig, ax = plt.subplots(
            1,
            5,
            figsize=(12, 1.5 * 2.5),
            sharex="all",
            sharey="all",
        )
        method = methods[0]
        prediction_path = prediction_dir / dataset / f"{model_name}_{method}.csv"
        df = pd.read_csv(prediction_path)
        if "train" in df:
            df = df[df["train"] == False]
        folds = sorted(df["fold"].unique())
        for i, fold in enumerate(folds):
            df_fold = df.loc[df["fold"] == fold]
            y_err = np.sqrt(df_fold["y_var"].values)
            # sns.scatterplot(
            # data=df_fold,
            # x="y_pred",
            # y="y",
            # color=COLORS[j],
            # ax=ax[i, j],
            # )
            ax[i].errorbar(
                y=df_fold["y_pred"],
                x=df_fold["y"],
                fmt="o",
                yerr=y_err,
                ecolor=COLORS[i],
                markerfacecolor=COLORS[i],
                markeredgecolor="white",
                capsize=2.5,
            )
            # Add dotted grey diagonal line
            y_min = min(df["y"].min(), df["y_pred"].min())
            y_max = max(df["y"].max(), df["y_pred"].max())

            ax[i].plot(
                [y_min, y_max],
                [y_min, y_max],
                "k--",
                linewidth=1,
                alpha=0.5,
                zorder=2.5,
            )
            ax[i].set_title(f"Test fold {fold}")
            if i == 0:
                ax[i].set_ylabel(f"{method}")

        plt.suptitle(f"{dataset} - {model_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fig_dir / f"pred_vs_true_{method}_{model_name}.png", dpi=125)
        plt.close()


def plot_reliability_diagram(
    dataset: list,
    folds: list,
    methods: list,
    model_name: str,
    fig_dir: Path,
    prediction_dir: Path,
):
    plt.rcParams["font.family"] = "serif"

    number_quantiles = 10

    fig, ax = plt.subplots(2, 3, figsize=(6.5, 5), sharex="all", sharey="all")

    for i, method in enumerate(methods):
        df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")

        if "train" in df:
            df = df[df["train"] == False]
        df_combined = pd.DataFrame(
            columns=[
                "percentile",
                "confidence",
                "fold",
                "ECE",
                "Sharpness",
                "chi_2",
            ]
        )
        for j, fold in enumerate(folds):
            df_fold = df.loc[df["fold"] == fold]

            y_target = df_fold["y"].values
            y_pred = df_fold["y_pred"].values
            y_var = df_fold["y_var"].values

            perc = np.arange(0, 1.1, 1 / number_quantiles)
            count_arr = np.vstack(
                [
                    np.abs(y_target - y_pred)
                    <= stats.norm.interval(
                        q, loc=np.zeros(len(y_pred)), scale=np.sqrt(y_var)
                    )[1]
                    for q in perc
                ]
            )
            count = np.mean(count_arr, axis=1)

            # Compute calibration metrics
            ECE = np.mean(np.abs(count - perc))
            Sharpness = np.std(y_var, ddof=1) / np.mean(y_var)

            marginal_var = np.var(y_target - y_pred)
            dof = np.sum(np.cov(y_pred, y_target)) / marginal_var
            chi_2 = np.sum((y_target - y_pred) ** 2 / y_var) / (len(y_target) - 1 - dof)

            df_fold_ = pd.DataFrame(
                {
                    "percentile": perc,
                    "confidence": count,
                    "fold": fold,
                    "ECE": ECE,
                    "Sharpness": Sharpness,
                    "chi_2": chi_2,
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
            # f"ECE: {df_combined['ECE'].mean():.2f} (std: {df_combined['ECE'].std():.2f})\nSharpness: {df_combined['Sharpness'].mean():.2f} (std: {df_combined['Sharpness'].std():.2f})\nchi^2: {df_combined['chi_2'].mean():.2f} (std: {df_combined['chi_2'].std():.2f})",
            f"ECE: {df_combined['ECE'].mean():.2f} (std: {df_combined['ECE'].std():.2f})",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[1, i].transAxes,
            fontsize=7,
            bbox=dict(
                # boxstyle="round", facecolor="white", edgecolor="black", alpha=0.5, pad=2
                facecolor="white",
                edgecolor="black",
                alpha=0.5,
                pad=2,
            ),
        )

        ax[0, i].set_ylabel("Confidence", fontsize=10)
        ax[0, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[0, i].set_yticks(np.arange(0, 1.1, 0.2))
        ax[0, i].set_title(f"{method[5:-2].capitalize()}")

        ax[1, i].set_xlabel("Percentile", fontsize=10)
        ax[1, i].set_ylabel("Confidence", fontsize=10)
        ax[1, i].set_xticks(np.arange(0, 1.1, 0.2))
        ax[1, i].set_yticks(np.arange(0, 1.1, 0.2))

        # Add dotted black line from (0,0) to (1,1)
        ax[0, i].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[1, i].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[0, i].set_aspect("equal", "box")
        ax[1, i].set_aspect("equal", "box")

    # plt.suptitle(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"rel_diag_{model_name}.png", dpi=125)
    plt.close()


def plot_variance_histogram(
    dataset: list,
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
    plt.close()


def plot_calibration_curve_fractions(
    dataset: list,
    folds: list,
    methods: list,
    model_name: str,
    fig_dir: Path,
    prediction_dir: Path,
):
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
    plt.close()


def plot_error_vs_variance(
    dataset, folds, methods, model_name, fig_dir, prediction_dir
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
    plt.close()


def plot_error_based_calibration_curve(
    dataset, folds, methods, model_name, fig_dir, prediction_dir
):
    n_bins = 5
    df = pd.DataFrame()
    for i, method in enumerate(methods):
        df_ = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")
        if "train" in df_:
            df_ = df_[df_["train"] == False]
        df_[f"method"] = method
        df_["abs_error"] = np.abs(df_["y"] - df_["y_pred"])
        df_["sq_error"] = (df_["y"] - df_["y_pred"]) ** 2
        df_["bin"] = np.nan
        for fold in folds:
            df_fold = df_[(df_["fold"] == fold)]

            df_.loc[df_fold.index, "bin"] = pd.qcut(
                df_fold["y_var"], n_bins, labels=False, duplicates="drop"
            )

        df = pd.concat([df, df_])
    df["bin"] = df["bin"].astype(int)
    df = df.reset_index(drop=True)
    df_avg = df.groupby(["method", "bin", "fold"], as_index=False).mean(
        numeric_only=True
    )
    df_avg["RMSE"] = np.sqrt(df_avg["sq_error"])
    df_avg["rmv"] = np.sqrt(df_avg["y_var"])

    fig, ax = plt.subplots(2, 3, figsize=(6.5, 5))

    for i, method in enumerate(methods):
        df_method = df_avg[df_avg["method"] == method]
        sns.scatterplot(
            data=df_method,
            x="rmv",
            y="RMSE",
            hue="fold",
            palette=COLORS,
            legend=False,
            ax=ax[0, i],
        )
        sns.lineplot(
            data=df_method,
            x="rmv",
            y="RMSE",
            hue="fold",
            palette=COLORS,
            legend=False,
            ax=ax[0, i],
        )
        ax[0, i].set_title(f"{method}")
        ax[0, i].set_xlabel("RMV")
        if i == 0:
            ax[0, i].set_ylabel("RMSE")
        else:
            ax[0, i].set_ylabel("")
            
        # Group by bin abd compute mean and std
        df_bin = df_method.groupby(["bin"], as_index=False).mean(numeric_only=True)
        df_bin["RMSE_std"] = df_method.groupby(["bin"], as_index=False).std(
            numeric_only=True
        )["RMSE"]
        df_bin["rmv_std"] = df_method.groupby(["bin"], as_index=False).std(
            numeric_only=True
        )["rmv"]

        ax[1, i].errorbar(
            y=df_bin["RMSE"],
            x=df_bin["rmv"],
            fmt="o",
            xerr=df_bin["rmv_std"],
            yerr=df_bin["RMSE_std"],
            ecolor=COLORS[i],
            markerfacecolor=COLORS[i],
            markeredgecolor="white",
            capsize=2.5,
        )

        # ax[i].set_title(f"{method}")
        ax[1, i].set_xlabel("RMV")
        if i == 0:
            ax[1, i].set_ylabel("RMSE")
        else:
            ax[1, i].set_ylabel("")

        # Compute ENCE
        ence_avg = (
            df_method.groupby(["fold"])
            .apply(lambda x: np.mean(np.abs(x["rmv"] - x["RMSE"]) / x["rmv"]))
            .mean()
        )
        ence_std = (
            df_method.groupby(["fold"])
            .apply(lambda x: np.mean(np.abs(x["rmv"] - x["RMSE"]) / x["rmv"]))
            .std()
        )

        ax[1, i].text(
            0.95,
            0.05,
            f"ENCE: {ence_avg:.2f} (std: {ence_std:.2f})",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[1, i].transAxes,
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.5, pad=2),
        )

        # Dotted diagonal line
        y_min, y_max = ax[1, i].get_ylim()
        ax[1, i].plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1, alpha=0.5)

        # Equal axes
        # ax[1, i].set_aspect("equal", "box")

    plt.suptitle(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / f"error_based_calibration_{model_name}.png", dpi=125)
    plt.close()


def plot_error_based_calibration_curve_multiples(dataset, model_name):
    fig_dir = Path("figures/ProteinGym/predictions") / dataset
    prediction_dir = Path("results/ProteinGym/predictions_multiples")

    n_bins = 5
    df = pd.read_csv(prediction_dir / dataset / f"{model_name}.csv")
    if "train" in df:
        df = df[df["train"] == False]
    df["abs_error"] = np.abs(df["y"] - df["y_pred"])
    df["sq_error"] = (df["y"] - df["y_pred"]) ** 2
    df["bin"] = np.nan
    df.loc[df.index, "bin"] = pd.qcut(
        df["y_var"], n_bins, labels=False, duplicates="drop"
    )

    df["bin"] = df["bin"].astype(int)
    df = df.reset_index(drop=True)
    df_avg = df.groupby(["bin"], as_index=False).mean(numeric_only=True)
    df_avg["RMSE"] = np.sqrt(df_avg["sq_error"])
    df_avg["RMV"] = np.sqrt(df_avg["y_var"])

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Compute ENCE
    ence = np.mean(np.abs(df_avg["RMV"] - df_avg["RMSE"]) / df_avg["RMV"])

    sns.scatterplot(
        data=df_avg,
        x="RMV",
        y="RMSE",
        color=COLORS[0],
        legend=False,
        ax=ax,
    )
    sns.lineplot(
        data=df_avg,
        x="RMV",
        y="RMSE",
        color=COLORS[0],
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Multiples")
    ax.set_xlabel("RMV")
    ax.set_ylabel("RMSE")

    ax.text(
        0.95,
        0.05,
        f"ENCE: {ence:.2f}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.5, pad=2),
    )

    # Dotted diagonal line
    y_min, y_max = ax.get_ylim()
    ax.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1, alpha=0.5)

    # Equal axes
    ax.set_aspect("equal", "box")

    plt.title(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        fig_dir / f"multiples_error_based_calibration_{model_name}.png", dpi=125
    )
    plt.close()


def plot_reliability_diagram_summary(
    dataset: list,
    folds: list,
    methods: list,
    model_name: str,
    prediction_dir: Path,
):
    plt.rcParams["font.family"] = "serif"
    number_quantiles = 10
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.6), sharex="all", sharey="all")

    for i, method in enumerate(methods):
        df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")

        if "train" in df:
            df = df[df["train"] == False]
        df_combined = pd.DataFrame(
            columns=[
                "percentile",
                "confidence",
                "fold",
                "ECE",
                "Sharpness",
                "chi_2",
            ]
        )
        for j, fold in enumerate(folds):
            df_fold = df.loc[df["fold"] == fold]

            y_target = df_fold["y"].values
            y_pred = df_fold["y_pred"].values
            y_var = df_fold["y_var"].values

            perc = np.arange(0, 1.1, 1 / number_quantiles)
            count_arr = np.vstack(
                [
                    np.abs(y_target - y_pred)
                    <= stats.norm.interval(
                        q, loc=np.zeros(len(y_pred)), scale=np.sqrt(y_var)
                    )[1]
                    for q in perc
                ]
            )
            count = np.mean(count_arr, axis=1)

            # Compute calibration metrics
            ECE = np.mean(np.abs(count - perc))
            Sharpness = np.std(y_var, ddof=1) / np.mean(y_var)

            marginal_var = np.var(y_target - y_pred)
            dof = np.sum(np.cov(y_pred, y_target)) / marginal_var
            chi_2 = np.sum((y_target - y_pred) ** 2 / y_var) / (len(y_target) - 1 - dof)

            df_fold_ = pd.DataFrame(
                {
                    "percentile": perc,
                    "confidence": count,
                    "fold": fold,
                    "ECE": ECE,
                    "Sharpness": Sharpness,
                    "chi_2": chi_2,
                }
            )
            df_combined = pd.concat([df_combined, df_fold_])

        df_combined = df_combined.reset_index(drop=True)
        sns.scatterplot(
            data=df_combined.groupby(["percentile"], as_index=False).agg(
                {"confidence": "mean"}
            ),
            x="percentile",
            y="confidence",
            # hue="fold",
            color=COLORS[i],
            ax=ax[i],
            legend=False,
        )
        sns.lineplot(
            data=df_combined,
            x="percentile",
            y="confidence",
            # hue="fold",
            color=COLORS[i],
            ax=ax[i],
            legend=False,
        )

        # Add box with EVE and sharpness value in top left corner of ax
        ax[i].text(
            0.95,
            0.05,
            # f"ECE: {df_combined['ECE'].mean():.2f} (std: {df_combined['ECE'].std():.2f})\nSharpness: {df_combined['Sharpness'].mean():.2f} (std: {df_combined['Sharpness'].std():.2f})\nchi^2: {df_combined['chi_2'].mean():.2f} (std: {df_combined['chi_2'].std():.2f})",
            f"ECE: {df_combined['ECE'].mean():.2f} (±{2*df_combined['ECE'].std():.2f})",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax[i].transAxes,
            fontsize=9,
            bbox=dict(
                # boxstyle="round", facecolor="white", edgecolor="black", alpha=0.5, pad=2
                facecolor="white",
                edgecolor="black",
                # alpha=0.5,
                pad=3,
            ),
        )

        ax[i].set_xlabel("Percentile", fontsize=10)
        ax[i].set_ylabel("Confidence", fontsize=10)
        ax[i].set_xticks(np.arange(0, 1.1, 0.2))

        ax[i].set_yticks(np.arange(0, 1.1, 0.2))
        ax[i].set_title(f"{method[5:-2].capitalize()}")
        ax[i].plot([0, 1], [0, 1], "k--", linewidth=1)
        ax[i].set_aspect("equal", "box")

    # plt.suptitle(f"{dataset}\n{model_name}", fontsize=12)
    plt.tight_layout()
    fig_dir = Path("figures/calibration/reliability")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"{dataset}_rel_diag_{model_name}_summary.pdf", dpi=125)
    plt.close()


def reliability_diagram_custom():
    plt.rcParams["font.family"] = "serif"
    datasets = [
        "BLAT_ECOLX_Stiffler_2015",
        "PA_I34A1_Wu_2015",
        "TCRG1_MOUSE_Rocklin_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
    ]
    prediction_dir = Path("results/ProteinGym/predictions")
    fig_dir = Path("figures/calibration/reliability")
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    folds = [0, 1, 2, 3, 4]
    number_quantiles = 10
    model_name = "kermut_distance_no_blosum"

    fig, ax = plt.subplots(4, 3, figsize=(6.5, 8.5), sharex="all", sharey="all")

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")
            df_combined = pd.DataFrame(
                columns=[
                    "percentile",
                    "confidence",
                    "fold",
                    "ECE",
                ]
            )
            for k, fold in enumerate(folds):
                df_fold = df.loc[df["fold"] == fold]
                y_target = df_fold["y"].values
                y_pred = df_fold["y_pred"].values
                y_var = df_fold["y_var"].values
                perc = np.arange(0, 1.1, 1 / number_quantiles)
                count_arr = np.vstack(
                    [
                        np.abs(y_target - y_pred)
                        <= stats.norm.interval(
                            q, loc=np.zeros(len(y_pred)), scale=np.sqrt(y_var)
                        )[1]
                        for q in perc
                    ]
                )
                count = np.mean(count_arr, axis=1)

                # Compute calibration metrics
                ECE = np.mean(np.abs(count - perc))

                df_fold_ = pd.DataFrame(
                    {
                        "percentile": perc,
                        "confidence": count,
                        "fold": fold,
                        "ECE": ECE,
                    }
                )
                df_combined = pd.concat([df_combined, df_fold_])

            df_combined = df_combined.reset_index(drop=True)
            sns.scatterplot(
                data=df_combined.groupby(["percentile"], as_index=False).agg(
                    {"confidence": "mean"}
                ),
                x="percentile",
                y="confidence",
                # hue="fold",
                color=COLORS[j],
                ax=ax[i, j],
                legend=False,
            )
            sns.lineplot(
                data=df_combined,
                x="percentile",
                y="confidence",
                # hue="fold",
                color=COLORS[j],
                ax=ax[i, j],
                legend=False,
                errorbar="sd",
            )

            # Add box with EVE and sharpness value in top left corner of ax
            ax[i, j].text(
                0.5,
                0.05,
                # f"ECE: {df_combined['ECE'].mean():.2f} (std: {df_combined['ECE'].std():.2f})\nSharpness: {df_combined['Sharpness'].mean():.2f} (std: {df_combined['Sharpness'].std():.2f})\nchi^2: {df_combined['chi_2'].mean():.2f} (std: {df_combined['chi_2'].std():.2f})",
                f"ECE: {df_combined['ECE'].mean():.2f} (±{2*df_combined['ECE'].std():.2f})",
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=ax[i, j].transAxes,
                fontsize=9,
                bbox=dict(
                    # boxstyle="round", facecolor="white", edgecolor="black", alpha=0.5, pad=2
                    facecolor="white",
                    edgecolor="black",
                    # alpha=0.5,
                    pad=3,
                ),
            )

            ax[i, j].set_xlim(0, 1)
            ax[i, j].set_ylim(0, 1)
            ax[i, j].plot([0, 1], [0, 1], "k--", linewidth=1)
            ax[i, j].set_aspect("equal", "box")

            if i == 0:
                ax[i, j].set_yticks(np.arange(0, 1.1, 0.2))
                ax[i, j].set_xticks(np.arange(0, 1.1, 0.2))
                ax[i, j].set_title(f"{method[5:-2].capitalize()}")
            if j == 0:
                ax[i, j].set_ylabel("Confidence", fontsize=10)
            # elif j == 2:
            # ax[i, j].set_ylabel(dataset, fontsize=10)
            # ax[i, j].yaxis.set_label_position("right")
            else:
                ax[i, j].set_ylabel("")

            if i == len(datasets) - 1:
                ax[i, j].set_xlabel("Percentile", fontsize=10)
            else:
                ax[i, j].set_xlabel("")

    # Reduce space between subplots
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.13, hspace=0.15)
    fig_dir = Path("figures/calibration/reliability")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"rel_diag_{model_name}_summary.png", dpi=125)
    plt.savefig(fig_dir / f"rel_diag_{model_name}_summary.pdf", dpi=125)
    plt.close()


def error_based_calibration_custom():
    plt.rcParams["font.family"] = "serif"
    datasets = [
        "BLAT_ECOLX_Stiffler_2015",
        "PA_I34A1_Wu_2015",
        "TCRG1_MOUSE_Rocklin_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
    ]
    prediction_dir = Path("results/ProteinGym/predictions")
    fig_dir = Path("figures/calibration/reliability")
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    folds = [0, 1, 2, 3, 4]
    n_bins = 5
    model_name = "kermut_distance_no_blosum"

    fig, ax = plt.subplots(4, 3, figsize=(6.5, 8.5), sharey="row", sharex="row")

    for i, dataset in enumerate(datasets):
        for j, method in enumerate(methods):
            df = pd.read_csv(prediction_dir / dataset / f"{model_name}_{method}.csv")
            df["sq_error"] = (df["y"] - df["y_pred"]) ** 2
            df["bin"] = np.nan

            for fold in folds:
                df_fold = df[(df["fold"] == fold)]

                df.loc[df_fold.index, "bin"] = pd.qcut(
                    df_fold["y_var"], n_bins, labels=False, duplicates="drop"
                )

            df["bin"] = df["bin"].astype(int)
            df_avg = df.groupby(["bin", "fold"], as_index=False).mean(numeric_only=True)
            df_avg["RMSE"] = np.sqrt(df_avg["sq_error"])
            df_avg["RMV"] = np.sqrt(df_avg["y_var"])
            df_ = df_avg.groupby(["bin"], as_index=False).mean(numeric_only=True)
            df_["RMSE_std"] = df_avg.groupby(["bin"], as_index=False).std(
                numeric_only=True
            )["RMSE"]
            df_["RMV_std"] = df_avg.groupby(["bin"], as_index=False).std(
                numeric_only=True
            )["RMV"]

            # Calculate ENCE from df_avg
            ence = (
                df_avg.groupby(["fold"])
                .apply(lambda x: np.mean(np.abs(x["RMV"] - x["RMSE"]) / x["RMV"]))
                .mean()
            )
            ence_sd = (
                df_avg.groupby(["fold"])
                .apply(lambda x: np.mean(np.abs(x["RMV"] - x["RMSE"]) / x["RMV"]))
                .std()
            )

            # Compute Coefficient of Variation
            cv = np.zeros(len(folds))
            df["y_std"] = np.sqrt(df["y_var"])
            for fold in folds:
                df_fold = df[df["fold"] == fold]
                mu_sig = np.mean(df_fold["y_std"])
                cv_ = np.sqrt(
                    np.sum((df_fold["y_std"] - mu_sig) ** 2 / (len(df_fold) - 1))
                    / mu_sig
                )
                cv[fold] = cv_
            cv_avg = np.mean(cv)
            cv_std = np.std(cv)

            ax[i, j].errorbar(
                y=df_["RMSE"],
                x=df_["RMV"],
                fmt="o",
                xerr=df_["RMV_std"],
                yerr=df_["RMSE_std"],
                ecolor=COLORS[j],
                markerfacecolor=COLORS[j],
                markeredgecolor="white",
                capsize=2.5,
            )

            if (i == 0) or (i == 3):
                y = 0.05
                verticalalignment = "bottom"
            elif (i == 1) or (i == 2):
                y = 0.95
                verticalalignment = "top"

            ax[i, j].text(
                x=0.875,
                y=y,
                s=f"ENCE: {ence:.2f} (±{2*ence:.2f})\n"
                + r"$c_v$"
                + f": {cv_avg:.2f} (±{2*cv_std:.2f})",
                horizontalalignment="right",
                verticalalignment=verticalalignment,
                transform=ax[i, j].transAxes,
                fontsize=9,
                bbox=dict(
                    # boxstyle="round", facecolor="white", edgecolor="black", alpha=0.5, pad=2
                    facecolor="white",
                    edgecolor="black",
                    # alpha=0.5,
                    pad=3,
                ),
            )

            # ax[i, j].set_aspect("equal", "box")

            # Equal axes
            if i in [0, 2]:
                ax[i, j].set_xlim(0.2, 0.9)
                ax[i, j].set_ylim(0.2, 0.9)
                ax[i, j].plot([0.2, 0.9], [0.2, 0.9], "k--", linewidth=1)
            elif i == 1:
                ax[i, j].plot([0.2, 0.8], [0.2, 0.8], "k--", linewidth=1)
                ax[i, j].set_xlim(0.35, 0.55)
                ax[i, j].set_ylim(0.2, 0.8)
            elif i == 3:
                ax[i, j].set_xlim(0.25, 1.55)
                ax[i, j].set_ylim(0.25, 1.55)
                # Add dashed diagonal line
                ax[i, j].plot([0.25, 1.55], [0.25, 1.55], "k--", linewidth=1)

            if i == 0:
                ax[i, j].set_title(f"{method[5:-2].capitalize()}")
            if j == 0:
                ax[i, j].set_ylabel("RMSE", fontsize=10)
            if i == len(datasets) - 1:
                ax[i, j].set_xlabel("RMV", fontsize=10)

    plt.tight_layout()
    # # Reduce space between subplots
    # plt.subplots_adjust(wspace=0.0)
    # plt.subplots_adjust(wspace=0.13, hspace=0.15)

    fig_dir = Path("figures/calibration/error_based")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f"error_based_{model_name}_summary.png", dpi=125)
    plt.savefig(fig_dir / f"error_based_{model_name}_summary.pdf")
    plt.close()


def y_true_vs_pred_custom():
    plt.rcParams["font.family"] = "serif"
    datasets = [
        "BLAT_ECOLX_Stiffler_2015",
        "PA_I34A1_Wu_2015",
        "TCRG1_MOUSE_Rocklin_2023_1E0L",
        "OPSD_HUMAN_Wan_2019",
    ]

    prediction_dir = Path("results/ProteinGym/predictions")
    fig_dir = Path("figures/calibration/predictions")
    fig_dir.mkdir(parents=True, exist_ok=True)
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    folds = [0, 1, 2, 3, 4]
    model_name = "kermut_distance_no_blosum"
    for dataset in datasets:
        fig, ax = plt.subplots(
            5,
            3,
            figsize=(6, 8),
            sharex="all",
            sharey="all",
        )

        for j, method in enumerate(methods):
            prediction_path = prediction_dir / dataset / f"{model_name}_{method}.csv"
            df = pd.read_csv(prediction_path)
            for i, fold in enumerate(folds):
                df_fold = df.loc[df["fold"] == fold]
                y_err = 2 * np.sqrt(df_fold["y_var"].values)
                ax[i, j].errorbar(
                    y=df_fold["y_pred"],
                    x=df_fold["y"],
                    fmt="o",
                    yerr=y_err,
                    ecolor=COLORS[j],
                    markerfacecolor=COLORS[j],
                    markeredgecolor="white",
                    capsize=2.5,
                    markersize=3.5,
                    alpha=0.7,
                )
                # Add dotted grey diagonal line
                y_min = min(df["y"].min(), df["y_pred"].min())
                y_max = max(df["y"].max(), df["y_pred"].max())

                ax[i, j].plot(
                    [y_min, y_max],
                    [y_min, y_max],
                    "k--",
                    linewidth=1,
                    alpha=0.5,
                    zorder=2.5,
                )
                if i == 0:
                    ax[i, j].set_title(f"{method[5:-2].capitalize()}", fontsize=10)
                # if i == (len(methods) - 1):
                if j == 0:
                    ax[i, j].set_ylabel(f"Prediction (fold {fold+1})", fontsize=10)
                if i == (len(folds) - 1):
                    ax[i, j].set_xlabel("Target", fontsize=10)
                # ax[i, j].set_aspect("equal", "box")
                # Set x and y limits
                # ax[i, j].set_xlim(y_min, y_max)
                # ax[i, j].set_ylim(y_min, y_max)

        # plt.suptitle(f"{dataset} - {model_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(
            fig_dir / f"{dataset}_pred_vs_true_{model_name}.png",
            dpi=125,
            bbox_inches="tight",
        )
        plt.savefig(
            fig_dir / f"{dataset}_pred_vs_true_{model_name}.pdf", bbox_inches="tight"
        )
        plt.close()


def compute_calibration_metrics():
    df_ref = pd.read_csv("data/processed/DMS_substitutions_reduced.csv")
    prediction_dir = Path("results/ProteinGym/predictions")
    folds = [0, 1, 2, 3, 4]
    model_name = "kermut_distance_no_blosum"
    methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    df_ece = pd.DataFrame(columns=["DMS_id", "fold_variable_name", "fold", "model_name", "ECE"])
    df_ence = pd.DataFrame(columns=["DMS_id", "fold_variable_name", "fold", "model_name", "ENCE"])
    
    num_quantiles = 10
    n_bins = 5
    perc = np.arange(0, 1.1, 1 / num_quantiles)
    
    for dataset in tqdm(df_ref["DMS_id"]):
        for method in methods:
            prediction_path = prediction_dir / dataset / f"{model_name}_{method}.csv"
            df_ = pd.read_csv(prediction_path)
            
            # ECE
            df_metrics = pd.DataFrame(columns=["DMS_id", "fold_variable_name", "fold", "model_name", "ECE"])
            for k, fold in enumerate(folds):
                df_fold = df_.loc[df_["fold"] == fold]
                y_target = df_fold["y"].values
                y_pred = df_fold["y_pred"].values
                y_var = df_fold["y_var"].values
                count_arr = np.vstack(
                    [
                        np.abs(y_target - y_pred)
                        <= stats.norm.interval(
                            q, loc=np.zeros(len(y_pred)), scale=np.sqrt(y_var)
                        )[1]
                        for q in perc
                    ]
                )
                count = np.mean(count_arr, axis=1)

                # Compute calibration metrics
                ECE = np.mean(np.abs(count - perc))
                df_metrics.loc[k] = [dataset, method, fold, model_name, ECE]
                df_ece= pd.concat([df_ece, df_metrics])
            
            # ENCE
            df_ = pd.read_csv(prediction_path)
            df_["y_std"] = np.sqrt(df_["y_var"])
            df_["sq_error"] = (df_["y"] - df_["y_pred"]) ** 2
            df_["bin"] = np.nan
            cv = np.zeros(len(folds))
            for fold in folds:
                df_fold = df_[(df_["fold"] == fold)]
                df_.loc[df_fold.index, "bin"] = pd.qcut(
                    df_fold["y_var"], n_bins, labels=False, duplicates="drop"
                )
                mu_sig = np.mean(df_fold["y_std"])
                cv_ = np.sqrt(
                    np.sum((df_fold["y_std"] - mu_sig) ** 2 / (len(df_fold) - 1))
                    / mu_sig
                )
                cv[fold] = cv_

            df_["bin"] = df_["bin"].astype(int)
            df_avg = df_.groupby(["bin", "fold"], as_index=False).mean(numeric_only=True)
            df_avg["RMSE"] = np.sqrt(df_avg["sq_error"])
            df_avg["RMV"] = np.sqrt(df_avg["y_var"])
            
            ence = (
                df_avg.groupby(["fold"])
                .apply(lambda x: np.mean(np.abs(x["RMV"] - x["RMSE"]) / x["RMV"]))
            )
            
            df_metrics = pd.DataFrame(columns=["DMS_id", "fold_variable_name", "fold", "model_name", "ENCE", "cv"])
            df_metrics["ENCE"] = ence
            df_metrics["fold"] = folds
            df_metrics["cv"] = cv
            df_metrics["DMS_id"] = dataset
            df_metrics["fold_variable_name"] = method
            df_metrics["model_name"] = model_name
            
            df_ence = pd.concat([df_ence, df_metrics])

    calibration_dir = Path("results/ProteinGym/calibration_metrics")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    df_ece.to_csv(calibration_dir / "ECE.csv", index=False)
    df_ence.to_csv(calibration_dir / "ENCE.csv", index=False)
            
            
def summarize_calibration_metrics():
    calibration_dir = Path("results/ProteinGym/calibration_metrics")
    df_ece = pd.read_csv(calibration_dir / "ECE.csv")
    df_ence = pd.read_csv(calibration_dir / "ENCE.csv")
    
    # Drop B2L11_HUMAN_Dutta_2010_binding-Mcl-1 from both
    df_ece = df_ece[df_ece["DMS_id"] != "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"]
    df_ence = df_ence[df_ence["DMS_id"] != "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"]
    
    df_ece = df_ece.groupby(["DMS_id", "model_name", "fold_variable_name"], as_index=False).mean(numeric_only=True)
    df_ence = df_ence.groupby(["DMS_id", "model_name", "fold_variable_name"], as_index=False).mean(numeric_only=True)
    
    # Group by model_name and fold_variable_name and compute mean and std
    df_ece = df_ece.groupby(["model_name", "fold_variable_name"], as_index=False).agg(
        {"ECE": ["mean", "std"]}
    )
    df_ence = df_ence.groupby(["model_name", "fold_variable_name"], as_index=False).agg(
        {"ENCE": ["mean", "std"],
        "cv": ["mean", "std"]}
    )
    
    df_ece.to_csv(calibration_dir / "ECE_summary.csv", index=False)
    df_ence.to_csv(calibration_dir / "ENCE_summary.csv", index=False)
    
    # Round to 3 decimals and print
    df_ece = df_ece.round(3)
    df_ence = df_ence.round(3)
    print(df_ece)
    print(df_ence)
    
                
    # Drop Unnamed: 0 col from df_ece
    # df_ece = df_ece.drop(columns=["Unnamed: 0"])
    # df_ece.to_csv(calibration_dir / "ECE.csv", index=False)
        
    


if __name__ == "__main__":
    # compute_calibration_metrics()
    summarize_calibration_metrics()
    # reliability_diagram_custom()
    # error_based_calibration_custom()
    # y_true_vs_pred_custom()
    # datasets = [
    #     "BLAT_ECOLX_Stiffler_2015",
    #     "NPC1_HUMAN_Erwood_2022_RPE1",
    #     "DNJA1_HUMAN_Rocklin_2023_2LO1",
    #     "A0A1I9GEU1_NEIME_Kennouche_2019",
    #     "NUSG_MYCTU_Rocklin_2023_2MI6",
    #     "PA_I34A1_Wu_2015",
    #     "EPHB2_HUMAN_Rocklin_2023_1F0M",
    #     "VKOR1_HUMAN_Chiasson_2020_activity",
    #     "TCRG1_MOUSE_Rocklin_2023_1E0L",
    #     "OPSD_HUMAN_Wan_2019",
    # ]
    # for dataset in datasets:
    #     prediction_dir = Path("results/ProteinGym/predictions")
    #     fig_dir = Path("figures/ProteinGym/predictions") / dataset
    #     fig_dir.mkdir(parents=True, exist_ok=True)
    #     # methods = [
    #     # "fold_random_5",
    #     # ]
    #     methods = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
    #     folds = [0, 1, 2, 3, 4]
    #     models = [
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT_1.0",
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT_0.1",
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT_0.01",
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT_0.001",
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT_HalfCauchy_0.1",
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT_HalfCauchy_0.01",
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT_HalfCauchy_0.001",
    #         # "kermut_ProteinMPNN_TranceptEVE_MSAT",
    #         # "MSAT_RBF",
    #         # "kermut_flatten",
    #         # "kermut_no_rbf",
    #         # "kermut_reduced",
    #         # "kermut_distance",
    #         "kermut_distance_no_blosum",
    #     ]

    #     for model_name in models:
    #         ############################
    #         # Plot predictions vs true #
    #         ############################

    #         # plot_y_vs_y_pred(
    #         # dataset, folds, methods, model_name, fig_dir, prediction_dir
    #         # )

    #         ###########################
    #         # Reliability diagram     #
    #         ###########################

    #         # plot_reliability_diagram(
    #         # dataset, folds, methods, model_name, fig_dir, prediction_dir
    #         # )

    #         ###########################
    #         # Calibration curve       #
    #         ###########################

    #         # plot_calibration_curve_fractions(
    #         # dataset, folds, methods, model_name, fig_dir, prediction_dir
    #         # )

    #         ###########################
    #         # Variance histogram      #
    #         ###########################

    #         # plot_variance_histogram(
    #         # dataset, folds, methods, model_name, fig_dir, prediction_dir
    #         # )

    #         ###########################
    #         # Error vs variance       #
    #         ###########################

    #         # plot_error_vs_variance(
    #         # dataset, folds, methods, model_name, fig_dir, prediction_dir
    #         # )

    #         ###########################
    #         # Error based calibration #
    #         ###########################

    #         # plot_error_based_calibration_curve(
    #         # dataset, folds, methods, model_name, fig_dir, prediction_dir
    #         # )

    #         ###########################
    #         # Error based calibration #
    #         ###########################

    #         # plot_error_based_calibration_curve_multiples(dataset, model_name)

    #         plot_reliability_diagram_summary(
    #             dataset, folds, methods, model_name, prediction_dir
    #         )
