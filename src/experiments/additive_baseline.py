import warnings

from collections import defaultdict

warnings.simplefilter(action="ignore", category=FutureWarning)
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
from scipy.stats import spearmanr
from ast import literal_eval
import numpy as np

# from sklearn.model_selection import train_test_split

sns.set_style("darkgrid")

from src import COLORS
from src.data.utils import load_regression_data
from pathlib import Path


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_gp",
)
def main(cfg: DictConfig) -> None:
    # Print config
    df = load_regression_data(cfg)
    fig_path = Path("figures")
    df_train = df[df["n_muts"] == 1]
    df_train = df_train.sample(
        n=min(cfg.experiment.n_train, len(df_train)),
        random_state=cfg.experiment.seeds[0],
    ).reset_index(drop=True)
    df_test = df[(df["n_muts"] <= cfg.n_muts) & (df["n_muts"] > 1)]
    y_test = df_test["delta_fitness"].values

    df_train = df_train[["delta_fitness", "mut2wt"]]
    df_train["mut2wt"] = df_train["mut2wt"].apply(literal_eval)
    df_train["mut2wt"] = df_train["mut2wt"].apply(lambda x: x[0])
    mut2fitness = defaultdict(lambda: 0.0)
    for i, row in df_train.iterrows():
        mut2fitness[row["mut2wt"]] = row["delta_fitness"]

    df_test["mut2wt"] = df_test["mut2wt"].apply(literal_eval)
    df_test["preds"] = np.NAN
    for i, row in df_test.iterrows():
        mut2wt = row["mut2wt"]
        additive_prediction = np.sum([mut2fitness[mut] for mut in mut2wt])
        df_test.loc[i, "preds"] = additive_prediction

    y_preds = df_test["preds"].values

    corr = spearmanr(y_test, y_preds)[0]

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        x=y_test,
        y=y_preds,
        ax=ax,
        color=COLORS[0],
        alpha=0.5,
        legend=False,
    )
    sns.scatterplot(x=y_test, y=y_preds, ax=ax, color=COLORS[0], s=10, linewidth=0)
    ax.set_xlim(min(y_test), max(y_test))
    ax.set_ylim(min(y_test), max(y_test))

    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")

    ax.text(
        0.05,
        0.95,
        f"Spearman = {corr:.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=COLORS[4], alpha=0.5),
        fontsize=12,
        weight="bold",
    )

    plt.suptitle(
        f"{cfg.experiment.dataset}: Predictions vs. target",
        fontsize=20,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        fig_path
        / f"additive_model_{cfg.experiment.dataset}_target_vs_predictions_{cfg.n_muts}_muts.png",
        dpi=250,
    )
    plt.show()

    if cfg.experiment.dataset == "SPG1":
        df_test = df_test.dropna(subset=["additive"])
        additive_corr = spearmanr(
            df_test.loc[df_test["additive"].astype(bool), "delta_fitness"],
            df_test.loc[df_test["additive"].astype(bool), "preds"],
        )[0]
        nonadditive_corr = spearmanr(
            df_test.loc[~df_test["additive"].astype(bool), "delta_fitness"],
            df_test.loc[~df_test["additive"].astype(bool), "preds"],
        )[0]
        fig, ax = plt.subplots(2, 1, figsize=(7, 10), sharey="col", sharex="col")
        sns.scatterplot(
            data=df_test[df_test["additive"]],
            x="delta_fitness",
            y="preds",
            ax=ax[0],
            color=COLORS[0],
            linewidth=0,
            alpha=0.8,
        )
        sns.scatterplot(
            data=df_test[df_test["additive"]],
            x="fitness_sum",
            y="preds",
            ax=ax[0],
            color=COLORS[4],
            linewidth=0,
            alpha=0.8,
        )
        ax[0].set_ylabel("Prediction")

        sns.scatterplot(
            data=df_test[~df_test["additive"].astype(bool)],
            x="delta_fitness",
            y="preds",
            ax=ax[1],
            color=COLORS[0],
            linewidth=0,
            alpha=0.8,
        )

        ax[0].set_title("Additive mutations (blue is true, orange is sum)")
        ax[1].set_title("Non-additive mutations")
        ax[1].set_xlabel("Target")
        ax[1].set_ylabel("Prediction")

        ax[0].text(
            0.5,
            0.05,
            f"Spearman = {additive_corr:.2f}",
            transform=ax[0].transAxes,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor=COLORS[0], alpha=0.5),
            fontsize=12,
            weight="bold",
        )
        ax[1].text(
            0.5,
            0.05,
            f"Spearman = {nonadditive_corr:.2f}",
            transform=ax[1].transAxes,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor=COLORS[0], alpha=0.5),
            fontsize=12,
            weight="bold",
        )
        # Add diagonal line dashed
        ax[0].plot(
            [min(df_test["delta_fitness"]), max(df_test["delta_fitness"])],
            [min(df_test["delta_fitness"]), max(df_test["delta_fitness"])],
            linestyle="--",
            color="black",
        )
        ax[1].plot(
            [min(df_test["delta_fitness"]), max(df_test["delta_fitness"])],
            [min(df_test["delta_fitness"]), max(df_test["delta_fitness"])],
            linestyle="--",
            color="black",
        )

        plt.suptitle(
            f"{cfg.experiment.dataset}: Predictions vs. target",
            fontsize=20,
            weight="bold",
        )
        plt.tight_layout()

        plt.savefig(
            fig_path
            / f"additive_model_{cfg.experiment.dataset}_target_vs_predictions_additivity_{cfg.experiment.n_train}_{cfg.n_muts}_muts.png",
            dpi=250,
        )

        plt.show()


#
# df_results.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
