import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import gpytorch
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr

# from sklearn.model_selection import train_test_split
from tqdm import trange

sns.set_style("darkgrid")

from src import COLORS
from src.data.utils import load_sampled_regression_data, load_regression_data
from src.model.gp import ExactGPKermut
from src.model.utils import load_conditional_probs


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_gp",
)
def main(cfg: DictConfig) -> None:
    # Print config
    print(OmegaConf.to_yaml(cfg))
    dataset = cfg.experiment.dataset

    # Prepare model args
    if "kermut" in cfg.gp.name:
        tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer)
        conditional_probs_method = cfg.gp.conditional_probs_method
        wt_df = pd.read_csv("data/processed/wt_sequences.tsv", sep="\t")
        wt_sequence = wt_df.loc[wt_df["dataset"] == dataset, "seq"].item()
        wt_sequence = tokenizer(wt_sequence).squeeze()
        conditional_probs = load_conditional_probs(dataset, conditional_probs_method)
        kwargs = {"wt_sequence": wt_sequence, "conditional_probs": conditional_probs}

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    df = load_regression_data(cfg)

    df_train_val = df[df["n_muts"] == 1]
    df_train_val = df_train_val.sample(
        n=min(cfg.experiment.n_train, len(df_train_val)),
        random_state=cfg.experiment.seeds[0],
    ).reset_index(drop=True)

    df_test = df[df["n_muts"] == 2]

    if "oh_mut" in cfg.gp.name:
        tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer, df["mut2wt"])
        seq = False
    else:
        tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer)
        seq = True

    y_train = df_train_val["delta_fitness"].values
    y_train = torch.tensor(y_train, dtype=torch.float32)
    train_seq = df_train_val["seq"].values if seq else df_train_val["mut2wt"]
    x_train = tokenizer(train_seq).squeeze()

    y_test = df_test["delta_fitness"].values
    test_seq = df_test["seq"].values if seq else df_test["mut2wt"]
    y_test = torch.tensor(y_test, dtype=torch.float32)
    x_test = tokenizer(test_seq).squeeze()

    # Setup model
    if "kermut" in cfg.gp.name:
        model = ExactGPKermut(
            train_x=x_train,
            train_y=y_train,
            likelihood=likelihood,
            gp_cfg=cfg.gp,
            **kwargs,
        )
    else:
        model = hydra.utils.instantiate(
            cfg.gp.model,
            train_x=x_train,
            train_y=y_train,
            likelihood=likelihood,
            **cfg.gp.kernel_params,
        )

    # Setup training
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gp.optim.lr)

    # Train
    for _ in trange(cfg.gp.optim.n_steps):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate model
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        predictive = likelihood(model(x_test))
        covariances = predictive.covariance_matrix
        variances = covariances.diag().detach().numpy()

        y_preds = predictive.mean.detach().numpy()
        y_test = y_test.detach().numpy()

        corr = spearmanr(y_test, y_preds)[0]

        df_test["preds"] = y_preds

        fig, ax = plt.subplots(figsize=(7, 7))
        sns.scatterplot(
            x=y_test,
            y=y_preds,
            size=variances,
            ax=ax,
            color=COLORS[0],
            alpha=0.5,
            sizes=(10, 100),
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

        plt.tight_layout()
        # plt.savefig("target_vs_predictions.png", dpi=300)
        plt.show()

        # df_test_copy = df_test.copy()
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

        plt.suptitle("SPG1: Predictions vs. target", fontsize=20, weight="bold")
        plt.tight_layout()
        plt.savefig(
            f"SPG1_target_vs_predictions_additive_{cfg.gp.name}_{cfg.experiment.n_train}.png",
            dpi=300,
        )
        plt.show()
    #
    # df_results.loc[i] = [
    #     seed,
    #     test_mse,
    #     test_spearmann,
    #     test_r2,
    #     test_pearson,
    # ]


#
# df_results.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
