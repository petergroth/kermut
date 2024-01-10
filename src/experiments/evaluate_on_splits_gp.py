from pathlib import Path

import gpytorch
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import trange

from src.data.utils import load_split_regression_data, load_sampled_regression_data


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluate_splits",
)
def main(cfg: DictConfig):
    dataset = cfg.dataset

    out_path = Path(
        "results/split",
        dataset,
        f"{cfg.n_train}_samples_gp_{cfg.gp.name}_{cfg.split}.tsv",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "oh_seq" in cfg.gp.name:
        tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer)
        seq = True
    elif "oh_mut" in cfg.gp.name:
        df = pd.read_csv(f"data/processed/{dataset}.tsv", sep="\t")
        tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer, df["mut2wt"])
        del df
        seq = False
    else:
        raise ValueError

    df_results = pd.DataFrame(
        columns=[
            "seed",
            "test_mse",
            "test_spearman",
            "test_r2",
            "test_pearson",
        ]
    )

    for i in trange(len(cfg.seeds)):
        df_train, df_test = load_split_regression_data(cfg, i)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # Prepare data
        y_train = df_train["delta_fitness"].values
        y_test = df_test["delta_fitness"].values
        if seq:
            train_seq = df_train["seq"].values
            test_seq = df_test["seq"].values
        else:
            train_seq = df_train["mut2wt"]
            test_seq = df_test["mut2wt"]
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        x_train = tokenizer(train_seq).squeeze()
        x_test = tokenizer(test_seq).squeeze()

        # Setup model
        model = hydra.utils.instantiate(
            cfg.gp.model,
            train_x=x_train,
            train_y=y_train,
            likelihood=likelihood,
            **cfg.gp.kernel_params,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gp.optim.lr)

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
            # Predictive distribution
            y_preds_dist = likelihood(model(x_test))
            y_preds_mean = y_preds_dist.mean

            # Compute metrics
            y_preds_mean_np = y_preds_mean.detach().numpy()
            y_test_np = y_test.detach().numpy()

            test_mse = mean_squared_error(y_test_np, y_preds_mean_np)
            test_spearman = spearmanr(y_test_np, y_preds_mean_np)[0]
            test_r2 = r2_score(y_test_np, y_preds_mean_np)
            test_pearson = pearsonr(y_test_np, y_preds_mean_np)[0]

            df_results.loc[i] = [
                cfg.seeds[i],
                test_mse,
                test_spearman,
                test_r2,
                test_pearson,
            ]

    df_results.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()