from pathlib import Path

import gpytorch
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from src.data.utils import load_conditional_probs, load_regression_data


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="gp_regression",
)
def main(cfg: DictConfig) -> None:
    dataset = cfg.experiment.dataset
    conditional_probs_method = cfg.gp.misc.conditional_probs_method
    out_path = Path("results/regression", f"{dataset}_gp_results.tsv")

    # Load data
    wt_df = pd.read_csv("data/processed/wt_sequences.tsv", sep="\t")
    wt_sequence = wt_df.loc[wt_df["dataset"] == dataset, "seq"].item()
    conditional_probs = load_conditional_probs(dataset, conditional_probs_method)
    assert len(conditional_probs) == len(wt_sequence)

    # Prepare data
    df = load_regression_data(cfg)

    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    wt_sequence = tokenizer(wt_sequence).squeeze()
    kwargs = {"wt_sequence": wt_sequence, "conditional_probs": conditional_probs}
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    df_results = pd.DataFrame(
        columns=[
            "seed",
            "test_mse",
            "test_spearman",
            "test_r2",
            "test_pearson",
        ]
    )

    for i, seed in enumerate(tqdm(cfg.experiment.seeds)):
        df_train, df_test = train_test_split(
            df, test_size=cfg.experiment.test_size, random_state=seed
        )

        y_train = df_train["delta_fitness"].values
        y_test = df_test["delta_fitness"].values
        train_seq = df_train["seq"].values
        test_seq = df_test["seq"].values

        # Prepare data
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
            **kwargs,
        )

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
            test_spearmann = spearmanr(y_test_np, y_preds_mean_np)[0]
            test_r2 = r2_score(y_test_np, y_preds_mean_np)
            test_pearson = pearsonr(y_test_np, y_preds_mean_np)[0]

            df_results.loc[i] = [
                seed,
                test_mse,
                test_spearmann,
                test_r2,
                test_pearson,
            ]

    df_results.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
