from pathlib import Path

import gpytorch
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import trange

from src.data.utils import load_split_regression_data
from src.model.utils import load_conditional_probs


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
        f"{cfg.n_train}_samples_mean_prediction_{cfg.split}.tsv",
    )

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

        # Prepare data
        y_train = df_train["delta_fitness"].values
        y_test = df_test["delta_fitness"].values
        y_test_pred = np.full(y_test.shape, np.mean(y_train))

        test_mse = mean_squared_error(y_test_pred, y_test)
        test_spearman = spearmanr(y_test_pred, y_test)[0]
        test_r2 = r2_score(y_test_pred, y_test)
        test_pearson = pearsonr(y_test_pred, y_test)[0]

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
