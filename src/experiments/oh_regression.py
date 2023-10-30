from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.utils import (
    load_regression_data,
    one_hot_encode_sequence,
    one_hot_encode_mutation,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="regression",
)
def run_cv(cfg: DictConfig) -> None:
    """Run regressor on one-hot encoded sequences.

    Run 5-fold cross-validation using one-hot encoded sequences as input representation. Predictor is a
    linear regressor with L2 regularization with alpha=1.0.

    """
    encoding = cfg.encoding
    dataset = cfg.experiment.dataset
    out_path = Path("results/regression", f"{dataset}_{encoding}_results.tsv")

    df = load_regression_data(cfg)
    x_oh = one_hot_encode_sequence(df)
    if encoding == "oh_mut":
        x_oh = one_hot_encode_mutation(df)
    elif encoding == "oh_seq":
        x_oh = one_hot_encode_sequence(df)

    df_results = pd.DataFrame(
        columns=[
            "seed",
            "train_mse",
            "test_mse",
            "train_spearman",
            "test_spearman",
            "train_r2",
            "test_r2",
            "train_pearson",
            "test_pearson",
        ]
    )

    for i, seed in enumerate(tqdm(cfg.experiment.seeds)):
        # Split data into train and test
        df_train, df_test = train_test_split(
            df, test_size=cfg.experiment.test_size, random_state=seed
        )

        y_train = df_train["delta_fitness"].values
        y_test = df_test["delta_fitness"].values
        x_train = x_oh[df_train.index.values]
        x_test = x_oh[df_test.index.values]

        # Train linear regressor
        model = Ridge()
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # Metrics
        train_mse = np.mean((y_train_pred - y_train) ** 2)
        test_mse = np.mean((y_test_pred - y_test) ** 2)
        train_spearman = spearmanr(y_train_pred, y_train)[0]
        test_spearman = spearmanr(y_test_pred, y_test)[0]
        train_r2 = model.score(x_train, y_train)
        test_r2 = model.score(x_test, y_test)
        train_pearson = pearsonr(y_train_pred, y_train)[0]
        test_pearson = pearsonr(y_test_pred, y_test)[0]

        df_results.loc[i] = [
            seed,
            train_mse,
            test_mse,
            train_spearman,
            test_spearman,
            train_r2,
            test_r2,
            train_pearson,
            test_pearson,
        ]

    df_results.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    run_cv()
