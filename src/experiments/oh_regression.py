"""NOT UPDATED. WILL NOT WORK."""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scipy
from omegaconf import DictConfig
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from tqdm import tqdm

from src.data.utils import (
    load_sampled_regression_data,
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
    out_path = Path(
        "results/regression",
        dataset,
        f"{cfg.experiment.n_train}_samples_{encoding}.tsv",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load full (filtered) dataset
    df = load_sampled_regression_data(cfg)

    if encoding == "oh_mut":
        x_oh = one_hot_encode_mutation(df)
    elif encoding == "oh_seq":
        x_oh = one_hot_encode_sequence(df)
    elif encoding == "mean_prediction":
        x_oh = None
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

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
        # Split data into train/val and test
        df_train_val, df_test = train_test_split(
            df, test_size=len(df) - cfg.experiment.n_train, random_state=seed
        )
        y_train = df_train_val["delta_fitness"].values
        y_test = df_test["delta_fitness"].values

        if x_oh is not None:
            # Model selection
            model = Ridge()
            random_search = RandomizedSearchCV(
                model,
                param_distributions={"alpha": scipy.stats.loguniform(1e-3, 5)},
                n_iter=cfg.n_steps,
                cv=5,
            )
            x_train = x_oh[df_train_val.index.values]
            x_test = x_oh[df_test.index.values]
            random_search.fit(x_train, y_train)

            # Fit on full train, predict on test
            model = Ridge(**random_search.best_params_)
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
        else:
            # Mean prediction
            y_train_pred = np.full(y_train.shape, np.mean(y_train))
            y_test_pred = np.full(y_test.shape, np.mean(y_train))

        # Metrics
        train_mse = mean_squared_error(y_train_pred, y_train)
        test_mse = mean_squared_error(y_test_pred, y_test)
        train_spearman = spearmanr(y_train_pred, y_train)[0]
        test_spearman = spearmanr(y_test_pred, y_test)[0]
        train_r2 = r2_score(y_train_pred, y_train)
        test_r2 = r2_score(y_test_pred, y_test)
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
