import ast
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src import AA_TO_IDX


def one_hot_encode(df: pd.DataFrame):
    """One-hot encoding mutations.

    Each position with a mutation is represented by a 20-dimensional vector, regardless of whether each
    mutation is actually observed.

    Args:
        df (pd.DataFrame): Dataset with list of mutations in the `mut2wt` column.

    Returns:
        np.ndarray: One-hot encoded mutations (shape: (n_samples, n_mutated_positions * 20)).
    """
    df["mut2wt"] = df["mut2wt"].apply(ast.literal_eval)
    mutated_positions = df["mut2wt"].explode().str[1:-1].astype(int).unique()
    mutated_positions = np.sort(mutated_positions)
    one_hot = np.zeros((len(df), len(mutated_positions), 20))
    pos_to_idx = {pos: i for i, pos in enumerate(mutated_positions)}
    for i, mut2wt in enumerate(df["mut2wt"]):
        for mut in mut2wt:
            pos = int(mut[1:-1])
            aa = mut[-1]
            one_hot[i, pos_to_idx[pos], AA_TO_IDX[aa]] = 1.0
    one_hot = one_hot.reshape(len(df), 20 * len(mutated_positions))
    return one_hot


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="coves",
)
def run_coves(cfg: DictConfig) -> None:
    """Run CoVES regressor.

    Run 5-fold cross-validation using one-hot encoded mutations as input representation. Predictor is a
    linear regressor with L2 regularization with alpha=1.0.

    """

    dataset = cfg.dataset
    data_path = Path("data/processed", f"{dataset}.tsv")
    out_path = Path("data/interim", dataset, f"{dataset}_coves_results.tsv")
    df = pd.read_csv(data_path, sep="\t")
    df = df.sample(cfg.n_samples, random_state=cfg.sample_seed)

    X = one_hot_encode(df)
    y = df[cfg.target_col].values

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
    for i, seed in enumerate(tqdm(cfg.seeds)):
        # Split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
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
    run_coves()
