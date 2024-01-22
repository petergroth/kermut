from pathlib import Path

import gpytorch
import hydra
import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange
from torch.nn.functional import softplus

from src.model.gp import ExactGPModelRBF
from src.data.data_utils import load_zero_shot, zero_shot_name_to_col


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="proteingym_gp_regression",
)
def main(cfg: DictConfig) -> None:
    print(f"--- {cfg.dataset} - {cfg.split_method}---")
    dataset = cfg.dataset
    split_method = cfg.split_method
    progress_bar = cfg.progress_bar
    model_name = cfg.gp.name
    embedding_type = cfg.gp.embedding_type
    use_zero_shot = cfg.use_zero_shot if "use_zero_shot" in cfg else False

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load reference data
    ref_path = Path("data/processed/DMS_substitutions.csv")
    emb_path = (
        Path("data/embeddings/substitutions_singles/MSA_Transformer") / f"{dataset}.h5"
    )
    df_ref = pd.read_csv(ref_path)
    wt_df = df_ref.loc[df_ref["DMS_id"] == dataset]
    sequence_col = "mutated_sequence"
    target_col = "DMS_score"

    # Zero-shot score as mean function
    if use_zero_shot:
        zero_shot_method = cfg.zero_shot_method
        zero_shot_col = zero_shot_name_to_col(zero_shot_method)
        df_zero = load_zero_shot(dataset, zero_shot_method)
        model_name = f"{model_name}_{zero_shot_method}"

    if "custom_name" in cfg:
        model_name = cfg.custom_name

    out_path = Path(
        "results/ProteinGym/per_dataset", dataset, f"{model_name}_{split_method}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_singles = True

    if use_singles:
        dms_path = Path(
            f"data/processed/proteingym_cv_folds_singles_substitutions/{dataset}.csv"
        )
        split_method_col = split_method
    else:
        dms_path = Path(
            f"data/processed/proteingym_cv_folds_multiples_substitutions/{dataset}.csv"
        )
        split_method_col = "fold_rand_multiples"
        raise NotImplementedError

    # Initialize results dataframe
    df_results = pd.DataFrame(
        columns=[
            "fold",
            "MSE",
            "Spearman",
        ]
    )

    # Load data
    df = pd.read_csv(dms_path)
    if use_zero_shot:
        df = pd.merge(left=df, right=df_zero, on="mutant")
    df = df.reset_index(drop=True)

    if embedding_type == "oh_seq":
        tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer)
        x_full = df[sequence_col].values
        x_full = tokenizer(x_full).squeeze()
    elif embedding_type == "MSAT":
        with h5py.File(emb_path, "r") as h5f:
            embeddings = torch.tensor(h5f["embeddings"][:]).float()
            mutants = [x.decode("utf-8") for x in h5f["mutants"][:]]
        embeddings = embeddings.mean(dim=1)
        idx = [df["mutant"].tolist().index(x) for x in mutants]
        x_full = embeddings[idx]
    else:
        raise NotImplementedError

    if use_zero_shot:
        zero_shot_full = torch.tensor(df[zero_shot_col].values, dtype=torch.float32)
        x_full = torch.cat([x_full, zero_shot_full.unsqueeze(-1)], dim=-1)

    y_full = torch.tensor(df[target_col].values, dtype=torch.float32)

    unique_folds = df[split_method_col].unique()
    for i, test_fold in enumerate(tqdm(unique_folds)):
        train_idx = (df[split_method_col] != test_fold).tolist()
        test_idx = (df[split_method_col] == test_fold).tolist()
        x_train = x_full[train_idx]
        x_test = x_full[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModelRBF(
            train_x=x_train,
            train_y=y_train,
            likelihood=likelihood,
            use_zero_shot=use_zero_shot,
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.gp.optim.lr)

        for _ in trange(cfg.gp.optim.n_steps, disable=not progress_bar):
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

            df_results.loc[i] = [
                test_fold,
                test_mse,
                test_spearman,
            ]

    df_results["assay_id"] = dataset
    df_results["model_name"] = model_name
    df_results["fold_variable_name"] = split_method
    df_results.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
