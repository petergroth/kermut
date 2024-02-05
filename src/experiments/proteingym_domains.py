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

from src.model.gp import ExactGPKermut
from src.data.data_utils import (
    load_proteinmpnn_proteingym,
    load_esmif1_proteingym,
    load_zero_shot,
    zero_shot_name_to_col,
    load_embeddings,
    prepare_kwargs,
    get_model_name,
    load_proteingym_dataset,
    get_wt_df
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="proteingym_gp_regression",
)
def main(cfg: DictConfig) -> None:
    print(f"--- {cfg.dataset} ---")
    
    # Experiment settings
    dataset = cfg.dataset
    standardize = cfg.standardize
    progress_bar = cfg.progress_bar
    log_params = cfg.log_params
    log_predictions = cfg.log_predictions
    model_name = get_model_name(cfg)
    overwrite = cfg.overwrite if "overwrite" in cfg else False
    sequence_col, target_col = "mutated_sequence", "DMS_score"
    
    # Model settings
    use_global_kernel = cfg.gp.use_global_kernel
    use_mutation_kernel = cfg.gp.use_mutation_kernel
    use_zero_shot = cfg.gp.use_zero_shot
    use_prior = cfg.gp.use_prior
    if use_prior:
        noise_prior_scale = cfg.gp.noise_prior_scale

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load WT details
    wt_df = get_wt_df(dataset)

    # Check if results already exist
    out_path = Path(
        "results/ProteinGym/per_dataset_multiples",
        dataset,
        f"{model_name}.csv",
    )
    if out_path.exists():
        if overwrite:
            print(f"Results already exist at {out_path}. Overwriting.")
        else:
            print(f"Results already exist at {out_path}. Exiting.\n")
            return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_proteingym_dataset(dataset, multiples=True)
    kwargs, tokenizer = prepare_kwargs(wt_df, cfg)

    if use_zero_shot:
        zero_shot_method = cfg.gp.zero_shot_method
        zero_shot_col = zero_shot_name_to_col(zero_shot_method)
        df_zero = load_zero_shot(dataset, zero_shot_method)
        df = pd.merge(left=df, right=df_zero, on="mutant")
        df = df.reset_index(drop=True)
        zero_shot_full = torch.tensor(df[zero_shot_col].values, dtype=torch.float32)

    y_full = torch.tensor(df[target_col].values, dtype=torch.float32)

    if use_global_kernel:
        embeddings = load_embeddings(dataset, df)

    if use_mutation_kernel:
        x_seq = df[sequence_col].values
        x_tokens = tokenizer(x_seq).squeeze()

    # Concatenate features for single tensor input (GPyTorch requirement)
    if use_mutation_kernel:
        # [B, 20*seq_len]
        x_full = x_tokens
        if use_zero_shot:
            # [B, 20*seq_len + 1]
            x_full = torch.cat([x_full, zero_shot_full.unsqueeze(-1)], dim=-1)
        if use_global_kernel:
            # [B, 20*seq_len (+ 1) + emb_dim]
            x_full = torch.cat([x_full, embeddings], dim=-1)
    elif use_global_kernel:
        # [B, emb_dim]
        x_full = embeddings
        if use_zero_shot:
            # [B, emb_dim + 1]
            x_full = torch.cat([x_full, zero_shot_full.unsqueeze(-1)], dim=-1)
    else:
        raise NotImplementedError

    train_idx = df[df["n_mutations"] == 1].index
    test_idx = df[~df.index.isin(train_idx)].index
    train_idx = train_idx.tolist()
    test_idx = test_idx.tolist()
    x_train = x_full[train_idx]
    x_test = x_full[test_idx]
    y_train = y_full[train_idx]
    y_test = y_full[test_idx]

    if use_prior:
        noise_prior = gpytorch.priors.HalfCauchyPrior(scale=noise_prior_scale)
    else:
        noise_prior = None
        
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
    model = hydra.utils.instantiate(
        cfg.gp.gp_model,
        train_x=x_train,
        train_y=y_train,
        likelihood=likelihood,
        _recursive_=False,
        **kwargs,
    )
    # Train model
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

        if log_params:
            params_opt = {}
            if model.use_rbf or model.use_matern:
                params_opt["alpha"] = torch.sigmoid(model.alpha).item()
            else:
                params_opt["alpha"] = None
            if use_zero_shot:
                params_opt["zero_shot_scale"] = softplus(model.zero_shot_scale).item()
            else:
                params_opt["zero_shot_scale"] = None
            params_opt["gp_mean"] = model.mean_module.constant.item()
            params_opt["likelihood_noise"] = model.likelihood.noise.item()
            params_opt["loss"] = loss.item()

    # Evaluate model
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Predictive distribution
        y_preds_dist = likelihood(model(x_test))
        y_preds_mean = y_preds_dist.mean.detach().numpy()
        y_preds_var = y_preds_dist.covariance_matrix.diag().detach().numpy()
        y_test = y_test.detach().numpy()

        df_predictions = pd.DataFrame(
            {
                "mutant": df.loc[test_idx, "mutant"],
                "n_mutations": df.loc[test_idx, "n_mutations"],
                "y": y_test,
                "y_pred": y_preds_mean,
                "y_var": y_preds_var,
            }
        )

        # Group by n_mutations and compute MSE and Spearman correlation
        df_results = df_predictions.groupby("n_mutations", as_index=False).apply(
            lambda x: pd.Series(
                {
                    "MSE": mean_squared_error(x["y"], x["y_pred"]),
                    "Spearman": spearmanr(x["y"], x["y_pred"])[0],
                }
            )
        )

        if log_params:
            df_params = pd.DataFrame(params_opt, index=[0])

    df_results["assay_id"] = dataset
    df_results["model_name"] = model_name

    pred_path = (
        Path("results/ProteinGym/predictions_multiples") / dataset / out_path.name
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(pred_path, index=False)
    df_results.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
