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
    split_method = cfg.split_method
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
        "results/ProteinGym/per_dataset", dataset, f"{model_name}_{split_method}.csv"
    )
    if out_path.exists():
        if overwrite:
            print(f"Results already exist at {out_path}. Overwriting.")
        else:
            print(f"Results already exist at {out_path}. Exiting.\n")
            return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_proteingym_dataset(dataset, multiples=False)
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


    # Initialize results dataframe
    df_results = pd.DataFrame(
        columns=[
            "fold",
            "MSE",
            "Spearman",
        ]
    )

    if log_params:
        df_params = pd.DataFrame(
            columns=[
                "fold",
                "alpha",
                "zero_shot_scale",
                "gp_mean",
                "likelihood_noise",
                "loss",
            ]
        )

    if log_predictions:
        df_predictions = pd.DataFrame(
            columns=["fold", "mutant", "y", "y_pred", "y_var", "train"]
        )
        pred_path = Path("results/ProteinGym/predictions") / dataset / out_path.name
        pred_path.parent.mkdir(parents=True, exist_ok=True)


    unique_folds = df[split_method].unique()
    for i, test_fold in enumerate(tqdm(unique_folds)):
        # Assign splits
        train_idx = (df[split_method] != test_fold).tolist()
        test_idx = (df[split_method] == test_fold).tolist()
        x_train = x_full[train_idx]
        x_test = x_full[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]
        
        # Standardize given current training set
        if standardize:
            y_train_mean = y_train.mean()
            y_train_std = y_train.std()
            y_train = (y_train - y_train_mean) / y_train_std
            y_test = (y_test - y_train_mean) / y_train_std

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

        # Evaluate model
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            # Predictive distribution
            y_preds_dist = likelihood(model(x_test))
            y_preds_mean = y_preds_dist.mean.detach().numpy()
            y_preds_var = y_preds_dist.covariance_matrix.diag().detach().numpy()
            y_test = y_test.detach().numpy()

            # Compute metrics
            test_mse = mean_squared_error(y_test, y_preds_mean)
            test_spearman = spearmanr(y_test, y_preds_mean)[0]

            df_results.loc[i] = [
                test_fold,
                test_mse,
                test_spearman,
            ]
            if log_params:
                # Gather
                params_opt = {}
                if use_global_kernel and use_mutation_kernel:
                    params_opt["alpha"] = torch.sigmoid(model.alpha).item()
                else:
                    params_opt["alpha"] = None
                if use_zero_shot:
                    params_opt["zero_shot_scale"] = model.zero_shot_scale.item()
                else:
                    params_opt["zero_shot_scale"] = None
                params_opt["gp_mean"] = model.mean_module.constant.item()
                params_opt["likelihood_noise"] = model.likelihood.noise.item()
                params_opt["loss"] = loss.item()
                
                # Save
                df_params.loc[i] = [
                    test_fold,
                    params_opt["alpha"],
                    params_opt["zero_shot_scale"],
                    params_opt["gp_mean"],
                    params_opt["likelihood_noise"],
                    params_opt["loss"],
                ]
                
            if log_predictions:
                df_pred_fold = pd.DataFrame(
                    {
                        "fold": test_fold,
                        "mutant": df.loc[test_idx, "mutant"],
                        "y": y_test,
                        "y_pred": y_preds_mean,
                        "y_var": y_preds_var,
                    }
                )
                df_predictions = pd.concat([df_predictions, df_pred_fold])

    df_results["assay_id"] = dataset
    df_results["model_name"] = model_name
    df_results["fold_variable_name"] = split_method

    if log_params:
        df_results = pd.merge(left=df_results, right=df_params, on="fold")

    if log_predictions:
        df_predictions.to_csv(pred_path, index=False)

    df_results.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
