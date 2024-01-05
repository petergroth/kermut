from pathlib import Path

import gpytorch
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange

from src.model.gp import ExactGPKermut

zero_shot_name_to_col = {
    "ProteinMPNN": "pmpnn_ll",
    "ESM_IF1": "esmif1_ll",
    "EVE": "evol_indices_ensemble",
}


def load_proteinmpnn_proteingym(wt: pd.Series):
    conditional_probs_path = Path(
        "data",
        "conditional_probs",
        wt["UniProt_ID"].item(),
        "proteinmpnn",
        "conditional_probs_only",
        f"{wt['UniProt_ID'].item()}.npz",
    )
    proteinmpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    proteinmpnn_tok_to_aa = {i: aa for i, aa in enumerate(proteinmpnn_alphabet)}

    raw_file = np.load(conditional_probs_path)
    log_p = raw_file["log_p"]
    wt_toks = raw_file["S"]

    # Load sequence from ProteinMPNN outputs
    wt_seq_from_toks = "".join([proteinmpnn_tok_to_aa[tok] for tok in wt_toks])
    assert wt_seq_from_toks == wt["target_seq"].item()

    # Process logits
    log_p_mean = log_p.mean(axis=0)
    p_mean = np.exp(log_p_mean)
    p_mean = p_mean[:, :20]  # "X" is included as 21st AA in ProteinMPNN alphabet

    p_mean = torch.tensor(p_mean).float()
    return p_mean


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="proteingym_gp_regression",
)
def main(cfg: DictConfig) -> None:
    print(f"--- {cfg.dataset} ---")
    dataset = cfg.dataset
    split_method = cfg.split_method
    progress_bar = cfg.progress_bar

    # Load reference data
    ref_path = Path("data/raw/DMS_substitutions.csv")
    df_ref = pd.read_csv(ref_path)
    wt_df = df_ref.loc[df_ref["DMS_id"] == dataset]
    sequence_col = "mutated_sequence"
    target_col = "DMS_score"

    # Zero-shot score as mean function
    if "use_zero_shot" in cfg:
        use_zero_shot = cfg.use_zero_shot
        zero_shot_method = cfg.zero_shot_method
        zero_shot_col = zero_shot_name_to_col[zero_shot_method]
        out_path = Path(
            "results/ProteinGym",
            dataset,
            f"{cfg.gp.name}_{split_method}_{zero_shot_method}.csv",
        )

        # Error in source data. Replace "Rocklin" with "Tsuboyama"
        try:
            df_zero = pd.read_csv(
                f"../../software/ProteinGym/zero_shot_substitution_scores/{zero_shot_method}/{dataset}.csv"
            )
        except FileNotFoundError:
            if "Rocklin" in dataset:
                dms_id = dataset.replace("Rocklin", "Tsuboyama")
                df_zero = pd.read_csv(
                    f"../../software/ProteinGym/zero_shot_substitution_scores/{zero_shot_method}/{dms_id}.csv"
                )
            else:
                raise FileNotFoundError
        df_zero = df_zero[["mutant", zero_shot_col]]
    else:
        use_zero_shot = False
        out_path = Path(
            "results/ProteinGym", dataset, f"{cfg.gp.name}_{split_method}.csv"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    multiples = wt_df["includes_multiple_mutants"].item()
    use_singles = not multiples or split_method in [
        "fold_modulo_5",
        "fold_contiguous_5",
    ]

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

    # Preprocess data
    tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer)
    wt_sequence = wt_df["target_seq"].item()
    wt_sequence = tokenizer(wt_sequence).squeeze()
    conditional_probs = load_proteinmpnn_proteingym(wt_df)
    kwargs = {"wt_sequence": wt_sequence, "conditional_probs": conditional_probs}
    df = pd.read_csv(dms_path)
    if use_zero_shot:
        df = pd.merge(left=df, right=df_zero, on="mutant")

    df_results = pd.DataFrame(
        columns=[
            "fold",
            "MSE",
            "Spearman",
        ]
    )

    df = df.reset_index(drop=True)
    x_full = df[sequence_col].values
    x_tokens = tokenizer(x_full).squeeze()
    y_full = torch.tensor(df[target_col].values, dtype=torch.float32)
    if use_zero_shot:
        zero_shot_full = torch.tensor(df[zero_shot_col].values, dtype=torch.float32)
        x_tokens = torch.cat([x_tokens, zero_shot_full.unsqueeze(-1)], dim=-1)

    unique_folds = df[split_method_col].unique()
    for i, test_fold in enumerate(tqdm(unique_folds)):
        train_idx = (df[split_method_col] != test_fold).tolist()
        test_idx = (df[split_method_col] == test_fold).tolist()
        x_train = x_tokens[train_idx]
        x_test = x_tokens[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPKermut(
            train_x=x_train,
            train_y=y_train,
            likelihood=likelihood,
            gp_cfg=cfg.gp,
            use_zero_shot=use_zero_shot,
            **kwargs,
        )
        # Train model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.gp.optim.lr)

        if cfg.gp.optim.log_to_wandb:
            import wandb

            wandb.init(project="kermut", group=f"{cfg.experiment.n_train}_samples")
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        for _ in trange(cfg.gp.optim.n_steps, disable=not progress_bar):
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

            if cfg.gp.optim.log_to_wandb:
                wandb.log({"loss": loss.item()})
                wandb.log(model.covar_module.get_params())
                if "use_rbf" in cfg.gp:
                    wandb.log({"alpha": torch.sigmoid(model.alpha).item()})
                wandb.log({"likelihood_noise": model.likelihood.noise.item()})
                wandb.log({"gp_mean": model.mean_module.constant.item()})

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

        if cfg.gp.optim.log_to_wandb:
            wandb.finish()

    df_results["assay_id"] = dataset
    df_results["model_name"] = cfg.gp.name
    df_results["fold_variable_name"] = split_method
    if use_zero_shot:
        df_results["model_name"] = df_results["model_name"] + "_" + zero_shot_method
    df_results.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
