from pathlib import Path

import gpytorch
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm, trange

from src.data.utils import load_sampled_regression_data
from src.model.utils import load_conditional_probs
from src.model.gp import ExactGPKermut


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="gp_regression",
)
def main(cfg: DictConfig) -> None:
    dataset = cfg.experiment.dataset

    out_path = Path(
        "results/regression",
        dataset,
        f"{cfg.experiment.n_train}_samples_gp_{cfg.gp.name}.tsv",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = hydra.utils.instantiate(cfg.gp.tokenizer)

    conditional_probs_method = cfg.gp.conditional_probs_method
    wt_df = pd.read_csv("data/processed/wt_sequences.tsv", sep="\t")
    wt_sequence = wt_df.loc[wt_df["dataset"] == dataset, "seq"].item()
    wt_sequence = tokenizer(wt_sequence).squeeze()
    conditional_probs = load_conditional_probs(dataset, conditional_probs_method)
    kwargs = {"wt_sequence": wt_sequence, "conditional_probs": conditional_probs}

    if cfg.gp.use_distance:
        distances = torch.load(Path("data/interim", dataset, "contact_map.pt"))
        kwargs["distances"] = distances.float()

    # Load full (filtered) dataset
    df = load_sampled_regression_data(cfg)
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
        df_train_val, df_test = train_test_split(
            df, test_size=len(df) - cfg.experiment.n_train, random_state=seed
        )
        # Extract data
        y_train = df_train_val["delta_fitness"].values
        y_test = df_test["delta_fitness"].values
        train_seq = df_train_val["seq"].values
        test_seq = df_test["seq"].values

        # Prepare data
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        x_train = tokenizer(train_seq).squeeze()
        x_test = tokenizer(test_seq).squeeze()

        model = ExactGPKermut(
            train_x=x_train,
            train_y=y_train,
            likelihood=likelihood,
            gp_cfg=cfg.gp,
            **kwargs,
        )

        # Train model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gp.optim.lr)

        if cfg.gp.optim.log_to_wandb:
            import wandb

            wandb.init(project="kermut", group=f"{cfg.experiment.n_train}_samples")
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

        for _ in trange(cfg.gp.optim.n_steps):
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

            if cfg.gp.optim.log_to_wandb:
                wandb.log({"loss": loss.item()})
                wandb.log(model.covar_module.get_params())
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
            test_r2 = r2_score(y_test_np, y_preds_mean_np)
            test_pearson = pearsonr(y_test_np, y_preds_mean_np)[0]

            df_results.loc[i] = [
                seed,
                test_mse,
                test_spearman,
                test_r2,
                test_pearson,
            ]

        if cfg.gp.optim.log_to_wandb:
            wandb.finish()

    df_results.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
