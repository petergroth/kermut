from pathlib import Path

import gpytorch
import hydra
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from src.data.utils import load_conditional_probs


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="train_gp",
)
def main(cfg: DictConfig) -> None:
    # Set paths
    dataset = cfg.fit.dataset
    conditional_probs_method = cfg.fit.conditional_probs_method

    assay_path = Path("data", "processed", f"{dataset}.tsv")

    # Load data
    df = pd.read_csv(assay_path, sep="\t")
    wt_df = pd.read_csv("data/processed/wt_sequences.tsv", sep="\t")
    wt_sequence = wt_df.loc[wt_df["dataset"] == dataset, "seq"].item()

    # Filter data
    if cfg.fit.filter_mutations:
        df = df[df["n_muts"] <= cfg.fit.max_mutations]
    df = df.sample(n=cfg.fit.n_samples, random_state=cfg.fit.seed)

    y = df["delta_fitness"].values
    sequences = df["seq"].tolist()
    conditional_probs = load_conditional_probs(dataset, conditional_probs_method)
    assert len(conditional_probs) == len(wt_sequence)

    # Prepare data
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    y = torch.tensor(y, dtype=torch.float32)
    x = tokenizer(sequences)
    wt_sequence = tokenizer(wt_sequence).squeeze()

    # Setup model
    torch.manual_seed(cfg.fit.seed)
    kwargs = {"wt_sequence": wt_sequence, "conditional_probs": conditional_probs}
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = hydra.utils.instantiate(
        cfg.gp.model,
        train_x=x,
        train_y=y,
        likelihood=likelihood,
        **cfg.gp.kernel_params,
        **kwargs,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.fit.lr)

    if cfg.fit.log_to_wandb:
        wandb.init(project="kermut")
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    for _ in tqdm(range(cfg.fit.training_iter)):
        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)

        loss.backward()
        optimizer.step()

        if cfg.fit.log_to_wandb:
            wandb.log({"negative_marginal_ll": loss.item()})
            wandb.log(model.covar_module.get_params())
            wandb.log({"likelihood_noise": model.likelihood.noise.item()})
            wandb.log({"GP mean": model.mean_module.constant.item()})
        # print(f"Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
