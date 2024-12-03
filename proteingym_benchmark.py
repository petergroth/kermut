from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from kermut.data import (
    filter_datasets,
    prepare_GP_inputs,
    prepare_GP_kwargs,
    split_inputs,
    standardize,
)
from kermut.gp import instantiate_gp, optimize_gp, predict


@hydra.main(
    version_base=None,
    config_path="kermut/hydra_configs",
    config_name="benchmark",
)
def main(cfg: DictConfig) -> None:
    df_ref = filter_datasets(cfg)
    if len(df_ref) == 0:
        print("All results exist.")
        return

    for i, (DMS_id, target_seq) in enumerate(df_ref.itertuples(index=False)):
        print(f"--- ({i+1}/{len(df_ref)}) {DMS_id} ---", flush=True)
        df, y, x_toks, x_embed, x_zero_shot = prepare_GP_inputs(cfg, DMS_id)
        gp_inputs = prepare_GP_kwargs(cfg, DMS_id, target_seq)

        df_out = df[["mutant"]].copy()
        df_out = df_out.assign(fold=np.nan, y=np.nan, y_pred=np.nan, y_var=np.nan)

        unique_folds = (
            df[cfg.split].unique() if cfg.data.test_index == -1 else [cfg.data.test_index]
        )
        for test_fold in unique_folds:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

            train_idx = (df[cfg.split] != test_fold).tolist()
            test_idx = (df[cfg.split] == test_fold).tolist()

            y_train, y_test = split_inputs(train_idx, test_idx, y)
            y_train, y_test = (
                standardize(y_train, y_test) if cfg.data.standardize else (y_train, y_test)
            )

            x_toks_train, x_toks_test = split_inputs(train_idx, test_idx, x_toks)
            x_embed_train, x_embed_test = split_inputs(train_idx, test_idx, x_embed)
            x_zero_shot_train, x_zero_shot_test = split_inputs(train_idx, test_idx, x_zero_shot)

            train_inputs = (x_toks_train, x_embed_train, x_zero_shot_train)
            test_inputs = (x_toks_test, x_embed_test, x_zero_shot_test)

            gp, likelihood = instantiate_gp(
                cfg=cfg, train_inputs=train_inputs, y_train=y_train, gp_inputs=gp_inputs
            )

            gp, likelihood = optimize_gp(
                gp=gp,
                likelihood=likelihood,
                train_inputs=train_inputs,
                y_train=y_train,
                lr=cfg.optim.lr,
                n_steps=cfg.optim.n_steps,
                progress_bar=cfg.optim.progress_bar,
            )

            df_out = predict(
                gp=gp,
                likelihood=likelihood,
                x_test=test_inputs,
                y_test=y_test,
                test_fold=test_fold,
                test_idx=test_idx,
                df_out=df_out,
            )

        spearman = df_out["y"].corr(df_out["y_pred"], "spearman")
        print(f"Spearman: {spearman:.3f} (DMS ID: {DMS_id})")

        out_path = (
            Path(cfg.data.paths.output_folder) / cfg.split / cfg.kernel.name / f"{DMS_id}.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
