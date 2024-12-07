from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr
from tqdm import tqdm


def _filter(cfg: DictConfig) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.data.paths.reference_file)
    match cfg.dataset:
        case "ablation":
            df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
        case "benchmark" | "reference":
            pass
        case _:
            raise ValueError(f"Unknown dataset type: {cfg.dataset}")

    return df_ref[["DMS_id"]]


def _process_single_model(cfg: DictConfig, df: pd.DataFrame, model_name: str) -> None:
    output_path = Path(cfg.data.paths.processed_scores) / f"{model_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not cfg.overwrite:
        print(f"Skipping {model_name} - result file exists at {output_path}")
        return
    if output_path.exists() and cfg.overwrite:
        print(f"Overwriting {model_name} - result file exists at {output_path}")

    out = {}
    for cv_scheme in cfg.cv_schemes:
        df_cv = df.copy()
        df_cv = df_cv.assign(cv_scheme=cv_scheme, Spearman=np.nan, MSE=np.nan)
        cv_dir = Path(cfg.data.paths.output_folder) / cv_scheme / model_name
        for DMS_id in df_cv["DMS_id"]:
            df_predictions = pd.read_csv(cv_dir / f"{DMS_id}.csv")
            corr, _ = spearmanr(
                df_predictions[cfg.target_col].values, df_predictions[cfg.pred_col].values
            )
            mse = ((df_predictions[cfg.target_col] - df_predictions[cfg.pred_col]) ** 2).mean()
            df_cv.loc[df_cv["DMS_id"] == DMS_id, "Spearman"] = corr
            df_cv.loc[df_cv["DMS_id"] == DMS_id, "MSE"] = mse
        out[cv_scheme] = df_cv

    df_out = pd.concat(out.values())
    df_out = df_out.sort_values(["DMS_id", "cv_scheme"])
    df_out.to_csv(output_path, index=False)


@hydra.main(
    version_base=None,
    config_path="../../hydra_configs",
    config_name="process_results",
)
def process_model_scores(cfg: DictConfig):
    df_ref = _filter(cfg)
    for model_name in tqdm(cfg.model_names):
        _process_single_model(cfg, df_ref, model_name)


if __name__ == "__main__":
    process_model_scores()
