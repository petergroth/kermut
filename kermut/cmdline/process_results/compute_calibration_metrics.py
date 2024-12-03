from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from kermut.calibration import (
    compute_confidence_interval_based_metrics,
    compute_error_based_metrics,
)


def _filter(cfg: DictConfig, cv_scheme: str) -> pd.DataFrame:
    df_ref = pd.read_csv(Path(cfg.data.paths.reference_file))
    match cfg.dataset:
        case "benchmark":
            if cv_scheme in ["fold_rand_multiples", "domain"]:
                df_ref = df_ref[df_ref["includes_multiple_mutants"]]
                df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
                df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]
        case "ablation":
            df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
        case _:
            raise ValueError(f"Unknown dataset type: {cfg.dataset}")

    return df_ref[["DMS_id"]]


def _compute_calibration_metrics_single_model(cfg: DictConfig, model_name: str) -> None:
    for cv_scheme in cfg.cv_schemes:
        df_ref = _filter(cfg, cv_scheme)
        predictions_dir = Path(cfg.data.paths.output_folder) / cv_scheme / model_name
        out_dir = Path(cfg.data.paths.calibration_metrics) / cv_scheme / model_name
        out_dir.mkdir(exist_ok=True, parents=True)

        df_metrics = pd.DataFrame(columns=["DMS_id", "fold", "ENCE", "CV", "ECE"])
        df_ci_curve = pd.DataFrame(columns=["DMS_id", "fold", "confidence", "percentile"])
        df_error_curve = pd.DataFrame(columns=["DMS_id", "fold", "bin", "RMSE", "RMV"])

        for i, row in df_ref.iterrows():
            DMS_id = row["DMS_id"]
            try:
                df = pd.read_csv(predictions_dir / f"{DMS_id}.csv")
            except FileNotFoundError:
                print(f"File not found: {DMS_id}.csv")
                continue
            _df_error_based_metrics, _df_error_based_curve = compute_error_based_metrics(
                df=df,
                n_bins=cfg.postprocessing.error_bins,
                DMS_id=DMS_id,
                split=cv_scheme,
                return_calibration_curve=True,
            )
            _df_ci_based_metrics, _df_ci_based_curve = compute_confidence_interval_based_metrics(
                df=df,
                n_bins=cfg.postprocessing.confidence_interval_bins,
                DMS_id=DMS_id,
                split=cv_scheme,
                return_calibration_curve=True,
            )
            _df_metrics = pd.merge(
                left=_df_error_based_metrics, right=_df_ci_based_metrics, on="fold"
            )

            _df_metrics["DMS_id"] = DMS_id
            _df_error_based_curve["DMS_id"] = DMS_id
            _df_ci_based_curve["DMS_id"] = DMS_id

            df_metrics = pd.concat([df_metrics, _df_metrics], ignore_index=True)
            df_error_curve = pd.concat([df_error_curve, _df_error_based_curve], ignore_index=True)
            df_ci_curve = pd.concat([df_ci_curve, _df_ci_based_curve], ignore_index=True)

        df_metrics.to_csv(out_dir / "metrics.csv", index=False)
        df_ci_curve.to_csv(out_dir / "ci_curve.csv", index=False)
        df_error_curve.to_csv(out_dir / "error_curve.csv", index=False)


@hydra.main(
    version_base=None,
    config_path="../../hydra_configs",
    config_name="process_results",
)
def compute_calibration_metrics(cfg: DictConfig):
    for model_name in cfg.model_names:
        _compute_calibration_metrics_single_model(cfg, model_name)


if __name__ == "__main__":
    compute_calibration_metrics()
