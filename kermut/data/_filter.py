from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def filter_datasets(cfg: DictConfig) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.data.paths.reference_file)

    # Datasets require >48GB VRAM
    large_datasets = [
        "POLG_CXB3N_Mattenberger_2021",
        "POLG_DEN26_Suphatrakul_2023",
    ]

    match cfg.dataset:
        case "benchmark":
            if cfg.cv_scheme in ["fold_rand_multiples", "domain"]:
                df_ref = df_ref[df_ref["includes_multiple_mutants"]]
                df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
                df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]
            else:
                df_ref = df_ref[~df_ref["DMS_id"].isin(large_datasets)]
        case "ablation":
            df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
        case "large":
            df_ref = df_ref[df_ref["DMS_id"].isin(large_datasets)]
        case "single":
            if cfg.single.use_id:
                df_ref = df_ref[df_ref["DMS_id"] == cfg.single.id]
            else:
                df_ref = df_ref.iloc[[cfg.single.id]]
        case _:
            raise ValueError(f"Unknown dataset type: {cfg.dataset}")

    df_ref = df_ref[["DMS_id", "target_seq"]]
    if not cfg.overwrite:
        output_dir = Path(cfg.data.paths.output_folder) / cfg.cv_scheme / cfg.kernel.name
        existing_results = []
        for DMS_id in df_ref["DMS_id"]:
            if (output_dir / f"{DMS_id}.csv").exists():
                existing_results.append(DMS_id)
        df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]

    return df_ref
