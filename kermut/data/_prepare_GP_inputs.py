import time
from pathlib import Path
from typing import Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from kermut.constants import ZERO_SHOT_NAME_TO_COL

from ._tokenizer import Tokenizer


def _load_zero_shot(cfg: DictConfig, df: pd.DataFrame, DMS_id: str) -> Union[torch.Tensor, None]:
    if cfg.kernel.use_zero_shot:
        zero_shot_col = ZERO_SHOT_NAME_TO_COL[cfg.kernel.zero_shot_method]
        df_zero = pd.read_csv(
            Path(cfg.data.paths.zero_shot) / cfg.kernel.zero_shot_method / f"{DMS_id}.csv"
        )[[zero_shot_col, "mutant"]]
        df = pd.merge(left=df, right=df_zero, on="mutant", how="left")
        df = df.groupby("mutant").mean(numeric_only=True).reset_index(drop=True)
        x_zero_shot = torch.tensor(df[zero_shot_col].values, dtype=torch.float32)
        return x_zero_shot
    else:
        return None


def _load_embeddings(cfg: DictConfig, df: pd.DataFrame, DMS_id: str) -> Union[torch.Tensor, None]:
    if not cfg.kernel.use_sequence_kernel:
        return None

    if cfg.cv_scheme in ["fold_rand_multiples", "domain"]:
        embedding_path = Path(cfg.data.paths.embeddings_multiples) / f"{DMS_id}.h5"
    else:
        embedding_path = Path(cfg.data.paths.embeddings_singles) / f"{DMS_id}.h5"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embeddings not found at {embedding_path}")

    # Occasional issues with reading the file due to concurrent access
    tries = 0
    while tries < 10:
        try:
            with h5py.File(embedding_path, "r", locking=True) as h5f:
                embeddings = torch.tensor(h5f["embeddings"][:]).float()
                mutants = [x.decode("utf-8") for x in h5f["mutants"][:]]
            break
        except OSError:
            tries += 1
            time.sleep(10)
            pass

    # If not already mean-pooled
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=1)

    # Keep entries that are in the dataset
    keep = [x in df["mutant"].tolist() for x in mutants]
    embeddings = embeddings[keep]
    mutants = np.array(mutants)[keep]
    # Ensures matching ordering
    idx = [df["mutant"].tolist().index(x) for x in mutants]
    embeddings = embeddings[idx]
    return embeddings


def _tokenize_data(cfg: DictConfig, df: pd.DataFrame) -> torch.Tensor:
    if not cfg.kernel.use_sequence_kernel:
        return None

    tokenizer = Tokenizer()
    x_toks = tokenizer(df[cfg.data.sequence_col])
    return x_toks


def prepare_GP_inputs(
    cfg: DictConfig, DMS_id: str
) -> Tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(Path(cfg.data.paths.DMS_input_folder) / f"{DMS_id}.csv")

    y = torch.tensor(df[cfg.data.target_col].values, dtype=torch.float32)
    x_toks = _tokenize_data(cfg, df)
    x_zero_shot = _load_zero_shot(cfg, df, DMS_id)
    x_embedding = _load_embeddings(cfg, df, DMS_id)

    if cfg.use_gpu and torch.cuda.is_available():
        x_toks = x_toks.cuda()
        if x_zero_shot is not None:
            x_zero_shot = x_zero_shot.cuda()
        if x_embedding is not None:
            x_embedding = x_embedding.cuda()
        y = y.cuda()

    return df, y, x_toks, x_embedding, x_zero_shot
