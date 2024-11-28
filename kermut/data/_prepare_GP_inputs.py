from pathlib import Path

import numpy as np
import torch

from omegaconf import DictConfig

from kermut.data import Tokenizer



def prepare_GP_inputs(cfg: DictConfig, DMS_id: str, wt_sequence: str) -> Dict:
    if not cfg.kernel.use_structure_kernel:
        return {}
        
    inputs = {}
    tokenizer = Tokenizer()
    wt_toks = tokenizer(wt_sequence)
    inputs["wt_sequence"] = wt_toks
    if cfg.kernel.structure_kernel.use_site_comparison or cfg.kernel.structure_kernel.use_mutation_comparison:
        conditional_probs = np.load(Path(cfg.data.conditional_probs) /f"{DMS_id}.npy")
        inputs["conditional_probs"] = torch.tensor(conditional_probs).float()
    if cfg.kernel.structure_kernel.use_distance_comparison:
        coords = np.load(Path(cfg.data.coords) / f"{DMS_id}.npy")
        inputs["coords"] = torch.tensor(coords).float()
        
    return inputs