"""Utility functions for data processing and modelling """

import ast
import time
from pathlib import Path
from typing import Sequence

import h5py
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from src import AA_TO_IDX, ALPHABET


def load_proteinmpnn_proteingym(wt: pd.Series):
    proteinmpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    proteinmpnn_tok_to_aa = {i: aa for i, aa in enumerate(proteinmpnn_alphabet)}

    conditional_probs_dir = Path(
        "data",
        "conditional_probs",
        wt["UniProt_ID"].item(),
        "proteinmpnn",
        "conditional_probs_only",
    )

    # BRCA2_HUMAN is large and requires special care
    if wt["UniProt_ID"].item() != "BRCA2_HUMAN":
        conditional_probs_path = (
            conditional_probs_dir / f"{wt['UniProt_ID'].item()}.npz"
        )
        raw_file = np.load(conditional_probs_path)
        log_p = raw_file["log_p"]
        wt_toks = raw_file["S"]

        # Process logits
        log_p_mean = log_p.mean(axis=0)
        p_mean = np.exp(log_p_mean)
        p_mean = p_mean[:, :20]  # "X" is included as 21st AA in ProteinMPNN alphabet

        # Load sequence from ProteinMPNN outputs
        wt_seq_from_toks = "".join([proteinmpnn_tok_to_aa[tok] for tok in wt_toks])

        # Mismatch between WT and PDB
        if wt["DMS_id"].item() == "CAS9_STRP1_Spencer_2017_positive":
            p_mean = p_mean[:1368]
            wt_seq_from_toks = wt_seq_from_toks[:1368]
        if wt["DMS_id"].item() in [
            "P53_HUMAN_Giacomelli_2018_Null_Etoposide",
            "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
            "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
        ]:
            # Replace index 71 with "R"
            wt_seq_from_toks = wt_seq_from_toks[:71] + "R" + wt_seq_from_toks[72:]

        # Special case where PDB is domain of a larger protein
        if wt["DMS_id"].item() in [
            "A0A140D2T1_ZIKV_Sourisseau_2019",
            "POLG_HCVJF_Qi_2014",
        ]:
            idx = wt["target_seq"].item().find(wt_seq_from_toks)
            assert idx != -1
            seq_len = len(wt["target_seq"].item())
            p_mean_pad = np.full((seq_len, 20), np.nan)
            p_mean_pad[idx : idx + len(wt_seq_from_toks)] = p_mean
            p_mean = p_mean_pad
        else:
            assert wt_seq_from_toks == wt["target_seq"].item()
    else:
        p_mean_full = np.zeros((2832, 20))
        suffixes = ["1-1000.npz", "1001-2085.npz", "2086-2832.npz"]
        idxs_1 = [0, 1000, 2085]
        idxs_2 = [1000, 2085, 2832]

        for suffix, idx_1, idx_2 in zip(suffixes, idxs_1, idxs_2):
            conditional_probs_path = (
                conditional_probs_dir / f"{wt['UniProt_ID'].item()}_{suffix}"
            )
            raw_file = np.load(conditional_probs_path)
            log_p = raw_file["log_p"]
            wt_toks = raw_file["S"]

            # Process logits
            log_p_mean = log_p.mean(axis=0)
            p_mean = np.exp(log_p_mean)
            p_mean = p_mean[
                :, :20
            ]  # "X" is included as 21st AA in ProteinMPNN alphabet
            p_mean_full[idx_1:idx_2] = p_mean
        p_mean = p_mean_full

    p_mean = torch.tensor(p_mean).float()
    return p_mean


def load_zero_shot(dataset: str, zero_shot_method: str):
    zero_shot_dir = Path("data/zero_shot_fitness_predictions") / zero_shot_method
    zero_shot_col = zero_shot_name_to_col(zero_shot_method)

    if zero_shot_method == "TranceptEVE":
        zero_shot_dir = zero_shot_dir / "TranceptEVE_L"
    if zero_shot_method == "ESM2":
        zero_shot_dir = zero_shot_dir / "650M"

    try:
        df_zero = pd.read_csv(zero_shot_dir / f"{dataset}.csv")
    except FileNotFoundError:
        if "Tsuboyama" in dataset:
            dataset_alt = dataset.replace("Tsuboyama", "Tsuboyama")
            df_zero = pd.read_csv(zero_shot_dir / f"{dataset_alt}.csv")
        else:
            raise FileNotFoundError

    # Average duplicates
    df_zero = df_zero[["mutant", zero_shot_col]].groupby("mutant").mean().reset_index()
    return df_zero


def zero_shot_name_to_col(key):
    return {
        "ProteinMPNN": "pmpnn_ll",
        "ESM_IF1": "esmif1_ll",
        "EVE": "evol_indices_ensemble",
        "TranceptEVE": "avg_score",
        "GEMME": "GEMME_score",
        "VESPA": "VESPA",
        "ESM2": "esm2_t33_650M_UR50D",
        "MSA_Transformer": "esm_msa1b_t12_100M_UR50S_ensemble",
    }[key]


def load_embeddings(
    dataset: str,
    df: pd.DataFrame,
    multiples: bool = False,
    embedding_type: str = "ESM2",
) -> torch.Tensor:

    if multiples:
        emb_path = (
            Path(f"data/embeddings/substitutions_multiples/{embedding_type}")
            / f"{dataset}.h5"
        )
    else:
        emb_path = (
            Path(f"data/embeddings/substitutions_singles/{embedding_type}")
            / f"{dataset}.h5"
        )
    # Check if file exists
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}.")

    # Occasional issues with reading the file due to concurrent access
    tries = 0
    while tries < 10:
        try:

            with h5py.File(emb_path, "r", locking=True) as h5f:
                embeddings = torch.tensor(h5f["embeddings"][:]).float()
                mutants = [x.decode("utf-8") for x in h5f["mutants"][:]]
            break
        except OSError:
            tries += 1
            time.sleep(10)
            pass

    embeddings = embeddings.mean(dim=1)
    # Keep entries that are in the dataset
    keep = [x in df["mutant"].tolist() for x in mutants]
    embeddings = embeddings[keep]
    mutants = np.array(mutants)[keep]
    # Ensures matching indices
    idx = [df["mutant"].tolist().index(x) for x in mutants]
    embeddings = embeddings[idx]
    return embeddings


def prepare_kwargs(wt_df: pd.DataFrame, cfg: DictConfig):
    # Prepare arugments for gp/kernel
    kwargs = {"use_zero_shot": cfg.gp.use_zero_shot}
    if cfg.gp.use_mutation_kernel:
        tokenizer = hydra.utils.instantiate(cfg.gp.mutation_kernel.tokenizer)
        wt_sequence = wt_df["target_seq"].item()
        wt_sequence = tokenizer(wt_sequence).squeeze()
        if cfg.gp.mutation_kernel.conditional_probs_method == "ProteinMPNN":
            # conditional_probs = load_proteinmpnn_proteingym(wt_df)
            conditional_probs = np.load(
                Path(f"data/conditional_probs/ProteinMPNN/{wt_df['DMS_id'].item()}.npy")
            )
        else:
            raise NotImplementedError

        kwargs["wt_sequence"] = wt_sequence
        kwargs["conditional_probs"] = torch.tensor(conditional_probs)
        kwargs["km_cfg"] = cfg.gp.mutation_kernel
        kwargs["use_global_kernel"] = cfg.gp.use_global_kernel

        if cfg.gp.mutation_kernel.use_distances:
            coords = np.load(f"data/structures/coords/{wt_df['DMS_id'].item()}.npy")
            kwargs["coords"] = torch.tensor(coords)
    else:
        tokenizer = None
    return kwargs, tokenizer


def get_model_name(cfg: DictConfig) -> str:
    return cfg.custom_name if "custom_name" in cfg else cfg.gp.name


def load_proteingym_dataset(dataset: str, multiples: bool = False) -> pd.DataFrame:
    if multiples:
        base_path = Path("data/substitutions_multiples")
        if "Tsuboyama" in dataset:
            # Depending on which ProteinGym version was used, the dataset name may differ
            dataset = dataset.replace("Tsuboyama", "Tsuboyama")
    else:
        base_path = Path("data/substitutions_singles")
    df = pd.read_csv(base_path / f"{dataset}.csv")

    df["n_mutations"] = df["mutant"].apply(lambda x: len(x.split(":")))
    return df.reset_index(drop=True)


def get_wt_df(dataset: str) -> pd.DataFrame:
    ref_path = Path("data/DMS_substitutions.csv")
    df_ref = pd.read_csv(ref_path)
    return df_ref.loc[df_ref["DMS_id"] == dataset]


def prepare_datasets(cfg: DictConfig, use_multiples: bool = False):

    # Datasets require too much RAM to use on GPU
    large_datasets = [
        "HMDH_HUMAN_Jiang_2019",
        "HSP82_YEAST_Flynn_2019",
        "MSH2_HUMAN_Jia_2020",
        "MTHR_HUMAN_Weile_2021",
        "POLG_CXB3N_Mattenberger_2021",
        "POLG_DEN26_Suphatrakul_2023",
        "Q2N0S5_9HIV1_Haddox_2018",
        "RDRP_I33A0_Li_2023",
        "S22A1_HUMAN_Yee_2023_abundance",
        "S22A1_HUMAN_Yee_2023_activity",
        "SC6A4_HUMAN_Young_2021",
        "SHOC2_HUMAN_Kwon_2022",
    ]

    # Use all datasets
    if cfg.dataset == "all":
        df_ref = pd.read_csv(Path("data", "DMS_substitutions.csv"))
        if use_multiples:
            df_ref = df_ref[df_ref["includes_multiple_mutants"]]
            df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
            # Remove GCN4_YEAST_Staller_2018 due to very high mutation count
            df_ref = df_ref[df_ref["DMS_id"] != "GCN4_YEAST_Staller_2018"]
        elif cfg.limit_mutants:
            # For ablation, use 174/217 datasets
            df_ref = df_ref[df_ref["DMS_number_single_mutants"] < 6000]
        datasets = df_ref["DMS_id"].tolist()
    elif cfg.dataset == "large":
        # Datasets requiring more than 48GB of VRAM
        datasets = large_datasets
    else:
        # Single dataset
        datasets = [cfg.dataset]

    # Sort datasets alphabetically
    datasets = sorted(datasets)

    if cfg.limit_mem and cfg.use_gpu:
        # Ignore datasets that require too much VRAM
        datasets = [d for d in datasets if d not in large_datasets]

    # Determine which datasets to process
    model_name = get_model_name(cfg)
    split_method = cfg.split_method
    overwrite = cfg.overwrite
    if not overwrite:
        # If not overwrite, run only on missing datasets
        output_dataset = []
        for dataset in datasets:
            out_path = (
                Path("results/predictions")
                / dataset
                / f"{model_name}_{split_method}.csv"
            )
            if not out_path.exists():
                # Dataset only processed if method does not require inter-residue distances
                if dataset == "BRCA2_HUMAN_Erwood_2022_HEK293T":
                    if cfg.gp.use_mutation_kernel:
                        if not cfg.gp.mutation_kernel.use_distances:
                            output_dataset.append(dataset)
                        else:
                            print(f"Skipping {dataset} (use_distances=True)")
                    else:
                        output_dataset.append(dataset)
                else:
                    output_dataset.append(dataset)
    else:
        output_dataset = datasets

    return output_dataset


def hellinger_distance(p: torch.tensor, q: torch.tensor):
    """Compute Hellinger distance between input distributions:

    HD(p, q) = sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))

    Args:
        x1 (torch.Tensor): Shape (n, 20)
        x2 (torch.Tensor): Shape (n, 20)

    Returns:
        torch.Tensor: Shape (n, n)
    """
    batch_size = p.shape[0]
    # Compute only the lower triangular elements if p == q
    if torch.allclose(p, q):
        tril_i, tril_j = torch.tril_indices(batch_size, batch_size, offset=-1)
        hellinger_tril = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[tril_i]) - torch.sqrt(q[tril_j])) ** 2, dim=1)
        )
        hellinger_matrix = torch.zeros((batch_size, batch_size))
        hellinger_matrix[tril_i, tril_j] = hellinger_tril
        hellinger_matrix[tril_j, tril_i] = hellinger_tril
    else:
        mesh_i, mesh_j = torch.meshgrid(
            torch.arange(batch_size), torch.arange(batch_size), indexing="ij"
        )
        mesh_i, mesh_j = mesh_i.flatten(), mesh_j.flatten()
        hellinger = torch.sqrt(
            0.5 * torch.sum((torch.sqrt(p[mesh_i]) - torch.sqrt(q[mesh_j])) ** 2, dim=1)
        )
        hellinger_matrix = hellinger.reshape(batch_size, batch_size)
    return hellinger_matrix


class Tokenizer:
    def __init__(self):
        super().__init__()
        self.alphabet = list(ALPHABET)
        self._aa_to_tok = AA_TO_IDX
        self._tok_to_aa = {v: k for k, v in self._aa_to_tok.items()}

    def encode(self, batch: Sequence[str]) -> torch.LongTensor:
        batch_size = len(batch)
        seq_len = len(batch[0])
        toks = torch.zeros((batch_size, seq_len))
        for i, seq in enumerate(batch):
            for j, aa in enumerate(seq):
                toks[i, j] = self._aa_to_tok[aa]

        return toks.long()

    def decode(self, toks: torch.LongTensor) -> Sequence[str]:
        seqs = []
        for tok in toks:
            seq = "".join([self.alphabet[int(t)] for t in tok])
            seqs.append(seq)
        return seqs

    def aa_to_tok(self, aa: str) -> int:
        return self._aa_to_tok[aa]

    def tok_to_aa(self, tok: int) -> str:
        return self._tok_to_aa[tok]

    def __call__(self, batch: Sequence[str]):
        return self.encode(batch)

    def __len__(self):
        return len(self.alphabet)


class Tokenizer_oh_seq:
    def __init__(self, flatten: bool = True):
        super().__init__()
        self.alphabet = list(ALPHABET)
        self.flatten = flatten
        self._aa_to_tok = AA_TO_IDX
        self._tok_to_aa = {v: k for k, v in self._aa_to_tok.items()}

    def encode(self, batch: Sequence[str]) -> torch.LongTensor:
        batch_size = len(batch)
        seq_len = len(batch[0])
        toks = torch.zeros((batch_size, seq_len, 20))
        for i, seq in enumerate(batch):
            for j, aa in enumerate(seq):
                toks[i, j, self._aa_to_tok[aa]] = 1

        if self.flatten:
            # Check if batch is str
            if isinstance(batch, str):
                return toks.squeeze().flatten().long()
            else:
                return toks.reshape(batch_size, seq_len * 20).long()
        else:
            return toks.squeeze().long()

    def __call__(self, batch: Sequence[str]):
        return self.encode(batch)


class Tokenizer_oh_mut:
    def __init__(self, mut2wt: pd.Series):
        super().__init__()
        self.alphabet = list(ALPHABET)
        self._aa_to_tok = AA_TO_IDX
        self._tok_to_aa = {v: k for k, v in self._aa_to_tok.items()}
        mut2wt = mut2wt.apply(ast.literal_eval)
        mutated_positions = np.sort(mut2wt.explode().str[1:-1].astype(int).unique())
        self._pos_to_idx = {pos: i for i, pos in enumerate(mutated_positions)}
        self.n_mutated_positions = len(mutated_positions)

    def encode(self, batch: pd.Series) -> torch.LongTensor:
        batch = batch.apply(ast.literal_eval)
        one_hot = np.zeros((len(batch), self.n_mutated_positions, 20))
        for i, mut2wt in enumerate(batch):
            for mut in mut2wt:
                pos = int(mut[1:-1])
                aa = mut[-1]
                one_hot[i, self._pos_to_idx[pos] - 1, self._aa_to_tok[aa]] = 1.0

        one_hot = one_hot.reshape(len(batch), 20 * self.n_mutated_positions)
        return torch.tensor(one_hot).long()

    def __call__(self, batch: pd.Series):
        return self.encode(batch)
