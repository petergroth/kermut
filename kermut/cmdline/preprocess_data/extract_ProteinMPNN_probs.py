from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


def _filter_datasets(cfg: DictConfig) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.data.paths.reference_file)
    probs_dir = Path(cfg.data.paths.conditional_probs)
    match cfg.dataset:
        case "all":
            pass
        case "single":
            if cfg.single.use_id:
                df_ref = df_ref[df_ref["DMS_id"] == cfg.single.id]
            else:
                df_ref = df_ref.iloc[[cfg.single.id]]
        case _:
            raise ValueError(f"Invalid dataset: {cfg.dataset}")

    if not cfg.overwrite:
        existing_results = []
        for DMS_id in df_ref["DMS_id"]:
            output_file = probs_dir / f"{DMS_id}.npy"
            if output_file.exists():
                existing_results.append(DMS_id)
        df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]

    return df_ref


@hydra.main(
    version_base=None,
    config_path="../hydra_configs",
    config_name="benchmark",
)
def extract_ProteinMPNN_probs(cfg: DictConfig) -> None:
    df_ref = _filter_datasets(cfg)

    if len(df_ref) == 0:
        print("All processed conditional probabilities exist. Exiting.")
        return

    proteinmpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    proteinmpnn_tok_to_aa = {i: aa for i, aa in enumerate(proteinmpnn_alphabet)}
    raw_proteinmpnn_dir = Path(cfg.data.paths.raw_conditional_probs)
    proteinmpnn_dir = Path(cfg.data.paths.conditional_probs)
    proteinmpnn_dir.mkdir(parents=True, exist_ok=True)

    for row in df_ref.itertuples():
        try:
            UniProt_ID = row.UniProt_ID
            DMS_id = row.DMS_id
            wt_sequence = row.target_seq
            if UniProt_ID != "BRCA2_HUMAN":
                file_path = (
                    raw_proteinmpnn_dir
                    / UniProt_ID
                    / f"proteinmpnn/conditional_probs_only/{UniProt_ID}.npz"
                )
                # Load and unpack
                raw_file = np.load(file_path)
                log_p = raw_file["log_p"]
                wt_toks = raw_file["S"]

                # Process logits ("X" is included as 21st AA in ProteinMPNN alphabet)
                log_p_mean = log_p.mean(axis=0)
                p_mean = np.exp(log_p_mean)
                p_mean = p_mean[:, :20]

                # Load sequence from ProteinMPNN outputs
                wt_seq_from_toks = "".join([proteinmpnn_tok_to_aa[tok] for tok in wt_toks])

                # Mismatch between WT and PDB
                if DMS_id == "CAS9_STRP1_Spencer_2017_positive":
                    p_mean = p_mean[:1368]
                    wt_seq_from_toks = wt_seq_from_toks[:1368]
                if DMS_id in [
                    "P53_HUMAN_Giacomelli_2018_Null_Etoposide",
                    "P53_HUMAN_Giacomelli_2018_Null_Nutlin",
                    "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
                ]:
                    # Replace index 71 with "R"
                    wt_seq_from_toks = wt_seq_from_toks[:71] + "R" + wt_seq_from_toks[72:]

                # Special case where PDB is domain of a larger protein
                if DMS_id in [
                    "A0A140D2T1_ZIKV_Sourisseau_2019",
                    "POLG_HCVJF_Qi_2014",
                ]:
                    idx = wt_sequence.find(wt_seq_from_toks)
                    assert idx != -1
                    seq_len = len(wt_sequence)
                    p_mean_pad = np.full((seq_len, 20), np.nan)
                    p_mean_pad[idx : idx + len(wt_seq_from_toks)] = p_mean
                    p_mean = p_mean_pad
                else:
                    assert wt_seq_from_toks == wt_sequence

            else:
                # Special case. The large protein is separated into three PDB files. All are loaded and combined.
                p_mean_full = np.zeros((2832, 20))
                suffixes = ["1-1000", "1001-2085", "2086-2832"]
                idxs_1 = [0, 1000, 2085]
                idxs_2 = [1000, 2085, 2832]

                for suffix, idx_1, idx_2 in zip(suffixes, idxs_1, idxs_2):
                    file_path = (
                        raw_proteinmpnn_dir
                        / UniProt_ID
                        / f"proteinmpnn/conditional_probs_only/{UniProt_ID}_{suffix}.npz"
                    )
                    raw_file = np.load(file_path)
                    log_p = raw_file["log_p"]
                    wt_toks = raw_file["S"]

                    # Process logits ("X" is included as 21st AA in ProteinMPNN alphabet)
                    log_p_mean = log_p.mean(axis=0)
                    p_mean = np.exp(log_p_mean)
                    p_mean = p_mean[:, :20]
                    p_mean_full[idx_1:idx_2] = p_mean
                p_mean = p_mean_full

            # SAVE
            np.save(proteinmpnn_dir / DMS_id, p_mean)
        except Exception as e:
            print(f"Error for {DMS_id}: {e}")


if __name__ == "__main__":
    extract_ProteinMPNN_probs()
