"""Adapted from https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py"""

from pathlib import Path

import hydra
import pandas as pd
import torch
from esm import pretrained
from omegaconf import DictConfig
from tqdm import tqdm


def _label_row(row, sequence, token_probs, alphabet, offset_idx):
    mutations = row.split(":")
    score = 0
    for mutation in mutations:
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()

    return score


def _filter_datasets(cfg: DictConfig) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.data.paths.reference_file)
    zero_shot_dir = Path(cfg.data.paths.zero_shot) / "ESM2" / "650M"
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
            output_file = zero_shot_dir / f"{DMS_id}.csv"
            if output_file.exists():
                existing_results.append(DMS_id)
        df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]

    return df_ref


@hydra.main(
    version_base=None,
    config_path="../hydra_configs",
    config_name="benchmark",
)
def extract_esm2_zero_shots(cfg: DictConfig) -> None:
    df_ref = _filter_datasets(cfg)
    DMS_dir = Path(cfg.data.paths.DMS_input_folder)
    score_key = "esm2_t33_650M_UR50D"

    if len(df_ref) == 0:
        print("All zero-shot score files already exist. Exiting.")
        return

    model_path = Path(cfg.data.embedding.model_path)
    model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
    model.eval()

    use_gpu = torch.cuda.is_available() and cfg.use_gpu
    if use_gpu:
        model = model.cuda()
        print("Transferred model to GPU.")

    for i, DMS_id in tqdm(enumerate(df_ref["DMS_id"])):
        print(f"--- Computing zero-shots for {DMS_id} ({i+1}/{len(df_ref)}) ---")
        df_ref_dms = df_ref.loc[df_ref["DMS_id"] == DMS_id].iloc[0]
        if (
            df_ref_dms["includes_multiple_mutants"]
            and df_ref_dms["DMS_total_number_mutants"] <= 7500
        ):
            file_in = DMS_dir / "cv_folds_multiples_substitutions" / f"{DMS_id}.csv"
        else:
            file_in = DMS_dir / "cv_folds_singles_substitutions" / f"{DMS_id}.csv"

        df = pd.read_csv(file_in)

        batch_converter = alphabet.get_batch_converter()
        data = [
            ("protein1", df_ref_dms["target_sequence"]),
        ]
        _, _, batch_tokens = batch_converter(data)

        all_token_probs = []
        for i in range(batch_tokens.size(1)):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                if use_gpu:
                    batch_tokens_masked = batch_tokens_masked.cuda()
                token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1)
            all_token_probs.append(token_probs[:, i])
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        df[score_key] = df.apply(
            lambda row: _label_row(
                row["mutations"],
                df_ref_dms["target_sequence"],
                token_probs,
                alphabet,
                1,
            ),
            axis=1,
        )

        file_out = Path(cfg.data.paths.zero_shot) / "ESM2" / "650M" / f"{DMS_id}.csv"
        df.to_csv(file_out, index=False)


if __name__ == "__main__":
    extract_esm2_zero_shots()
