"""Script to extract ESM-2 embeddings for ProteinGym DMS assays.
Adapted from https://github.com/facebookresearch/esm/blob/main/scripts/extract.py"""

from pathlib import Path

import h5py
import hydra
import pandas as pd
import torch
from esm import FastaBatchedDataset, pretrained
from omegaconf import DictConfig
from tqdm import tqdm


def _filter_datasets(cfg: DictConfig, embedding_dir: Path) -> pd.DataFrame:
    df_ref = pd.read_csv(cfg.data.paths.reference_file)
    match cfg.dataset:
        case "all":
            if cfg.data.embedding.mode == "multiples":
                df_ref = df_ref[df_ref["includes_multiple_mutants"]]
                df_ref = df_ref[df_ref["DMS_total_number_mutants"] < 7500]
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
            output_file = embedding_dir / f"{DMS_id}.h5"
            if output_file.exists():
                existing_results.append(DMS_id)
        df_ref = df_ref[~df_ref["DMS_id"].isin(existing_results)]

    return df_ref


@hydra.main(
    version_base=None,
    config_path="../hydra_configs",
    config_name="benchmark",
)
def extract_esm2_embeddings(cfg: DictConfig) -> None:
    match cfg.data.embedding.mode:
        case "singles":
            embedding_dir = Path(cfg.data.paths.embeddings_singles)
            DMS_dir = Path(cfg.data.paths.DMS_input_folder) / "cv_folds_singles_substitutions"
        case "multiples":
            embedding_dir = Path(cfg.data.paths.embeddings_multiples)
            DMS_dir = Path(cfg.data.paths.DMS_input_folder) / "cv_folds_multiples_substitutions"
        case _:
            raise ValueError(f"Invalid mode: {cfg.data.embedding.mode}")

    df_ref = _filter_datasets(cfg, embedding_dir)

    if len(df_ref) == 0:
        print("All embeddings already exist. Exiting.")
        return

    model_path = Path(cfg.data.embedding.model_path)
    model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
    model.eval()

    use_gpu = torch.cuda.is_available() and cfg.use_gpu
    if use_gpu:
        model = model.cuda()
        print("Transferred model to GPU.")

    for i, DMS_id in tqdm(enumerate(df_ref["DMS_id"])):
        print(f"--- Extracting embeddings for {DMS_id} ({i+1}/{len(df_ref)}) ---")
        df = pd.read_csv(DMS_dir / f"{DMS_id}.csv")

        mutants = df["mutant"].tolist()
        sequences = df["mutated_sequence"].tolist()
        batched_dataset = FastaBatchedDataset(sequence_strs=sequences, sequence_labels=mutants)

        batches = batched_dataset.get_batch_indices(
            cfg.data.embedding.toks_per_batch, extra_toks_per_seq=1
        )
        data_loader = torch.utils.data.DataLoader(
            batched_dataset,
            collate_fn=alphabet.get_batch_converter(truncation_seq_length=1022),
            batch_sampler=batches,
        )

        repr_layers = [33]
        all_labels = []
        all_representations = []

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                if use_gpu:
                    toks = toks.to(device="cuda", non_blocking=True)

                out = model(toks, repr_layers=repr_layers, return_contacts=False)
                representations = {
                    layer: t.to(device="cpu") for layer, t in out["representations"].items()
                }

                for i, label in enumerate(labels):
                    truncate_len = min(1022, len(strs[i]))
                    all_labels.append(label)
                    all_representations.append(
                        representations[33][i, 1 : truncate_len + 1].mean(axis=0).clone().numpy()
                    )

        assert mutants == all_labels
        embeddings_dict = {
            "embeddings": all_representations,
            "mutants": mutants,
        }

        # Store data as HDF5
        with h5py.File(embedding_dir / f"{DMS_id}.h5", "w") as h5f:
            for key, value in embeddings_dict.items():
                h5f.create_dataset(key, data=value)


if __name__ == "__main__":
    extract_esm2_embeddings()
