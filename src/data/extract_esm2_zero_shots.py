"""Adapted from https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py"""

import argparse
import pathlib

import torch

from esm import pretrained
import pandas as pd
from tqdm import tqdm

from pathlib import Path


def label_row(row, sequence, token_probs, alphabet, offset_idx):
    mutations = row.split(":")
    score = 0
    for mutation in mutations:
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert (
            sequence[idx] == wt
        ), "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score += (
            token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
        ).item()

    return score


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert (
        sequence[idx] == wt
    ), "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens_masked.cuda())["logits"], dim=-1
            )
        log_probs.append(
            token_probs[0, i, alphabet.get_idx(sequence[i])].item()
        )  # vocab size
    return sum(log_probs)


def compute_zero_shot(dataset: str, model, alphabet, nogpu: bool, overwrite: bool):
    file_out = Path(
        "data", "zero_shot_fitness_predictions", "ESM2/650M", f"{dataset}.csv"
    )
    if file_out.exists() and not overwrite:
        print(f"Predictions for {dataset} already exist. Skipping.")
        return
    else:
        print(f"--- {dataset} ---")

    # Load data
    df_ref = pd.read_csv(Path("data", "DMS_substitutions.csv"))
    df_wt = df_ref.loc[df_ref["DMS_id"] == dataset]
    reference_seq = df_wt["target_seq"].iloc[0]

    if (
        df_wt["includes_multiple_mutants"].iloc[0]
        and df_wt["DMS_total_number_mutants"].iloc[0] <= 7500
    ):
        file_in = Path("data", "substitutions_multiples", f"{dataset}.csv")
    else:
        file_in = Path("data", "substitutions_singles", f"{dataset}.csv")

    df = pd.read_csv(file_in)

    score_key = "esm2_t33_650M_UR50D"
    batch_converter = alphabet.get_batch_converter()
    data = [
        ("protein1", reference_seq),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    if args.scoring_strategy == "wt-marginals":
        with torch.no_grad():
            if torch.cuda.is_available() and not nogpu:
                batch_tokens = batch_tokens.cuda()
            token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
        df[score_key] = df.apply(
            lambda row: label_row(
                row[args.mutation_col],
                args.sequence,
                token_probs,
                alphabet,
                args.offset_idx,
            ),
            axis=1,
        )
    elif args.scoring_strategy == "masked-marginals":
        all_token_probs = []
        for i in tqdm(range(batch_tokens.size(1))):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = alphabet.mask_idx
            with torch.no_grad():
                if torch.cuda.is_available() and not nogpu:
                    batch_tokens_masked = batch_tokens_masked.cuda()
                token_probs = torch.log_softmax(
                    model(batch_tokens_masked)["logits"], dim=-1
                )
            all_token_probs.append(token_probs[:, i])  # vocab size
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        df[score_key] = df.apply(
            lambda row: label_row(
                row["mutations"],
                reference_seq,
                token_probs,
                alphabet,
                1,
            ),
            axis=1,
        )
    elif args.scoring_strategy == "pseudo-ppl":
        tqdm.pandas()
        df[score_key] = df.progress_apply(
            lambda row: compute_pppl(
                row[args.mutation_col],
                args.sequence,
                model,
                alphabet,
                args.offset_idx,
            ),
            axis=1,
        )

    df.to_csv(file_out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="masked-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help="",
    )
    parser.add_argument(
        "--nogpu", action="store_true", help="Do not use GPU even if available"
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    model_path = Path("models") / "esm2_t33_650M_UR50D.pt"
    model, alphabet = pretrained.load_model_and_alphabet_local(model_path)
    model.eval()

    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU.")

    if args.dataset == "all":
        df_ref = pd.read_csv(Path("data", "DMS_substitutions.csv"))
        datasets = df_ref["DMS_id"].tolist()
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        try:
            compute_zero_shot(
                dataset=dataset,
                model=model,
                alphabet=alphabet,
                nogpu=args.nogpu,
                overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"Error in {dataset}: {e}")
            continue
