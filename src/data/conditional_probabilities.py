import argparse

import torch

from esm import pretrained
import pandas as pd
from tqdm import trange
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np
from pathlib import Path
from src import AA_TO_IDX


def compute_ESM_masked_plls(dataset: str):
    # Load wildtype sequence
    output_path = Path("data", "interim", dataset, "esm2_masked_probs.pt")
    wt_df = pd.read_csv("data/processed/wt_sequences.tsv", sep="\t")
    wt_sequence = wt_df.loc[wt_df["dataset"] == dataset, "seq"].item()
    # Load model
    model_name = "esm2_t33_650M_UR50D"
    model_path = Path("models", f"{model_name}.pt")
    model, alphabet = pretrained.load_model_and_alphabet_local(str(model_path))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    # Tokenize wt
    batch_converter = alphabet.get_batch_converter()
    data = [("wt", wt_sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Compute masked pseudo-probabilities at each position conditioned on the rest
    log_probs = torch.zeros((len(wt_sequence), len(alphabet)))
    for i in trange(1, len(wt_sequence) + 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        if torch.cuda.is_available():
            batch_tokens_masked = batch_tokens_masked.cuda()
        with torch.no_grad():
            token_probs = torch.softmax(model(batch_tokens_masked)["logits"], dim=-1)
        log_probs[i - 1] = token_probs[0, i]

    # Alphabet reordering to align with Kermut
    order = [AA_TO_IDX[aa] for aa in alphabet.all_toks if aa in AA_TO_IDX.keys()]
    p = log_probs[:, order].clone()
    torch.save(p, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-dataset", type=str, required=True)
    args = parser.parse_args()
    compute_ESM_masked_plls(args.dataset)
