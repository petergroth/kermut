"""Adapted from https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/gvp_transformer.py#L88"""
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from esm.inverse_folding.util import rotate, CoordBatchConverter
from tqdm import trange

from src import ALPHABET


if __name__ == "__main__":
    model_name = "esm_if1_gvp4_t16_142M_UR50"
    model_path = Path("models", f"{model_name}.pt")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(model_path))
    aa_indices = [alphabet.get_idx(aa) for aa in ALPHABET]
    model.eval()
    df = pd.read_csv("data/interim/DMS_substitutions_reduced.csv")
    with torch.no_grad():
        for UniProt_ID in df["UniProt_ID"].unique()[15:]:
            print(f"--- {UniProt_ID} ---")
            fpath = f"data/raw/pdbs/{UniProt_ID}.pdb"
            chain_id = "A"
            try:
                structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
                coords, seq = esm.inverse_folding.util.extract_coords_from_structure(
                    structure
                )

                seq = list(seq)
                partial_seq = seq
                confidence = None
                device = None

                L = len(coords)
                # Convert to batch format
                batch_converter = CoordBatchConverter(model.decoder.dictionary)
                batch_coords, confidence, _, _, padding_mask = batch_converter(
                    [(coords, confidence, None)], device=device
                )

                # Run encoder only once
                encoder_out = model.encoder(batch_coords, padding_mask, confidence)

                # Start with prepend token
                mask_idx = model.decoder.dictionary.get_idx("<mask>")
                tokens = torch.full((1, 1 + L), mask_idx, dtype=int)
                tokens[0, 0] = model.decoder.dictionary.get_idx("<cath>")
                if partial_seq is not None:
                    for i, c in enumerate(partial_seq):
                        tokens[0, i + 1] = model.decoder.dictionary.get_idx(c)
                conditional_probabilities = torch.zeros(
                    (L, len(model.decoder.dictionary))
                )
                for i in trange(1, L + 1):
                    # Distribution at position i is conditioned on positions < i
                    logits, _ = model.decoder(
                        tokens[:, :i],
                        encoder_out,
                    )
                    logits = logits[0].transpose(0, 1)

                    probs = F.softmax(logits, dim=-1)
                    conditional_probabilities[i - 1] = probs[-1]

                # Reorder
                conditional_probabilities = conditional_probabilities[:, aa_indices]
                conditional_probabilities = conditional_probabilities.detach().numpy()

                # Save
                np.save(
                    f"data/conditional_probs/{UniProt_ID}/{UniProt_ID}_ESM_IF1.npy",
                    conditional_probabilities,
                )
            except FileNotFoundError:
                print(f"File {fpath} not found.")
