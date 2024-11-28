from typing import Sequence

import torch

from kermut.constants import ALPHABET, AA_TO_IDX


class Tokenizer:
    """Tokenizer for amino acid sequences. Converts sequences to one-hot encoded tensors."""

    def __init__(self, flatten: bool = True):
        super().__init__()
        # Uses the standard 20 amino acids: ACDEFGHIKLMNPQRSTVWY
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
