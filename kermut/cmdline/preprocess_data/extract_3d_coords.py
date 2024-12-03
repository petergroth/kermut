from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path="../hydra_configs",
    config_name="benchmark",
)
def extract_3d_coords(cfg: DictConfig) -> None:
    """Extracts the coordinates of the alpha carbons for all proteins in the ProteinGym benchmark"""

    df_ref = pd.read_csv(Path(cfg.data.paths.reference_file))
    pdb_dir = Path(cfg.data.paths.pdbs)
    distance_dir = Path(cfg.data.paths.coords)
    distance_dir.mkdir(exist_ok=True, parents=True)

    match cfg.dataset:
        case "single":
            if cfg.single.use_id:
                df_ref = df_ref[df_ref["DMS_id"] == cfg.single.id]
            else:
                df_ref = df_ref.iloc[[cfg.single.id]]
        case "all":
            pass
        case _:
            raise ValueError("Invalid dataset argument. Must be 'single' or 'all'.")

    for row in tqdm(df_ref.itertuples()):
        uniprot_id = row.UniProt_ID
        dms_id = row.DMS_id

        out_path = distance_dir / f"{dms_id}.npy"
        pdb_path = pdb_dir / f"{uniprot_id}.pdb"

        if out_path.exists():
            print(f"Skipping {dms_id} as it already exists.")
            continue

        try:
            # Fails for BRCA2_HUMAN
            structure = PDBParser().get_structure(uniprot_id, pdb_path)

        except FileNotFoundError:
            print(f"Could not find PDB file for {uniprot_id}")
            continue

        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "CA":
                            coords.append(atom.get_coord())

        coords = np.array(coords)

        # Special cases
        if dms_id == "A0A140D2T1_ZIKV_Sourisseau_2019":
            min_pos = 291 - 1
            max_pos = min_pos + len(coords)
            seq_len = len(row.target_seq)
            full_coords = np.full((seq_len, 3), np.nan)
            full_coords[min_pos:max_pos] = coords
            coords = full_coords
        elif dms_id == "POLG_HCVJF_Qi_2014":
            min_pos = 1982 - 1
            max_pos = min_pos + len(coords)
            seq_len = len(row.target_seq)
            full_coords = np.full((seq_len, 3), np.nan)
            full_coords[min_pos:max_pos] = coords
            coords = full_coords

        np.save(out_path, coords)


if __name__ == "__main__":
    extract_3d_coords()
