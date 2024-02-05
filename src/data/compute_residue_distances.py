import pandas as pd
import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
from tqdm import tqdm


if __name__ == "__main__":
    df_ref = pd.read_csv("data/processed/DMS_substitutions_reduced.csv")
    pdb_dir = Path("data/raw/pdbs")
    distance_dir = Path("data/interim/coords")
    distance_dir.mkdir(exist_ok=True, parents=True)

    for uniprot_id in tqdm(df_ref["UniProt_ID"].unique()):
        pdb_path = pdb_dir / f"{uniprot_id}.pdb"
        try:
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
        out_path = distance_dir / f"{uniprot_id}.npy"
        np.save(out_path, coords)
