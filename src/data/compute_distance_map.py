import torch
from pathlib import Path
from src.model.utils import get_coords_from_pdb


def main(dataset: str):
    out_path = Path("data/interim", dataset, "contact_map.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = get_coords_from_pdb(dataset, as_tensor=True)
    distances = torch.cdist(coords, coords, p=2)
    # Save
    torch.save(distances, out_path)


if __name__ == "__main__":
    datasets = ["SPG1", "GFP", "BLAT_ECOLX", "PARD3_10"]
    for dataset in datasets:
        main(dataset)
