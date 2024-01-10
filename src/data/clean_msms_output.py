import argparse
import pandas as pd
from pathlib import Path


def main(wt_key: str):
    in_path = Path(f"data", "interim", wt_key, f"{wt_key}.vert")
    out_path = Path(f"data", "processed", f"{wt_key}_surface.csv")
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "face_idx",
        "closest_sphere",
        "misc",
        "name",
    ]
    df = pd.read_csv(in_path, header=None, delim_whitespace=True, names=names)
    # Position kept in final col
    df["pos"] = df["name"].str.split("_").str[-1].astype(int)
    # Average of all vertices for each position
    df_sum = df.groupby("pos").agg("mean", numeric_only=True).reset_index()
    df_summary = df_sum[["x", "y", "z", "pos"]]
    # Save to disc
    df_summary.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wt_key",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(**vars(args))
