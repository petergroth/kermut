# Based on mGPfusion by Jokinen et al. (2018)

from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pickle


def main():
    output_path = Path("data", "interim", "substitution_matrices.pkl")
    matrix_path = Path("data", "raw", "subMats.mat")
    matrix_file = loadmat(str(matrix_path))["subMats"]
    names = [name.item() for name in matrix_file[:, 1]]
    descriptions = [description.item() for description in matrix_file[:, 2]]

    full_matrix = np.zeros((21, 20, 20))
    for i in range(21):
        full_matrix[i] = matrix_file[i, 0]

    substitution_dict = {name: matrix for name, matrix in zip(names, full_matrix)}
    # Save
    with open(output_path, "wb") as f:
        pickle.dump(substitution_dict, f)


if __name__ == "__main__":
    main()
