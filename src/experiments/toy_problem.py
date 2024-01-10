import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hydra

from src.model.utils import hellinger_distance
from src.model.gp import ExactGPKermut

sns.set_style("dark")


def eval_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    hellinger_matrix: torch.Tensor,
    wt_sequence: torch.LongTensor,
    blosum_matrix: torch.Tensor = None,
    h_scale: float = 0.9,
    h_lengthscale: float = 0.5,
    **kwargs,
):
    x1_idx = torch.argwhere(wt_sequence != x1)
    x2_idx = torch.argwhere(wt_sequence != x2)
    x1_toks = x1[x1_idx[:, 0], x1_idx[:, 1]]
    x2_toks = x2[x2_idx[:, 0], x2_idx[:, 1]]
    x1_wt_toks = wt_sequence[x1_idx[:, 1]]
    x2_wt_toks = wt_sequence[x2_idx[:, 1]]

    # Hellinger kernel
    d_hn = hellinger_matrix[x1_idx[:, 1].unsqueeze(1), x2_idx[:, 1].unsqueeze(0)]
    k_hn = h_scale * torch.exp(d_hn * h_lengthscale)

    # Blosum kernel
    if blosum_matrix is not None:
        bl_src = blosum_matrix[x1_wt_toks][:, x2_wt_toks]
        bl_tar = blosum_matrix[x1_toks][:, x2_toks]
        k_bl = bl_src * bl_tar
    else:
        k_bl = torch.ones_like(k_hn)

    # Combine kernels
    k_mult = k_hn + (k_bl * k_hn)

    # Sum over sequences
    one_hot_x1 = torch.zeros(x1_idx[:, 0].size(0), x1_idx[:, 0].max().item() + 1)
    one_hot_x2 = torch.zeros(x2_idx[:, 0].size(0), x2_idx[:, 0].max().item() + 1)
    one_hot_x1.scatter_(1, x1_idx[:, 0].unsqueeze(1), 1)
    one_hot_x2.scatter_(1, x2_idx[:, 0].unsqueeze(1), 1)

    K_x1x2 = torch.transpose(
        torch.transpose(k_mult @ one_hot_x2, 0, 1) @ one_hot_x1, 0, 1
    )

    # norm = torch.sum(one_hot_x1, dim=0).unsqueeze(1) @ torch.sum(
    #     one_hot_x2, dim=0
    # ).unsqueeze(0)
    # K_x1x2 = K_x1x2 / norm

    return K_x1x2


def imshow_matrix(
    hellinger_matrix: torch.Tensor,
    blosum_matrix: torch.Tensor = None,
):
    # To numpy
    hellinger_matrix_np = hellinger_matrix.numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.heatmap(
        data=hellinger_matrix_np, cmap="magma_r", annot=True, ax=ax[0], square=True
    )
    ax[0].set_title('Position "distance"')
    # Set x-ticks to 12345
    ax[0].set_xticklabels(["1", "2", "3", "4", "5"])
    ax[0].set_yticklabels(["1", "2", "3", "4", "5"], rotation=0)
    if blosum_matrix is not None:
        blosum_matrix_np = blosum_matrix.numpy()
        sns.heatmap(
            data=blosum_matrix_np, cmap="magma", annot=True, ax=ax[1], square=True
        )
        ax[1].set_title('AA "similarity"')
        # Set x-ticks to ABCD
        ax[1].set_xticklabels(["A", "B", "C", "D"])
        ax[1].set_yticklabels(["A", "B", "C", "D"], rotation=90)
    plt.tight_layout()
    plt.show()


def imshow_kernel(kernel_matrix: torch.Tensor, annotation: str = None):
    fig, ax = plt.subplots(
        figsize=(kernel_matrix.shape[1] * 1.5, kernel_matrix.shape[0] * 1.5)
    )
    sns.heatmap(
        data=kernel_matrix, annot=True, fmt=".2f", ax=ax, square=True, cmap="magma_r"
    )
    if annotation is not None:
        ax.set_title(annotation)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # WT
    wt_sequence = torch.LongTensor([0, 0, 0, 0, 0])  # AAAAA
    # TRAIN
    a = torch.LongTensor([1, 0, 0, 0, 0])  # BAAAA
    b = torch.LongTensor([0, 2, 0, 0, 0])  # ACAAA
    c = torch.LongTensor([0, 0, 1, 0, 0])  # AABAA
    # TEST
    d_1 = torch.LongTensor([1, 2, 0, 0, 0])  # BCAAA
    d_2 = torch.LongTensor([1, 2, 1, 0, 0])  # BCBAA
    d_3 = torch.LongTensor([1, 2, 1, 0, 3])  # BCBAD
    d_4 = torch.LongTensor([1, 1, 0, 0, 0])  # BBAAA

    # Labels
    y_a = torch.tensor(0.50)
    y_b = torch.tensor(1.00)
    y_c = torch.tensor(0.25)

    overwrites = []
    noise = 0.01
    kernel_name = "kermutBH"
    show_matrices = True
    show_kernels = True
    x_train = torch.stack([a, b, c])
    x_test = torch.stack([d_1, d_2, d_3, d_4])
    Y_X = torch.tensor([y_a, y_b, y_c])

    # SIMILAR:
    # (0, 1), (2, 3) (4)
    # DISSIMILAR:
    # (4, *)
    # MEDIUM:
    # (0, 2), (0, 3), (1, 2), (1, 3)

    # Define probability distributions
    conditional_probs = torch.tensor(
        [
            [0.6, 0.2, 0.15, 0.05],
            [0.55, 0.25, 0.15, 0.05],
            [0.4, 0.2, 0.3, 0.1],
            [0.35, 0.25, 0.3, 0.1],
            [0.97, 0.01, 0.01, 0.01],
        ]
    )

    hellinger_matrix = hellinger_distance(conditional_probs, conditional_probs)
    # eigs = torch.linalg.eigvals(torch.exp(-hellinger_matrix))
    # print(eigs)

    # SIMILAR:
    # (0, 1), (2, 3)
    # DISSIMILAR:
    # (3, *)
    # MEDIUM:
    # (0, 3)
    blosum_matrix = torch.tensor(
        [
            [
                0.6956521739130435,
                0.30434782608695654,
                0.391304347826087,
                0.21739130434782608,
            ],
            [
                0.30434782608695654,
                0.6521739130434783,
                0.43478260869565216,
                0.17391304347826086,
            ],
            [
                0.391304347826087,
                0.43478260869565216,
                0.6086956521739131,
                0.17391304347826086,
            ],
            [
                0.21739130434782608,
                0.17391304347826086,
                0.17391304347826086,
                0.6521739130434783,
            ],
        ]
    )
    # blosum_matrix = torch.eye(4)

    # eigs = torch.linalg.eigvals(blosum_matrix)
    # print(eigs)

    if show_matrices:
        imshow_matrix(hellinger_matrix, blosum_matrix)

    # Load config file with hydra
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(
            config_name="gp_regression",
            overrides=[
                f"gp={kernel_name}",
                "experiment=regression_BLAT_ECOLX",
                *overwrites,
            ],
        )

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(noise)
    x_train_ = x_train
    x_test_ = x_test
    y_train_ = Y_X
    GP = ExactGPKermut(
        train_x=x_train_,
        train_y=y_train_,
        likelihood=likelihood,
        gp_cfg=cfg.gp,
        conditional_probs=conditional_probs,
        wt_sequence=wt_sequence,
    )
    GP.eval()
    likelihood.eval()

    K_XX = GP.covar_module(x_train).evaluate()
    K_YX = GP.covar_module(x_test, x_train).evaluate()
    K_YY = GP.covar_module(x_test).evaluate()

    # GP predictions
    pred = likelihood(GP(x_test_))
    GP_mean = pred.mean
    GP_var = pred.variance
    GP_covar = pred.covariance_matrix

    # to numpy
    GP_mean = GP_mean.detach().squeeze().numpy()
    GP_var = GP_var.detach().squeeze().numpy()
    GP_covar = GP_covar.detach().numpy()
    K_XX = K_XX.detach().numpy()
    K_YY = K_YY.detach().numpy()
    K_YX = K_YX.detach().numpy()
    K_XY = K_YX.T
    np.set_printoptions(precision=2)
    if len(GP_mean.shape) == 0:
        print(f"Mean: {GP_mean}, std: {np.sqrt(GP_var)}")
    else:
        print(f"Mean: \n{GP_mean}")
        print(f"Standard deviation: \n{np.sqrt(GP_var)}")
        print(f"Covariance matrix: \n{GP_covar}")

    if show_kernels:
        imshow_kernel(K_XX, r"$K(X, X)$")
        imshow_kernel(K_YY, r"$K(X_*, X_*)$")
        imshow_kernel(K_YX, r"$K(X_*, X)$")
