from typing import List

import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood

from kermut.gp import KermutGP


def predict(
    gp: KermutGP,
    likelihood: GaussianLikelihood,
    test_inputs: tuple[torch.Tensor, ...],
    test_targets: torch.Tensor,
    test_fold: int,
    test_idx: List[bool],
    df_out: pd.DataFrame,
) -> pd.DataFrame:
    """Makes predictions using a trained Gaussian Process and records results.

    Evaluates the GP model on test data and stores predictions, true values, and
    uncertainty estimates in a DataFrame. The function handles model evaluation mode,
    prediction generation, and proper CPU/numpy conversion of results.

    Args:
        gp: Trained KermutGP model to use for predictions.
        likelihood: Trained Gaussian likelihood function associated with the GP model.
        x_test: Tuple of input tensors for testing. None values in the tuple
            will be filtered out.
        y_test: Target values tensor for testing.
        test_fold: Integer indicating the current test fold number for cross-validation.
        test_idx: List of boolean values indicating which rows in df_out correspond
            to the current test set.
        df_out: DataFrame to store results, must contain columns:
            - 'fold': For cross-validation fold number
            - 'y': For true target values
            - 'y_pred': For predicted mean values
            - 'y_var': For prediction variances

    Returns:
        pd.DataFrame: Updated DataFrame containing the original data plus:
            - Model predictions (mean)
            - Prediction uncertainties (variance)
            - True values
            - Fold assignments

    """
    gp.eval()
    likelihood.eval()

    x_test = tuple([x for x in test_inputs if x is not None])
    y_test = test_targets.detach().cpu().numpy()

    with torch.no_grad():
        # Predictive distribution
        y_preds_dist = likelihood(gp(*x_test))
        y_preds_mean = y_preds_dist.mean.detach().cpu().numpy()
        y_preds_var = y_preds_dist.covariance_matrix.diag().detach().cpu().numpy()

        df_out.loc[test_idx, "fold"] = test_fold
        df_out.loc[test_idx, "y"] = y_test
        df_out.loc[test_idx, "y_pred"] = y_preds_mean
        df_out.loc[test_idx, "y_var"] = y_preds_var

    return df_out
