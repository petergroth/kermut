
from typing import List

import pandas as pd
import torch

from gpytorch.likelihoods import GaussianLikelihood

from kermut.gp import KermutGP


def predict(
    gp: KermutGP, 
    likelihood: GaussianLikelihood, 
    x_test: torch.Tensor, 
    y_test: torch.Tensor, 
    test_fold: int, 
    test_idx: List[bool],
    df_out: pd.DataFrame
    ) -> pd.DataFrame:
    gp.eval()
    likelihood.eval()

    x_test = tuple([x for x in x_test if x is not None])

    with torch.no_grad():
        # Predictive distribution
        y_preds_dist = likelihood(gp(*x_test))
        y_preds_mean = y_preds_dist.mean.detach().cpu().numpy()
        y_preds_var = (
            y_preds_dist.covariance_matrix.diag().detach().cpu().numpy()
        )
        y_test = y_test.detach().cpu().numpy()

        df_out.loc[test_idx, "fold"] = test_fold
        df_out.loc[test_idx, "y"] = y_test
        df_out.loc[test_idx, "y_pred"] = y_preds_mean
        df_out.loc[test_idx, "y_var"] = y_preds_var
        
    return df_out