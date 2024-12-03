from typing import Tuple, Union

import numpy as np
import pandas as pd


def compute_error_based_metrics(
    df: pd.DataFrame,
    n_bins: int = 5,
    DMS_id: str = None,
    cv_scheme: str = None,
    return_calibration_curve: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes error-based calibration metrics and optionally returns calibration curve data.

    This function calculates the Expected Normalized Calibration Error (ENCE) and
    Coefficient of Variation (CV) for predictive uncertainty estimates. It bins predictions
    by their predicted variance and compares the Root Mean Squared Error (RMSE) to the
    Root Mean Variance (RMV) in each bin.

    Args:
        df: DataFrame containing columns:
            - 'y': Ground truth values
            - 'y_pred': Predicted values
            - 'y_var': Predicted variances
            - 'fold': (Optional) Cross-validation fold indices
        n_bins: Number of bins to use for variance stratification. Defaults to 5.
        DMS_id: Dataset identifier for error reporting. Defaults to None.
        cv_scheme: Data split identifier for error reporting. Defaults to None.
        return_calibration_curve: If True, returns both metrics and calibration curve data.
            If False, returns only metrics. Defaults to True.

    Returns:
        If return_calibration_curve=False:
            DataFrame with columns ['fold', 'ENCE', 'CV']
        If return_calibration_curve=True:
            Tuple containing:
            - Original DataFrame with added columns ['sq_error', 'bin']
            - Calibration curve DataFrame with columns ['bin', 'fold', 'RMSE', 'RMV']
    """

    _df = df.copy()
    try:
        _df["sq_error"] = (_df["y"] - _df["y_pred"]) ** 2
        if "fold" not in _df.columns:
            _df["fold"] = 0
        _df["fold"] = _df["fold"].astype(int)
        _df["bin"] = np.nan

        for fold in _df["fold"].unique():
            df_fold = _df[_df["fold"] == fold]
            _df.loc[df_fold.index, "bin"] = pd.qcut(
                df_fold["y_var"], n_bins, labels=False, duplicates="drop"
            )
        _df["bin"] = _df["bin"].astype(int)

        df_curve = _df.groupby(["bin", "fold"], as_index=False).agg(
            {"sq_error": "mean", "y_var": "mean"}
        )
        df_curve["RMSE"] = np.sqrt(df_curve["sq_error"])
        df_curve["RMV"] = np.sqrt(df_curve["y_var"])
        df_curve = df_curve[["bin", "fold", "RMSE", "RMV"]]

        # Compute expected normalized calibration error
        ence = df_curve.groupby(["fold"]).apply(
            lambda x: np.mean(np.abs(x["RMV"] - x["RMSE"]) / x["RMV"]), include_groups=False
        )
        # Compute coefficient of variation
        cv = np.zeros(len(_df["fold"].unique()))
        _df["y_std"] = np.sqrt(_df["y_var"])
        for fold in _df["fold"].unique():
            df_fold = _df[_df["fold"] == fold]
            mu_sig = np.mean(df_fold["y_std"])
            _cv = np.sqrt(np.sum((df_fold["y_std"] - mu_sig) ** 2 / (len(df_fold) - 1)) / mu_sig)
            cv[fold] = _cv

        df_metrics = pd.DataFrame(dict(fold=_df["fold"].unique(), ENCE=ence.values, CV=cv))

    except ValueError:
        if DMS_id is None and cv_scheme is None:
            print("Error-based calibration metrics could not be computed")
        else:
            print(
                f"Error-based calibration metrics could not be computed for {DMS_id} ({cv_scheme})"
            )

        df_metrics = pd.DataFrame(dict(fold=_df["fold"].unique(), ENCE=np.nan, CV=np.nan))
        df_curve = pd.DataFrame(dict(bin=np.nan, fold=np.nan, RMSE=np.nan, RMV=np.nan))

    df_metrics["fold"] = df_metrics["fold"].astype(int)
    df_curve["fold"] = df_curve["fold"].astype(int)

    if return_calibration_curve:
        return df_metrics, df_curve
    return df_metrics
