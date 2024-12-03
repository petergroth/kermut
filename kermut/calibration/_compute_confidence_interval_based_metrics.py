from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_interval_based_metrics(
    df: pd.DataFrame,
    n_bins: int = 10,
    DMS_id: str = None,
    cv_scheme: str = None,
    return_calibration_curve: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Computes confidence interval based calibration metrics and calibration curves.

    This function calculates the Expected Calibration Error (ECE) based on confidence
    intervals. For each confidence level, it checks if the true value falls within
    the predicted confidence interval and compares the empirical coverage to the
    expected coverage.

    Args:
        df: DataFrame containing columns:
            - 'y': Ground truth values
            - 'y_pred': Predicted values
            - 'y_var': Predicted variances
            - 'fold': Cross-validation fold indices
        n_bins: Number of confidence levels to evaluate between 0 and 1. Defaults to 10.
        DMS_id: Dataset identifier for error reporting. Defaults to None.
        cv_scheme: Data split identifier for error reporting. Defaults to None.
        return_calibration_curve: If True, returns both metrics and calibration curve data.
            If False, returns only metrics. Defaults to True.

    Returns:
        If return_calibration_curve=False:
            DataFrame with columns ['fold', 'ECE']
        If return_calibration_curve=True:
            Tuple containing:
            - Original DataFrame
            - Calibration curve DataFrame with columns ['fold', 'confidence', 'percentile']
    """

    _df = df.copy()
    perc = np.arange(0, 1.1, 1 / n_bins)

    try:
        df_metrics = pd.DataFrame(_df["fold"].unique(), columns=["fold"])
        df_metrics = df_metrics.assign(ECE=np.nan)
        n = (n_bins + 1) * len(_df["fold"].unique())
        df_curve = pd.DataFrame(
            np.full((n, 3), fill_value=np.nan), columns=["fold", "confidence", "percentile"]
        )

        for i, fold in enumerate(_df["fold"].unique()):
            df_fold = _df[_df["fold"] == fold]
            y_target = df_fold["y"].values
            y_pred = df_fold["y_pred"].values
            y_var = df_fold["y_var"].values

            count_arr = np.vstack(
                [
                    np.abs(y_target - y_pred)
                    <= stats.norm.interval(q, loc=np.zeros(len(y_pred)), scale=np.sqrt(y_var))[1]
                    for q in perc
                ]
            )
            count = np.mean(count_arr, axis=1)
            ECE = np.mean(np.abs(count - perc))
            df_metrics.loc[df_metrics["fold"] == fold, "ECE"] = ECE

            slice_start = i * (n_bins + 1)
            slice_end = (i + 1) * (n_bins + 1) - 1

            df_curve.loc[slice_start:slice_end, "fold"] = fold
            df_curve.loc[slice_start:slice_end, "confidence"] = count
            df_curve.loc[slice_start:slice_end, "percentile"] = perc
    except ValueError:
        if DMS_id is None and cv_scheme is None:
            print("CI-based calibration metrics could not be computed")
        else:
            print(f"CI-based calibration metrics could not be computed for {DMS_id} ({cv_scheme})")
        df_metrics = pd.DataFrame(dict(fold=_df["fold"].unique(), ECE=np.nan))
        df_curve = pd.DataFrame(
            dict(fold=_df["fold"].unique(), confidence=np.nan, percentile=np.nan)
        )

    df_metrics["fold"] = df_metrics["fold"].astype(int)
    df_curve["fold"] = df_curve["fold"].astype(int)

    if return_calibration_curve:
        return df_metrics, df_curve
    return df_metrics
