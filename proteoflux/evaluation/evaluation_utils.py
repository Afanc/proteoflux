import numpy as np
import pandas as pd
import warnings
from typing import Callable, List, Dict

def geometric_cv(values: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Computes the Geometric Coefficient of Variation (GCV) from log-transformed values.

    GCV = sqrt(e ** (σ²) - 1), where σ is the std of the log-transformed values.

    Parameters:
        values (np.ndarray): Log-transformed matrix (e.g. log2 intensities)
        axis (int): Axis along which to compute std (default = 1, rows = features)

    Returns:
        np.ndarray: GCV values, one per feature.
    """
    log2_std = np.nanstd(values, axis=axis, ddof=1)
    ln2_squared = np.log(2)**2
    return np.sqrt(np.exp(ln2_squared * log2_std**2) - 1)

# TODO sketchy, abandon
def log_cv(values: np.ndarray, axis: int = 1, log_base: float = 2) -> np.ndarray:
    """
    Computes %CV from log-transformed values.

    Parameters:
        values: Log-transformed intensity matrix.
        axis: Axis for computation.
        log_base: Log base of the transformation (default: 2).

    Returns:
        Vector of %CV estimates.
    """
    log_std = np.nanstd(values, axis=axis)

    # Prevent exponential explosion
    factor = np.log(log_base)
    unlog_variance = np.power(log_base, log_std * factor) - 1

    # Optionally clip extreme values if needed (e.g. when std is crazy)
    unlog_variance = np.clip(unlog_variance, 0, 1e4)

    return unlog_variance

def compute_metrics(mat: np.ndarray, metrics: List[str] = ["CV", "MAD"]) -> Dict[str, np.ndarray]:
    """
    Compute per-feature metrics across samples.

    Parameters:
        mat (np.ndarray): 2D array (features x samples).
        metrics (List[str], optional): List of metric names to compute.
            Supported metrics include:
              - "CV": Coefficient of variation (std / mean)
              - "MAD": Median absolute deviation (with respect to the median)
              - "PEV": Population explained variance (i.e. variance)
              - "Mean": Mean value
              - "Median": Median value
              - "STD": Standard deviation
            Defaults to ["CV", "MAD", "PEV"].

    Returns:
        Dict[str, np.ndarray]: Dictionary with metric names as keys and 1D arrays
                               (one value per feature) as values.
    """
    if metrics is None:
        metrics = ["CV", "MAD", "PEV"]

    result = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        means   = np.nanmean(mat, axis=1)
        medians = np.nanmedian(mat, axis=1)
        stds    = np.nanstd(mat, axis=1, ddof=1)
        log_cvs = log_cv(mat, axis=1, log_base=2)
        geometric_cvs = geometric_cv(mat, axis=1)
        mads = np.nanmedian(np.abs(mat - medians[:, None]), axis=1)
        rmads = mads/medians

        if "CV" in metrics:
            result["CV"] = stds / means
        if "log_CV" in metrics:
            result["log_CV"] = log_cvs
        if "geometric_CV" in metrics:
            result["geometric_CV"] = geometric_cvs
        if "MAD" in metrics:
            result["MAD"] = mads
        if "RMAD" in metrics:
            result["RMAD"] = rmads
        if "PEV" in metrics:
            result["PEV"] = np.nanvar(mat, axis=1, ddof=1)
        if "Mean" in metrics:
            result["Mean"] = means
        if "Median" in metrics:
            result["Median"] = medians
        if "STD" in metrics:
            result["STD"] = stds

    return result

def create_long_metric_df(
    series_before: pd.Series,
    series_after: pd.Series,
    metric_name: str,
    cond_label: str,
    label_col: str,
    before_label: str,
    after_label: str,
) -> pd.DataFrame:
    """
    Create a long-format DataFrame for comparing a given metric.

    Args:
        series_before (pd.Series): Metric values before processing.
        series_after (pd.Series): Metric values after processing.
        metric_name (str): Name of the metric (e.g., "CV", "MAD", "PEV", etc.).
        cond_label (str): Condition label (e.g., "Total" or a specific condition).
        label_col (str): Column name for the processing label (e.g., "Imputation" or "Normalization").
        before_label (str): Label for the pre-processed data (e.g., "Original" or "Before").
        after_label (str): Label for the processed data (e.g., "Imputed" or "After").

    Returns:
        pd.DataFrame: Long-format DataFrame with columns 'Condition', 'Value', label_col, and 'Metric'.
    """
    n_total = len(series_before) + len(series_after)
    df = pd.DataFrame({
        "Condition": [cond_label] * n_total,
        "Value": np.concatenate([series_before.values, series_after.values]),
        label_col: ([before_label] * len(series_before)) +
                   ([after_label] * len(series_after)),
        "Metric": [metric_name] * n_total
    })
    return df

def prepare_long_df(
    df: pd.DataFrame,
    label: str,
    label_col: str,
    condition_mapping: pd.DataFrame = None,
    sample_col: str = "Sample",
    intensity_col: str = "Intensity",
) -> pd.DataFrame:
    """ Convert a wide-format DataFrame (samples as columns) to long-format, filter out non-positive intensities, optionally merge with a condition mapping, and add a processing label.
    Args:
        df (pd.DataFrame): Wide-format DataFrame.
        label (str): The label to assign (e.g., "Imputed" or "Before/After").
        label_col (str): The column name for the label (e.g., "Imputation" or "Normalization").
        condition_mapping (pd.DataFrame, optional): Mapping to merge on 'Sample'.

    Returns:
        pd.DataFrame: Long-format DataFrame with 'Sample', 'Intensity', and the label_col.
    """
    df_long = df.melt(var_name=sample_col, value_name=intensity_col)
    df_long = df_long[df_long[intensity_col] > 0].copy()
    if condition_mapping is not None:
        df_long = df_long.merge(condition_mapping, on=sample_col, how="left")
    df_long[label_col] = label
    return df_long

def aggregate_metrics(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    condition_mapping: pd.DataFrame,
    metrics: List[str] = None,
    label_col: str = "Imputation",
    before_label: str = "Original",
    after_label: str = "Imputed",
    condition_key: str = "Condition",
    sample_key: str = "Sample",
    long_df_creator: Callable[
        [pd.Series, pd.Series, str, str, str, str, str], pd.DataFrame
    ] = create_long_metric_df,
) -> Dict[str, pd.DataFrame]:
    """
    Compute global and per-condition metrics and aggregate them into a dictionary.

    Args:
        before_df (pd.DataFrame): Wide-format DataFrame before processing.
        after_df (pd.DataFrame): Wide-format DataFrame after processing.
        condition_mapping (pd.DataFrame): Mapping DataFrame containing at least the columns
            specified by condition_key and sample_key.
        compute_metrics_func (Callable): Function that takes a numpy array and a list of metric names,
            and returns a dictionary with metric names as keys.
        metrics (List[str], optional): List of metric names to aggregate. Defaults to ["CV", "MAD", "PEV"].
        label_col (str): Column name to label the processing (e.g., "Imputation" or "Normalization").
        before_label (str): Label for the original data.
        after_label (str): Label for the processed data.
        condition_key (str): Column name in condition_mapping that indicates condition.
        sample_key (str): Column name in condition_mapping that indicates sample.
        long_df_creator (Callable): Function to create the long-format DataFrame. Defaults to create_long_metric_df.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys as metric names and values as aggregated long-format DataFrames.
    """
    if metrics is None:
        metrics = ["CV", "MAD", "PEV"]

    global_metrics_before = compute_metrics(before_df.to_numpy(), metrics=metrics)
    global_metrics_after = compute_metrics(after_df.to_numpy(), metrics=metrics)

    result = {}
    for metric in metrics:
        # Global metrics labeled as "Total"
        df_global = long_df_creator(
            pd.Series(global_metrics_before[metric]),
            pd.Series(global_metrics_after[metric]),
            metric_name=metric,
            cond_label="Total",
            label_col=label_col,
            before_label=before_label,
            after_label=after_label,
        )
        data_list = [df_global]

        # Per-condition metrics
        for cond in condition_mapping[condition_key].unique():
            samples = condition_mapping.loc[
                condition_mapping[condition_key] == cond, sample_key
            ].tolist()
            before_subset = before_df.loc[:, before_df.columns.isin(samples)].to_numpy()
            after_subset = after_df.loc[:, after_df.columns.isin(samples)].to_numpy()
            cond_metrics_before = compute_metrics(before_subset, metrics=metrics)[metric]
            cond_metrics_after = compute_metrics(after_subset, metrics=metrics)[metric]

            df_cond = long_df_creator(
                pd.Series(cond_metrics_before),
                pd.Series(cond_metrics_after),
                metric_name=metric,
                cond_label=cond,
                label_col=label_col,
                before_label=before_label,
                after_label=after_label,
            )
            data_list.append(df_cond)
        result[metric] = pd.concat(data_list, ignore_index=True)
    return result
