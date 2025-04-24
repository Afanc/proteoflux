from typing import Any
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer  # noqa ? don't remember what I meant but check
import sklearn.ensemble

def get_imputer(**kwargs) -> Any:
    """
    Returns an imputer instance based on the given method.

    Valid methods:
        - "mean": RowMeanImputer from simpleimputers.
        - "median": RowMedianImputer from simpleimputers.
        - "knn": KNNImputer from scikit-learn.
        - "nsknn": NSkNNImputer.
        - "tnknn": KNNTNImputer.
        - "randomforest": IterativeImputer using a HistGradientBoostingRegressor.

    kwargs are passed to the imputer constructors.
    """
    method = kwargs.pop("method", None)

    if method == "mean":
        from proteoflux.workflow.imputers.simpleimputers import RowMeanImputer
        return RowMeanImputer(**kwargs)
    elif method == "median":
        from proteoflux.workflow.imputers.simpleimputers import RowMedianImputer
        return RowMedianImputer(**kwargs)
    elif method == "knn":
        return sklearn.impute.KNNImputer(n_neighbors=kwargs.get("knn_k", 6))
    elif method == "hybridknn":
        from proteoflux.workflow.imputers.hybridimputer import HybridImputer
        return HybridImputer(
            condition_map=kwargs.get("condition_map"),
            group_column=kwargs.get("group_column", "CONDITION"),
            sample_index=kwargs.get("sample_index"),
            knn_k=kwargs.get("knn_k", 6),
            knn_weights=kwargs.get("knn_weights", "distance"),
            knn_metric=kwargs.get("knn_metric", "nan_euclidean"),
            gaussian_clip=kwargs.get("gaussian_clip", 0.5),
            gaussian_left_shift=kwargs.get("gaussian_left_shift", 1.8),
            random_state=kwargs.get("random_state", 42),
        )
    elif method == "nsknn":
        from proteoflux.workflow.imputers.nsknnimputer import NSkNNImputer
        return NSkNNImputer(n_neighbors=kwargs.get("knn_k", 6))
    elif method == "tnknn":
        from proteoflux.workflow.imputers.tnknnimputer import KNNTNImputer
        return KNNTNImputer(
            n_neighbors=kwargs.get("knn_k", 10),
            perc=kwargs.get("knn_tn_perc", 0.75)
        )
    elif method == "randomforest":
        return sklearn.impute.IterativeImputer(
            estimator=sklearn.ensemble.HistGradientBoostingRegressor(),
            random_state=kwargs.get("rf_random_state", 42),
            max_iter=kwargs.get("rf_max_iter", 20),
        )
    else:
        raise ValueError(f"Invalid imputation method: {method}. And one is needed...")

