from typing import Any
import sklearn.impute
import sklearn.ensemble
#from sklearn.experimental import enable_iterative_imputer  # noqa ? don't remember what I meant but check

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
        return RowMeanImputer()
    elif method == "median":
        from proteoflux.workflow.imputers.simpleimputers import RowMedianImputer
        return RowMedianImputer()
    elif method == "knn":
        return sklearn.impute.KNNImputer(n_neighbors=kwargs.get("knn_k", 6))
    elif method == "lc_conmed":
        from proteoflux.workflow.imputers.lc_conmed_imputer import LC_ConMedImputer
        return LC_ConMedImputer(
            condition_map=kwargs.get("condition_map"),
            sample_index=kwargs.get("sample_index"),
            group_column=kwargs.get("group_column", "CONDITION"),
            lod_k=kwargs.get("lc_conmed_lod_k", 10),
            jitter_frac=kwargs.get("lc_conmed_jitter_frac", 0.20),
            lod_shift=kwargs.get("lc_conmed_lc_shift", 0.20),
            lod_sd_width=kwargs.get("lc_conmed_lc_sd_width", 0.05),
            q_lower=kwargs.get("lc_conmed_q_lower", 0.25),
            q_upper=kwargs.get("lc_conmed_q_upper", 0.75),
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
            estimator=sklearn.ensemble.HistGradientBoostingRegressor(
                max_depth=kwargs.get("rf_max_depth", 6),
                learning_rate=kwargs.get("rf_learning_rate", 0.1),
                max_iter=kwargs.get("rf_n_estimators", 100)
            ),
            random_state=kwargs.get("random_state", 42),
            max_iter=kwargs.get("rf_max_iter", 20),
            tol=kwargs.get("rf_tol", 5e-2),
            n_nearest_features=kwargs.get("rf_nearest_features", 50),
        )
    elif method == "mindet":
        from proteoflux.workflow.imputers.min_imputers import MinDetImputer
        return MinDetImputer(
            quantile=kwargs.get("lc_quantile", 0.01),
            shift=kwargs.get("lc_shift", 0.2),
        )
    elif method == "minprob":
        from proteoflux.workflow.imputers.min_imputers import MinProbImputer
        return MinProbImputer(
            quantile=kwargs.get("lc_quantile", 0.01),
            mu_shift=kwargs.get("lc_mu_shift", 1.8),
            sd_width=kwargs.get("lc_sd_width", 0.3),
            random_state=kwargs.get("random_state", 42),
            clip_to_quantile=kwargs.get("lc_clip", True),
        )
    else:
        raise ValueError(f"Invalid imputation method: {method}. And one is needed...\n"
                           "Options: lc_conmed, mean, median, knn, tnknn, mindet, minprob, randomforest")

