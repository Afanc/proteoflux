import numpy as np
import pandas as pd
from proteoflux.utils.utils import logger, log_time

class LinearModelFitter:
    def __init__(self, expression: np.ndarray, design_matrix: np.ndarray):
        """
        Parameters:
        - expression: (n_samples x n_proteins) matrix (adata.X or a layer)
        - design_matrix: (n_samples x n_covariates) matrix from DesignMatrixBuilder
        """
        self.Y = expression
        self.X = design_matrix
        self.coefficients = None
        self.residuals = None
        self.residual_variance = None
        self.df_residual = self.X.shape[0] - np.linalg.matrix_rank(self.X)
        self.xtx_inv = None  # (X^T X)^(-1)

    @log_time("Linear Regressions")
    def fit(self):
        """
        Fits OLS for all proteins simultaneously.
        Vectorized across proteins.
        """
        X = self.X
        Y = self.Y  # shape: (n_samples x n_proteins)

        # Precompute pseudoinverse
        self.xtx_inv = np.linalg.inv(X.T @ X)

        # Fit coefficients for all proteins
        betas = self.xtx_inv @ X.T @ Y  # shape: (n_covariates x n_proteins)
        self.coefficients = betas.T     # shape: (n_proteins x n_covariates)

        # Compute residuals
        fitted = X @ betas              # shape: (n_samples x n_proteins)
        resid = Y - fitted
        self.residuals = resid.T        # shape: (n_proteins x n_samples)

        # Compute residual variances
        rss = np.sum(resid**2, axis=0)  # shape: (n_proteins,)
        self.residual_variance = rss / self.df_residual

        return self

    def get_results(self) -> dict:
        """
        Returns a dictionary of results.
        """
        return {
            "coefficients": self.coefficients,
            "residuals": self.residuals,
            "residual_variance": self.residual_variance,
            "df_residual": self.df_residual,
            "xtx_inv": self.xtx_inv
        }

