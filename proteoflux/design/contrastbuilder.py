import itertools
import numpy as np
import pandas as pd

class ContrastBuilder:
    def __init__(self, design_info, baseline=None):
        """
        Parameters:
        - design_info: patsy DesignInfo object from the design matrix
        - baseline: str (optional), if you want to force a specific baseline
        """
        self.design_info = design_info
        self.column_names = design_info.column_names
        self.factor_infos = design_info.factor_infos
        self.levels = self._extract_levels()
        self.baseline = baseline or self.levels[0]

    def _extract_levels(self):
        # Support only one categorical factor for now
        for factor, info in self.factor_infos.items():
            if info.type == "categorical":
                return list(info.categories)
        raise ValueError("No categorical factor found in design matrix.")

    def _contrast_vector(self, group1, group2):
        """
        Create contrast vector for group1 - group2
        """
        vec = np.zeros(len(self.column_names))

        # Intercept is always 0
        for i, name in enumerate(self.column_names):
            if name == f"C(CONDITION)[T.{group1}]":
                vec[i] = 1
            elif name == f"C(CONDITION)[T.{group2}]":
                vec[i] = -1

        return vec

    def make_all_pairwise_contrasts(self):
        """
        Generate all pairwise contrasts between levels (not just vs baseline)
        Returns:
        - contrast_matrix: np.ndarray (p x m)
        - contrast_names: list of str (e.g., "B_vs_A")
        """
        pairs = list(itertools.combinations(self.levels, 2))
        contrast_matrix = []
        contrast_names = []

        for g1, g2 in pairs:
            vec = self._contrast_vector(g1, g2)
            contrast_matrix.append(vec)
            contrast_names.append(f"{g1}_vs_{g2}")

        contrast_matrix = np.vstack(contrast_matrix).T  # shape: (p x m)
        return contrast_matrix, contrast_names

