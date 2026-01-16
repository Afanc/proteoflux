"""
Centralized AnnData key schema for ProteoFlux.

This module is intentionally small and declarative: it defines the canonical
keys used in .uns / .varm / .obsm written by analysis/QC components.
"""

# -----------------------
# .uns (analysis metadata)
# -----------------------
UNS_CONTRAST_NAMES = "contrast_names"
UNS_PILOT_MODE = "pilot_study_mode"
UNS_HAS_COVARIATE = "has_covariate"
UNS_N_FULLY_IMPUTED_CELLS = "n_fully_imputed_cells"
UNS_RESIDUAL_VARIANCE = "residual_variance"

# Missingness
UNS_MISSINGNESS = "missingness"
UNS_MISSINGNESS_SOURCE = "missingness_source"
UNS_MISSINGNESS_RULE = "missingness_rule"

# -----------------------
# .varm (analysis outputs)
# -----------------------
VARM_LOG2FC = "log2fc"

VARM_SE_RAW = "se_raw"
VARM_T_RAW = "t_raw"
VARM_P_RAW = "p_raw"
VARM_Q_RAW = "q_raw"

VARM_SE_EBAYES = "se_ebayes"
VARM_T_EBAYES = "t_ebayes"
VARM_P_EBAYES = "p_ebayes"
VARM_Q_EBAYES = "q_ebayes"

# Covariate branch (phospho FT / ANCOVA)
VARM_RAW_LOG2FC = "raw_log2fc"
VARM_RAW_Q_EBAYES = "raw_q_ebayes"
VARM_COV_PART = "cov_part"

VARM_COVARIATE_BETA = "covariate_beta"
VARM_COVARIATE_SE = "covariate_se"
VARM_COVARIATE_T = "covariate_t"
VARM_COVARIATE_P = "covariate_p"
VARM_COVARIATE_Q = "covariate_q"

VARM_FT_LOG2FC = "ft_log2fc"
VARM_FT_P_EBAYES = "ft_p_ebayes"
VARM_FT_Q_EBAYES = "ft_q_ebayes"

# -----------------------
# .uns (clustering / QC)
# -----------------------
UNS_NEIGHBORS = "neighbors"
UNS_PCA = "pca"
UNS_UMAP = "umap"
UNS_MDS = "mds"

# Missingness clustering
UNS_MISSING_SAMPLE_LINKAGE = "missing_sample_linkage"
UNS_MISSING_SAMPLE_ORDER = "missing_sample_order"
UNS_MISSING_FEATURE_LINKAGE = "missing_feature_linkage"
UNS_MISSING_FEATURE_ORDER = "missing_feature_order"
UNS_MISSING_FEATURE_SUBSET_INDICES = "missing_feature_subset_indices"
UNS_MISSING_FEATURE_SELECTION = "missing_feature_selection"
UNS_MISSING_MAX_FEATURES = "missing_max_features"
