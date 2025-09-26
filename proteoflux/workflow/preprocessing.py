import time
import sys
from typing import Optional, List, Tuple

import numpy as np
import polars as pl

import sklearn.impute
import sklearn.preprocessing
import sklearn.ensemble
from skmisc.loess import loess

from proteoflux.workflow.normalizers.regression_normalization import regression_normalization
from proteoflux.workflow.imputer_factory import get_imputer
from proteoflux.dataset.preprocessresults import PreprocessResults
from proteoflux.dataset.intermediateresults import IntermediateResults
from proteoflux.evaluation.normalization_evaluator import NormalizerPlotter
from proteoflux.evaluation.imputation_evaluator import ImputeEvaluator
from proteoflux.utils.utils import load_contaminant_accessions, log_time, logger, log_info
from proteoflux.utils.directlfq import estimate_protein_intensities
import proteoflux.utils.directlfq_config as dlcfg

class Preprocessor:
    """Handles filtering, normalization, and imputation for proteomics data."""

    available_normalization = ["zscore", "vst", "linear"]
    available_imputation = ["mean", "median", "knn", "randomforest"]

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config (dict) with all the params
        """

        self.intermediate_results = IntermediateResults()

        # Filtering
        self.filtering = config.get("filtering")
        self.remove_contaminants = self.filtering.get("contaminants_files", [])
        self.filter_qvalue = self.filtering.get("qvalue", 0.01)
        self.filter_pep = self.filtering.get("pep", 0.2)
        self.filter_run_evidence_count = self.filtering.get("run_evidence_count", 1)
        self.pep_direction = (self.filtering.get("pep_direction", "lower") or "lower").lower()
        if self.pep_direction not in {"lower", "higher"}:
            raise ValueError(f"Invalid pep_direction='{self.pep_direction}'. Use 'lower' or 'higher'.")


        # Pivoting
        self.pivot_signal_method = config.get("pivot_signal_method", "sum")
        self.directlfq_cores = config.get("directlfq_cores", 4)

        # Normalization
        self.normalization = config.get("normalization")

        # Imputation
        self.imputation = config.get("imputation")

        self.exports = config.get("exports")

    def fit_transform(self, df: pl.DataFrame) -> PreprocessResults:
        """Applies preprocessing steps in sequence and returns structured results."""
        # Step 1: Filtering
        self._filter(df)

        # Fetch protein metadata
        self._get_protein_metadata()

        # Fech sample metadata
        self._get_condition_map()

        # Step 2: Pivoting
        self._pivot_data()

        # Step 3: Normalization
        self._normalize()

        # Step 4: Imputation
        self._impute()

        # Final step: returning the structured data class
        return PreprocessResults(
            filtered=self.intermediate_results.dfs.get("raw_df"),
            lognormalized=self.intermediate_results.dfs.get("postlog"),
            normalized=self.intermediate_results.dfs.get("normalized"),
            processed=self.intermediate_results.dfs.get("imputed"),
            qvalues=self.intermediate_results.dfs.get("qvalues"),
            pep=self.intermediate_results.dfs.get("pep"),
            spectral_counts=self.intermediate_results.dfs.get("spectral_counts"),
            condition_pivot=self.intermediate_results.dfs.get("condition_pivot"),
            protein_meta=self.intermediate_results.dfs.get("protein_metadata"),
            peptides_wide=self.intermediate_results.dfs.get("peptides_wide"),
            peptides_centered=self.intermediate_results.dfs.get("peptides_centered"),
            meta_cont = self.intermediate_results.metadata.get("filtering").get("meta_cont"),
            meta_qvalue = self.intermediate_results.metadata.get("filtering").get("meta_qvalue"),
            meta_pep =  self.intermediate_results.metadata.get("filtering").get("meta_pep"),
            meta_rec =  self.intermediate_results.metadata.get("filtering").get("meta_rec"),
        )

    @log_time("Filtering")
    def _filter(self, df: pl.DataFrame) -> None:
        df_filtered = self._filter_contaminants(df)
        self.intermediate_results.add_df("filtered_contaminants", df_filtered)

        df_filtered = self._filter_by_stat(df_filtered, "QVALUE")
        self.intermediate_results.add_df("filtered_QVALUE", df_filtered)

        df_filtered = self._filter_by_stat(df_filtered, "PEP")
        self.intermediate_results.add_df("filtered_PEP", df_filtered)

        df_filtered = self._filter_by_run_evidence(df_filtered)
        self.intermediate_results.add_df("filtered_final/RE", df_filtered)

    def _filter_contaminants(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.remove_contaminants or "INDEX" not in df.columns:
            return df

        # Load all contaminant accessions
        all_contaminants = set()
        for path in self.remove_contaminants:
            all_contaminants |= load_contaminant_accessions(path)

        mask_keep = ~pl.col("INDEX").is_in(list(all_contaminants))

        # keep & drop
        df_kept    = df.filter(mask_keep)
        df_dropped = df.filter(~mask_keep)
        mask_bools = df.select(mask_keep.alias("keep")).get_column("keep").to_list()

        dropped_dict = {
            "files": self.remove_contaminants,
            "values": mask_bools,
            "number_kept": len(df_kept),
            "number_dropped": len(df_dropped),
        }

        self.intermediate_results.add_metadata(
            "filtering",
            "meta_cont",
            dropped_dict
        )

        return df_kept

    def _filter_by_stat(self, df: pl.DataFrame, stat: str) -> pl.DataFrame:
        """Filters out rows based on Q-value and PEP thresholds."""
        if stat not in ["PEP", "QVALUE"]:
            raise ValueError(f"Invalid filtering stat '{stat}'")

        # Column missing → skip, record why, return pass-through
        if stat not in df.columns:
            skipped = {
                "skipped": True,
                "reason": f"'{stat}' column not present",
                "threshold": None,              # help the viewer fail-soft
                "direction": "lower_or_equal" if stat=="QVALUE" else self.pep_direction.replace("higher","greater_or_equal"),
                "raw_values": np.array([]),
                "number_kept": len(df),
                "number_dropped": 0,
            }
            self.intermediate_results.add_metadata("filtering", f"meta_{stat.lower()}", skipped)
            log_info(f"{stat} filtering: skipped (column not present).")
            return df

        # Configure threshold + direction
        if stat == "QVALUE":
            threshold = self.filter_qvalue
            keep_mask = pl.col(stat) <= threshold
            direction_note = "lower_or_equal"
        else:  # PEP
            threshold = self.filter_pep
            if self.pep_direction == "lower":
                keep_mask = pl.col(stat) <= threshold
                direction_note = "lower_or_equal"
            else:  # higher is better (FragPipe MaxPepProb)
                keep_mask = pl.col(stat) >= threshold
                direction_note = "greater_or_equal"

        values = df[stat].to_numpy()
        self.intermediate_results.add_array(f"{stat.lower()}_array", values)

        df_kept    = df.filter(keep_mask)
        df_dropped = df.filter(~keep_mask)

        dropped_dict = {
            "threshold": threshold,
            "direction": direction_note,
            "raw_values": values,
            "number_kept": len(df_kept),
            "number_dropped": len(df_dropped),
        }
        self.intermediate_results.add_metadata("filtering", f"meta_{stat.lower()}", dropped_dict)
        log_info(f"{stat} filtering: kept={len(df_kept)} dropped={len(df_dropped)} (direction={direction_note}, thr={threshold}).")

        return df_kept

    def _filter_by_stat_old(self, df: pl.DataFrame, stat: str) -> pl.DataFrame:
        """Filters out rows based on Q-value and PEP thresholds."""
        if stat not in ["PEP", "QVALUE"]:
            raise ValueError(f"Invalid filtering stat '{stat}'")

        if stat in df.columns:
            threshold = self.filter_qvalue if stat == "QVALUE" else self.filter_pep
            values = df[stat].to_numpy()
            self.intermediate_results.add_array(f"{stat.lower()}_array", values)

            df_kept    = df.filter(df[stat] <= threshold)
            df_dropped = df.filter(df[stat] >  threshold)
            dropped_dict = {
                "threshold": threshold,
                "raw_values": values,
                "number_kept": len(df_kept),
                "number_dropped": len(df_dropped),
            }

            self.intermediate_results.add_metadata("filtering", f"meta_{stat.lower()}", dropped_dict)

        return df_kept

    def _filter_by_run_evidence(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters out proteins with insufficient run evidence."""
        if "RUN_EVIDENCE_COUNT" not in df.columns:
            skipped = {
                "skipped": True,
                "reason": "'RUN_EVIDENCE_COUNT' column not present",
                "threshold": None,
                "raw_values": np.array([]),
                "number_kept": len(df),
                "number_dropped": 0,
            }
            self.intermediate_results.add_metadata("filtering", "meta_rec", skipped)
            log_info("Run-evidence filtering: skipped (column not present).")
            return df

        values = df["RUN_EVIDENCE_COUNT"].to_numpy()
        self.intermediate_results.add_array("run_evidence_count_array", values)

        df_kept    = df.filter(pl.col("RUN_EVIDENCE_COUNT") >= self.filter_run_evidence_count)
        df_dropped = df.filter(pl.col("RUN_EVIDENCE_COUNT") <  self.filter_run_evidence_count)
        dropped_dict = {
            "threshold": self.filter_run_evidence_count,
            "raw_values": values,
            "number_kept": len(df_kept),
            "number_dropped": len(df_dropped),
        }

        self.intermediate_results.add_metadata("filtering", "meta_rec", dropped_dict)
        log_info(f"Run-evidence filtering: kept={len(df_kept)} dropped={len(df_dropped)} (thr={self.filter_run_evidence_count}).")

        return df_kept

    def _filter_by_run_evidence_old(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filters out proteins with insufficient run evidence."""
        if "RUN_EVIDENCE_COUNT" in df.columns:
            values = df["RUN_EVIDENCE_COUNT"].to_numpy()
            self.intermediate_results.add_array("run_evidence_count_array", values)

            df_kept    = df.filter(pl.col("RUN_EVIDENCE_COUNT") >= self.filter_run_evidence_count)
            df_dropped = df.filter(pl.col("RUN_EVIDENCE_COUNT") <  self.filter_run_evidence_count)
            dropped_dict = {
                "threshold": self.filter_run_evidence_count,
                "raw_values": values,
                "number_kept": len(df_kept),
                "number_dropped": len(df_dropped),
            }

            self.intermediate_results.add_metadata("filtering", f"meta_rec", dropped_dict)

        return df_kept

    def _get_protein_metadata(self) -> None:
        df_full = self.intermediate_results.dfs["filtered_final/RE"]  # long format

        # Base meta: "first" per protein (works for Spectronaut)
        keep_cols = {
            "INDEX","FASTA_HEADERS","GENE_NAMES",
            "PROTEIN_DESCRIPTIONS","PROTEIN_WEIGHT","IBAQ","PRECURSORS_EXP"
        }
        existing = [c for c in df_full.columns if c in keep_cols]

        base_meta = (
            df_full
            .select(existing)
            .group_by("INDEX")
            .agg([pl.first(c).alias(c) for c in existing if c != "INDEX"])
            .sort("INDEX")
        )

        # Detect FragPipe (we tagged it in the harmonizer)
        is_fp = ("SOURCE" in df_full.columns) and (
            "fragpipe" in df_full.select(pl.col("SOURCE").unique()).to_series().to_list()
        )

        # For FragPipe: PRECURSORS_EXP is per-peptide (Spectrum Number).
        # Compute protein-level sum over unique peptides to avoid multiplying by #samples.
        if is_fp and {"PEPTIDE_LSEQ","PRECURSORS_EXP"}.issubset(df_full.columns):
            rec_df = (
                df_full
                .select(["INDEX","PEPTIDE_LSEQ","PRECURSORS_EXP"])
                .drop_nulls("PRECURSORS_EXP")
                .unique()                                # dedupe across samples
                .group_by("INDEX")
                .agg(pl.col("PRECURSORS_EXP").sum().cast(pl.Int64).alias("PRECURSORS_EXP"))
            )
            base_meta = base_meta.drop("PRECURSORS_EXP", strict=False).join(rec_df, on="INDEX", how="left")

        # Numeric casts (same as before)
        casts = []
        if "IBAQ" in base_meta.columns:
            casts.append(pl.col("IBAQ").cast(pl.Float64, strict=False))
        if "PROTEIN_WEIGHT" in base_meta.columns:
            casts.append(pl.col("PROTEIN_WEIGHT").cast(pl.Float64, strict=False))
        if "PRECURSORS_EXP" in base_meta.columns:
            casts.append(pl.col("PRECURSORS_EXP").cast(pl.Int64, strict=False))
        if casts:
            base_meta = base_meta.with_columns(casts)

        self.intermediate_results.add_df("protein_metadata", base_meta)

    def _get_protein_metadata_old(self) -> None:
        """
        Extract protein-level metadata such as FASTA_HEADERS, GENE_NAMES, DESCRIPTION, etc.,
        grouped by protein INDEX. Assumes all column names are capitalized.
        """
        df = self.intermediate_results.dfs["filtered_final/RE"]
        keep_cols = {"INDEX",
                     "FASTA_HEADERS",
                     "GENE_NAMES",
                     "PROTEIN_DESCRIPTIONS",
                     "PROTEIN_WEIGHT",
                     "IBAQ",
                     "PRECURSORS_EXP"}
        existing = [col for col in df.columns if col in keep_cols]

        # --- make sure numeric columns are truly numeric ---
        df = df.select(existing)
        casts = []
        if "IBAQ" in df.columns:
            casts.append(pl.col("IBAQ").cast(pl.Float64, strict=False))
        if "PROTEIN_WEIGHT" in df.columns:
            casts.append(pl.col("PROTEIN_WEIGHT").cast(pl.Float64, strict=False))
        if "RUN_EVIDENCE_COUNT" in df.columns:
            casts.append(pl.col("RUN_EVIDENCE_COUNT").cast(pl.Int64, strict=False))
        if "PRECURSORS_EXP" in df.columns:
            casts.append(pl.col("PRECURSORS_EXP").cast(pl.Int64, strict=False))

        if casts:
            df = df.with_columns(casts)

        df = df.select(existing).unique(subset=["INDEX"]).sort("INDEX")

        self.intermediate_results.add_df("protein_metadata", df)

    def _get_condition_map(self):
        df = self.intermediate_results.dfs["filtered_final/RE"]
        condition_mapping = df.select(["FILENAME", "CONDITION", "REPLICATE"]).unique()
        condition_mapping = condition_mapping.rename({"FILENAME": "Sample"}).sort("Sample")

        self.intermediate_results.add_df("condition_pivot", condition_mapping)

    def _pivot_df(
        self,
        df: pl.DataFrame,
        sample_col: str,
        protein_col: str,
        values_col: str,
        aggregate_fn: str
    ) -> pl.DataFrame:
        # 1) Map the requested agg name to a Polars expression
        fn_map = {
            "sum":    pl.col(values_col).sum(),
            "mean":   pl.col(values_col).mean(),
            "min":    pl.col(values_col).min(),
            "max":    pl.col(values_col).max(),
            "median": pl.col(values_col).median(),
        }
        if aggregate_fn not in fn_map:
            raise ValueError(f"Unsupported aggregate_fn '{aggregate_fn}'")

        agg_expr = fn_map[aggregate_fn].alias(values_col)
        # Count non-null entries per group
        n_valid = pl.col(values_col).is_not_null().sum().alias("_NVALID")

        # 2) Pre-aggregate (protein, sample) with a null-preserving guard:
        #    if a group has 0 valid values, force the aggregate to null
        df_agg = (
            df
            .group_by([protein_col, sample_col], maintain_order=True)
            .agg([agg_expr, n_valid])
            .with_columns(
                pl.when(pl.col("_NVALID") == 0)
                  .then(pl.lit(None))
                  .otherwise(pl.col(values_col))
                  .alias(values_col)
            )
            .drop("_NVALID")
        )

        # 3) Pivot WITHOUT a second aggregation step: missing groups → nulls
        pivot_df = df_agg.pivot(
            index=protein_col,
            columns=sample_col,
            values=values_col,
        )

        # 4) Ensure those nulls become np.nan in NumPy
        pivot_df = pivot_df.fill_null(np.nan)

        return pivot_df

    # using a trick because polars>=1.31 does aggregate filling with 0s ! Instead of nan ! 
    @log_time("DirectLFQ")
    def _pivot_df_LFQ(
        self,
        df: pl.DataFrame,
        sample_col: str,
        protein_col: str,
        values_col: str,
        ion_col: str,
    ) -> pl.DataFrame:

        #import directlfq.config as dlcfg
        # 0) sanity
        for c in (sample_col, protein_col, values_col, ion_col):
            if c not in df.columns:
                raise ValueError(f"LFQ requires column '{c}' in input DataFrame.")

                # 1) Count valid (non-null AND non-zero) per (protein, ion, sample)
        is_valid = (pl.col(values_col).is_not_null() & (pl.col(values_col) != 0))
        n_valid  = is_valid.sum().alias("_NVALID")

        # 2) Pre-aggregate per (protein, ion, sample) with a null-preserving guard:
        df_ion_agg = (
            df
            .group_by([protein_col, ion_col, sample_col], maintain_order=True)
            .agg([
                pl.col(values_col).first().alias(values_col),   # you said there are no dups
                n_valid
            ])
            .with_columns(
                pl.when(pl.col("_NVALID") == 0)
                  .then(pl.lit(None))
                  .otherwise(pl.col(values_col))
                  .alias(values_col)
            )
            .drop("_NVALID")
        )

        # 3) Build a (protein × sample) boolean mask: any valid ion in this run?
        has_valid = (
            df
            .group_by([protein_col, sample_col], maintain_order=True)
            .agg(is_valid.any().alias("_HAS_VALID"))
            .pivot(index=protein_col, columns=sample_col, values="_HAS_VALID")
            .fill_null(False)
        )

        # 4) Peptide/ion × sample wide matrix (missing groups -> NaNs)
        runs_in_order = (
            df.select(sample_col).unique(maintain_order=True).to_series().to_list()
        )
        pep_wide = (
            df_ion_agg
            .pivot(index=[protein_col, ion_col], columns=sample_col, values=values_col)
            .fill_null(np.nan)
        )

        # 5) → pandas, log2 as expected by directLFQ
        pw = pep_wide.to_pandas().set_index([protein_col, ion_col]).sort_index(level=0)
        pw = np.log2(pw)  # NaNs stay NaNs

        # 6) Run directLFQ
        dlcfg.PROTEIN_ID = protein_col
        dlcfg.QUANT_ID   = ion_col
        prot_df, _ = estimate_protein_intensities(
            normed_df=pw,
            min_nonan=1,
            num_samples_quadratic=2,
            num_cores=self.directlfq_cores,
        )
        if protein_col not in prot_df.columns:
            prot_df = prot_df.reset_index()
        prot_df = prot_df.set_index(protein_col).astype(float)  # linear scale

        # 7) Apply the mask: where no ion existed in a run, force NaN (not 0)
        mask_pd = has_valid.to_pandas().set_index(protein_col)
        # align mask rows/cols to prot_df
        mask_pd = mask_pd.reindex(index=prot_df.index, columns=prot_df.columns, fill_value=False)
        prot_df = prot_df.mask(~mask_pd)  # False → NaN

        # 8) Back to Polars; preserve sample order; turn NaN -> null
        out = pl.DataFrame(prot_df.reset_index())
        keep_cols = [protein_col] + [c for c in runs_in_order if c in out.columns]
        out = out.select(keep_cols)
        out = out.with_columns([
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c in out.columns if c != protein_col
        ])
        return out

    def _build_peptide_tables(self) -> None:
        df = self.intermediate_results.dfs["filtered_final/RE"]

        needed = {"INDEX", "FILENAME", "SIGNAL", "PEPTIDE_LSEQ"}
        if not needed.issubset(df.columns):
            return

        # normalize sequence: drop Oxidation[...] and trim outer "_"
        seq_clean = (
            pl.col("PEPTIDE_LSEQ")
              .str.replace_all(r"(?i)\[Oxidation[^\]]*\]", "")
              .str.replace_all(r"^_+|_+$", "")
              .alias("PEPTIDE_SEQ")
        )

        base = df.select(
            pl.col("INDEX"),
            pl.col("FILENAME"),
            pl.col("SIGNAL"),
            seq_clean,
        ).with_columns(
            (pl.col("INDEX").cast(pl.Utf8) + pl.lit("|") + pl.col("PEPTIDE_SEQ")).alias("PEPTIDE_ID")
        )

        # pivot with your helper (sum duplicates)
        pep_pivot = self._pivot_df(
            df=base.select(["PEPTIDE_ID", "FILENAME", "SIGNAL"]),
            sample_col="FILENAME",
            protein_col="PEPTIDE_ID",
            values_col="SIGNAL",
            aggregate_fn="sum",
        )

        # readable columns
        id_map = base.select(["PEPTIDE_ID", "INDEX", "PEPTIDE_SEQ"]).unique()
        pep_wide = pep_pivot.join(id_map, on="PEPTIDE_ID", how="left")

        # centered (row / row-mean ignoring NaNs)
        sample_cols = [c for c in pep_wide.columns if c not in ("PEPTIDE_ID","INDEX","PEPTIDE_SEQ")]
        row_sum = pl.sum_horizontal(
            *[pl.when(pl.col(c).is_nan()).then(0.0).otherwise(pl.col(c)) for c in sample_cols]
        )
        # Row-wise count of non-NaN cells
        pep_centered = (
            pep_wide
            .with_columns(
                pl.mean_horizontal([
                    pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c))
                    for c in sample_cols
                ]).alias("__rowmean__")
            )
            .with_columns([(pl.col(c) / pl.col("__rowmean__")).alias(c) for c in sample_cols])
            .drop("__rowmean__")
        )

        return pep_wide, pep_centered

    def _pivot_data(self) -> None:
        """
        Converts the dataset from long format (one row per protein per sample)
        to a wide format (one row per protein, with sample-specific intensity & q-value columns).
        """
        df = self.intermediate_results.dfs["filtered_final/RE"]

        # QVALUE is optional (e.g., FragPipe TMT). Require only these:
        required_cols = {"INDEX", "FILENAME", "SIGNAL"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for pivoting: {missing_cols}")

        if self.pivot_signal_method.lower() == "directlfq":
            intensity_pivot = self._pivot_df_LFQ(
                df=df,
                sample_col="FILENAME",
                protein_col="INDEX",
                values_col="SIGNAL",
                ion_col="PEPTIDE_LSEQ",
            )
        else:
            intensity_pivot = self._pivot_df(
                df=df,
                sample_col='FILENAME',
                protein_col='INDEX',
                values_col='SIGNAL',
                aggregate_fn=self.pivot_signal_method,
            )

        # Pivot Q-values ONLY if present
        qvalue_pivot = None
        if "QVALUE" in df.columns:
            qvalue_pivot = self._pivot_df(
                df=df,
                sample_col='FILENAME',
                protein_col='INDEX',
                values_col='QVALUE',
                aggregate_fn='mean'
            )
        else:
            log_info("Pivot: QVALUE matrix not built (column not present).")

        # Pivot PEP if present
        pep_pivot = None
        if "PEP" in df.columns:
            pep_pivot = self._pivot_df(
                df=df,
                sample_col='FILENAME',
                protein_col='INDEX',
                values_col='PEP',
                aggregate_fn='mean'
            )

        # Pivot RUN_EVIDENCE_COUNT if present (as spectral_counts proxy)
        rec_pivot = None
        if "RUN_EVIDENCE_COUNT" in df.columns:
            rec_pivot = self._pivot_df(
                df=df,
                sample_col='FILENAME',
                protein_col='INDEX',
                values_col='RUN_EVIDENCE_COUNT',
                aggregate_fn='mean'
            )

        # Peptide drill-down (optional)
        pep_tables = self._build_peptide_tables()
        if pep_tables is not None:
            raw_pep_pivot, centered_pep_pivot = pep_tables
        else:
            raw_pep_pivot = None
            centered_pep_pivot = None

        # --- ALIGN secondary pivots (qvalue/pep/rec) to the intensity order ---
        order_idx = intensity_pivot.select("INDEX")
        def _align_to_intensity(pvt: Optional[pl.DataFrame]) -> Optional[pl.DataFrame]:
            if pvt is None:
                return None
            # left join keeps intensity order and ensures identical index vector
            return order_idx.join(pvt, on="INDEX", how="left")

        qvalue_pivot = _align_to_intensity(qvalue_pivot)
        pep_pivot    = _align_to_intensity(pep_pivot)
        rec_pivot    = _align_to_intensity(rec_pivot)

        self.intermediate_results.set_columns_and_index(intensity_pivot)
        self.intermediate_results.add_df("raw_df", intensity_pivot)
        self.intermediate_results.add_df("qvalues", qvalue_pivot)
        self.intermediate_results.add_df("pep", pep_pivot)
        self.intermediate_results.add_df("spectral_counts", rec_pivot)
        self.intermediate_results.dfs["peptides_wide"]     = raw_pep_pivot
        self.intermediate_results.dfs["peptides_centered"] = centered_pep_pivot

    @log_time("Normalization")
    def _normalize(self) -> None:
        """
        Normalize intensity values and generate normalization diagnostic plots.

        Args:
            df (pl.DataFrame): Input dataframe.
            condition_pivot (pl.DataFrame): Mapping from sample to condition.

        Returns:
            pl.DataFrame: Normalized dataframe (Polars).
        """
        self._normalize_matrix()

    def _normalize_matrix(self) -> None:

        """
        Apply selected normalization(s) to intensity matrix extracted from a DataFrame.
        Supports chained normalization steps (e.g., log + loess).

        Args:
            df (pl.DataFrame): Input dataframe with intensity columns.
            condition_pivot (pl.DataFrame): Mapping of sample to condition.
            span (float): Smoothing span for loess.

        Returns:
            pl.DataFrame: DataFrame with normalized intensity values.
        """

        normalization_method = self.normalization.get("method")
        numeric_cols = self.intermediate_results.columns

        df = self.intermediate_results.dfs.get("raw_df")

        mat = df.select(numeric_cols).to_numpy()
        original_mat = mat.copy()
        postlog_mat = mat.copy()
        postlog_df = df.clone()
        models = None
        regression_scale_used = None
        regression_type_used = None
        tags = None
        tag_matches = None

        if isinstance(normalization_method, str):
            normalization_method = [normalization_method]

        for method in normalization_method:
            if method == "log10":
                mat = np.clip(mat, 1e-10, None)
                mat = np.log10(mat)
                postlog_mat = mat.copy()

            elif method == "log2":
                mat = np.clip(mat, 1e-10, None)
                mat = np.log2(mat)
                postlog_mat = mat.copy()

            elif method == "median_equalization":
                global_median = np.nanmedian(mat)
                medians = np.nanmedian(mat, axis=0, keepdims=True)
                mat = mat * (global_median / medians)

            elif method == "quantile":
                qt = sklearn.preprocessing.QuantileTransformer(random_state=42)
                mat = qt.fit_transform(mat)

            elif method in {"global_linear", "global_loess", "local_linear", "local_loess"}:
                regression_scale_used = "global" if "global" in method else "local"
                regression_type_used = "loess" if "loess" in method else "linear"

                # build condition_labels in same column order as mat
                sample_names = numeric_cols
                cond_df = self.intermediate_results.dfs["condition_pivot"].to_pandas()
                cond_df.columns = [c.capitalize() for c in cond_df.columns]
                cond_map = cond_df.set_index("Sample")["Condition"]

                condition_labels = [cond_map[s] for s in sample_names]

                mat, models = regression_normalization(
                    mat,
                    scale=regression_scale_used,
                    regression_type=regression_type_used,
                    span=self.normalization.get("loess_span"),
                    condition_labels=condition_labels
                )
            elif method == "median_equalization_by_tag":
                # --- config ---
                tag_value  = self.normalization.get("reference_tag")
                fasta_col  = self.normalization.get("fasta_column", "FASTA_HEADERS")

                if tag_value is None or (isinstance(tag_value, str) and not tag_value.strip()):
                    raise ValueError("median_equalization_by_tag requires normalization.reference_tag")

                tags = [tag_value] if isinstance(tag_value, str) else list(tag_value)
                tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]

                # --- get FASTA headers aligned to the current protein order ---
                meta_df = self.intermediate_results.dfs.get("protein_metadata")
                if meta_df is None or fasta_col not in meta_df.columns:
                    raise ValueError(f"median_equalization_by_tag requires protein_metadata with column '{fasta_col}'")

                # join FASTA headers onto the current df to preserve row order
                idx_col = "INDEX"
                fasta_map = meta_df.select([idx_col, fasta_col])
                df_with_fasta = df.join(fasta_map, on=idx_col, how="left")

                # boolean mask of proteins used as reference (case-insensitive substring)
                fasta_vals = df_with_fasta.get_column(fasta_col).to_list()
                ref_mask = np.array([
                    (s is not None) and any(t in str(s).lower() for t in [t.lower() for t in tags])
                    for s in fasta_vals
                ], dtype=bool)

                n_ref = int(ref_mask.sum())
                if n_ref == 0:
                    log_info(f"median_equalization_by_tag: no proteins matched tag(s) {tags} in '{fasta_col}'. Step skipped.")
                    # leave mat as-is and continue
                else:
                    # compute medians on the reference subset (current space: log or linear, consistent with your existing method)
                    ref_sub = mat[ref_mask, :]

                    ref_counts = np.sum(np.isfinite(ref_sub), axis=0)  # per-sample non-NaN counts in the reference set
                    if np.any(ref_counts == 0):
                        sample_names = self.intermediate_results.columns
                        missing_samples = [sample_names[i] for i, c in enumerate(ref_counts) if c == 0]
                        log_info(f"median_equalization_by_tag: no reference signal in samples: {missing_samples} → leaving those samples unscaled (factor=1).")

                    ref_col_meds = np.nanmedian(ref_sub, axis=0, keepdims=True)      # shape (1, n_samples)
                    ref_global   = np.nanmedian(ref_sub)                              # scalar

                    # build safe scale factors (don’t touch columns with NaN medians)
                    scale = np.where(np.isfinite(ref_col_meds), ref_global / ref_col_meds, 1.0)
                    mat = mat * scale

                    self.normalization["tag_matches"] = n_ref

                    log_info(f"median_equalization_by_tag: matched={n_ref} proteins by {tags} in '{fasta_col}'.")

            elif method == "none":
                log_info(f"Skipping normalization (raw data)")

            else:
                raise ValueError(f"Invalid normalization method: {method}")

        df = df.with_columns(pl.DataFrame(mat, schema=numeric_cols))
        postlog_df = df.with_columns(pl.DataFrame(postlog_mat, schema=numeric_cols))

        # Decide if any log step was applied in the normalization chain
        has_log_step = any(
            (m or "").lower() in {"log2", "log10", "log", "ln"}
            for m in (self.normalization.get("method") or [])
        )

        if not has_log_step:
            # original_mat is the pre-normalization matrix (log2 when input is FP)
            raw_lin = np.where(np.isnan(original_mat), np.nan, np.exp2(original_mat))
            # 1) overwrite raw_df's numeric columns with linear values
            numeric_cols = self.intermediate_results.columns
            self.intermediate_results.dfs["raw_df"] = (
                self.intermediate_results.dfs["raw_df"]
                .with_columns(pl.DataFrame(raw_lin, schema=numeric_cols))
            )
            # 2) store linear matrix as raw_mat
            original_mat = raw_lin
            log_info("No log step detected → saving RAW DATA in linear space (assumed input log2).")


        self.intermediate_results.add_matrix("raw_mat", original_mat)
        self.intermediate_results.add_matrix("postlog", postlog_mat)
        self.intermediate_results.add_matrix("normalized", mat)
        self.intermediate_results.add_df("postlog", postlog_df)
        self.intermediate_results.add_df("normalized", df)
        self.intermediate_results.add_model("normalization", models)
        self.intermediate_results.add_metadata("normalization",
                                               "method",
                                               self.normalization.get("method"))
        self.intermediate_results.add_metadata("normalization",
                                               "regression_scale_used",
                                               regression_scale_used)
        self.intermediate_results.add_metadata("normalization",
                                               "regression_type_used",
                                               regression_type_used)
        self.intermediate_results.add_metadata("normalization",
                                               "span",
                                               self.normalization.get("loess_span"))
        self.intermediate_results.add_metadata("normalization",
                                               "tags",
                                               tags)
        self.intermediate_results.add_metadata("normalization",
                                               "tag_matches",
                                               tag_matches)

    @log_time("Imputation")
    def _impute(self) -> None:
        """
        Impute missivg values and generate diagnostic plots.

        Args:
            df (pl.DataFrame): Input dataframe.
            condition_pivot (pl.DataFrame): Mapping from sample to condition.

        Returns:
            pl.DataFrame: Normalized dataframe (Polars).
        """
        results = self._impute_matrix()

    def _impute_matrix(self) -> None:
        """Applies imputation to missing values."""

        numeric_cols = self.intermediate_results.columns
        df = self.intermediate_results.dfs["normalized"]
        not_imputed_mat = df.select(numeric_cols).to_numpy().copy()

        #imputer = get_imputer(**self.imputation)
        condition_map = self.intermediate_results.dfs["condition_pivot"].to_pandas()
        imputer = get_imputer(
            **self.imputation,
            condition_map=condition_map,
            sample_index=numeric_cols
        )
        if imputer is None:
            imputed_mat = not_imputed_mat
            #mask = np.isnan(imputed_mat)
            #imputed_mat[mask] = 0.0
        else:
            imputed_mat = imputer.fit_transform(not_imputed_mat)

        df = df.with_columns(pl.DataFrame(imputed_mat,
                                          schema=numeric_cols))
        imputed_only = np.where(np.isnan(not_imputed_mat),
                                imputed_mat,
                                np.nan)

        self.intermediate_results.add_matrix("imputed_only", imputed_only)
        self.intermediate_results.add_matrix("imputed", imputed_mat)
        self.intermediate_results.add_df("imputed", df)
        self.intermediate_results.add_metadata("imputation",
                                               "method",
                                               self.imputation.get("method"))

