import re
import polars as pl
from typing import Dict
from proteoflux.utils.utils import log_time, logger, log_info, log_warning


class DataHarmonizer:
    """Harmonizes input data by renaming columns to a common format."""

    DEFAULT_COLUMN_MAP = {
        "index_column": "INDEX",
        "signal_column": "SIGNAL",
        "qvalue_column": "QVALUE",
        "pep_column": "PEP",
        "condition_column": "CONDITION",
        "replicate_column": "REPLICATE",
        "filename_column": "FILENAME",
        "spectral_counts_column": "SPECTRAL_COUNTS",
        "fasta_column": "FASTA_HEADERS",
        "protein_weight": "PROTEIN_WEIGHT",
        "protein_descriptions": "PROTEIN_DESCRIPTIONS",
        "gene_names": "GENE_NAMES",
        "precursors_exp_column": "PRECURSORS_EXP",
        "ibaq_column": "IBAQ",
        "peptide_seq_column": "PEPTIDE_LSEQ",
        "uniprot_column": "UNIPROT", #this should be default, and then index_column is redefined as additional to that
        "modified_seq_column": "MODIFIED_SEQUENCE",
        "charge_column": "CHARGE",
        "ptm_positions_column": "PTM_POSITIONS_STR",
        "ptm_probabilities_column": "PTM_PROBS_STR",
        "ptm_sites_column": "PTM_SITES_STR",
        "stripped_seq_column": "PEP_SEQUENCE",
    }

    def __init__(self, column_config: dict):
        """Initialize column mappings with user-defined config."""
        self.column_map: Dict[str, str] = {}
        self.annotation_file = column_config.get("annotation_file", None)

        # vendor-agnostic layout toggle (required if data are pre-pivoted)
        # Accepted: "long" (default), "wide"
        self.input_layout = (column_config.get("input_layout") or "long").strip().lower()
        self.analysis_type = (column_config.get("analysis_type") or "DIA").strip().lower()

        for config_key, std_name in self.DEFAULT_COLUMN_MAP.items():
            original_col = column_config.get(config_key)
            if original_col:
                self.column_map[original_col] = std_name

    def _load_annotation(self) -> pl.DataFrame:
        if not self.annotation_file:
            raise ValueError(
                "annotation_file is required for this input layout but was not provided."
            )
        ann = pl.read_csv(self.annotation_file, separator="\t", ignore_errors=True)

        # Pick a single join key (first match wins), then rename only that one -> 'FILENAME'
        join_col = None
        for cand in ("FILENAME", "File Name", "Channel"):
            if cand in ann.columns:
                join_col = cand
                break

        if join_col is None:
            raise ValueError(
                f"Annotation file {self.annotation_file!r} must contain a join column: "
                "'FILENAME', 'File Name', or 'Channel'."
            )

        if join_col != "FILENAME":
            # Rename only the chosen column to avoid creating duplicates
            ann = ann.rename({join_col: "FILENAME"})

        # Be tolerant about whitespace on the join key
        ann = ann.with_columns(pl.col("FILENAME").cast(pl.Utf8).str.strip_chars())
        return ann

    def _fmt_diff(self, missing: list[str], extra: list[str], what: str) -> None:
        if not missing and not extra:
            return
        def _fmt(lst, cap=20):
            return lst[:cap] + ([f"... (+{len(lst)-cap} more)"] if len(lst) > cap else [])
        parts = []
        if missing:
            parts.append(f"Annotated {what} not found in data ({len(missing)}): {_fmt(missing)}")
        if extra:
            parts.append(f"Data has {what} not in annotation ({len(extra)}): {_fmt(extra)}")
        msg = f"{what.capitalize()} / annotation mismatch:\n  " + "\n  ".join(parts)
        logger.error(msg)
        raise ValueError(msg)

    def _validate_long_annotation(self, df: pl.DataFrame, ann: pl.DataFrame) -> None:
        df_files  = set(df.select("FILENAME").unique().to_series().to_list())
        ann_files = set(ann.select("FILENAME").unique().to_series().to_list())
        missing = sorted(ann_files - df_files)
        extra   = sorted(df_files - ann_files)
        self._fmt_diff(missing, extra, "filenames")

    def _validate_wide_annotation(self, df: pl.DataFrame, ann: pl.DataFrame) -> None:
        ann_channels = set(ann.select("FILENAME").unique().to_series().to_list())

        # Infer candidate intensity columns (unchanged logic)
        float_types = {pl.Float64, pl.Float32}
        schema = df.schema
        float_cols = {c for c, t in schema.items() if t in float_types}
        meta_float_exclude = {"ReferenceIntensity", "MaxPepProb"}
        data_channels = float_cols - meta_float_exclude

        missing = sorted([c for c in ann_channels if c not in df.columns])
        extra   = sorted([c for c in data_channels if c not in ann_channels])
        self._fmt_diff(missing, extra, "channels")

    def _rename_columns_safely(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rename to standardized names from config, but raise only if we'd clobber
        a different existing column (e.g., FILENAME after melt).
        """
        rename_map: Dict[str, str] = {}

        for original, target in self.column_map.items():
            if original in df.columns:
                if target in df.columns and original != target:
                    raise ValueError(
                        f"Cannot rename '{original}' to standardized '{target}' because "
                        f"'{target}' already exists in the dataset."
                    )
                rename_map[original] = target
            else:
                log_warning(f"Column '{original}' not found in input data, skipping harmonization for it.")

        return df.rename(rename_map) if rename_map else df

    def _standardize_then_inject(self, df: pl.DataFrame) -> pl.DataFrame:
        """Common path: rename → strict filename check → annotation join → coalesce condition/replicate."""
        df = self._rename_columns_safely(df)

        if not self.annotation_file:
            return df  # nothing to inject

        log_info("Injecting annotation")
        ann = self._load_annotation()

        # Require FILENAME for long join; for wide we will already have created it via melt.
        if "FILENAME" not in df.columns:
            raise ValueError("Cannot inject annotation: 'FILENAME' column not found in data.")

        # Strict filename parity (long)
        self._validate_long_annotation(df, ann)

        # Join
        df = df.join(ann, on="FILENAME", how="left")

        # Coalesce condition/replicate from common aliases
        condition_sources = [c for c in ["R.Condition", "Condition"] if c in df.columns]
        replicate_sources = [r for r in ["R.Replicate", "Replicate"] if r in df.columns]

        if condition_sources:
            src = condition_sources[0]
            df = df.with_columns(
                pl.when(pl.col(src).is_not_null())
                  .then(pl.col(src))
                  .otherwise(pl.col("CONDITION") if "CONDITION" in df.columns else None)
                  .alias("CONDITION")
            ).drop(src, strict=False)

        if replicate_sources:
            src = replicate_sources[0]
            df = df.with_columns(
                pl.when(pl.col(src).is_not_null())
                  .then(pl.col(src))
                  .otherwise(pl.col("REPLICATE") if "REPLICATE" in df.columns else None)
                  .alias("REPLICATE")
            ).drop(src, strict=False)

        return df

    def _melt_wide_to_long(self, df: pl.DataFrame) -> pl.DataFrame:
        """Use annotation to melt a wide matrix into the canonical long schema."""
        ann = self._load_annotation()
        channels = ann.select("FILENAME").unique().to_series().to_list()

        # Strict two-way validation before melt
        self._validate_wide_annotation(df, ann)

        id_vars = [c for c in df.columns if c not in channels]
        if not id_vars:
            raise ValueError("No identifier columns left after selecting channels; check your annotation and table.")

        long_df = df.melt(
            id_vars=id_vars,
            value_vars=channels,
            variable_name="FILENAME",
            value_name="SIGNAL",
        ).with_columns(
            pl.col("FILENAME").cast(pl.Utf8),
            pl.col("SIGNAL").cast(pl.Float64, strict=False),
        )

        # Generic post-melt aliases (no vendor strings):
        # - Some exporters name per-peptide spectrum count as 'Spectrum Number'
        alias_map = {}
        if "Spectrum Number" in long_df.columns:
            alias_map["Spectrum Number"] = "PRECURSORS_EXP"
        if alias_map:
            long_df = long_df.rename(alias_map)

        # Tag layout for downstream heuristics (e.g., metadata aggregation)
        long_df = long_df.with_columns(pl.lit("wide").alias("INPUT_LAYOUT"))

        return long_df

    def _strip_mods(self, s: str) -> str:
        if s is None:
            return None
        # Spectronaut often wraps with underscores and brackets: _ACD[Carbamidomethyl (C)]EF_
        out = s.strip("_")
        out = re.sub(r"\[.*?\]", "", out)   # drop bracketed annotations
        out = out.replace("(", "").replace(")", "")  # safety for exotic exporters
        return out

    def _assert_required_phospho_columns(self, df: pl.DataFrame) -> None:
        required = {"PEPTIDE_LSEQ", "PTM_POSITIONS_STR"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                "[PHOSPHO] Missing required columns after harmonization: "
                f"{missing}. Ensure your config maps:\n"
                "  peptide_seq_column -> PEPTIDE_LSEQ (e.g. 'PEP.StrippedSequence')\n"
                "  ptm_positions_column -> PTM_POSITIONS_STR (e.g. 'EG.PTMPositions [Phospho (STY)]')"
            )

        # Prefer vendor-specific per-site probs; fall back to the generic localization probs if present
        has_vendor_probs = "PTM_PROBS_STR" in df.columns
        has_generic_probs = "EG.PTMLocalizationProbabilities" in df.columns  # present in your phospho head
        if not has_vendor_probs and has_generic_probs:
            df = df.rename({"EG.PTMLocalizationProbabilities": "PTM_PROBS_STR"})
        elif not has_vendor_probs and not has_generic_probs:
            log_warning("[PHOSPHO] No per-site probability column found; LOC_PROB will be null.")

        # Strongly suggested but not fatal... yet
        if "UNIPROT" not in df.columns:
            log_warning("[PHOSPHO] UNIPROT not present; parent protein mapping will be limited.")
        return df

    def _first_existing(self, df: pl.DataFrame, cols: list[str]) -> str | None:
        for c in cols:
            if c in df.columns:
                return c
        return None

    @log_time("Building Peptide Index")
    def _build_peptido_index(self, df: pl.DataFrame) -> pl.DataFrame:
        # choose a real sequence source (std or vendor)
        cand_cols = [
            "PEPTIDE_LSEQ", "PEP_SEQUENCE", "MODIFIED_SEQUENCE",  # standardized names (if mapped)
            "PEP.StrippedSequence", "EG.ModifiedSequence", "FG.LabeledSequence"  # vendor fallbacks
        ]
        seq_col = self._first_existing(df, cand_cols)
        if seq_col is None:
            raise ValueError("[PEPTIDOMICS] No sequence column found (tried: "
                             + ", ".join(cand_cols) + "). Map one in the config.")

        df = df.with_columns(
            pl.col(seq_col).cast(pl.Utf8).alias("_seq_candidate")
        ).with_columns(
            pl.col("_seq_candidate").map_elements(self._strip_mods, return_dtype=str).alias("STRIPPED_SEQ")
        ).drop(["_seq_candidate"], strict=False)

        df = df.with_columns(
            pl.col("STRIPPED_SEQ").alias("PARENT_PEPTIDE_ID"),
            pl.when(pl.col("UNIPROT").is_not_null()).then(pl.col("UNIPROT")).otherwise(None).alias("PARENT_PROTEIN"),
            pl.col("STRIPPED_SEQ").alias("INDEX"),
        )

        if "ASSAY" not in df.columns:
            df = df.with_columns(pl.lit("PEPTIDOMICS").alias("ASSAY"))

        return df

    @log_time("Building Phospho Index")
    def _build_phospho_index(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._assert_required_phospho_columns(df)

        cand_cols = [
            "PEPTIDE_LSEQ", "PEP_SEQUENCE", "MODIFIED_SEQUENCE",
            "PEP.StrippedSequence", "EG.ModifiedSequence", "FG.LabeledSequence"
        ]
        seq_col = self._first_existing(df, cand_cols)
        if seq_col is None:
            raise ValueError("[PHOSPHO] No sequence column found (tried: " + ", ".join(cand_cols) + ").")

        df = df.with_columns(
            pl.col(seq_col).cast(pl.Utf8).alias("_seq_candidate")
        ).with_columns(
            pl.col("_seq_candidate").map_elements(self._strip_mods, return_dtype=str).alias("STRIPPED_SEQ")
        ).drop(["_seq_candidate"], strict=False)

        # Split lists
        with_lists = df.with_columns(
            pl.when(pl.col("PTM_POSITIONS_STR").is_not_null())
              .then(pl.col("PTM_POSITIONS_STR").cast(pl.Utf8).str.split(";"))
              .otherwise(pl.lit([])).alias("_pos_list"),
            pl.when(pl.col("PTM_PROBS_STR").is_not_null())
              .then(pl.col("PTM_PROBS_STR").cast(pl.Utf8).str.split(";"))
              .otherwise(pl.lit([])).alias("_prob_list"),
        )

        # Explode in lockstep
        exploded = with_lists.explode(["_pos_list", "_prob_list"]).with_columns(
            pl.col("_pos_list").cast(pl.Utf8).str.strip_chars().replace("", None).cast(pl.Int64, strict=False).alias("SITE_POS"),
            pl.col("_prob_list").cast(pl.Utf8).str.strip_chars().replace("", None).cast(pl.Float64, strict=False).alias("LOC_PROB"),
        ).filter(pl.col("SITE_POS").is_not_null())

        # Normalize probs and drop zero-prob candidates (keeps only truly localized sites)
        exploded = exploded.with_columns(
            pl.when((pl.col("LOC_PROB") > 1.0) & (pl.col("LOC_PROB").is_not_null()))
              .then(pl.col("LOC_PROB") / 100.0)
              .otherwise(pl.col("LOC_PROB"))
              .alias("LOC_PROB")
        ).filter((pl.col("LOC_PROB").is_null()) | (pl.col("LOC_PROB") > 0.0))

        # Build final keys and parents from STRIPPED_SEQ
        out = exploded.with_columns(
            (pl.col("STRIPPED_SEQ").cast(pl.Utf8) + pl.lit("|p") + pl.col("SITE_POS").cast(pl.Utf8)).alias("INDEX"),
            pl.col("STRIPPED_SEQ").alias("PARENT_PEPTIDE_ID"),
            pl.when(pl.col("UNIPROT").is_not_null()).then(pl.col("UNIPROT")).otherwise(None).alias("PARENT_PROTEIN"),
            pl.lit("PHOSPHO").alias("ASSAY"),
        ).drop(["_pos_list", "_prob_list"], strict=False)

        return out


    @log_time("Data Harmonizing")
    def harmonize(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.input_layout not in {"long", "wide"}:
            raise ValueError("dataset.input_layout must be 'long' or 'wide'.")

        if self.input_layout == "wide":
            df = self._melt_wide_to_long(df)
            df = self._standardize_then_inject(df)
            return df

        # long
        df = self._standardize_then_inject(df)

        if self.analysis_type == "phospho":
            df = self._build_phospho_index(df)
        elif self.analysis_type == "peptidomics":
            df = self._build_peptido_index(df)

        return df
