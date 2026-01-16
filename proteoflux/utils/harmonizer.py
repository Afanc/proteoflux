import re
import polars as pl
from typing import Dict, List
from proteoflux.utils.utils import log_time, logger, log_info, log_warning
from proteoflux.utils.ptm_map import PTM_MAP
from proteoflux.utils.sequence_ops import (
    strip_mods as _strip_mods_impl,
    convert_numeric_ptms as _convert_numeric_ptms_impl,
    normalize_peptido_index_seq as _normalize_peptido_index_seq_impl,
)


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
        "protein_descriptions_column": "PROTEIN_DESCRIPTIONS",
        "gene_names_column": "GENE_NAMES",
        "precursors_exp_column": "PRECURSORS_EXP",
        "ibaq_column": "IBAQ",
        "peptide_seq_column": "PEPTIDE_LSEQ",
        "uniprot_column": "UNIPROT", #this should be default, and then index_column is redefined as additional to that
        "charge_column": "CHARGE",
        "peptide_start_column": "PEPTIDE_START",
        "ptm_proteinlocations_column": "PTM_PROTEINLOCATIONS",
        "ptm_probabilities_column": "PTM_PROBS_STR",
        "ptm_positions_column": "PTM_POSITIONS_STR",
        "ptm_sites_column": "PTM_SITES_STR",
        "stripped_seq_column": "PEP_SEQUENCE",
    }

    def __init__(self, column_config: dict):
        """Initialize column mappings with user-defined config."""
        self.column_map: Dict[str, str] = {}
        self.annotation_file = column_config.get("annotation_file", None)

        self.input_layout = (column_config.get("input_layout") or "long").strip().lower()
        self.analysis_type = (column_config.get("analysis_type") or "DIA").strip().lower()

        self.signal_key = column_config.get("signal_column")
        self.spectral_counts_key = column_config.get("spectral_counts_column")

        self.convert_numeric_ptms = bool(column_config.get("convert_numeric_ptms", True))
        self.collapse_all_ptms = bool(column_config.get("collapse_all_ptms", False))

        raw_excl = column_config.get("exclude_runs")
        self.exclude_runs = set()
        if raw_excl:
            if isinstance(raw_excl, str):
                self.exclude_runs = {raw_excl.strip()} if raw_excl.strip() else set()
            else:
                self.exclude_runs = {str(x).strip() for x in raw_excl if str(x).strip()}

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

        required = {"Condition", "Replicate"}
        missing_req = sorted(required - set(ann.columns))

        # Pick a single join key (first match wins), then rename only that one -> 'FILENAME'
        join_col = None
        for cand in ("FILENAME", "File Name", "File name", "filename", "Filename", "Channel"): #could be smarter
            if cand in ann.columns:
                join_col = cand
                break

        if join_col is None:
            raise ValueError(
                f"Annotation file {self.annotation_file!r} must contain a join column: "
                "'FILENAME', 'File Name', or 'Channel'."
            )

        if missing_req:
            raise ValueError(
                f"Annotation file {self.annotation_file!r} missing required columns: {missing_req}. "
                "Provide canonical columns: Condition, Replicate."
            )

        if join_col != "FILENAME":
            # Rename only the chosen column to avoid creating duplicates
            ann = ann.rename({join_col: "FILENAME"})

        # Be tolerant about whitespace on the join key
        ann = ann.with_columns(pl.col("FILENAME").cast(pl.Utf8).str.strip_chars())

        # Strip common MS file extensions to make join robust
        def _strip_ext(s: str | None) -> str | None:
            if s is None:
                return None
            s = s.strip()
            # remove common vendor suffixes
            for ext in (".raw", ".d", ".mzML", ".mzXML", ".wiff"):
                if s.lower().endswith(ext.lower()):
                    return s[: -len(ext)]
            return s

        ann = ann.with_columns(
            pl.col("FILENAME").map_elements(_strip_ext, return_dtype=pl.Utf8).alias("FILENAME")
        )

        ann = ann.with_columns(
            pl.col("Condition").cast(pl.Utf8).str.strip_chars(),
            pl.col("Replicate").cast(pl.Int64, strict=False),
        )
        if ann.select(pl.col("Replicate").is_null().any()).item():
            raise ValueError(f"Annotation file {self.annotation_file!r}: Replicate contains non-integer or null values.")

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
        excl = set(getattr(self, "exclude_runs", set()) or set())
        df_files  = df_files  - excl
        ann_files = ann_files - excl
        missing = sorted(ann_files - df_files)
        extra   = sorted(df_files - ann_files)
        self._fmt_diff(missing, extra, "filenames")

    def _validate_wide_annotation(self, df: pl.DataFrame, ann: pl.DataFrame) -> None:
        ann_filenames = set(ann.select("FILENAME").unique().to_series().to_list())
        excl = set(getattr(self, "exclude_runs", set()) or set())
        df_files  = df_files  - excl
        ann_filenames = ann_filenames - excl

        # Infer candidate intensity columns (unchanged logic)
        float_types = {pl.Float64, pl.Float32}
        schema = df.schema
        float_cols = {c for c, t in schema.items() if t in float_types}
        meta_float_exclude = {"ReferenceIntensity", "MaxPepProb"}
        data_channels = float_cols - meta_float_exclude

        missing = sorted([c for c in ann_filenames if c not in df.columns])
        extra   = sorted([c for c in data_channels if c not in ann_filenames])
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
                # For wide input without annotation, 'signal_column' and 'spectral_counts_column'
                # have already been consumed when melting (e.g. 'A_1 Intensity' -> SIGNAL).
                if (
                    self.input_layout == "wide"
                    and not self.annotation_file
                    and original in {self.signal_key, self.spectral_counts_key}
                ):
                    continue
                log_warning(f"Column '{original}' not found in input data, skipping harmonization for it.")

        return df.rename(rename_map) if rename_map else df

    def _standardize_then_inject(self, df: pl.DataFrame) -> pl.DataFrame:
        """Common path: rename → strict filename check → annotation join → coalesce condition/replicate."""
        df = self._rename_columns_safely(df)

        # Always enforce canonical CONDITION/REPLICATE (annotation or not).
        condition_sources = [c for c in ["CONDITION"] if c in df.columns]
        replicate_sources = [r for r in ["REPLICATE"] if r in df.columns]

        if not condition_sources:
            raise ValueError("Missing condition column: expected 'CONDITION'.")
        if not replicate_sources:
            raise ValueError("Missing replicate column: expected 'REPLICATE'.")

        # Pick a single source deterministically (prefer canonical if present)
        cond_src = condition_sources[0]
        rep_src = replicate_sources[0]

        if cond_src != "CONDITION":
            df = df.with_columns(pl.col(cond_src).alias("CONDITION")).drop(cond_src, strict=False)
        if rep_src != "REPLICATE":
            df = df.with_columns(pl.col(rep_src).alias("REPLICATE")).drop(rep_src, strict=False)

        # Strong typing: CONDITION must be string, REPLICATE must be integer.
        df = df.with_columns(
            pl.col("CONDITION").cast(pl.Utf8),
            pl.col("REPLICATE").cast(pl.Int64),
        )

        # Hard fail on nulls (these break imputation downstream)
        if df.select(pl.col("CONDITION").is_null().any()).item():
            raise ValueError("CONDITION contains nulls after standardization. Fix input/annotation; refusing to proceed.")
        if df.select(pl.col("REPLICATE").is_null().any()).item():
            raise ValueError("REPLICATE contains nulls after standardization. Fix input/annotation; refusing to proceed.")

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
        #df = df.join(ann, on="FILENAME", how="left")
        ann_join = ann.select(
            [
                pl.col("FILENAME"),
                pl.col("Condition").alias("_ANN_CONDITION"),
                pl.col("Replicate").alias("_ANN_REPLICATE"),
            ]
        )
        df = df.join(ann_join, on="FILENAME", how="left")

        # Hard fail if any row did not match annotation
        if df.select(pl.col("_ANN_CONDITION").is_null().any()).item():
            raise ValueError(
                "Annotation join produced NULL CONDITION values. "
                "This means some FILENAME values in the data did not match the annotation (after stripping extensions)."
            )
        if df.select(pl.col("_ANN_REPLICATE").is_null().any()).item():
            raise ValueError(
                "Annotation join produced NULL REPLICATE values. "
                "This means some FILENAME values in the data did not match the annotation (after stripping extensions)."
            )

        # Explicit overwrite from annotation (canonical source of truth when provided)
        df = df.with_columns(
            pl.col("_ANN_CONDITION").alias("CONDITION"),
            pl.col("_ANN_REPLICATE").alias("REPLICATE"),
        ).drop(["_ANN_CONDITION", "_ANN_REPLICATE"], strict=False)

        # Handling of CONDITION/REPLICATE:
        required = {"CONDITION", "REPLICATE"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns after harmonization: {sorted(missing)}. "
                "Both CONDITION and REPLICATE must be present."
            )

        # No implicit/empty conditions: fail fast.
        if df.select(pl.col("CONDITION").is_null().any()).item():
            raise ValueError("CONDITION contains NULL values after harmonization/injection.")
        if df.select(pl.col("REPLICATE").is_null().any()).item():
            raise ValueError("REPLICATE contains NULL values after harmonization/injection.")

        return df

    def _melt_wide_to_long_no_annotation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Melt a wide matrix without external annotation.

        Uses dataset.signal_column (signal_key) to detect intensity columns,
        and expects column names of the form '<CONDITION>_<N> <signal_key>',
        for example: 'A_1 Intensity', 'B_4 Intensity', ...
        """
        if not self.signal_key:
            raise ValueError(
                "Wide input without annotation: dataset.signal_column must be set "
                "(for FragPipe this is typically 'Intensity')."
            )

        pattern = re.compile(rf"^(.+?)\s+{re.escape(self.signal_key)}$")

        signal_cols: list[str] = []
        labels: list[str] = []

        for col in df.columns:
            m = pattern.match(col)
            if m:
                signal_cols.append(col)
                labels.append(m.group(1).strip())

        if not signal_cols:
            raise ValueError(
                f"Wide input without annotation: no columns match the pattern "
                f"'<LABEL> {self.signal_key}'\"".rstrip()
                + f". Example columns: {df.columns[:10]}"
            )

        # Parse labels as CONDITION + numeric replicate index (e.g. A_1, B_4, ...)
        label_re = re.compile(r"^([A-Za-z]+)_(\d+)$")

        parsed: list[tuple[str, int]] = []
        for lab in labels:
            m = label_re.match(lab)
            if not m:
                raise ValueError(
                    "Wide input without annotation: cannot parse sample label "
                    f"'{lab}' as '<CONDITION>_<number>'. "
                    f"Example labels: {labels[:10]}"
                )
            parsed.append((m.group(1), int(m.group(2))))

        # Renumber replicates per CONDITION: 1..N within each condition (order by original number)
        by_cond: dict[str, list[tuple[int, int]]] = {}
        for i, (cond, num) in enumerate(parsed):
            by_cond.setdefault(cond, []).append((num, i))

        repls = [0] * len(labels)
        for pairs in by_cond.values():
            pairs.sort(key=lambda x: x[0])
            for new_r, (_, idx) in enumerate(pairs, start=1):
                repls[idx] = new_r

        if any(r <= 0 for r in repls):
            raise RuntimeError("Replicate renumbering failed: encountered unset replicate(s).")

        conds = [cond for cond, _ in parsed]

        # Build small channel metadata table
        channel_meta = pl.DataFrame(
            {
                "CHANNEL": signal_cols,
                "FILENAME": labels,
                "CONDITION": conds,
                "REPLICATE": repls,
            }
        )
        # Optionally detect spectral-count channels using the same LABEL set
        count_cols: list[str] = []
        if self.spectral_counts_key:
            pattern_counts = re.compile(rf"^(.+?)\s+{re.escape(self.spectral_counts_key)}$")

            count_labels: list[str] = []
            for col in df.columns:
                m = pattern_counts.match(col)
                if m:
                    count_cols.append(col)
                    count_labels.append(m.group(1).strip())

            if count_cols:
                # Require exact match of labels between intensity and spectral counts
                if set(count_labels) != set(labels):
                    raise ValueError(
                        "Wide input without annotation: intensity and spectral-count channel labels do not match.\n"
                        f"  Intensity labels: {sorted(set(labels))}\n"
                        f"  Spectral-count labels: {sorted(set(count_labels))}"
                    )
            else:
                # Config requested spectral_counts_column, but no matching wide columns
                raise ValueError(
                    "Wide input without annotation: 'spectral_counts_column' was set "
                    f"to '{self.spectral_counts_key}', but no columns match the pattern "
                    f"'<LABEL> {self.spectral_counts_key}'."
                )

        # Identifier columns are everything except the intensity and spectral-count channels
        id_vars = [c for c in df.columns if c not in signal_cols and c not in count_cols]
        if not id_vars:
            raise ValueError(
                "Wide input without annotation: no identifier columns left after "
                "selecting signal channels. Check your input table."
            )

        # Identifier columns are everything except the intensity channels
        if not id_vars:
            raise ValueError(
                "Wide input without annotation: no identifier columns left after "
                "selecting signal channels. Check your input table."
            )

        # Melt intensities
        long_int = df.melt(
            id_vars=id_vars,
            value_vars=signal_cols,
            variable_name="CHANNEL_INT",
            value_name="SIGNAL",
        ).with_columns(
            pl.col("CHANNEL_INT").cast(pl.Utf8),
            pl.col("SIGNAL").cast(pl.Float64, strict=False),
        )

        # Attach FILENAME, CONDITION, REPLICATE for intensities
        long_int = long_int.join(
            channel_meta.rename({"CHANNEL": "CHANNEL_INT"}),
            on="CHANNEL_INT",
            how="left",
        ).drop("CHANNEL_INT")

        # Optionally melt spectral counts and join on id_vars + FILENAME
        if self.spectral_counts_key and count_cols:
            channel_counts_meta = pl.DataFrame(
                {
                    "CHANNEL_CNT": count_cols,
                    "FILENAME": count_labels,
                }
            )

            long_cnt = df.melt(
                id_vars=id_vars,
                value_vars=count_cols,
                variable_name="CHANNEL_CNT",
                value_name="SPECTRAL_COUNTS",
            ).with_columns(
                pl.col("CHANNEL_CNT").cast(pl.Utf8),
                pl.col("SPECTRAL_COUNTS").cast(pl.Float64, strict=False),
            )

            long_cnt = long_cnt.join(channel_counts_meta, on="CHANNEL_CNT", how="left").drop("CHANNEL_CNT")

            join_keys = id_vars + ["FILENAME"]
            long_df = long_int.join(long_cnt, on=join_keys, how="left")
        else:
            long_df = long_int

        # Tag layout
        long_df = long_df.with_columns(pl.lit("wide").alias("INPUT_LAYOUT"))

        # Informative log (same as in 1.)
        preview_df = (
            long_df.select(["FILENAME", "CONDITION", "REPLICATE"])
                   .unique()
                   .sort(["CONDITION", "REPLICATE"])
        )
        lines = []
        for row in preview_df.iter_rows(named=True):
            lines.append(
                f"    FILENAME={row['FILENAME']} | "
                f"CONDITION={row['CONDITION']} | "
                f"REPLICATE={row['REPLICATE']}"
            )

        msg = (
            f"Wide input without annotation: inferred {len(signal_cols)} runs "
            f"from columns ending with ' {self.signal_key}'.\n"
            f"  Mappings:\n" + "\n".join(lines)
        )
        log_info(msg)

        return long_df

    def _melt_wide_to_long(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Melt a wide matrix into the canonical long schema.

        If an annotation file is provided, it is used to define filenames and metadata.
        If not, filenames are inferred from columns ending with '<signal_key>'.
        """
        if self.annotation_file:
            ann = self._load_annotation()
            filenames = ann.select("FILENAME").unique().to_series().to_list()

            # Strict two-way validation before melt
            self._validate_wide_annotation(df, ann)

            id_vars = [c for c in df.columns if c not in filenames]
            if not id_vars:
                raise ValueError(
                    "No identifier columns left after selecting filenames; "
                    "check your annotation and table."
                )

            long_df = df.melt(
                id_vars=id_vars,
                value_vars=filenames,
                variable_name="FILENAME",
                value_name="SIGNAL",
            ).with_columns(
                pl.col("FILENAME").cast(pl.Utf8),
                pl.col("SIGNAL").cast(pl.Float64, strict=False),
            )

            alias_map = {}
            if "Spectrum Number" in long_df.columns:
                alias_map["Spectrum Number"] = "PRECURSORS_EXP"
            if alias_map:
                long_df = long_df.rename(alias_map)

            long_df = long_df.with_columns(pl.lit("wide").alias("INPUT_LAYOUT"))
            return long_df

        # No annotation: infer FILENAME, CONDITION, REPLICATE from headers
        return self._melt_wide_to_long_no_annotation(df)

    def _strip_mods(self, s: str) -> str:
        return _strip_mods_impl(s)

    def _convert_numeric_ptms(self, s: str | None) -> str | None:
        return _convert_numeric_ptms_impl(
            s,
            enabled=self.convert_numeric_ptms,
            ptm_map=PTM_MAP,
        )
    def _normalize_peptido_index_seq(self, s: str | None) -> str | None:
        return _normalize_peptido_index_seq_impl(
            s,
            convert_numeric_ptms_enabled=self.convert_numeric_ptms,
            collapse_all_ptms=self.collapse_all_ptms,
            ptm_map=PTM_MAP,
        )

    def _assert_required_phospho_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        # For phospho indexing we need both:
        # - peptide-local PTM positions (Spectronaut: "EG.PTMPositions [Phospho (STY)]")
        # - absolute protein PTM locations (Spectronaut: "EG.ProteinPTMLocations")
        required = {
            "PEPTIDE_START",
            "PEPTIDE_LSEQ",
            "PTM_SITES_STR",
            "PTM_POSITIONS_STR",
            "PTM_PROTEINLOCATIONS",
        }
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                "[PHOSPHO] Missing required columns after harmonization: "
                f"{missing}. Ensure your config maps:\n"
                "  peptide_start_column -> PEPTIDE_START (e.g. 'PEP.PeptidePosition')\n"
                "  peptide_seq_column -> PEPTIDE_LSEQ (e.g. 'EG.ModifiedSequence')\n"
                "  ptm_positions_str -> PTM_POSITIONS_STR (e.g. 'EG.PTMPositions [Phospho (STY)]')\n"
                "  ptm_proteinlocations_column -> PTM_PROTEINLOCATIONS (e.g. 'EG.ProteinPTMLocations')\n"
                "  ptm_sites_column -> PTM_SITES_STR (e.g. 'EG.PTMSites [Phospho (STY)]')\n"
            )

        # Prefer vendor-specific per-site probs; fall back to the generic localization probs if present
        has_vendor_probs = "PTM_PROBS_STR" in df.columns
        has_generic_probs = "EG.PTMLocalizationProbabilities" in df.columns
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
        if "PEPTIDE_LSEQ" not in df.columns:
            raise ValueError(
                "[PEPTIDOMICS] Missing required column 'PEPTIDE_LSEQ'. "
                "Map 'peptide_seq_column' to the modified peptide sequence in the config."
            )

        df = df.with_columns(
            pl.col("PEPTIDE_LSEQ").cast(pl.Utf8).alias("_seq_candidate")
        ).with_columns(
            pl.col("_seq_candidate")
              .map_elements(self._strip_mods, return_dtype=str)
              .alias("STRIPPED_SEQ"),
            pl.col("_seq_candidate")
              .map_elements(self._normalize_peptido_index_seq, return_dtype=str)
              .alias("_INDEX_SEQ"),
        ).drop(["_seq_candidate"], strict=False)

        df = df.with_columns(
            pl.col("STRIPPED_SEQ").alias("PARENT_PEPTIDE_ID"),
            pl.when(pl.col("UNIPROT").is_not_null())
              .then(pl.col("UNIPROT"))
              .otherwise(None)
              .alias("PARENT_PROTEIN"),
            pl.col("_INDEX_SEQ").alias("INDEX"),
        ).drop(["_INDEX_SEQ"], strict=False)

        if "ASSAY" not in df.columns:
            df = df.with_columns(pl.lit("FLOWTHROUGH").alias("ASSAY"))

        return df

    @log_time("Building Phospho Index")
    def _build_phospho_index(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self._assert_required_phospho_columns(df)

        # ------------------------------------------------------------------
        # 1) Gate phospho rows
        #    Trust Spectronaut: PTMSites [Phospho (STY)] is the declaration of phospho
        # ------------------------------------------------------------------
        n0 = df.height
        df = df.filter(
            pl.col("PTM_SITES_STR").is_not_null()
            & (pl.col("PTM_SITES_STR").cast(pl.Utf8).str.strip_chars() != "")
        )
        n1 = df.height
        log_info(f"Removing non-phospho PSMs, kept: {n1}/{n0}")

        # ------------------------------------------------------------------
        # 2) Parse peptide-local candidate positions + probabilities
        # ------------------------------------------------------------------
        df = df.with_columns(
            pl.col("PTM_POSITIONS_STR")
              .cast(pl.Utf8)
              .str.strip_chars()
              .str.split(";")
              .list.eval(pl.element().cast(pl.Int64, strict=False))
              .alias("_ptmpos_list")
        )

        df = df.with_columns(
            pl.when(
                pl.col("PTM_PROBS_STR").is_not_null()
                & (pl.col("PTM_PROBS_STR").cast(pl.Utf8).str.strip_chars() != "")
            )
            .then(
                pl.col("PTM_PROBS_STR")
                  .cast(pl.Utf8)
                  .str.split(";")
                  .list.eval(pl.element().cast(pl.Float64, strict=False))
            )
            .otherwise(pl.lit([]))
            .alias("_prob_list")
        )

        # Align probabilities to positions (or nulls if vendor gave nonsense)
        df = df.with_columns(
            pl.when(pl.col("_prob_list").list.len() == pl.col("_ptmpos_list").list.len())
              .then(pl.col("_prob_list"))
              .otherwise(pl.col("_ptmpos_list").list.eval(pl.lit(None, dtype=pl.Float64)))
              .alias("_prob_list_aligned")
        )

        # Explode peptide-local candidates
        df2 = (
            df.with_columns(
                pl.int_ranges(pl.lit(0), pl.col("_ptmpos_list").list.len()).alias("_idx")
            )
            .explode("_idx")
            .with_columns(
                pl.col("_ptmpos_list").list.get(pl.col("_idx")).alias("SITE_POS"),
                pl.col("_prob_list_aligned").list.get(pl.col("_idx")).alias("LOC_PROB"),
            )
            .drop("_idx")
        )

        # Normalize probabilities (0–100 → 0–1) and drop explicit zeros
        df2 = df2.with_columns(
            pl.when((pl.col("LOC_PROB") > 1.0) & pl.col("LOC_PROB").is_not_null())
              .then(pl.col("LOC_PROB") / 100.0)
              .otherwise(pl.col("LOC_PROB"))
              .alias("LOC_PROB")
        ).filter(
            pl.col("LOC_PROB").is_null() | (pl.col("LOC_PROB") > 0.0)
        )

        # ------------------------------------------------------------------
        # 3) Parse ProteinPTMLocations → phospho-only biological sites
        # ------------------------------------------------------------------
        df2 = df2.with_columns(
            pl.col("PTM_PROTEINLOCATIONS")
              .cast(pl.Utf8)
              .str.extract_all(r"[STY]\d+")
              .alias("_prot_sites")
        ).explode("_prot_sites")

        df2 = df2.with_columns(
            pl.col("_prot_sites").str.slice(0, 1).alias("_AA"),
            pl.col("_prot_sites").str.extract(r"(\d+)").cast(pl.Int64).alias("_ABS_POS"),
        )

        # ------------------------------------------------------------------
        # 4) Map protein position → peptide-local position
        # ------------------------------------------------------------------
        df2 = df2.with_columns(
            (
                pl.col("_ABS_POS")
                - pl.col("PEPTIDE_START").cast(pl.Int64, strict=False)
                + 1
            ).alias("SITE_POS")
        )

        # Keep only real phospho sites:
        #  - valid peptide-local position
        #  - present in PTMPositions [Phospho]
        df2 = df2.filter(
            pl.col("SITE_POS").is_not_null()
            & (pl.col("SITE_POS") >= 1)
            & pl.col("_ptmpos_list").list.contains(pl.col("SITE_POS"))
        )

        # ------------------------------------------------------------------
        # 5) Build indices
        # ------------------------------------------------------------------
        df2 = df2.with_columns(
            (pl.col("PEP_SEQUENCE") + pl.lit("|p") + pl.col("SITE_POS").cast(pl.Utf8))
                .alias("PEPTIDE_INDEX"),
            (
                pl.col("UNIPROT")
                + pl.lit("|")
                + pl.col("_prot_sites").str.replace_all(r"[()]", "")
                #+ pl.col("_AA")
                #+ pl.col("_ABS_POS").cast(pl.Utf8)
            ).alias("INDEX"),
            pl.col("UNIPROT").alias("PARENT_PROTEIN"),
            pl.col("PEP_SEQUENCE").alias("PARENT_PEPTIDE_ID"),
            pl.lit("PHOSPHO").alias("ASSAY"),
        )

        # ------------------------------------------------------------------
        # 6) Diagnostics
        # ------------------------------------------------------------------
        log_info(
            f"Unique phosphosites: {df2.select('INDEX').n_unique()}"
        )

        # ------------------------------------------------------------------
        # 7) Hard invariant: INDEX must never be null
        # ------------------------------------------------------------------
        null_idx = df2.filter(pl.col("INDEX").is_null())
        if null_idx.height > 0:
            log_info("[PHOSPHO] Null INDEX examples:")
            log_info(
                null_idx.select(
                    "UNIPROT",
                    "PEP_SEQUENCE",
                    "PEPTIDE_START",
                    "PTM_POSITIONS_STR",
                    "PTM_PROTEINLOCATIONS",
                    "SITE_POS",
                    "_AA",
                    "_ABS_POS",
                    "LOC_PROB",
                ).head(20)
            )
            raise ValueError("[PHOSPHO] Null INDEX produced — phospho indexing is broken.")

        return df2

    @log_time("Data Harmonizing")
    def harmonize(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.input_layout not in {"long", "wide"}:
            raise ValueError("dataset.input_layout must be 'long' or 'wide'.")

        if self.input_layout == "wide":
            df = self._melt_wide_to_long(df)

        df = self._standardize_then_inject(df)

        if self.analysis_type == "phospho":
            df = self._build_phospho_index(df)
        elif self.analysis_type == "peptidomics":
            df = self._build_peptido_index(df)

        return df
