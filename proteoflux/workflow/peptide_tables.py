from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING, List

import numpy as np
import polars as pl

from proteoflux.utils.sequence_ops import expr_peptide_index_seq

if TYPE_CHECKING:
    from proteoflux.workflow.preprocessing import Preprocessor

def _center_by_rowmean(df: pl.DataFrame, sample_cols: list[str]) -> pl.DataFrame:
    """
    Mechanical extraction of the current centering logic:
      - compute row mean using mean_horizontal after converting NaN -> null
      - divide each sample column by that mean
    """
    return (
        df.with_columns(
            pl.mean_horizontal(
                [
                    pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c))
                    for c in sample_cols
                ]
            ).alias("__rowmean__")
        )
        .with_columns([(pl.col(c) / pl.col("__rowmean__")).alias(c) for c in sample_cols])
        .drop("__rowmean__")
    )

def _build_consistency_precursors(
    *,
    self: "Preprocessor",
    df: pl.DataFrame,
    cond_to_runs: dict[str, list[str]],
) -> None:
    """Mechanical extraction of precursor consistency logic (peptidomics/phospho)."""

    prec_long = (
        df.select(["INDEX", "PEPTIDE_LSEQ", "CHARGE", "FILENAME", "SIGNAL"])
        .drop_nulls(["INDEX", "CHARGE"])
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("INDEX").cast(pl.Utf8),
                        pl.lit("/+"),
                        pl.col("CHARGE").cast(pl.Utf8),
                    ]
                ).alias("PREC_KEY"),
                (pl.col("SIGNAL").is_not_null() & ~pl.col("SIGNAL").is_nan()).alias(
                    "HAS_SIGNAL"
                ),
            ]
        )
        .group_by(["PREC_KEY", "FILENAME"], maintain_order=True)
        .agg(pl.col("HAS_SIGNAL").any().alias("HAS_SIGNAL"))
    )

    prec2idx = (
        df.select(["INDEX", "CHARGE"])
        .drop_nulls(["INDEX", "CHARGE"])
        .with_columns(
            pl.concat_str(
                [
                    pl.col("INDEX").cast(pl.Utf8),
                    pl.lit("/+"),
                    pl.col("CHARGE").cast(pl.Utf8),
                ]
            ).alias("PREC_KEY")
        )
        .select(["PREC_KEY", "INDEX"])
        .unique()
    )

    wide_prec = (
        prec_long.pivot(index="PREC_KEY", columns="FILENAME", values="HAS_SIGNAL")
        .fill_null(False)
    )
    bool_sample_cols = [c for c in wide_prec.columns if c != "PREC_KEY"]

    per_cond_prec: List[pl.DataFrame] = []

    for cond, runs in cond_to_runs.items():
        cond_cols = [c for c in bool_sample_cols if c in runs]
        if not cond_cols:
            continue

        cons_flag = pl.all_horizontal([pl.col(c) for c in cond_cols]).alias("_cons")

        cons_prec = (
            wide_prec.select(["PREC_KEY"] + cond_cols)
            .with_columns(cons_flag)
            .select(["PREC_KEY", pl.col("_cons").cast(pl.Int64).alias("_CONS_INT")])
        )

        cons_counts = (
            cons_prec.join(prec2idx, on="PREC_KEY", how="left")
            .group_by("INDEX", maintain_order=True)
            .agg(pl.col("_CONS_INT").sum().alias(f"CONSISTENT_PRECURSOR_{cond}"))
        )

        per_cond_prec.append(cons_counts)

    if per_cond_prec:
        cons_df = per_cond_prec[0]
        for extra in per_cond_prec[1:]:
            cons_df = cons_df.join(extra, on="INDEX", how="outer")
            keep_cols = ["INDEX"] + [
                c for c in cons_df.columns if c.startswith("CONSISTENT_PRECURSOR_")
            ]
            cons_df = cons_df.select(keep_cols)

        self.intermediate_results.add_df("consistent_precursors_per_condition", cons_df)


def _build_consistency_peptides(
    *,
    self: "Preprocessor",
    pep_wide: pl.DataFrame,
    sample_cols: list[str],
    cond_to_runs: dict[str, list[str]],
) -> None:
    """Mechanical extraction of peptide consistency logic (proteomics)."""

    per_cond_pep: List[pl.DataFrame] = []

    for cond, runs in cond_to_runs.items():
        cond_cols = [c for c in sample_cols if c in runs]
        if not cond_cols:
            continue

        consistent_flag = pl.all_horizontal(
            [pl.col(c).is_not_null() & ~pl.col(c).is_nan() for c in cond_cols]
        ).alias("_cons")

        tmp = (
            pep_wide.select(["INDEX"] + cond_cols)
            .with_columns(consistent_flag)
            .group_by("INDEX", maintain_order=True)
            .agg(
                pl.col("_cons")
                .sum()
                .cast(pl.Int64)
                .alias(f"CONSISTENT_PEPTIDE_{cond}")
            )
        )
        per_cond_pep.append(tmp)

    if per_cond_pep:
        cons_df = per_cond_pep[0]
        for extra in per_cond_pep[1:]:
            cons_df = cons_df.join(extra, on="INDEX", how="outer")
            keep_cols = ["INDEX"] + [
                c for c in cons_df.columns if c.startswith("CONSISTENT_PEPTIDE_")
            ]
            cons_df = cons_df.select(keep_cols)

        self.intermediate_results.add_df("consistent_peptides_per_condition", cons_df)

def build_peptide_tables(
    self: "Preprocessor",
) -> Optional[Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Extracted from Preprocessor._build_peptide_tables verbatim.
    This function is intentionally *not* a redesign: same logic, same side-effects,
    same IntermediateResults keys.
    """

    df = self.intermediate_results.dfs["filtered_final_censored"]

    # Compute missed cleavages once, on the post-filter peptide universe.
    df_mc = df
    if "IS_COVARIATE" in df_mc.columns:
        df_mc = df_mc.filter(
            ~pl.col("IS_COVARIATE").cast(pl.Boolean, strict=False).fill_null(False)
        )
    self.intermediate_results.dfs["missed_cleavages_per_sample"] = (
        self._compute_missed_cleavages_per_sample(df_mc)
    )

    needed = {"INDEX", "FILENAME", "SIGNAL", "PEPTIDE_LSEQ"}
    if not needed.issubset(df.columns):
        return None

    analysis = self.analysis_type
    precursor_only = analysis in {"peptidomics", "phospho"}

    #seq_clean = expr_clean_peptide_seq("PEPTIDE_LSEQ").alias("PEPTIDE_SEQ")
    seq_clean = expr_peptide_index_seq(
        "PEPTIDE_LSEQ",
        collapse_met_oxidation=getattr(self, "collapse_met_oxidation", True),
        drop_ptms=getattr(self, "drop_ptms", False),
    ).alias("PEPTIDE_SEQ")

    # 1) Clean peptide sequence
    pep_wide = None
    pep_centered = None
    if not precursor_only:
        base = (
            df.select(
                pl.col("INDEX"),
                pl.col("FILENAME"),
                pl.col("SIGNAL"),
                seq_clean,
            ).with_columns(
                (
                    pl.col("INDEX").cast(pl.Utf8)
                    + pl.lit("|")
                    + pl.col("PEPTIDE_SEQ")
                ).alias("PEPTIDE_ID")
            )
        )

    # 2) Distributions: precursors per peptide / per protein, peptides per protein
    if not precursor_only:
        # Proteomics: peptide_id = INDEX|PEPTIDE_SEQ; precursor = peptide_id + charge
        prec_stats = (
            df.select(
                pl.col("INDEX"),
                seq_clean,
                pl.col("CHARGE"),
            ).with_columns(
                (
                    pl.col("INDEX").cast(pl.Utf8)
                    + pl.lit("|")
                    + pl.col("PEPTIDE_SEQ")
                ).alias("PEPTIDE_ID")
            )
        )

        prec_per_pep = (
            prec_stats.select(["PEPTIDE_ID", "CHARGE"])
            .drop_nulls(["PEPTIDE_ID", "CHARGE"])
            .unique()
            .group_by("PEPTIDE_ID", maintain_order=True)
            .agg(pl.len().alias("N_PREC_PER_PEP"))
        )
        self.intermediate_results.add_array(
            "num_precursors_per_peptide",
            prec_per_pep["N_PREC_PER_PEP"].to_numpy(),
        )

        pep_per_prot = (
            base.select(["INDEX", "PEPTIDE_ID"])
            .drop_nulls(["INDEX", "PEPTIDE_ID"])
            .unique()
            .group_by("INDEX", maintain_order=True)
            .agg(pl.len().alias("N_PEP_PER_PROT"))
        )
        self.intermediate_results.add_array(
            "num_peptides_per_protein",
            pep_per_prot["N_PEP_PER_PROT"].to_numpy(),
        )

        prec_per_prot = (
            prec_stats.select(["INDEX", "PEPTIDE_ID", "CHARGE"])
            .drop_nulls(["PEPTIDE_ID", "CHARGE"])
            .unique()
            .group_by("INDEX", maintain_order=True)
            .agg(pl.len().alias("N_PREC_PER_PROT"))
        )
        self.intermediate_results.add_array(
            "num_precursors_per_protein",
            prec_per_prot["N_PREC_PER_PROT"].to_numpy(),
        )
    else:
        # peptidomics/phospho: precursor = peptide_key + charge
        analysis_lc = (self.analysis_type or "").strip().lower()

        _require_cols = {"FILENAME", "SIGNAL", "CHARGE"}
        missing = _require_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns for precursor distributions: {sorted(missing)!r}"
            )

        if analysis_lc == "phospho":
            # peptide key is PEPTIDE_INDEX; protein key is PARENT_PROTEIN
            req = {"PEPTIDE_INDEX", "PARENT_PROTEIN"}
            missing = req - set(df.columns)
            if missing:
                raise ValueError(
                    f"[PHOSPHO] Missing required columns for distributions: {sorted(missing)!r}"
                )

            base_prec = (
                df.select(
                    pl.col("PEPTIDE_INDEX").cast(pl.Utf8).alias("PEP_KEY"),
                    pl.col("PARENT_PROTEIN").cast(pl.Utf8).alias("PROT_KEY"),
                    pl.col("CHARGE"),
                )
                .drop_nulls(["PEP_KEY", "PROT_KEY", "CHARGE"])
                .unique()
            )
        else:
            # peptidomics: peptide key is INDEX (strip anything after '|'); protein key is PARENT_PROTEIN
            req = {"INDEX", "PARENT_PROTEIN"}
            missing = req - set(df.columns)
            if missing:
                raise ValueError(
                    f"[PEPTIDO] Missing required columns for distributions: {sorted(missing)!r}"
                )

            pep_key = (
                pl.col("INDEX")
                .cast(pl.Utf8)
                .str.split("|")
                .list.get(0)
                .alias("PEP_KEY")
            )
            base_prec = (
                df.select(
                    pep_key,
                    pl.col("PARENT_PROTEIN").cast(pl.Utf8).alias("PROT_KEY"),
                    pl.col("CHARGE"),
                )
                .drop_nulls(["PEP_KEY", "PROT_KEY", "CHARGE"])
                .unique()
            )

        # a) Precursors per peptide: count unique charge per peptide key
        prec_per_pep = (
            base_prec.group_by("PEP_KEY", maintain_order=True)
            .agg(pl.len().alias("N_PREC_PER_PEP"))
        )
        self.intermediate_results.add_array(
            "num_precursors_per_peptide",
            prec_per_pep["N_PREC_PER_PEP"].to_numpy(),
        )

        # b) Peptides per protein: count unique peptide keys per protein
        pep_per_prot = (
            base_prec.select(["PROT_KEY", "PEP_KEY"])
            .unique()
            .group_by("PROT_KEY", maintain_order=True)
            .agg(pl.len().alias("N_PEP_PER_PROT"))
        )
        self.intermediate_results.add_array(
            "num_peptides_per_protein",
            pep_per_prot["N_PEP_PER_PROT"].to_numpy(),
        )

        # c) Precursors per protein: count unique (peptide,charge) per protein
        prec_per_prot = (
            base_prec.group_by("PROT_KEY", maintain_order=True)
            .agg(pl.len().alias("N_PREC_PER_PROT"))
        )
        self.intermediate_results.add_array(
            "num_precursors_per_protein",
            prec_per_prot["N_PREC_PER_PROT"].to_numpy(),
        )

    # 3) Peptide-wide matrices (sum duplicates)
    if not precursor_only:
        pep_pivot = self._pivot_df(
            df=base.select(["PEPTIDE_ID", "FILENAME", "SIGNAL"]),
            sample_col="FILENAME",
            protein_col="PEPTIDE_ID",
            values_col="SIGNAL",
            aggregate_fn=self.peptide_rollup_method,
        )

        id_map = base.select(["PEPTIDE_ID", "INDEX", "PEPTIDE_SEQ"]).unique()
        pep_wide = pep_pivot.join(id_map, on="PEPTIDE_ID", how="left")

        sample_cols = [
            c
            for c in pep_wide.columns
            if c not in ("PEPTIDE_ID", "INDEX", "PEPTIDE_SEQ")
        ]

        pep_centered = _center_by_rowmean(pep_wide, sample_cols)

    # 4) Precursor-wide matrices (peptidomics/phospho: same feature + charge)
    if analysis == "phospho":
        if "PEPTIDE_INDEX" not in df.columns:
            raise ValueError("[PHOSPHO] Missing required column 'PEPTIDE_INDEX' for precursor tables.")

        prec_base = (
            df.select(
                pl.col("INDEX").cast(pl.Utf8).alias("SITE_ID"),
                pl.col("PEPTIDE_INDEX").cast(pl.Utf8).alias("PEPTIDE_INDEX"),
                pl.col("FILENAME"),
                pl.col("SIGNAL"),
                pl.col("CHARGE"),
            )
            .drop_nulls(["SITE_ID", "PEPTIDE_INDEX", "CHARGE"])
            .with_columns(
                (
                    pl.col("PEPTIDE_INDEX")
                    + pl.lit("/+")
                    + pl.col("CHARGE").cast(pl.Utf8)
                ).alias("PREC_ID")
            )
        )
    else:
        index_clean = (
            pl.col("INDEX")
            .cast(pl.Utf8)
            .str.split("|")
            .list.get(0)
            .alias("_INDEX_FEATURE")
        )
        prec_base = (
            df.select(
                index_clean,
                pl.col("FILENAME"),
                pl.col("SIGNAL"),
                pl.col("CHARGE"),
            )
            .drop_nulls(["_INDEX_FEATURE", "CHARGE"])
            .with_columns(
                (
                    pl.col("_INDEX_FEATURE")
                    + pl.lit("/+")
                    + pl.col("CHARGE").cast(pl.Utf8)
                ).alias("PREC_ID")
            )
            .rename({"_INDEX_FEATURE": "INDEX"})
            .with_columns(pl.col("INDEX").alias("PEPTIDE_SEQ"))
        )

    prec_pivot = self._pivot_df(
        df=prec_base.select(["PREC_ID", "FILENAME", "SIGNAL"]),
        sample_col="FILENAME",
        protein_col="PREC_ID",
        values_col="SIGNAL",
        aggregate_fn=self.peptide_rollup_method,
    )

    if analysis == "phospho":
        prec_id_map = (
            prec_base
            .select(["PREC_ID", "SITE_ID", "PEPTIDE_INDEX", "CHARGE"])
            .unique(subset=["PREC_ID"], maintain_order=True)
        )
    else:
        prec_id_map = (
            prec_base
            .select(["PREC_ID", "INDEX", "PEPTIDE_SEQ", "CHARGE"])
            .unique(subset=["PREC_ID"], maintain_order=True)
        )

    prec_wide = prec_pivot.join(prec_id_map, on="PREC_ID", how="left")

    id_cols = {"PREC_ID", "INDEX", "PEPTIDE_SEQ", "CHARGE", "SITE_ID", "PEPTIDE_INDEX"}
    prec_sample_cols = [
        c
        for c, dt in zip(prec_wide.columns, prec_wide.dtypes)
        if (c not in id_cols) and (dt in pl.NUMERIC_DTYPES)
    ]
    prec_centered = _center_by_rowmean(prec_wide, prec_sample_cols)

    self.intermediate_results.dfs["precursors_wide"] = prec_wide
    self.intermediate_results.dfs["precursors_centered"] = prec_centered

    # 5) Per-condition consistency (peptides vs precursors)
    cond_map = self.intermediate_results.dfs.get("condition_pivot")
    if cond_map is None:
        return pep_wide, pep_centered

    cond_map = cond_map.select(
        ["Sample", "CONDITION"]
        + (["IS_COVARIATE"] if "IS_COVARIATE" in cond_map.columns else [])
    ).rename({"Sample": "FILENAME"})

    if "IS_COVARIATE" in cond_map.columns:
        cond_map = cond_map.filter(~pl.col("IS_COVARIATE")).drop("IS_COVARIATE")

    if cond_map.height == 0:
        return pep_wide, pep_centered

    conditions = cond_map.select("CONDITION").unique().to_series().to_list()
    cond_to_runs = {
        cond: (
            cond_map.filter(pl.col("CONDITION") == cond)
            .select("FILENAME")
            .to_series()
            .to_list()
        )
        for cond in conditions
    }

    if analysis in ["peptidomics", "phospho"]:
        _build_consistency_precursors(
            self=self,
            df=df,
            cond_to_runs=cond_to_runs,
        )

    else:
        _build_consistency_peptides(
            self=self,
            pep_wide=pep_wide,
            sample_cols=sample_cols,
            cond_to_runs=cond_to_runs,
        )

    return pep_wide, pep_centered

