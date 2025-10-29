"""Export differential-expression results to Excel/CSV and write cleaned .h5ad.

This module assembles a single “Summary” table (metadata, log2FC, q/p, missingness,
intensities, and phospho-specific fields when relevant) and optional identification
tables. Behavior, column names, and sheet names are kept identical to the existing
implementation—only readability and documentation are improved.
"""
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
from proteoflux.utils.utils import logger, log_time
from importlib.metadata import version as _pkg_version, PackageNotFoundError


class DEExporter:
    def __init__(
        self,
        adata,
        output_path,
        use_xlsx=True,
        sig_threshold=0.05,
        annotate_matrix=False,
        config: dict | None = None,
    ):
        """Excel/CSV and .h5ad exporter for a processed `AnnData`."""
        self.adata = adata
        self.output_path = Path(output_path)
        self.use_xlsx = use_xlsx
        self.sig_threshold = sig_threshold
        self.annotate_matrix = False  # enforced off from version 1.7 on
        self.contrasts = self.adata.uns.get("contrast_names", [])
        self.config = config or self.adata.uns.get("config", {})

    def _get_dataframe(self, matrix_name: str) -> Optional[pd.DataFrame]:
        """Return varm[matrix_name] as a (features × contrasts) DataFrame, or None."""
        if matrix_name in self.adata.varm:
            return pd.DataFrame(self.adata.varm[matrix_name],
                                index=self.adata.var.index,
                                columns=self.contrasts)
        return None

    def _peptide_frames(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Return (raw_peptides_df, centered_peptides_df) built from `uns['peptides']`."""
        pep = self.adata.uns.get("peptides")
        if pep is None:
            return None, None
        meta = pd.DataFrame(
            {
                "PEPTIDE_ID": pep["rows"],
                "PROTEIN_UNIPROT_AC": pep["protein_index"],
                "PEPTIDE_SEQ": pep["peptide_seq"],
            },
            index=pep["rows"],
        )

        raw_df = pd.DataFrame(pep["raw"], index=pep["rows"], columns=pep["cols"])
        centered_df = pd.DataFrame(pep["centered"], index=pep["rows"], columns=pep["cols"])

        # PEPTIDES (raw): drop PEPTIDE_ID and reorder to [PEPTIDE_SEQ, PROTEIN_INDEX, samples...]
        raw_df = pd.concat([meta, raw_df], axis=1)
        raw_df = raw_df.drop(columns=["PEPTIDE_ID"], errors="ignore")
        leading = [c for c in ["PEPTIDE_SEQ", "PROTEIN_INDEX"] if c in raw_df.columns]
        others = [c for c in raw_df.columns if c not in leading]
        raw_df = raw_df[leading + others]

        # PEPTIDES (centered): keep same leading order (even if we don't export for phospho)
        centered_df = pd.concat([meta, centered_df], axis=1)
        centered_df = centered_df.drop(columns=["PEPTIDE_ID"], errors="ignore")
        leading_c = [c for c in ["PEPTIDE_SEQ", "PROTEIN_INDEX"] if c in centered_df.columns]
        others_c = [c for c in centered_df.columns if c not in leading_c]
        centered_df = centered_df[leading_c + others_c]

        return raw_df, centered_df

    def _export_excel(self, tables: Dict[str, Optional[pd.DataFrame]], readme: str) -> None:
        """Write selected tables to a single XLSX with a README sheet."""
        out_file = self.output_path.with_suffix(".xlsx")
        with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
            writer.book.use_zip64()

            # README like before: one line per row
            pd.DataFrame({"README": readme.split("\n")}).to_excel(
                writer, index=False, sheet_name="README"
            )

            header_fmt = writer.book.add_format({"bold": False, "align": "left", "border": 0})

            # Write tables; peptides sheet without index so A1 is the first header
            for name, df in tables.items():
                if df is None:
                    continue

                ws = writer.book.add_worksheet(name)

                if name == "Peptides (raw)":
                    columns = list(df.columns)
                    ws.write_row(0, 0, columns, header_fmt)

                    df.to_excel(writer, sheet_name=name, startrow=1, index=False, header=False)
                else:
                    df_out = df.reset_index()
                    columns = list(df_out.columns)
                    ws.write_row(0, 0, columns, header_fmt)
                    df_out.to_excel(writer, sheet_name=name, startrow=1, index=False, header=False)

                ws.set_column(0, len(columns)-1, 14)

    def _export_csvs(self, tables: Dict[str, Optional[pd.DataFrame]]) -> None:
        """Write each table to a separate CSV with a shared filename prefix."""
        prefix = self.output_path.with_suffix("")
        for name, df in tables.items():
            if df is not None:
                df.to_csv(f"{prefix}_{name}.csv")

    @log_time("Differential Expression - exporting table")
    def export(self) -> Path:
        """Export Summary + auxiliary sheets as xlsx (or csv)."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        ad = self.adata
        preproc = ad.uns.get("preprocessing", {})
        analysis_type = str(preproc.get("analysis_type", "DIA")).lower()

        # Core statistics
        log2fc = self._get_dataframe("log2fc")
        q_ebayes = self._get_dataframe("q_ebayes")
        p_ebayes = self._get_dataframe("p_ebayes")
        q_raw = self._get_dataframe("q_raw")
        p_raw = self._get_dataframe("p_raw")

        # Metadata (rename PRECURSORS_EXP → NUM_PRECURSORS; add NUM_UNIQUE_PEPTIDES placeholder if proteomics)
        meta_cols = [c for c in ["GENE_NAMES", "FASTA_HEADERS", "PROTEIN_DESCRIPTIONS", "PRECURSORS_EXP", "PARENT_PROTEIN"] if c in ad.var.columns]
        meta_df = ad.var[meta_cols].copy() if meta_cols else pd.DataFrame(index=ad.var.index)

        if "PRECURSORS_EXP" in meta_df.columns:
            meta_df = meta_df.rename(columns={"PRECURSORS_EXP": "NUM_PRECURSORS", "INDEX": "PHOSPHOSITE", "PARENT_PROTEIN": "PARENT_PROTEIN_UNIPROT_AC"})

        if analysis_type != "phospho":
            # Placeholder column
            if "NUM_UNIQUE_PEPTIDES" not in meta_df.columns:
                meta_df["NUM_UNIQUE_PEPTIDES"] = np.nan

        # Intensities (samples × features) -> (features × samples)
        raw = pd.DataFrame(ad.layers["raw"], index=ad.obs_names, columns=ad.var_names).T if "raw" in ad.layers else None
        X = pd.DataFrame(ad.X, index=ad.obs_names, columns=ad.var_names).T

        # Identification-level optional
        id_qval = pd.DataFrame(ad.layers.get("qvalue"), index=ad.obs_names, columns=ad.var_names).T if "qvalue" in ad.layers else None
        id_pep  = pd.DataFrame(ad.layers.get("pep"),    index=ad.obs_names, columns=ad.var_names).T if "pep"    in ad.layers else None
        spectral_counts = pd.DataFrame(ad.layers.get("spectral_counts"), index=ad.obs_names, columns=ad.var_names).T if "spectral_counts" in ad.layers else None

        # Missingness per group
        miss_ratios = ad.uns.get("missingness", None)
        miss_renamed = None
        max_missing_col = None
        if miss_ratios is not None:
            miss_renamed = miss_ratios.add_prefix("Missingness_")
            max_missing_col = miss_ratios.max(axis=1).rename("MAX_MISSINGNESS")

        # Build base Summary
        if log2fc is None:
            raise AssertionError("Missing one of required varm matrices: log2fc")

        log2fc_pref = log2fc.add_prefix("log2FC_")

        pval_pref = qval_pref = None
        if q_ebayes is not None and p_ebayes is not None:
            qval_pref = q_ebayes.add_prefix("QVALUE_")
            pval_pref = p_ebayes.add_prefix("PVALUE_")

        # Processed & Raw intensities integrated
        log2_int_cols = X.add_prefix("processed_log2_") if X is not None else None
        raw_int_cols  = raw.add_prefix("Raw_") if raw is not None else None

        blocks = [meta_df, log2fc_pref]

        if qval_pref is not None: blocks.append(qval_pref)
        if pval_pref is not None: blocks.append(pval_pref)

        if miss_renamed is not None:
            blocks.append(miss_renamed)
            if max_missing_col is not None:
                blocks.append(max_missing_col)

        if log2_int_cols is not None:
            blocks.append(log2_int_cols)
        if raw_int_cols is not None:
            blocks.append(raw_int_cols)

        # Phospho-specific: add FT_* (if flowthrough exists) & LocScores per sample; Covariate part columns
        # Detect FT by config (dataset.inject_runs.flowthrough) OR presence of covariate layers
        has_ft_cfg = False

        # Safe guard: treat missing as False
        try:
            inj = (self.config or {}).get("dataset", {}).get("inject_runs", {})
            has_ft_cfg = any(k.lower() == "flowthrough" for k in inj.keys()) if isinstance(inj, dict) else False
        except Exception:
            has_ft_cfg = False

        has_cov_layers = ("processed_covariate" in ad.layers) or ("raw_covariate" in ad.layers)
        is_phospho = analysis_type == "phospho"

        if is_phospho:
            # Add covariate-part if present
            if "cov_part" in ad.varm:
                cov_part = pd.DataFrame(ad.varm["cov_part"], index=ad.var.index, columns=self.contrasts).add_prefix("COVARIATE_PART_")
                blocks.append(cov_part)

            # Loc scores per sample (no max)
            if "locprob" in ad.layers:
                loc = pd.DataFrame(ad.layers["locprob"], index=ad.obs_names, columns=ad.var_names).T
                loc_pref = loc.add_prefix("LOCSCORE_")
                blocks.append(loc_pref)

            # FT stats & intensities
            if has_ft_cfg or has_cov_layers:
                if "ft_log2fc" in ad.varm:
                    ft_fc = pd.DataFrame(ad.varm["ft_log2fc"], index=ad.var.index, columns=self.contrasts).add_prefix("FT_log2FC_")
                    blocks.append(ft_fc)
                if "ft_q_ebayes" in ad.varm:
                    ft_q = pd.DataFrame(ad.varm["ft_q_ebayes"], index=ad.var.index, columns=self.contrasts).add_prefix("FT_QVALUE_")
                    blocks.append(ft_q)
                if "ft_p_ebayes" in ad.varm:
                    ft_p = pd.DataFrame(ad.varm["ft_p_ebayes"], index=ad.var.index, columns=self.contrasts).add_prefix("FT_PVALUE_")
                    blocks.append(ft_p)
                if "processed_covariate" in ad.layers:
                    ft_proc = pd.DataFrame(ad.layers["processed_covariate"], index=ad.obs_names, columns=ad.var_names).T.add_prefix("FT_processed_log2_")
                    blocks.append(ft_proc)
                if "raw_covariate" in ad.layers:
                    ft_raw = pd.DataFrame(ad.layers["raw_covariate"], index=ad.obs_names, columns=ad.var_names).T.add_prefix("FT_Raw_")
                    blocks.append(ft_raw)

        idx_name = "PHOSPHOSITE" if is_phospho else "UNIPROT_AC"
        summary_df = pd.concat([b for b in blocks if b is not None], axis=1)

        # Ensure phospho gets PARENT_PEPTIDE_ID column and preferred metadata order
        if is_phospho:
            if "PARENT_PEPTIDE_ID" in ad.var.columns and "PARENT_PEPTIDE_ID" not in summary_df.columns:
                summary_df.insert(0, "PARENT_PEPTIDE_ID", ad.var["PARENT_PEPTIDE_ID"])
            order = [c for c in ["PARENT_PEPTIDE_ID", "PARENT_PROTEIN", "GENE_NAMES", "FASTA_HEADERS", "PROTEIN_DESCRIPTIONS"] if c in summary_df.columns]
            other = [c for c in summary_df.columns if c not in order]
            summary_df = summary_df[order + other]

        # Index header A1
        summary_df.index.name = "PHOSPHOSITE" if is_phospho else "UNIPROT_AC"

        # Compute NUM_UNIQUE_PEPTIDES for proteomics by var-position
        if analysis_type != "phospho":
            pep_uns = ad.uns.get("peptides")
            assert pep_uns is not None, "uns['peptides'] is required to compute NUM_UNIQUE_PEPTIDES"
            pep_meta = pd.DataFrame({
                "PEPTIDE_ID": pep_uns["rows"],
                "PROTEIN_INDEX": pep_uns["protein_index"],
                "PEPTIDE_SEQ": pep_uns["peptide_seq"],
            }, index=pep_uns["rows"])
            counts = pep_meta.groupby("PROTEIN_INDEX")["PEPTIDE_SEQ"].nunique()

            # Map counts by label to var index
            mapped = counts.reindex(ad.var.index).astype("Int64")
            if "NUM_UNIQUE_PEPTIDES" in summary_df.columns:
                summary_df["NUM_UNIQUE_PEPTIDES"] = mapped
            else:
                summary_df.insert(0, "NUM_UNIQUE_PEPTIDES", mapped)

        # Peptide tables
        pep_wide_df, pep_centered_df = self._peptide_frames()

        # proper indexing
        for df_ in (id_qval, id_pep, spectral_counts):
            if df_ is not None:
                df_.index.name = idx_name

        # Prepare README reflecting new layout
        readme = (
            "ProteoFlux Differential Expression Export\n\n"
            "Sheet Descriptions:\n"
            "- Summary: metadata, Log2FC, intensities, Q/P-values, missingness.\n"
            "- Identification Qvalue: PSM-level q-values (from search engine), if available.\n"
            "- Identification PEP: Posterior error probability (from search engine), if available.\n"
            "- Spectral Counts: mean run-evidence counts per (protein × sample), if available.\n"
            "- Peptides (raw): wide peptide-by-sample matrix.\n"
        )

        # Tables to export
        tables = {
            "Summary": summary_df,
            "Identification Qvalue": id_qval if id_qval is not None else None,
            "Identification PEP": id_pep if id_pep is not None else None,
            "Spectral Counts": spectral_counts if spectral_counts is not None else None,
            "Peptides (raw)": (None if (str(preproc.get("analysis_type", "DIA")).lower()=="phospho") else pep_wide_df),
        }

        if self.use_xlsx:
            self._export_excel(tables, readme)
        else:
            self._export_csvs(tables)

        return self.output_path.with_suffix(".xlsx" if self.use_xlsx else ".csv")

    @log_time("Exporting .h5ad")
    def export_adata(self, h5ad_path: str) -> None:
        """Write a compact .h5ad with categorical metadata and summarized preprocessing config."""
        for col in ["CONDITION", "GENE_NAMES", "PROTEIN_WEIGHT", "PROTEIN_DESCRIPTIONS",
                    "FASTA_HEADERS", "ASSAY", "PARENT_PEPTIDE_ID", "PARENT_PROTEIN",
                    "CONDITION_ORIG"]:
            if col in self.adata.obs.columns:
                self.adata.obs[col] = self.adata.obs[col].astype("category")
            if col in self.adata.var.columns:
                self.adata.var[col] = self.adata.var[col].astype("category")

        meta = self.adata.uns.get("proteoflux", {})
        if not isinstance(meta, dict):
            meta = {}
        try:
            pf_version = _pkg_version("proteoflux")
        except PackageNotFoundError:
            pf_version = "0+unknown"
        meta.setdefault("pf_version", pf_version)
        meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds") + "Z")
        self.adata.uns["proteoflux"] = meta

        def _strip_values(d):
            if isinstance(d, dict) and "values" in d:
                try:
                    v = d["values"]
                    d["values_count"] = int(np.sum([bool(x) for x in v if x is not None]))
                except Exception:
                    d["values_count"] = None
                d.pop("values", None)

        pre = self.adata.uns.get("preprocessing", {})
        flt = (pre or {}).get("filtering", {})

        for key in ("cont", "qvalue", "pep", "prec", "meta_cont", "meta_qvalue", "meta_pep", "meta_prec"):
            sub = flt.get(key)
            _strip_values(sub)

        self.adata.write(h5ad_path, compression="gzip")

