import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from proteoflux.utils.utils import logger, log_time

class DEExporter:
    def __init__(
        self,
        adata,
        output_path,
        use_xlsx=True,
        sig_threshold=0.05,
        uniprot_mapper=None,
    ):
        self.adata = adata
        self.output_path = Path(output_path)
        self.use_xlsx = use_xlsx
        self.sig_threshold = sig_threshold
        self.uniprot_mapper = uniprot_mapper
        self.contrasts = self.adata.uns.get("contrast_names", [])

    def _annotate_matrix(self, base_df, q_ebayes=None, miss_ratios=None):
        """
        Annotates base_df with:
        *  → significant (q < threshold)
        †  → fully imputed in one group
        ~  → partially imputed in one group
        """
        annotated = base_df.astype(str)

        # Extract contrast → (group1, group2)
        contrast_map = self.adata.uns.get("contrast_names", [])
        if isinstance(contrast_map, list):
            contrast_map = {c: tuple(c.split("_vs_")) for c in contrast_map}

        for prot in base_df.index:
            for contrast in base_df.columns:
                s = ""
                g1, g2 = contrast_map.get(contrast, (None, None))

                if miss_ratios is not None and g1 in miss_ratios.columns and g2 in miss_ratios.columns:
                    r1 = miss_ratios.loc[prot, g1]
                    r2 = miss_ratios.loc[prot, g2]
                    if r1 == 1.0 or r2 == 1.0:
                        s += "†"
                    elif 0.0 < r1 < 1.0 or 0.0 < r2 < 1.0:
                        s += "~"

                if q_ebayes is not None and contrast in q_ebayes.columns:
                    if q_ebayes.loc[prot, contrast] < self.sig_threshold:
                        s += "*"

                annotated.loc[prot, contrast] += s
        return annotated

    def _get_dataframe(self, matrix_name):
        if matrix_name in self.adata.varm:
            return pd.DataFrame(self.adata.varm[matrix_name],
                                index=self.adata.var.index,
                                columns=self.contrasts)
        return None

    def _export_excel(self, tables: dict):
        description = (
            "ProteoFlux Differential Expression Export\n"
            "\n"
            "Sheet Descriptions:\n"
            "- Summary sheet: Log2 fold changes across contrasts, q and p values associated and raw Intensities\n"
            "- Log2FC: Annotated log2 fold changes across contrasts\n"
            "- Quantification Qvalue: Corresponding q-values (eBayes)\n"
            "- T-statistics (raw): Classical t-statistics (pre-eBayes)\n"
            "- P-values (raw): Classical unmoderated p-values\n"
            "- Q-values (raw): Classical unmoderated q-values\n"
            "- Missingness: Ratio of missing values per group\n"
            "- Metadata: FASTA headers, gene names, UniProt info\n"
            "- Raw Intensities: Untransformed signal matrix\n"
            "- Processed Log2 Intensities: Normalized, imputed data\n"
            "- Identification Qvalue: PSM-level q-values (from search engine)\n"
            "- Identification PEP: Posterior error probability (optional)\n"
            "\n"
            "Annotations:\n"
            f"- * = Quantification Q value < {self.sig_threshold}\n"
            "- ~ = Partially missing in one group (partial imputation)\n"
            "- † = Fully missing in one group (full imputation)\n"
        )

        out_file = self.output_path.with_suffix(".xlsx")
        with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
            pd.DataFrame({"README": description.split("\n")}).to_excel(writer, index=False, sheet_name="README")
            for name, df in tables.items():
                if df is not None:
                    df.to_excel(writer, sheet_name=name)

    def _export_csvs(self, tables: dict):
        prefix = self.output_path.with_suffix("")
        for name, df in tables.items():
            if df is not None:
                df.to_csv(f"{prefix}_{name}.csv")

    @log_time("Differential Expression - exporting table")
    def export(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Gather dataframes
        log2fc = self._get_dataframe("log2fc")
        q_ebayes = self._get_dataframe("q_ebayes")
        p_ebayes = self._get_dataframe("p_ebayes")

        miss_ratios = self.adata.uns.get("missingness", None)

        # Annotate log2fc and q_ebayes
        log2fc_annot = self._annotate_matrix(log2fc, q_ebayes, miss_ratios) if log2fc is not None else None
        q_ebayes_annot = self._annotate_matrix(q_ebayes, q_ebayes, miss_ratios) if q_ebayes is not None else None

        # Metadata sheet
        meta_cols = ["GENE_NAMES", "FASTA_HEADERS", "PROTEIN_DESCRIPTIONS"]
        meta_df = self.adata.var[meta_cols].copy() if all(c in self.adata.var.columns for c in meta_cols) else None

        # Raw & X intensities
        raw = pd.DataFrame(self.adata.layers["raw"], index=self.adata.obs_names, columns=self.adata.var_names)
        X = pd.DataFrame(self.adata.X, index=self.adata.obs_names, columns=self.adata.var_names)

        qval = pd.DataFrame(self.adata.layers.get("qvalue"), index=self.adata.obs_names, columns=self.adata.var_names)
        pep = None
        if "pep" in self.adata.layers:
            pep = pd.DataFrame(self.adata.layers["pep"], index=self.adata.obs_names, columns=self.adata.var_names)

        t_raw = self._get_dataframe("t")
        p_raw = self._get_dataframe("p")
        q_raw = self._get_dataframe("q")

        # Summary sheet
        summary_df = None
        #log2fc_renamed = log2fc.copy().add_prefix("log2FC_")
        log2fc_renamed = log2fc_annot.copy().add_prefix("log2FC_")
        qval_renamed = q_ebayes.copy().add_prefix("Q_")
        pval_renamed = p_ebayes.copy().add_prefix("P_")
        raw_renamed = raw.T.copy().add_prefix("Raw_")

        summary_df = pd.concat([meta_df,
                                log2fc_renamed,
                                qval_renamed,
                                pval_renamed,
                                raw_renamed], axis=1)

        # Tables to export
        tables = {
            "Summary": summary_df,
            "Log2FC": log2fc_annot,
            "Quantification Qvalue": q_ebayes_annot,
            "Metadata": meta_df,
            "Processed Log2 Intensities": X.T,
            "Missingness": miss_ratios,
            "Identification Qvalue": qval.T,
            "Identification PEP": pep.T if pep is not None else None,
            "Raw Intensities": raw.T,
            "T-statistics (raw)": t_raw,
            "P-values (raw)": p_raw,
            "Q-values (raw)": q_raw,
        }

        if self.use_xlsx:
            self._export_excel(tables)
        else:
            self._export_csvs(tables)

        return self.output_path.with_suffix(".xlsx" if self.use_xlsx else ".csv")

    @log_time("Exporting .h5ad")
    def export_adata(self, h5ad_path: str):
        for col in ["CONDITION", "GENE_NAMES", "PROTEIN_WEIGHT", "PROTEIN_DESCRIPTIONS",
                    "FASTA_HEADERS"]:
            if col in self.adata.obs.columns:
                self.adata.obs[col] = self.adata.obs[col].astype("category")
            if col in self.adata.var.columns:
                self.adata.var[col] = self.adata.var[col].astype("category")

        self.adata.write(h5ad_path, compression="gzip")
