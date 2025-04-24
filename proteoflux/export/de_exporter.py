import pandas as pd
import numpy as np
from pathlib import Path

class DEExporter:
    def __init__(
        self,
        adata,
        output_path,
        use_xlsx=True,
        sig_threshold=0.05,
        include_uniprot_map=False,
        uniprot_mapper=None,
    ):
        self.adata = adata
        self.output_path = Path(output_path)
        self.use_xlsx = use_xlsx
        self.sig_threshold = sig_threshold
        self.include_uniprot_map = include_uniprot_map
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

    def _annotate_matrix2(self, base_df, q_ebayes=None, miss_ratios=None):
        annotated = base_df.astype(str)
        for prot in base_df.index:
            for col in base_df.columns:
                s = ""
                if miss_ratios is not None:
                    group_ratios = miss_ratios.get(col, {}).get(prot, {})
                    if any(v == 1.0 for v in group_ratios.values()):
                        print("yay")
                        s += "†"
                    elif any(0.0 < v < 1.0 for v in group_ratios.values()):
                        s += "~"
                if q_ebayes is not None and q_ebayes.loc[prot, col] < self.sig_threshold:
                    s += "*"
                annotated.loc[prot, col] += s
        return annotated

    def _get_dataframe(self, matrix_name):
        if matrix_name in self.adata.varm:
            return pd.DataFrame(self.adata.varm[matrix_name],
                                index=self.adata.var.index,
                                columns=self.contrasts)
        return None

    def _export_excel(self, tables: dict):
        out_file = self.output_path.with_suffix(".xlsx")
        with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
            for name, df in tables.items():
                if df is not None:
                    df.to_excel(writer, sheet_name=name)

    def _export_csvs(self, tables: dict):
        prefix = self.output_path.with_suffix("")
        for name, df in tables.items():
            if df is not None:
                df.to_csv(f"{prefix}_{name}.csv")

    def export(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Gather dataframes
        log2fc = self._get_dataframe("log2fc")
        q_ebayes = self._get_dataframe("q_ebayes")

        miss_ratios = self.adata.uns.get("missingness", None)

        # Annotate log2fc and q_ebayes
        log2fc_annot = self._annotate_matrix(log2fc, q_ebayes, miss_ratios) if log2fc is not None else None
        q_ebayes_annot = self._annotate_matrix(q_ebayes, q_ebayes, miss_ratios) if q_ebayes is not None else None

        # Metadata sheet
        meta_cols = ["FASTA_HEADERS", "GENE_NAMES", "PROTEIN_DESCRIPTIONS"]
        meta_df = self.adata.var[meta_cols].copy() if all(c in self.adata.var.columns for c in meta_cols) else None

        # Uniprot conversion placeholder
        if self.include_uniprot_map and meta_df is not None and self.uniprot_mapper:
            meta_df["UNIPROT_ID"] = meta_df["FASTA_HEADERS"].apply(self.uniprot_mapper)

        # Raw & X intensities
        raw = pd.DataFrame(self.adata.layers["raw"], index=self.adata.obs_names, columns=self.adata.var_names)
        X = pd.DataFrame(self.adata.X, index=self.adata.obs_names, columns=self.adata.var_names)

        qval = pd.DataFrame(self.adata.layers.get("qvalue"), index=self.adata.obs_names, columns=self.adata.var_names)
        pep = None
        if "pep" in self.adata.layers:
            pep = pd.DataFrame(self.adata.layers["pep"], index=self.adata.obs_names, columns=self.adata.var_names)

        # Tables to export
        tables = {
            "Log2FC": log2fc_annot,
            "Quantification Qvalue": q_ebayes_annot,
            "Missingness": miss_ratios,
            "Metadata": meta_df,
            "Raw Intensities": raw.T,
            "Processed Log2 Intensities": X.T,
            "Identification Qvalue": qval.T,
            "Identification PEP": pep.T if pep is not None else None,
        }

        if self.use_xlsx:
            self._export_excel(tables)
        else:
            self._export_csvs(tables)

        return self.output_path.with_suffix(".xlsx" if self.use_xlsx else ".csv")
