"""Small authored experiments, plus a reduction of an already tracked example.

No private/local input or generated result is copied into the test suite.
All files produced by these builders belong in pytest's temporary directory.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES = ["ACDEFGHIK", "LMNPQRSTK", "AGHSTVLER", "VLGANQIPR",
             "MELGSTNAR", "ATVDEQFGR", "GLQNSTVAK", "AEFQGHLIR",
             "NSTVQELAR", "ASGHLTVEK", "VELNQGATR", "ATQEFGSLK",
             "QNSTALVEK", "DEFGHILMR", "LTQNASVEK", "GHLQSTVER"]


def base_config(directory, analysis_type="proteomics"):
    directory = Path(directory)
    return {
        "dataset": {"input_file": str(directory / "input.tsv"), "input_layout": "long",
                    "analysis_type": analysis_type, "index_column": "protein",
                    "uniprot_column": "accession", "signal_column": "intensity",
                    "qvalue_column": "qvalue", "pep_column": "pep",
                    "filename_column": "run", "condition_column": "condition",
                    "replicate_column": "replicate", "peptide_seq_column": "sequence",
                    "charge_column": "charge", "gene_names_column": "gene",
                    "fasta_column": "fasta", "precursors_exp_column": "n_precursors"},
        "preprocessing": {"protein_rollup_method": "sum", "peptide_rollup_method": "sum",
                          "directlfq_cores": 1, "robust_regression_cores": 1,
                          "filtering": {"qvalue": .01, "pep": .05, "min_linear_intensity": 2},
                          "normalization": {"method": ["log2", "median_equalization"]},
                          "imputation": {"method": "lc_conmed", "random_state": 42}},
        "analysis": {"sign_threshold": .05, "clustering_max": 12},
        "exports": {"path_h5ad": str(directory / "result.h5ad"),
                    "path_table": str(directory / "result.xlsx"), "path_pdf_report": None},
    }


def protein_rows():
    rows = []
    noise = np.random.default_rng(814).normal(size=(14, 3, 3))
    for g in range(14):
        for c, condition in enumerate("ABC"):
            for r in range(3):
                for peptide in range(2):
                    # Unequal residual variances and treatment effects across proteins.
                    value = 2 ** (8 + .3*g + (c * ((g % 5)-2) * .6)
                                  + noise[g, c, r]*(.025*1.5**g))
                    value *= (1, 2)[peptide]
                    missing = (g == 0 and c == 0 and r == 2) or (g == 1 and c == 0)
                    missing |= g == 2 and c < 2
                    if missing:
                        value = np.nan
                    if g == 3 and c == 0 and r == 0:
                        value = 0
                    rows.append({"protein": f"P{g:03}", "accession": f"P{g:03}",
                                 "sequence": SEQUENCES[g] + ("" if peptide == 0 else "A"),
                                 "charge": 2 + peptide, "intensity": value,
                                 "qvalue": .1 if g == 12 or (g == 4 and c == 1 and r == 2) else .001,
                                 "pep": .4 if g == 13 else .002,
                                 "run": f"{condition}_{r+1}.raw", "condition": condition,
                                 "replicate": r+1, "gene": f"GENE{g}",
                                 "fasta": f"sp|P{g:03}|SYNTHETIC", "n_precursors": 2})
    # Deliberately scramble source order: labels, not position, must drive alignment.
    return pd.DataFrame(rows).sample(frac=1, random_state=17).reset_index(drop=True)


def protein_workflow(directory, analysis_type="proteomics"):
    config = base_config(directory, analysis_type)
    protein_rows().to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    return config


def phospho_workflow(directory, reuse=False, kinase=True):
    config = base_config(directory, "phospho")
    config["dataset"].update({"stripped_seq_column": "stripped", "peptide_start_column": "start",
                              "ptm_sites_column": "sites", "ptm_positions_column": "positions",
                              "ptm_probabilities_column": "probabilities",
                              "ptm_proteinlocations_column": "locations"})
    config["preprocessing"]["phospho"] = {"covariate_protein_rollup_method": "sum"}
    config["analysis"]["only_contrasts"] = ["B_vs_A", "C_vs_A"]
    rows, background = [], []
    noise = np.random.default_rng(912).normal(size=(7, 3, 3))
    for g in range(7):
        for c, condition in enumerate("ABC"):
            for r in range(3):
                ft = 2 ** (8 + .3*g + .25*c + noise[g, c, r]*(.015*2**g))
                # All A background missing for one parent, partial missing for another.
                if (g == 0 and c == 0 and r == 1) or (g == 1 and c == 0):
                    ft = np.nan
                base = {"protein": f"P{g:03}", "accession": f"P{g:03}",
                        "qvalue": .001, "pep": .002, "charge": 2,
                        "run": f"{condition}_{r+1}.raw", "condition": condition,
                        "replicate": r+1, "gene": f"GENE{g}", "fasta": f"sp|P{g:03}|SYNTHETIC",
                        "n_precursors": 2, "start": 1}
                bg = dict(base, sequence=SEQUENCES[g], stripped=SEQUENCES[g], intensity=ft,
                          sites="", positions="", probabilities="", locations="")
                if not reuse:
                    bg["run"] = f"background_{condition}_{r+1}.raw"
                background.append(bg)
                for site in (0, 1):
                    # Two distinct sites per parent, with full sequence context.
                    sequence = "A" + SEQUENCES[g][:4] + ("S" if site == 0 else "T") + "PEPK"
                    value = 2 ** (9 + .2*g + .6*site + c*((g%3)-1)*.7
                                  + noise[g, c, r]*(.03*2**g) + .02*((r+g+c) % 3))
                    if g == 2 and site == 0 and c == 0:
                        value = np.nan
                    rows.append(dict(base, sequence=sequence[:6]+"[Phospho (STY)]"+sequence[6:],
                                     stripped=sequence, intensity=value, sites="1", positions="6",
                                     probabilities="0.6" if g == 6 else ("0.8" if r == 0 else "0.99"),
                                     locations=("S6" if site == 0 else "T6")))
    frame = pd.DataFrame(rows + background if reuse else rows).sample(frac=1, random_state=19)
    frame.to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    injection = {"indexing_type": "peptidomics", "is_covariate": True}
    if reuse:
        injection["reuse_main_input_file"] = True
    else:
        path = Path(directory)/"background.tsv"
        pd.DataFrame(background).sample(frac=1, random_state=23).to_csv(path, sep="\t", index=False)
        injection["input_file"] = str(path)
    config["dataset"]["inject_runs"] = {"flowthrough": injection}
    if kinase:
        db = Path(directory)/"substrates.tsv"
        pd.DataFrame([{"kinase": "KIN_A" if g < 3 else "KIN_B", "accession": f"P{g:03}",
                       "site": site} for g in range(6) for site in ("S6", "T6")]).to_csv(db, sep="\t", index=False)
        config["analysis"]["kinase_activity"] = {
            "method": "ksea", "substrate_database": str(db), "min_substrates": 3,
            "database_columns": {"kinase_gene": "kinase", "substrate_uniprot": "accession", "substrate_site": "site"}}
    return config


def pelsa_workflow(directory):
    config = base_config(directory, "pelsa")
    config["preprocessing"]["normalization"]["method"] = ["log2"]
    config["preprocessing"]["imputation"]["method"] = None
    config["analysis"]["clustering_max"] = 16
    config["analysis"]["pelsa"] = {"fit_steps": 300, "fit_n_starts": 5,
                                   "fit_loss": "huber", "fit_huber_delta": .5,
                                   "fit_num_threads": 1, "min_control_points": 2}
    rows, annotation = [], []
    concentrations = [0, .01, .1, 1, 10]
    for j, concentration in enumerate(concentrations):
        for r in range(2):
            run = f"dose{j}_{r+1}"
            annotation.append({"File Name": run, "Condition": f"dose{j}",
                               "Replicate": r+1, "Concentration": concentration})
            for g in range(16):
                ec50 = .04 * 1.35**g
                amplitude = (1 if g%2 else -1) * (1 + .1*g)
                # Independently expressed saturating dose response on the log2 scale.
                response = amplitude * concentration/(ec50 + concentration)
                log2_value = 12 + .1*g + response + (r-.5)*(.02+.003*g)
                value = 2**log2_value
                if g == 14 and j == 0 and r == 1:
                    value = np.nan  # insufficient controls, failure row
                if g == 15 and j == 0:
                    value = np.nan  # no reference, no curve row
                rows.append({"protein": f"P{g//2:03}", "accession": f"P{g//2:03}",
                             "sequence": SEQUENCES[g], "charge": 2, "intensity": value,
                             "qvalue": .001, "pep": .002, "run": run+".raw",
                             "condition": f"dose{j}", "replicate": r+1,
                             "gene": f"GENE{g//2}", "fasta": "synthetic", "n_precursors": 1})
    pd.DataFrame(rows).to_csv(config["dataset"]["input_file"], sep="\t", index=False)
    path=Path(directory)/"annotation.tsv"
    pd.DataFrame(annotation).to_csv(path, sep="\t", index=False)
    config["dataset"]["annotation_file"] = str(path)
    return config


def searle_workflow(directory):
    """Reduce in temporary storage, using only data already distributed in examples."""
    root = ROOT / "examples/searle_small"
    config = yaml.safe_load((root/"config.yaml").read_text())
    columns = list(dict.fromkeys(v for k,v in config["dataset"].items()
                                if k.endswith("_column") and isinstance(v, str)))
    frame = pd.read_csv(root/"searle_test.tsv", sep="\t", usecols=columns)
    counts = frame.groupby("PG.ProteinGroups")["R.FileName"].nunique()
    # Include complete coverage and naturally sparse proteins; no local files involved.
    complete = sorted(counts[counts == counts.max()].index)[:16]
    sparse = sorted(counts[counts < counts.max()].index)[:4]
    frame = frame[frame["PG.ProteinGroups"].isin(complete+sparse)]
    path = Path(directory)/"input.tsv"
    frame.to_csv(path, sep="\t", index=False)
    config["dataset"]["input_file"] = str(path)
    config["dataset"]["annotation_file"] = str(root/"annotation_searle.tsv")
    config["preprocessing"]["directlfq_cores"] = 1
    config["preprocessing"]["imputation"]["random_state"] = 42
    config["analysis"]["clustering_max"] = 18
    config["exports"] = base_config(directory)["exports"]
    return config


WORKFLOWS = ("proteomics", "peptidomics", "pilot", "batch_pairwise", "phospho",
             "phospho_reuse", "phospho_unadjusted", "phosx", "pelsa", "searle_directlfq")


def make_workflow(name, directory):
    if name in {"proteomics", "peptidomics"}:
        return protein_workflow(directory, name)
    if name == "pilot":
        config = protein_workflow(directory)
        path = config["dataset"]["input_file"]
        frame = pd.read_csv(path, sep="\t")
        frame = frame[~((frame.condition == "C") & (frame.replicate > 1))]
        frame.to_csv(path, sep="\t", index=False)
        # Include singleton C in the selected analysis: every contrast is pilot.
        config["analysis"]["only_contrasts"] = ["A_vs_B", "A_vs_C"]
        return config
    if name == "batch_pairwise":
        config = protein_workflow(directory)
        config["analysis"].update(only_contrasts=["B_vs_A", "C_vs_A"], separate_contrasts=True,
                                   batch_effect_columns=["REPLICATE"])
        return config
    if name in {"phospho", "phospho_reuse"}:
        return phospho_workflow(directory, reuse=name.endswith("reuse"))
    if name == "phospho_unadjusted":
        config = phospho_workflow(directory)
        del config["dataset"]["inject_runs"]
        return config
    if name == "phosx":
        config = phospho_workflow(directory, kinase=False)
        # Exercise both independent PSSM/FDR families in the PhosX workflow.
        path = config["dataset"]["input_file"]
        frame = pd.read_csv(path, sep="\t")
        tyrosine = frame.locations == "T6"
        frame.loc[tyrosine, "locations"] = "Y6"
        frame.loc[tyrosine, "sequence"] = frame.loc[tyrosine, "sequence"].str.replace("T[Phospho", "Y[Phospho", regex=False)
        frame.loc[tyrosine, "stripped"] = frame.loc[tyrosine, "stripped"].str[:5]+"Y"+frame.loc[tyrosine, "stripped"].str[6:]
        frame.to_csv(path, sep="\t", index=False)
        config["analysis"]["kinase_activity"] = {"method": "phosx", "n_permutations": 127,
                                                  "min_substrates": 2, "random_seed": 47, "n_processes": 1}
        return config
    if name == "pelsa":
        return pelsa_workflow(directory)
    if name == "searle_directlfq":
        return searle_workflow(directory)
    raise ValueError(name)
