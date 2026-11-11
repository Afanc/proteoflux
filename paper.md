---
title: 'ProteoFlux: a modality-aware framework for downstream analysis of quantitative proteomics data'
tags:
  - Python
  - Bioinformatics
  - Proteomics
  - Mass spectrometry
  - Differential Expression
authors:
  - name: Dariush Mollet
    affiliation: 1
    orcid: 0009-0003-9911-025X
  - name: Alexander Schmidt
    affiliation: 1
    orcid: 0000-0002-3149-2381
affiliations:
  - name: Biozentrum, University of Basel, Basel, Switzerland
    index: 1
date: 9 November 2026
bibliography: paper.bib
---

# Summary

Mass spectrometry can measure thousands of proteins and peptides across samples, but deciding which measurements are reliable, how to handle missing values, and which groups to compare remains a separate analytical task. ProteoFlux is an open-source Python pipeline for that downstream task. It accepts quantitative tables from upstream identification and quantification software and provides configured workflows for protein- and peptide-centric proteomics, phosphoproteomics, and peptide-centric local stability assay (PELSA) dose-response experiments. It produces an annotated data object containing measurements, statistical results, quality-control summaries, and selected provenance, with optional spreadsheet and PDF reports. ProteoViewer is a separate companion application for interactive examination of these outputs.

![Figure 1: Overview of the ProteoFlux workflow. Quantitative outputs from upstream identification tools are harmonized via configuration into a canonical representation, followed by preprocessing, statistical modeling, and standardized exports for downstream visualization and reporting.](assets/proteoflux_overview.png)

# Statement of need

Upstream tools for peptide identification and quantification have matured, yet downstream analysis remains modality-specific and tightly coupled to individual tools or experiment types. Preprocessing, statistical modeling, and visualization are frequently intertwined, making analyses hard to inspect, modify, or reproduce consistently. Re-analysis of existing datasets, as well as analysis of new experiments following similar designs, often requires re-implementing custom scripts or adapting workflows originally developed for a different context. This variability complicates comparison across experiments and increases the risk of unintentional analytical bias.

These challenges are particularly acute for proteomics core facilities and advanced users, who routinely process large numbers of related experiments under comparable designs. Maintaining consistent preprocessing and statistical assumptions across analyses is critical to limit analytical variance and avoid selective parameter tuning. This requires software that makes preprocessing and modeling decisions explicit, produces stable and inspectable outputs, and clearly separates analysis from visualization and reporting.

ProteoFlux was developed to address these needs by providing a unified downstream framework with explicit configuration, deterministic input handling, early error detection, and transparent reporting at each stage of the pipeline.


# State of the field

Established tools support both individual analysis steps and complete workflows. MSstats and MSstatsPTM provide models for protein abundance and post-translational modifications, including adjustment for changes in parent-protein abundance [@Kohler2023; @KohlerPTM2023]. DEP2 and protti combine preprocessing and statistical analysis in R [@Feng2023; @Quast2022], while FragPipe-Analyst connects analysis and interactive visualization for FragPipe outputs [@Hsiao2024].

ProteoFlux was designed to provide a single modality-aware downstream framework that standardizes harmonization, preprocessing, statistical modeling, and output generation across experiment types. Rather than replacing established statistical methodology, it integrates established approaches (e.g., limma-based modeling and directLFQ [@Ammar2023; @Ritchie2015]) within an explicitly configured Python pipeline and exports standardized analysis objects intended for downstream consumers, including ProteoViewer.


# Software design

ProteoFlux separates input harmonization, preprocessing, statistical analysis, and export. Conditions, replicates, feature identities, and quantitative scales are declared through configuration rather than inferred from signal magnitudes. Invalid scales and confounded model designs raise errors rather than yielding fitted results. If any selected condition has only one sample, the selected analysis enters pilot mode and omits significance tests.

## Input processing and quantities

The pipeline reads long or wide CSV/TSV tables and Parquet tables. Packaged mappings cover common exports from Spectronaut, DIA-NN, and FragPipe [@Bruderer2015; @Demichev2020; @Yu2020]; users must check the mappings against their actual export columns. Configured q-value and posterior-error-probability filters act on input rows, when those columns exist, before feature aggregation. Protein and peptide quantities are formed by explicit rollup rules before imputation. Summation is the runtime default; protein-level alternatives include directLFQ [@Ammar2023] and robust regression.

Packaged templates generally apply log2 transformation and median equalization [@Callister2006]; other normalization methods require explicit selection. Normalization precedes missing-value treatment so imputed low signals do not influence sample-median shifts. Inference requires declared log2 quantities, rather than a scale guessed from signal magnitudes.

Peptide-centric analyses retain peptide identities. Phosphoproteomics constructs site features from declared protein locations and localization evidence; the default multisite policy assigns a measured peptide intensity to each declared site, so these site measurements are not independent. Localization thresholds can remove entire site features. Optional protein-background data may be supplied separately or recovered from unmodified peptides in a suitable main input. These choices change the tested feature set and are recorded with the output.

## Missing values and statistical analysis

With LC-ConMed selected, its default `mnar_only` mode leaves partial gaps missing in the differential matrix. It draws low-intensity replacements only when a feature has no observation in one condition but is observed in another; this assumes left-censored signal rather than measured abundance. Other imputation methods and LC-ConMed `standard` mode are optional. Exploratory embeddings and clustering use a separate, condition-independent fill of a normalized-layer copy; this does not alter the differential matrix.

For protein, peptide, and phosphosite contrasts, ProteoFlux fits condition coefficients with the InMoose implementation of limma [@Ritchie2015; @Colange2025]. A maintained InMoose fork supplies corrections for fits with missing observations and for empirical Bayes moderation when features have unequal residual degrees of freedom. Each feature is fitted using its nonmissing values, with feature-specific residual degrees of freedom where estimable; empirical Bayes moderation follows. Categorical batch factors can enter the design, but confounded designs are rejected. The default fit pools residual information across the experiment; an option fits each contrast using only its two conditions. Benjamini–Hochberg correction is applied separately across finite feature p-values for each contrast. For a contrast with one wholly unobserved condition, the low-intensity replacement yields a finite, censoring-based fold-change estimate.

## Specialized analyses and outputs

For phosphoproteomics, optional protein-level adjustment subtracts centered log2 parent-protein abundance with a fixed slope of one before fitting the residuals [@Wu2011]. This assumes that a one-unit change in parent-protein abundance contributes one unit to the measured site signal. If either condition has no observed protein background, the contrast uses an unadjusted fit. The workflow can also estimate kinase activity from configured kinase–substrate relationships using KSEA [@Casado2013] or from sequence-based assignments using PhosX [@Lussana2024]. PELSA follows a separate path: peptide ratios to mean linear-scale control abundance are fitted with bounded four-parameter logistic curves [@Li2025]. Its curve significance is model-based and should not be interpreted as independent proof of target engagement.

The required H5AD output uses AnnData [@Virshup2024] to store samples, features, processed and selected pre-imputation quantities, differential or curve results, embeddings, quality-control summaries, and analysis metadata. Optional XLSX and PDF outputs support inspection and reporting. Stored metadata do not replace the input files, complete configuration, or dependency versions. The project archive is available through its stable concept DOI [@ProteoFluxArchive].

# Research impact statement

ProteoFlux is used in the Biozentrum Proteomics Core Facility's downstream analyses. Three published or reviewed reports document concrete research use: an eLife reviewed preprint on expansion-microscopy-guided microdissection reports analysis with v1.8.3 [@Franziscus2026]; a study of soft-nanocarrier protein coronas reports v1.8.6 for normalization and differential testing [@Nikoletic2026]; and a chromatin-tracing study in *Nature Structural & Molecular Biology* reports v1.8.0 for a proteomic comparison [@PulidoBarrera2026]. The repository also provides small public examples for protein, phosphosite, and PELSA analyses; these are workflow demonstrations, not calibration benchmarks.

# AI usage disclosure

OpenAI ChatGPT (including GPT-5-family models; individual versions were not consistently recorded) and Codex (GPT-6 for the November 2026 revision) assisted with code review, test scaffolding, documentation, and manuscript review and wording suggestions. The authors made the conceptual, statistical, and architectural decisions and reviewed, edited, and validated all AI-assisted contributions to the software, documentation, and paper.

# Acknowledgements

We thank members of the Biozentrum Proteomics Core Facility for feedback and routine testing of the software.

# References
