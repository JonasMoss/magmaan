# Benchmark Zoo Source Shelf

Use this directory for papers, source notes, and license reviews that justify
benchmark cases. Do not use it as a raw-data cache. Downloaded datasets belong
under the ignored `benchmarks/data/` cache unless redistribution is explicitly
cleared.

Suggested subfolders:

- `papers/` for local PDFs you manually download. This folder ignores paper
  files by default; do not commit copyrighted PDFs.
- `licenses/` for copied or summarized source terms.
- `notes/` for source audits and model-translation notes.

Initial paper/source checklist:

- lavaan built-in datasets: Holzinger-Swineford, Political Democracy,
  `Demo.growth`.
- psychTools: `bfi` and related psychometric item datasets.
- Rogers 2024 CFA tutorial materials and WHOQOL-BREF data.
- Mackinnon, Curtis, and O'Connor longitudinal MI / CLPM tutorial materials.
- OpenMx RAM examples, especially MIMIC and mixed continuous/ordinal CFA.
- Stata Press SEM Reference Manual datasets.
- Mplus User's Guide Chapter 5 examples and Monte Carlo input files.
- semTools simulated datasets for latent interactions, categorical invariance,
  repeated indicators, and parcels.
