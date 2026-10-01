# Consolidated study

This is one experiment leaf with one report and parent runner. Lanes retain
separate populations, estimator-level targets, seed schedules, failures and
metadata. The results symlinks stay inside this leaf. Do not pool different
runs or reinterpret historical smoke output as substantive calibration.

Bootstrap edits change source fingerprints. Keep resume guards strict and
use fresh absolute output directories for verification. Reports read only
this study's results; rendering never fits, simulates or downloads inputs.
Core/bindings own shared SEM primitives; lane modules are study-specific.

The compact `results/overview/*.csv` evidence is a deliberate tracked exception:
it makes the report portable without large ignored raw/checkpoint trees.
`scripts/summarize.R` reads original results and records source hashes. These
are retained summaries, not a new run or an inference-policy decision.

Active: audit sparse-summary and endpoint failures, interval connectedness and independent pseudo-target precision before choosing a correction.
