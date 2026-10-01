# Study structure

This is one experiment leaf. `run_experiment.R --lane NAME` dispatches to
self-contained method lanes; rendering reads only this study's `results/`.
Lane-local `results` symlinks point to the corresponding directory under that
root. Keep historical outputs, seeds, metadata and method definitions separate.
The parent `report.qmd` is the reader-facing report. Source modules stay in
`lanes/`; shared SEM methods belong in the library, never in another study.

Do not reinterpret frozen outputs as a new experiment, pool different designs,
or overwrite them with verification runs. Use a fresh run name or output path.

This study is banked. Reopen only for a concrete finite-sample GOF consumer,
with an independently validated nuisance-influence/moment reference. The two
lanes have different ranks and nonnormal DGPs; their cells must not be pooled.
