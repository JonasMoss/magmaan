# Study structure

This is one experiment leaf. `run_experiment.R --lane NAME` dispatches to
self-contained method lanes; rendering reads only this study's `results/`.
Lane-local `results` symlinks point to the corresponding directory under that
root. Keep historical outputs, seeds, metadata and method definitions separate.
The parent `report.qmd` is the reader-facing report. Source modules stay in
`lanes/`; shared SEM methods belong in the library, never in another study.

Do not reinterpret frozen outputs as a new experiment, pool different designs,
or overwrite them with verification runs. Use a fresh run name or output path.

The original normal and robust lane AGENTS instructions apply within their
respective lanes. Prespecified designs are in `design/normal.md` and
`design/robust.md`. The robust lane's prohibition of bootstrap Bartlett work
applies to robust inference; the normal lane retains its original Bartlett
pilot. Preserve original source fingerprints so frozen runs remain resumable.
