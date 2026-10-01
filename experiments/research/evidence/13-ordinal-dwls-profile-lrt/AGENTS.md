# Retained native simulation

This completed study deliberately uses a native C++ simulator and `justfile`
rather than an R entry point. `report.md` is the retained reader-facing report.
Build with this leaf's `just build`; verification invokes `build/ordinal_dwls_profile_lrt_sim`
with a fresh `--out=/tmp/...` path, never the frozen calibration file.
No new simulation is queued. Reopen for a named ordinal nested-test defect or
new target regime. This is an explicit legacy exception to the ordinary live
QMD/R-runner layout, not a second maintained implementation.

The small checked-in `results/calibration.csv` is frozen evidence for the
exact pseudo-null comparison described in the report. Preserve its bytes;
the simulator owns the original population/seed conventions. Nothing imports
another numbered experiment, and no render/build should run calibration.
