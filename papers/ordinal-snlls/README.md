# ordinal-snlls

Target journal: Structural Equation Modeling or Psychometrika note.

Short memory jog: separable nonlinear least squares for ordinal SEM is not a
drop-in reuse of the continuous SNLLS paper. The ordinal least-squares target
contains thresholds and latent-response correlations, and the full WLS weight
couples those two blocks. Free thresholds can often be profiled, but profiling
changes the effective correlation weight. That is the note.

The magmaan connection is future-facing: `magmaan` already has ordinal
partable rows, ordinal sample statistics, and DWLS/WLS estimation, while the
continuous SNLLS path expects covariance/mean moments. This paper sketches the
adapter layer before any C++ implementation is attempted.

## Layout

- `ordinal-snlls.tex` - main theory sketch.
- `ordinal-snlls.bib` - paper-local bibliography.
- `Makefile` - LaTeX build helper.

## Current Scope

This is intentionally theory-only for now. No simulations, no benchmarks, and
no `magmaan` source changes belong in this first draft. Later work can add:

- a small worked ordinal CFA example;
- benchmarks separating ordinal-stat construction from fitting;
- ULS, DWLS, profiled-WLS, and full-WLS comparisons;
- tests for fixed, labeled, and multi-group threshold constraints.

## Build

```sh
make pdf
```
