### Testing and validation

Validation has three deliberately separate surfaces:

- **Corpus golden tests** — breadth. 26 small synthetic models in
  `cpp/tests/fixtures/corpus.json` exercise every parser, lavaanify, matrix, fit,
  and inference stage against checked-in lavaan fixtures. Oracle:
  `cpp/tests/tools/regen_oracle.R`.
- **Parity golden tests** — depth. `cpp/tests/golden/lavaan_parity_golden_test.cpp`
  gates magmaan against lavaan on the real-data benchmark cases
  (HolzingerSwineford1939, PoliticalDemocracy, Demo.growth, bfi, Mplus ex5.1):
  real sample sizes, real conditioning, real SEs and fit measures, and the
  raw-data ingestion path. The FIML tranche includes both a quick two-factor
  bfi missing-data sentinel with robust MLR post-fit reporting and a full
  25-item five-factor bfi missing-data convergence/global-fit gate.
  Self-contained fixtures live in
  `cpp/tests/fixtures/parity/`; oracle: `cpp/tests/tools/regen_parity_fixtures.R`.
  The same parity executable also includes the Mplus SEM corpus golden,
  generated from local `external/textbook-corpus/raw/mplus_sem` into
  `cpp/tests/fixtures/mplus_sem/`.
  The textbook WLSMV gate covers 19 supported all-ordinal cases, including
  Mplus categorical growth examples 6.4 and 6.15 with released DELTA scales.
  It compares DELTA associations using live prepared `~*~` multipliers
  (`delta_i * sigma_ij * delta_j`), and THETA correlations using the implied
  diagonal. The additive evaluator's DELTA diagonal is not the derived
  response variance and is excluded. Tolerances and the two known gaps stay
  unchanged.
- **Benchmarks** — `benchmarks/` fits *live lavaan* on every run and gates
  timing, not CI correctness. Active advisory cases include complete-data ML,
  controlled-missingness FIML, and continuous ULS/GLS smoke paths. The harness
  is R-dependent and advisory; the parity layer is the bridge that freezes its
  correctness checks into the gated C++ suite.

The suite builds as eight doctest executables — `magmaan_test_{smoke, spec,
estimate, inference, ordinal, api, parity, robcat}` — so areas build and run
independently (`ctest -L parity`). CI never invokes R; fixture regeneration is
a manual developer step. Property and boundary tests are expected to catch
structural mistakes early, before they surface as hard-to-debug parity
failures.

The ordinary R package tests local prepared-handle reuse and serialization
independently of its PSOCK worker check. The worker check probes socket access
and skips with the socket error when a check sandbox disallows listeners;
local serialization assertions still run in that environment.

The lab's live DELTA response-scale comparisons carry their model declarations
in `r-package/tests/testthat/fixtures/delta_scale_models.json`, so portable
source-package checks run the same seven cases without a checkout of C++ test
fixtures. Estimates, standard errors, global tests, standardized reporting and
factor scores are still compared with installed lavaan.
