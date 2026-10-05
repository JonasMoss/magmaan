### Parser, lavaanify, and matrix representation

- EQS model-section frontend (2026-10-01): `parse::EqsParser` and
  `api::model_from_eqs` lower explicit single-group continuous equations,
  variances and covariances into the existing model triple. Fixed/free values,
  starts, error ownership, default-zero covariances and explicit identification
  are preserved; EQS job/estimator settings are not imported. The lab exposes
  `eqs_model()`; EQS remains excluded from the ordinary-user simulation
  prerelease. Ordinary-user integration is deferred until the C++/lab
  language-extension and round-trip gates pass.
  The [EQS contract](../../grammar/eqs.md) defines the subset, unsupported cases
  and evidence limits. Offline pinned-lavaan fixtures gate rows, starts, ML
  estimates, implied covariance, expected SEs, df and chi-square; R tests
  compare installed lavaan. There is no live EQS oracle.
  The [manual source inventory](../../grammar/eqs_source_inventory.md) records
  the full documented SEM model-language target, source pages, model-triple
  adapter requirements and focused runtime checks for unresolved cases.
  The inventory does not promote additional parser or fitting capabilities.
  The [planned implementation sequence](../../grammar/eqs.md#implementation-sequence-planned)
  orders C++ resolution, general-equation representation, shorthand, means/
  groups and restrictions, with lab exposure after each validated increment.
  The [backlog](../../backlog/todo.md#eqs-language-extension) records the later
  `r-magmaan` task separately. This is planning, not additional implementation.
- Mplus input frontend (0.3.0): `parse::MplusParser::read()` reads whole
  input files, classifies commands/options, expands NAMES and selects
  USEVARIABLES, preserves source spans, and aggregates classified rejections.
  Schema settings, execution/data-description notes and explicit family
  boundaries are checked by independent unit expectations and a 142-variant
  Demo input gate (with documented LX02, NM02 and MS11 deviations).
  `parse::MplusParser::parse()` lowers continuous and categorical single- and multiple-group BY/ON/WITH,
  PON/PWITH, variances, means, modifiers, line-local labels and ranges into
  an owned `MplusModel`. It materializes marker, role, mean and covariance
  defaults explicitly, including thresholds, DELTA scales and THETA residuals.
  Polynomial/free-time and shared-intercept piecewise growth materialize BY,
  zero outcome intercepts, growth means and categorical time-invariance rows.
  MODEL CONSTRAINT imports NEW starts/free coordinates/derived quantities,
  explicit/implicit linear and continuous nonlinear equalities, and nested DO
  expansion. Auxiliary `new` rows have no variable IDs or matrix cell but retain
  their free index, label and start through partable round trips. Moment derivative
  consumers skip their missing matrix locations; noniterative CFA and profiled
  SNLLS charts reject auxiliary NEW coordinates explicitly. MODEL INDIRECT
  enumerates simple directed indirect paths, supports specific written orders
  (absent paths give zero) and VIA filters, including paths through factors.
  Per-group defined names follow `ind_g<group>_<outcome>_ind_<names>` and
  `ind_g<group>_<outcome>_via_<mediator>_<predictor>`; unlabelled regression
  coefficients receive `mi_g<group>_<outcome>_<predictor>` labels.
  Equality restrictions reach the existing affine/nonlinear fit machinery;
  all-ordinal nonlinear equalities use constrained LS fitting and expected-
  information lavaan inference (TASK-54.2). Inequalities follow
  the permanent scope boundary and redirect admissibility to PSD/barrier fits.
  Categorical CONFIGURAL/SCALAR shortcuts follow Mplus 9.1; METRIC is
  rejected. Data-driven threshold completion preserves these defaults.
  `compat::mplus::build_options()` disables lavaan
  automatic defaults and retains fixed-x sample moments for conditional sources;
  complete observed-X mentions lower the joint random-X model (P-MS08b).
  Ordinary construction accepts explicit lab Mplus specs, preserves source/schema
  and records fittability; conditional-X, NOMEANSTRUCTURE and summary-without-MEANS
  fits raise one classed refusal listing all input edits in order, with vector
  `reason` and named vector `edit` fields. Printing the ordinary model shows
  fittability and each needed edit. Combined starts and
  labels use duplicate formula rows, as supported by `spec::build()`; the
  gate compares unique parameter keys and checks both modifier components.
  `api::model_from_mplus()` retains the whole source and diagnostics;
  `magmaanlab::mplus_model()` exposes source-based rebuilding, portable specs,
  notes and a lossless lavaan row projection. Observed names retain NAMES
  spelling and factors their first BY spelling; BY factor sets are rejected.
  Live lavaan rows/ML/expected SEs and fresh/prepared/serialized round trips
  gate the lab surface. Ten independent categorical WLSMV goldens gate
  both parameterizations and grouped defaults; the lab fits all-ordinal DWLS
  and reports mixed, conditional and categorical ML boundaries. The local
  corpus gate accepts 50 of 68 cases: 44 matched, six unsupported fit routes,
  and five verified test-convention differences (see the validation ledger).
  The TASK-85 output-only sweep scans 1,933 outputs / 683 distinct
  Mplus inputs; all 130 accepted inputs match groups, free counts
  and df. Independent equality-Jacobian rank and printed category counts
  complete the dimensions. Nine Demo variants and independent row regressions
  gate the repaired ownership of later bracket-group labels and modifiers.
  Free-parameter count and df now also gate the six unsupported routes through
  independent conditional/mixed moment dimensions. The
  [Mplus plan](../../grammar/mplus.md) reads whole input files, lowers the
  linear SEM subset with Mplus's model defaults into the model triple, gates
  meaning against Mplus (manual, Demo TECH1, corpus `.out`) and numerics
  against lavaan, and targets ordinary integration after a stability bar.
  The [source inventory](../../grammar/mplus_source_inventory.md) and the
  continuous/categorical [grammar](../../grammar/mplus_grammar.ebnf) own the rules;
  checked-in Demo probes supply independent behavioral evidence.
  Stability closeout publishes one [coverage matrix](../../grammar/mplus.md#coverage-matrix)
  for all 121 primary inventory IDs and their settled aliases, with rule-specific
  rejection contracts and matching lab help. Standalone ASan/UBSan reader/parser
  and API-lowering drivers gate all 2,440 corpus inputs with ten-second per-input
  deadlines; reader acceptance is 658 and parser/API acceptance 614. All 33 firing
  rule IDs are classified, with zero crashes, sanitizer reports or hangs.
  Seven lab model kinds gate partable projections, fresh/prepared fits,
  original-source rebuild/refit and saveRDS/reload in a fresh R process.
  GROUPING imports ordered integer codes and retains source labels separately.
  Per-group modifier vectors implement defaults, releases, identifier ties,
  cumulative sections and asymmetric rows (fixed zero elsewhere, with generated
  provenance). Group-specific variable-role changes are an explicit boundary.
  CONFIGURAL/METRIC/SCALAR support marker and variance identification; 14
  independent grouped goldens gate TECH1, rows, ML/SEs and implied moments at
  1e-5. Lab specs retain grouping on rebuild/serialization and reject unlisted
  data codes with row counts. Same-manifest input acceptance rises 253→291
  for the reader and 189→217 for MODEL lowering.
  Increment 5 parses a typed data plan and exposes `mplus_data()` for free/fixed
  individual files and summary covariance/correlation/means/SD. FILE groups use
  labels in declaration order and reserved `.mplus_group`; NGROUPS summaries
  use g1, g2, ... . Summary data without MEANS omit mean rows and reject explicit
  means; correlations without SD have unit variances. Missing flags compare
  scaled values, NOBSERVATIONS limits raw records, and unlisted GROUPING codes
  are dropped with counts. LISTWISE/sample-selection rules are reported only.
  Independent frames/matrices, Demo printed moments and two corpus fit round
  trips gate the reader; summary input covariances use N-1 and are converted
  by (N-1)/N to the lab divisor-N convention. Saturated ML agrees with Demo
  and lavaan default covariance rescaling (P-DA5).
  Independent increment-1 meaning/numeric evidence is frozen in
  `cpp/tests/fixtures/mplus/golden.json` by `regen_oracle_mplus.R`: 13
  hand-written paired models fitted to identical complete observations by
  pinned lavaan and the Mplus 9.1 Demo. Exact parameter-count/df checks and
  fixed chi-square/printed-estimate tolerances gate serialization; negative
  lavaan variances and population-fit p-values below 0.001 stop generation.
  The fixture includes explicit rows, starts, expected-information SEs and
  sample/implied moments. The C++ consumer compares complete row sets,
  equality partitions, explicit starts, ML estimates and implied moments,
  expected-information SEs (1e-5), parameter counts, df and chi-square.
  Demo TECH1 gates compare parameter counts, cell status and equality
  partitions; the optional local corpus sweep reports rule-ID tallies.
  Reader refinements exclude TITLE/comment text from the column-90 check,
  classify group/class MODEL sections using all declarations, reject
  duplicate analysis names and bound generated NAMES at 100000.
- Lavaan-style syntax parser with normative grammar in `project/grammar/`,
  including fixed numeric intercept shorthand (`x ~ 0`) and parenthesized
  modifier labels (`(label)*x`), signed numeric modifiers/starts, chained
  modifiers, multi-LHS regressions, tolerated repeated `+` separators, and
  RHS continuations after the operator newline.
- Checked-in lexer, parser, partable, matrix, and fit oracle fixtures.
- Single- and multi-group LISREL matrix representation.
- Reduced LISREL lowers measurement rows whose RHS observed variable is already
  promoted into `lv_ext_order` into the structural `Beta` matrix, while keeping
  the mechanically inserted `Lambda[observed, phantom] = 1` identity. This
  keeps latent-change-score and single-indicator state chains on the same
  state vector as their autoregressions.
- Fixed.x resolution, mean structures, marker/std.lv/effect-coding
  identification, lavaan-style single-indicator residual fixing, start hints,
  and linear equality constraints.
- **Scaling conventions are changes of coordinates, and are held to that —
  wherever the loadings are actually free to move.** The marker and `std_lv`
  parameterizations of the same model then reach the same optimum, hence the same
  df and chi-square; only the raw parameter count moves. Pinned for two-level
  (per-level application, npar unchanged), single-indicator latents (`std_lv` and
  `auto_fix_single` are orthogonal, npar unchanged), second-order and
  endogenous-latent models (fmin to 13+ digits), and multi-group metric
  invariance. **The exception is a model whose loadings are all user-fixed**:
  `std_lv` does not override a user fix, so it pins the latent variances without
  freeing anything in exchange, and the result is a strictly more restricted
  model. `growth(std_lv = TRUE)` is the case that bites — lavaan itself goes npar
  9 → 7, df 5 → 7, χ² 8.07 → 106.85 on `Demo.growth`, silently. magmaan matches
  that, so this is faithful behaviour rather than a bug, but "std_lv is just a
  reparameterization" is false there and the mean structure is never
  standardized by `std_lv` at all.
  The one place the coordinate-change property is not automatic is multi-group
  `group_equal = Loadings`:
  `apply_std_lv` fixes `lv ~~ lv` at 1.0 per group, which with Λ tied across
  groups also forbids group differences in factor variance, making the std.lv
  invariance model strictly more restrictive than its marker twin. Step 8a-bis in
  `spec/build` therefore releases the latent variances in groups 2..G when
  `Loadings ∈ group_equal` and `LvVariances ∉ group_equal`, matching lavaan
  (`lav_partable_flat.R`, upstream `fecaf6b7`, 2019-06-27 — behaviour stable from
  0.6-4 through 0.7-2). Net effect: df unchanged across conventions, raw npar
  higher by (G−1)·n_lv under `std_lv`, the surplus absorbed by the extra
  cross-group loading equalities. Gated both by the `fit_stdlv` multi-group
  golden fixture and self-consistently by "std.lv multi-group metric invariance agrees with marker scaling"
  (`cpp/tests/unit/constraints_test.cpp`), which fits one deliberately misspecified
  two-group model both ways and requires equal df/chi²/fmin — an oracle-free
  check, and the cheapest guard against any future scaling convention silently
  changing the fitted model rather than its coordinates.
  Numerical scaling remains a research question. The one-factor measurements in
  [project/design/parameterization-geometry.md](../../design/parameterization-geometry.md)
  and experiment research/46 show marker sensitivity to a weak indicator; they do not
  establish a globally optimal chart. For endogenous latents, `std_lv` fixes
  disturbance variance, which can make coordinates extreme at high explained
  variance. Experiment _archive/sem-total-variance compares marker, disturbance-unit, and total-variance-unit
  coordinates across six recursive structures using an experiment-local R
  prototype, common starts, analytic derivatives, and the same strictly PD
  component domain. Native fitting behavior is unchanged. Boundary solutions,
  practical starts, feedback and shared/equality-constrained parameters remain
  outside the initial pilot; a fit gap across unequal admissible domains is not
  evidence of optimizer failure. Its Bollen follow-up adds diagonal/frozen full
  information scaling, a local chart selector, and existing native PSD references
  from two starts. It distinguishes interior audit failures from matching
  cone-stationary boundary candidates; no native parameterization policy changed.
- Linear equality constraints through affine reparameterization (θ = θ₀ + K·α)
  for the ML, GMM/GLS, and bounded ordinal LS paths; per-θ box bounds fold onto
  the reduced α for the pure-merge case.
- Nonlinear equality constraints (`a == b*c`, `b1 == (b2+b3)^2`) for the ML
  and complete-data LS paths: compiled to name-free expression trees by
  `resolve_lin_constraints`, enforced by NLopt SLSQP or the optional IPOPT
  interior-point backend, with the constrained vcov / df projected through the
  constraint Jacobian H(θ̂). They may be combined with linear equality
  constraints in the same model — the constrained optimizer then runs in the
  linear-constraint-reduced α-space. FIML has the same NLopt SLSQP / IPOPT
  nonlinear constraint support. All-ordinal ULS/DWLS/WLS bounded paths
  enforce these restrictions, with expected-information covariance, global and
  nested tests, scores and df using the fitted constraint tangent. The ordinary
  DWLS policy and observed/IJ/misspecification profile sensitivity refuse
  nonlinear equalities until Lagrangian curvature is implemented. SNLLS,
  ordinal association ML and pairwise composite paths remain unsupported.
- The expression sub-language shared by `:=` defined parameters and `==`
  constraints supports `+ - * / ^`, unary `+ -`, and the unary functions
  `exp`, `log`, `sqrt`, `pnorm` (Mplus PHI) and `log10`; both the defined-parameter evaluator and the
  nonlinear-constraint evaluator evaluate them with forward-mode AD.
- Effect coding for loadings.
