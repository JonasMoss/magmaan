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
  Schema settings, execution/data-description notes and explicit later-increment
  boundaries are checked by independent unit expectations and a 100-variant
  Demo input gate (with documented LX02, NM02, MS11 and CL11 deviations).
  MODEL bodies are delimited; MODEL parsing/lowering and API/R exposure remain
  pending. The
  [Mplus plan](../../grammar/mplus.md) reads whole input files, lowers the
  linear SEM subset with Mplus's model defaults into the model triple, gates
  meaning against Mplus (manual, Demo TECH1, corpus `.out`) and numerics
  against lavaan, and targets ordinary integration after a stability bar.
  The [source inventory](../../grammar/mplus_source_inventory.md) and the
  increment-1 [grammar](../../grammar/mplus_grammar.ebnf) own the rules;
  checked-in Demo probes supply independent behavioral evidence.
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
  nonlinear constraint support; ordinal and the separable (SNLLS) path reject
  them.
- The expression sub-language shared by `:=` defined parameters and `==`
  constraints supports `+ - * / ^`, unary `+ -`, and the unary functions
  `exp` / `log`; both the defined-parameter evaluator and the
  nonlinear-constraint evaluator evaluate them with forward-mode AD.
- Effect coding for loadings.
