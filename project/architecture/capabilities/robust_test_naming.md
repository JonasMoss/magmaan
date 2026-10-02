### Robust-test naming and compatibility

The core robust-test API should use statistical names for the object being
computed, not historical SEM estimator labels. The preferred core surface is:

- weighted chi-square mixture tests from explicit eigenvalues or trace
  summaries;
- moment reductions of that same mixture: mean-scaled, mean/variance-adjusted,
  and scaled/shifted;
- likelihood-ratio / nested-model helpers named for the statistical contract
  they implement, for example exact restriction-map LRT rather than a default
  public name centred on "Satorra-2000";
- robust SEs as named sandwich constructions.

Historical labels remain important for lavaan/Mplus parity, but they belong in
`compat::lavaan` or similarly explicit compatibility wrappers. This includes
estimator shortcuts such as `MLM`, `MLMV`, `MLMVS`, and `MLR`, lavaan test
labels such as `satorra.bentler`, `scaled.shifted`,
`mean.var.adjusted`, `yuan.bentler`, and `yuan.bentler.mplus`, and legacy
nested-test formulas such as Satorra-Bentler 2001/2010. The compatibility
wrappers should bind each label to the exact lavaan/Mplus bundle it denotes;
they should not expose a combinatorial menu that lets callers freely pair a
test label with unrelated bread, gamma, vcov, or scaling choices.

`yuan.bentler.mplus` is especially a compatibility target rather than a core
weighted-mixture reducer. Lavaan's MLR default computes a scalar trace
difference, H1 minus H0, relying on Mplus-style approximations such as
`A0 ~= Delta' A1 Delta` and the analogous first-order matrix. The plain
Yuan-Bentler formula instead constructs the H1 residual-space trace directly.
Both should be documented when exposed, but neither should force historical
names into the core naming scheme.

The R boundary now stages this policy with `robust_nested_lrt()` as the
friendly statistical name for nested robust likelihood-ratio work. Its default
`method = "restriction_map"` names the exact restriction-map contract, while
`"lavaan_sb2001"` and `"lavaan_sb2010"` are explicit compatibility methods.
The restriction-map route also exposes `computation = "streaming"` versus
`"materialized"` so paper-local benchmarks can compare the algebra without
using lavaan as the timing denominator. When both fits are FIML,
`robust_nested_lrt()` dispatches to the restriction-map or scalar
`"lavaan_sb2001"`/`"lavaan_sb2010"` engines and uses the retained FIML
`raw_data` rather than a caller-supplied complete-data argument. Mixed
FIML/complete-data pairs are rejected. These scalar engines use
spectrum-derived single-model scales; their historical names do not establish
the ordinary lavaan MLR default bundle, which still needs convention-matched
scale and full-report gates. The same scalar lab methods exist for NT ML2S;
other Stage-2 weights reject them and retain the restriction-map route.
Satorra-2000 rank checks and solves use symmetric diagonal equilibration of
pooled expected or observed parameter information and the restriction companion
matrix (2026-10-01). The same congruence is applied to both matrices of the
reduced spectral pencil; returned `C` and `S` retain their original coordinates.
This removes measurement-unit effects without regularization or relaxing the
singularity tolerance. The ordinary prepared NTML nested score/LR route has its
own information and restriction-metric checks; these now use the same
normalization, including the casewise whitening consumed by `policy_nested()`
and `anova()`. `satorra2000_test.cpp` gates grouped covariance/mean models,
streaming/materialized/dense spectra, moment/parameter/restriction rescaling and
singular controls. `policy_test.cpp` carries fitted CFA points through exact
unit transformations. `test_nested_units.R` checks both R routes for a two-group
path model with a roughly 300-fold variance contrast and a 100-fold unit change.
The complete-data ML restriction map handles mean structure: when either fit
carries free intercepts / latent means the per-group moment vector is augmented
to `[μ_g; vech(Σ_g)]` (mean rows on top), so `lr_test_satorra2000_from_data`
stacks `dmu_dtheta` over `dsigma_dtheta` for both the per-group `Pi_alpha` and
the delta restriction, and `compute_satorra2000` carries the augmented block
through `SatorraGroup.mu_dim` (μ-block of `V` is `Σ⁻¹`, the casewise meat picks
up the full μ×σ cross-block). This lifts the former covariance-only limitation
(mean-structured pairs used to error with a rank-deficient pooled `P`); it
matches lavaan's mean-structured `lavTestLRT(method = "satorra.2000")` and is
validated by `cpp/tests/unit/satorra2000_test.cpp` (augmented streaming/materialized/
dense agreement, an independent full-UΓ oracle, and the NT collapse) and by
`r-package/examples/nested_test_satorra2000.R` (meanstructure configural/metric
and a genuine intercept-invariance restriction vs lavaan).
The older `nestedTest()` spelling and lavaan labels (`"satorra.2000"`,
`"satorra.bentler.2001"`, `"satorra.bentler.2010"`) remain as compatibility
aliases during exploration. Low-level `magmaan_core` exposes both
`robust_nested_lrt_restriction_map` and `compat_lavaan_nested_lrt_*` aliases
over the same C++ primitives so methods scripts can choose the intended
surface directly.

2026-10-02: FMG `blocks_effective` and policy `peba_blocks` report actual nonempty PEBA blocks; ordinary global/nested output adds a footnote below four formed blocks without changing tails or default columns.
