# Oracle Defects Ledger

`AGENTS.md` makes lavaan the oracle: magmaan matches installed lavaan output to
documented tolerances. This file is the deliberate exception list — the small
set of cases where lavaan (or another oracle: Mplus, semTests, robcat, ...) is
**provably wrong** and magmaan is right. It exists so that:

1. we do not re-litigate a known oracle defect every time a parity check
   "fails";
2. we do not gate a test against output we know to be incorrect (gate
   transitively or self-consistently instead — see each entry);
3. we have a written, reproducible case to file upstream (PR / bug report) when
   we get around to it.

This is the *opposite* of [`test_ledger.md`](test_ledger.md), which records
magmaan bugs we fixed. Here magmaan is correct and the oracle is not.

## Standard of proof

"Lavaan is the oracle" is a non-negotiable, so the bar to declare an oracle
defect is high. A bare "magmaan differs from lavaan" is **not** enough — that is
almost always a magmaan bug. Require at least:

- an **independent** reference (a from-scratch implementation of the textbook
  definition, an analytic value, or a second tool) that magmaan matches and the
  oracle does not; and
- a **first-principles** argument for why the oracle output is wrong (e.g. it
  violates a defining property: a posterior mode whose gradient is not zero, a
  probability that does not integrate to one, a statistic that is not invariant
  where it must be).

For a **scaled or robust test statistic**, "defensible convention difference" is
not defensible on a single-dataset comparison. magmaan and the oracle can use
different finite-sample conventions that agree on one draw yet imply different
rejection rates. Before exempting such a row, prove calibration in the target
regime (nonnormal / missing / ordinal): a null Monte Carlo whose magmaan
rejection rate tracks the oracle and sits near nominal. Normal-data calibration
does not license a nonnormal-data method. The nested Satorra-2000 scaling was a
near-miss: exempted here as "believed correct," it over-rejected 5x under
nonnormality. See [calibration-parity.md](calibration-parity.md).

If you investigate a divergence and the oracle turns out to be right (or the
call is a defensible convention difference), record it in the **Investigated —
not a defect** section so the next person does not redo the work.

## Entry format

```text
Defect: <one-line symptom: which oracle, which feature, what is wrong>.
Scope: <when it bites — versions, model shapes, options>.
Proof: <the independent reference + first-principles property that magmaan
        satisfies and the oracle violates; how to reproduce>.
magmaan: <what magmaan does instead, and the test that protects it>.
Upstream: <not filed / issue link / PR link / fixed-in-version>.
```

## Confirmed defects

### Ordinal nonlinear nested and release-score starting tangents (TASK-54.2)

**Scope:** pinned lavaan 0.7-2, all-ordinal WLSMV nonlinear equality
restrictions; nested Satorra-2000 delta geometry and release scores with combined
affine/nonlinear equalities. Expected covariance and global tests agree without
this exception. This does not authorize observed/IJ or ordinary-policy inference.

**Proof:** `cpp/tests/tools/prove_ordinal_nonlinear_tangent.R` generates seed
542072, N=800 three-category observations and fits `l2 == l3^2`. The exact
constraint row is H=(1,-2*l3); a fitted tangent K must satisfy H(theta_hat)K=0.
The independent reconstruction obtains K from QR, maps D0 K into the alternative
moment Jacobian D1, and takes the orthogonal restriction A. With I=D1' W D1 and
B=D1' W Gamma W D1, its scaling is
`tr(A I^-1 B I^-1 A') / tr(A I^-1 A')`. Public lavaan components give 6.3070077,
matching magmaan. Default-start lavaan gives 6.2846959431. Refitting from its own
fitted partable gives 6.3070077 while estimates change by at most 1.2e-10.
A statistic conditional on data and fitted models cannot depend on an optimizer
start. The original basis uses derivative 1.626897 (twice starting l3=.8134487),
rather than 1.949846 (twice fitted l3=.9749229); its fitted constraint residual
is .169113204. These are component observations, not an upstream code port.

**Transitive gate:** `regen_ordinal_nonlinear.R` refits *both* models with
`start=parTable(fit)` before freezing nested values, and refits the null before
freezing release scores. For combined `a == b*c` and `b == c`, the second
release score changes from 4.731 to 5.283 and then agrees with the fitted-tangent
core (with the documented N/(N-G) score divisor). If the alternative is also
nonlinear its fitted-start refit is essential. Default-start values remain
observations in the fixture. The live R gate separately starts lavaan from
magmaan estimates and checks the same nested statistic. Six synthetic fixture
cases cover DELTA, THETA, binary, product, combined and cross-group restrictions.

**Target-regime calibration:** reproducible runner
`cpp/tests/tools/calibrate_ordinal_nonlinear.R 500`, seeds 542201–542700, N=800,
three-category Gaussian latent-response CFA, true relative loadings
l3=.875 and l2=.875^2. On 2026-10-04 all 500 fits succeeded in 145.848 seconds,
with BLAS/OpenMP pinned to one thread under nice. Magmaan and fitted-start
lavaan both rejected 26/500 at 5%: 5.2%, exact binomial 95% interval
[.03424570,.07526637], p=.837 against .05. Maximum absolute statistic difference
was 1.409237e-5; no rejection decisions differed. This licenses the documented
expected-information ordinal regime only; it is not evidence for native
misspecification-robust Lagrangian sensitivity.

**Replacement:** fitted constraint tangents for expected-information nested and
score inference. Frozen C++ and live R gates enforce transitive lavaan agreement
at existing tolerances. Upstream: not filed.

```text
Defect: lavaan multi-group categorical lavPredict(type="lv", method="EBM")
        returns a non-stationary point for non-reference groups (the returned
        score is not the posterior mode).
Scope: lavaan 0.7-1.2691 (and earlier); ordered/categorical multi-group fits.
        The reference group is correct; group 2+ drift (~0.2 max|diff|,
        corr ~0.996 on a 2-group 3-cat one-factor CFA) even with theta matched
        to 1e-7. Single-group categorical EBM is correct.
Proof: (1) magmaan's group-2 EBM matches an independent R optimize()
        posterior-mode scorer built from lavaan's OWN extracted group-2
        parameters to 1.9e-5; (2) at lavaan's group-2 score the posterior
        gradient is O(1) and the posterior density is LOWER than at magmaan's
        score (gradient ~1e-7) — lavaan is not at the mode (defining property of
        EBM violated); (3) every scorer ingredient lavaan uses (VETAx prior,
        THETA, TH(delta=FALSE), loadings, data, th.idx) is identical to
        magmaan's. Consistent with the FIXME in lavaan R/lav_predict.R
        (lav_predict_eta_ebm_ml) that categorical scores are "not identical (but
        close) to Mplus". Repro: regenerate a 2-group ordinal CFA, compare
        lavPredict(EBM)[[2]] to an optimize() over
        [ordinal log-lik + log N(alpha, psi) prior] using lavInspect(.,"est").
magmaan: factor_scores_ordinal / _mixed_ordinal compute the true posterior mode.
        Because lavaan is not a usable oracle here, the multi-group scorer is
        gated TRANSITIVELY: for an unconstrained two-group fixture the per-group
        multi-group EBM equals an independent single-group fit on that group's
        data (~3e-8), and single-group EBM is lavaan-gated. Test:
        cpp/tests/golden/ordinal_golden_test.cpp
        "ordinal/mixed factor scores (EBM/ML) match lavaan".
Upstream: not filed. Found 2026-06-14.
```

```text
Defect: semfindr::est_change_approx() (the one-step, no-refit case-influence
        approximation) applies the finite-sample factor N/(N-1) twice to the
        standardized change (DFTHETAS) and only once inside the approximate
        generalized Cook's distance (gcd_approx) — one too many and one too
        few, respectively. est_change_raw_approx() is correct (one factor).
Scope: semfindr 0.2.0. Both errors are exactly O(1/N) constant factors, so they
        are immaterial relative to the one-step approximation's own error, but
        they have no first-principles basis. Affects only the *_approx engine,
        not the exact leave-one-out est_change().
Proof: the exact definitions are Pek & MacCallum (2011;
        external/refs/pek-2011-case-influence-sem-sensitivity-analysis.pdf),
        Eq. 7 (DFTHETAS = (θ̂ⱼ − θ̂ⱼ₍ᵢ₎) / SE(θ̂ⱼ₍ᵢ₎), the leave-one-out SE)
        and Eq. 6 (gCD = Δ'[V̂AR(θ̂₍ᵢ₎)]⁻¹Δ, the reduced-
        sample covariance). A one-step approximation must approximate these.
        Influence-function derivation — removing case i gives
        θ̂ − θ̂₍ᵢ₎ ≈ (N/(N-1))·V·s_i (one N/(N-1), the factor semfindr's
        est_change_raw_approx already carries). Hence DFTHETAS = Δ/SE carries
        exactly one such factor and gCD = Δ'V⁻¹Δ carries it squared. semfindr's
        est_change_approx multiplies the already-factored raw change by N/(N-1)
        AGAIN for DFTHETAS, and forms gcd_approx = (N-1)·xᵀ(V⁻¹/N)x =
        (N/(N-1))·sᵀVs instead of (N/(N-1))²·sᵀVs. Independent reference: the
        exact leave-one-out engine (est_change(), itself gated against
        semfindr::est_change() to ~1e-5) — magmaan's corrected one-step tracks
        the exact gcd marginally better than semfindr's (0.2643 vs 0.2654 on a
        2-factor HS CFA, the rest being one-step error).
magmaan: est_change_approx() uses the correct scaling (DFTHETAS = Δ/SE,
        gcd_approx = Δ'V_sel⁻¹Δ with Δ = (N/(N-1))Vs). Gated transitively in
        r-package/examples/case_influence_semfindr.R: matched up to the two
        documented constant factors (dftheta_magmaan = dftheta_semfindr·(N-1)/N,
        gcd_magmaan = gcd_semfindr·N/(N-1)) to machine precision, not against the
        raw semfindr output.
Upstream: not filed (PR to semfindr planned — see project/backlog/todo.md). Found
        2026-06-23.
```

```text
Defect: lavaan cannot fit a multi-group two-level model: sem(model, data,
        cluster=, group=) on a `level:` model errors with "subscript out of
        bounds" in lav_data_cl_patterns. lavaanify drops the group axis for
        `level:` models (it emits only n_levels blocks, not n_groups*n_levels),
        so no multigroup-twolevel oracle output exists at all.
Scope: lavaan 0.7-1.2691. Any model combining `level:` (two-level) with a
        grouping variable. Single-group two-level and single-level multi-group
        both work; only the crossing breaks.
Proof: not "wrong output" but "no output" — so magmaan is gated SELF-
        CONSISTENTLY rather than against lavaan. First principles: with no
        cross-group equality constraints a K-group fit is K independent fits,
        so its likelihood is block-diagonal and (i) each group's θ̂/SE must
        equal the lavaan-validated single-group fit on that group's data and
        (ii) the LRT statistic and df both scale by K exactly. Independent
        reference: the single-group two-level path, which IS lavaan-gated
        (cpp/tests/golden/twolevel_golden_test.cpp first case + the *.json oracles).
        Repro: lavaan::sem(level-syntax model, data, cluster="c", group="g").
magmaan: multi-group two-level works end to end (the objective/H1/information
        and level_block_pairs already loop over cs.groups; cluster_sample_stats
        and data_from_cluster build one ClusterGroupStats per group; df =
        Σ_g [p_g(p_g+1)+p_g] − q). Gated by a two-group-on-duplicated-data
        self-consistency check: per-group θ̂/SE equal the single-group oracle
        and the LRT/df double. Tests:
        cpp/tests/golden/twolevel_golden_test.cpp
        "twolevel multigroup: two-group self-consistency vs single-group oracle"
        and r-package/tests/testthat/test-twolevel.R (the direct multigroup
        lavaan-parity case skips with this reason). Mplus could serve as a
        future oracle (see cpp/tests/tools/regen_oracle_twolevel_mplus.R).
Upstream: not filed. Found 2026-06-27 (multi-group two-level finish-up).
```

```text
Defect: lavaan's standardized optimizer scaling can return a converged fit
        that violates an affine equality in the original parameter
        units. It enforces the equality in scaled coordinates instead.
Scope: reproduced in installed lavaan 0.7.2, complete continuous ML,
        ceq.simple=FALSE, an equality not preserved by the projected scale and
        optim.parscale="standardized". The preset's standardized retry uses
        the same coordinate convention. This is an estimation-feasibility
        defect, not a robust/scaled-statistic or sampling-law discrepancy.
        Zero RHS alone is insufficient; homogeneous ratios are also affected.
Proof: independent analytic reference: a+b=1.5 is satisfied by a=t,
        b=1.5-t for every t. On S=outer((1,.8,.6,.9),(1,.8,.6,.9))+.7*I,
        N=200, model "f =~ x1+a*x2+b*x3+x4; a+b == 1.5", with
        sample.cov.rescale=FALSE, meanstructure=FALSE, fixed.x=FALSE and
        se=test="none", ordinary scaling returns a+b=1.4999999999999998,
        fmin=0.0010516528432651384. A from-scratch Gaussian covariance
        objective and analytic gradient, optimized by BFGS under b=1.5-a,
        return a+b=1.5, fmin=0.0010516528432646943, max|gradient|=8.06e-9.
        That reference uses neither lavaan nor magmaan for its covariance,
        objective or derivatives. Standardized lavaan reports converged=TRUE
        but a+b=2.0183524702172848, original-constraint residual 0.5183524702
        and fmin=0.026351340769849241. Its scaled-constraint residual is zero.
        A second witness uses a==2*b on the same S: ordinary lavaan returns
        a/b=2.0000000000000004, fmin=0.016244306065005709; independent BFGS
        under a=2*t, b=t gives fmin=0.016244306064985281. Standardized lavaan
        reports convergence but returns a/b=0.99999999999999989 and
        a-2*b=-0.69088432321583448. Thus a zero RHS does not fix the defect.
        First principles: under z=D*theta, z=K*alpha+k0, preserving the
        original A*theta=d requires A*D^-1*K=0 and A*D^-1*k0=d. Retaining the
        original affine basis/offset while dividing by D need not preserve
        that surface. The ratio witness has max|A*D^-1*K|=0.8284731975.
        Reproduction and independent reference:
        Rscript cpp/tests/checks/lavaan_affine_scaling_defect.R.
magmaan: the pinned preset returns an explicit unsupported error on entry
        to a standardized retry if any ordered affine RHS is nonzero. It
        retains the preceding attempt's iterations/objective in the error.
        Zero-RHS standardized retries additionally require finite nonzero
        scales and A*D^-1*K=0, checked with per-row/per-column norm-scaled
        roundoff tolerance 64*epsilon*n. First unstandardized affine fits
        remain supported and are checked for A*theta=d; shared-label/group
        equality retries preserve that surface and remain supported.
        Replacement gates (no incorrect oracle endpoint is accepted):
        cpp/tests/unit/configured_ml_test.cpp,
        "lavaan preset rejects nonzero affine RHS only when a standardized
        retry is needed" and "lavaan preset rejects homogeneous ratio retries
        that change the constraint surface";
        r-package/tests/testthat/test_fitting_options.R,
        "unsafe affine standardized retries error while shared-label retries
        remain valid". The independent reproduction above protects the
        diagnosis; affine feasibility and the typed refusal protect the
        supported fitting contract. No inference-statistic exemption is made.
Upstream: not filed externally. Found and independently verified 2026-10-02.
```

## Investigated — not a defect

### Mixed ordinal MI criterion scale (TASK-33.4)

The factor-two difference in score fixture 0005 was a magmaan defect, not an
oracle defect: mixed MI multiplied both -N J'r and N J'J by two, while
mixed equality releases already used the correct units of the fitter's N F/2.
Removing that factor halves ordinary and fixed-weight robust MI and preserves
EPC. The independent residual-space df=1 reconstruction and frozen-moment
oracle comparison are described in [the MI inventory](capabilities.md#mi-and-equality-release-score-components).
Lavaan's separate (N-1)/N score divisor is transported only for the comparison;
no oracle exemption is claimed and no estimated mixed-weight support is added.


### Constrained ML retry gradients at different endpoints (2026-10-02)

For the same synthetic covariance with x1 multiplied by 100 and x3 divided
by 100, the model `f =~ x1+a*x2+a*x3+x4` takes an unstandardized attempt and
a standardized retry. The two implementations' full QR bases are bit-identical,
and their starts, scales, endpoint/objective tolerances and attempt verdicts
agree. PORT nevertheless stops after different iteration counts (1165 versus
1185, then 249 versus 253). Reduced gradient vectors at those different
endpoints differ by maxima 0.00268556 and 6.29886e-6.

Evaluating both implementations at the **same** frozen oracle endpoints gives
gradient differences below 1.86e-13; evaluating installed lavaan at magmaan's
endpoints gives differences below 1.97e-12 and objective differences below
1.34e-15. This is floating-point search-path variation, not evidence of a
derivative or oracle defect. The approved compatibility contract gates
derivatives at identical points, starts/scales/coordinates, endpoint and
objective tolerances, and each actual endpoint's declared acceptance rule.
It does not require final gradients at different endpoints to be identical.
The ×100 witness remains in the fitting fixture and C++ gates. Installed-lavaan
R gates use a fixed ×200 shared-label retry witness, which enters the
standardized route in the R binary and retains the same numerical comparisons.
No endpoint-parity exclusion or relaxed numerical tolerance is used.

### Native lavaan pEBA-4: absolute integration accuracy in a tiny tail (2026-09-20)

Lavaan 0.7-2's HS three-factor NTML example yields `T = 85.305521769973225`,
`df = 24`, and default `peba4_ml` p-value `1.642986754424314e-7`.
With the **identical** statistic and spectrum, magmaan gives
`1.529608811641211e-7`. Both use expected information; parameter estimates,
covariances, and the spectrum are not the source of this discrepancy.

The native `lav_test_fmg_imhof` uses `epsabs = epsrel = 1e-6` for an integral
whose probability is recovered as `0.5 + integral/pi`. Relative accuracy in
the integral does not imply relative accuracy in this small probability.
Tightening both tolerances to `1e-13` gives `1.529608875672217e-7`.
The default difference is about `1.13e-8` in absolute probability, within the
requested absolute integration accuracy; it is about 6.9% of the default
reported probability. This is a numerical precision limitation, not a change
of robust test, Hessian convention, or a proven integration-algorithm defect.

Independent reference: all four penalized pEBA weights occur six times.
Thus each block is `w * chi-square(6) = Erlang(shape=3, rate=1/(2w))`.
The sum is the absorption time of a twelve-phase exponential chain. Form the
upper-bidiagonal transient generator with diagonal `-rate_i` and superdiagonal
`rate_i`; its survival is `e_1' exp(Q*T) 1`. At 70 decimal digits this gives
`1.52960890009620744e-7`; a separate 50-digit computation agrees to more than
40 relative decimal places. This calculation uses neither Imhof quadrature
nor magmaan's positive-series implementation. The reproducible tail audit is
kept with experiment showcases/08, under its `scripts/` directory and experiment-local
results. The benchmark retains its original rejected rows pending a timed
accuracy-matched comparator; no statistical parity gate is waived.


- **Satorra-2000 scaled-difference parity** (2026-05-17): a divergence first
  suspected to be a lavaan bug was resolved as a magmaan-side issue / convention.
  See [`satorra2000_parity.md`](satorra2000_parity.md). Kept here as a reminder
  that most "lavaan is wrong" hunches are not.
- **ULS standard(Browne) vs robust(2N·fmin) test base**: lavaan-faithful, not a
  bug (see the test ledger / numerical-conventions notes).
