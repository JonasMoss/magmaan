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
        tests/golden/ordinal_golden_test.cpp
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
Proof: the exact definitions are Pek & MacCallum (2011, external/refs/
        case-influence/pek2011.pdf) Eq. 7 (DFTHETAS = (θ̂ⱼ − θ̂ⱼ₍ᵢ₎) / SE(θ̂ⱼ₍ᵢ₎),
        the leave-one-out SE) and Eq. 6 (gCD = Δ'[V̂AR(θ̂₍ᵢ₎)]⁻¹Δ, the reduced-
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
Upstream: not filed (PR to semfindr planned — see docs/backlog/todo.md). Found
        2026-06-23.
```

## Investigated — not a defect

- **Satorra-2000 scaled-difference parity** (2026-05-17): a divergence first
  suspected to be a lavaan bug was resolved as a magmaan-side issue / convention.
  See [`satorra2000_parity.md`](satorra2000_parity.md). Kept here as a reminder
  that most "lavaan is wrong" hunches are not.
- **ULS standard(Browne) vs robust(2N·fmin) test base**: lavaan-faithful, not a
  bug (see the test ledger / numerical-conventions notes).
