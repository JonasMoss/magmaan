# Calibration parity for robust and scaled statistics

A testing principle for magmaan, prompted by a near-miss: a miscalibrated nested
Satorra-2000 scaling that the single-dataset parity gate passed as a "known
convention difference" while it over-rejected by 5x under nonnormality. This note
records the gap, the case study, and the discipline that closes it. It is a
companion to [`local_hardening.md`](local_hardening.md) (the validation program),
[`oracle-defects.md`](oracle-defects.md) (the standard of proof for divergence),
and [`test_ledger.md`](test_ledger.md) (the per-subsystem surface).

## The gap

Single-dataset lavaan parity is magmaan's contract: match the oracle on the same
data to a documented tolerance. That is necessary and sufficient for quantities
with one correct value: point estimates, standard errors, the chi-square, df,
fit indices. It is **not sufficient for robust or scaled test statistics**.

A robust statistic carries a finite-sample *convention*: which moment weight
`Gamma`, which information `V` (observed vs expected, structured vs unstructured
H1), which restriction map (exact vs delta), which estimator-specific metric
under missing data. More than one convention can be defensible and produce a
*different but plausible per-dataset number*. So a single-dataset check has a
fork: match (fine) or differ. When it differs, the tempting move is to record a
"convention difference, magmaan believed correct" and exempt the row.

That exemption is only valid if magmaan's convention is **calibrated**: the
statistic must have the right sampling distribution, so the test has nominal size
and correct power. A single realization cannot establish that. Per-dataset
agreement does not imply equal rejection rates, and a "believed correct"
convention can be quietly mis-sized.

## Case study: nested Satorra-2000 scaling (2026-06)

magmaan's native nested difference test scaled in a saturated-EM eta-space metric
(`V = SaturatedMoments::H`, `Gamma = acov`, `A.method = "exact"`). Its unscaled
`T_diff` matched lavaan exactly, so the model and LRT were right. The *scaled*
difference differed from lavaan's delta/`WLS.V` convention, and the parity gate
recorded those rows as a KNOWN, non-blocking convention difference, "believed
correct" on the strength of normal-data calibration (a prior experiment).

Under nonnormal (ig2) data the native convention over-rejected: the strict
(residual-invariance) difference test rejected at .45 vs lavaan .085 on identical
complete-data draws, and the over-rejection worsened with model size and did not
shrink with N. None of this was visible to the single-dataset gate, which kept
passing. It surfaced only from a paired magmaan-vs-lavaan **rejection-rate** Monte
Carlo on the same draws. The fix (`fbb8914`) added a `convention = "lavaan"`
selector that reproduces lavaan's metric per dataset, so calibration parity then
holds by construction. The native convention remains available as a diagnostic.

The lesson is not "the convention was wrong." It is that the *test that would
have caught it was missing*, and that "believed correct" rested on the wrong
regime (normal data for a method whose job is nonnormal data).

## The discipline

1. **Calibration parity is mandatory for robust and scaled test statistics.**
   Beyond per-dataset value parity, run a small null Monte Carlo and require the
   magmaan rejection rate to agree with the oracle and to sit near nominal, **in
   the regime the statistic exists to handle** (nonnormal, missing, ordinal, as
   applicable). This catches convention and scaling errors that value parity
   cannot.

2. **No "convention difference, believed correct" exemption without a
   calibration proof in the target regime.** Normal-data calibration does not
   license a nonnormal-data method. This raises the
   [`oracle-defects.md`](oracle-defects.md) bar: a divergence-but-magmaan-right
   claim for a scaled statistic must include calibration in the target regime,
   not just an independent single-value reference.

3. **Default parity-critical quantities to the oracle-compatible convention.**
   Offer native conventions as opt-in diagnostics gated by their own calibration
   evidence. The `nestedTest(..., convention = "lavaan" | "magmaan")` selector is
   the template: the lavaan convention is the default and the parity target, the
   native one is a labeled diagnostic.

4. **Two gates, cheap then expensive.** Per-dataset value parity stays the first,
   exact, fast gate. Calibration parity is the second gate: Monte Carlo, advisory
   or periodic, run before trusting a robust statistic in a new regime or before
   a production simulation that depends on it.

## Scope: which statistics carry a convention surface

These are the magmaan quantities where a wrong-but-plausible convention can pass
value parity yet mis-size the test, so each wants a calibration check in its
target regime:

- Satorra-Bentler scaled and scaled-shifted GOF (continuous nonnormal).
- Satorra-2000 scaled-difference nested tests (the case above), FIML and ML2S.
- FMG / pEBA eigenvalue-spectrum tests (GOF and nested).
- Robust (sandwich) standard errors and Wald / z tests.
- Two-stage ML (ML2S) tests under missing data.
- Ordinal DWLS / WLSMV mean and mean-and-variance corrections.

## Implementation sketch

A reusable advisory harness under `tests/checks/` (Monte Carlo, outside the
default `ctest` suite, like the existing simulation checks), parameterized by
(estimator, statistic, data regime). Per cell it simulates under the null,
computes the magmaan and lavaan rejection rates on the same draws, and reports
the pair gap plus the nominal deviation. The fiml-fmg paper's
`analysis/parity_calibration.R` is a working prototype of exactly this shape and
can be ported inward once generalized. See [`docs/backlog/todo.md`](../backlog/todo.md)
for the active task.
