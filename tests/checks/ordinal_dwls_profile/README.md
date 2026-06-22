# Categorical DWLS Profile-Hessian Verification

Advisory local check for the categorical (all-ordinal) DWLS estimated-weight
profile-Hessian law: `estimate::ordinal_dwls_profile_rmsea` /
`ordinal_dwls_profile_lrt`. It is outside the default test suite and CI; it
links the optimized static libs and runs a population calculation.

## What it verifies

**Verification 1 — estimated-weight channel dormant at correct spec (eps = 0).**
At eps = 0 the C4 population is compound-symmetric, so both models fit perfectly
(residual 0). The residual-driven gamma channel must then vanish: the check
confirms `gap = tr_full - tr_fixed = 0` and `tr_full == tr_fixed` to machine
precision, i.e. the full extended-`(u, gamma)` law collapses onto the
fixed-weight DWLS law. This is the end-to-end analogue of the `r = 0` reduction
unit-tested in `weighted_inference_test.cpp`. Note the residual law is **not**
chi^2_3: DWLS (`W = diag(Gamma)^-1 != Gamma^-1`) needs scaling even under the
null, so by symmetry the three contrast eigenvalues are all the single
Satorra-Bentler scaling constant `c ~ 0.954`, not 1 — exactly the standard
DWLS/WLSMV behaviour.

**Verification 2 — misspecified population law vs the prototype.**
It rebuilds the symmetry-protected four-cycle (C4) binary pseudo-null from
`docs/research/notes/ordinal_dwls_profile_exploration.tex` (p = 4 binary
indicators, thresholds 0, latent-response correlations `0.3025` on the two
opposite pairs and `0.3025 + eps` on the four C4 edges), fits the congeneric
(H1) and tau-equivalent (H0) one-factor models with magmaan's all-ordinal DWLS,
and calls `ordinal_dwls_profile_lrt`. It compares, against the note's published
population values:

- `tr(Q_fixed Gamma_u)` — the fixed-weight (u-block only) contrast trace;
- `tr(Q Gamma_x)` — the full extended `(u, gamma)` contrast trace, whose excess
  over `tr_fixed` is the residual-driven estimated-weight (gamma) channel;
- the top-3 LRT spectrum at `eps = 0.10`.

The 16 cell probabilities are tallied from a large latent-normal simulation and
expanded to an exact proportional integer dataset, so the fitted moments equal
the tallied population proportions (Monte-Carlo noise enters only through the
tally, not through a fresh fit sample).

## Result

At `n_tally = 8e6`, both verifications pass. Verification 1 (eps = 0):

```
Reduction at eps=0 (correct spec):
  tr_full=2.8620  tr_fixed=2.8620  gap=0.0000
  spectrum top3: 0.9547 0.9541 0.9531  -> DWLS scaling c=0.9540 (all equal, != 1)
```

Verification 2 — magmaan reproduces the misspecified population law:

```
 eps   magmaan tr_fixed (ref)    magmaan tr_full (ref)   channel gap
 0.02     2.8537   (2.8544)        2.8571   (2.8577)       0.0034
 0.04     2.8477   (2.8475)        2.8610   (2.8606)       0.0133
 0.06     2.8410   (2.8408)        2.8704   (2.8510)       0.0294
 0.08     2.8334   (2.8331)        2.8849   (2.8842)       0.0515
 0.10     2.8235   (2.8234)        2.9029   (2.8959)       0.0793

 eps=0.10 spectrum: magmaan (1.1160, 1.1153, 0.6716) vs ref (1.1151, 1.1151, 0.6656)
```

`tr_fixed` matches to ~1e-3 and the degenerate eigenvalue pair to ~1e-3. The
estimated-weight channel is reproduced (positive, growing with `eps`), which is
the discriminating signal: a fixed-weight-only law would land on `tr_fixed`.

The check's pass criteria assert what is actually true rather than treating the
prototype as a gold oracle: tight `tr_fixed`, a positive and `eps`-monotone
gamma channel, and matching `eps=0.10` eigenvalues. They deliberately do **not**
require `tr_full` to match the prototype point-for-point, because the prototype's
`Gamma_x` is built by *double* finite-differencing (an FD Jacobian of a quantity
that is itself an FD `diag(Gamma_u)`), which the note flags as "useful for
orientation, not for final numerical claims." Its published `tr_full` at
`eps = 0.06` (`2.8510`) is non-monotonic — a clear FD artifact — while magmaan's
analytic-influence law gives the smooth, monotone `2.8704`. magmaan is the
corrected reference; this check is the realization of the note's own remark that
"a production version should reuse the analytic ordinal moment influence and
DWLS weight machinery in magmaan."

## Run

Requires `cmake --build --preset opt` first (links `build/opt/*.a`).

```sh
just quick   # n_tally=2e6, n_fit=2e5  (fast smoke)
just all     # n_tally=8e6, n_fit=4e5  (tighter convergence)
just clean
```

Override per-run: `build/ordinal_dwls_profile_verify --n-tally=… --n-fit=… --seed=…`.
