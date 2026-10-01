# Published conventions and reference design

Retained from the definitions audit. The 192-cell grid and verification expansion
are prospective reference designs; no run is queued. The merged report owns the
current evidence and explains why the written convention is not an EQS identity.

## Evidence

### The tests, reconstructed

Write

$$
\beta=(\operatorname{vech}\Sigma',\mu')',\qquad
\Omega=A^{-1}BA^{-1},\qquad
\Delta=\frac{\partial\beta(\theta)}{\partial\theta'}.
$$

Here $A$ is observed information and $B$ is the covariance of individual
likelihood scores. The article says explicitly that $A$ must be observed for
consistency under MAR, that all simulation computations used observed
information, and that Appendix B's `SE=EXACT` computed it analytically rather
than by numerical differentiation (article pp. 282, 285, 301--302).

| Item | Robust FIML | Robust two-stage |
|---|---|---|
| Generated $H_0$ | Structured CFA mean/covariance model, true in every cell | Same structured CFA, true in every cell |
| Saturated $H_1$ role | Supplies $\hat\beta$ and the maximized observed-data likelihood | Supplies Stage-1 EM $\hat\beta$ and $\hat\Omega$ |
| Structured estimate | Direct incomplete-data ML $\hat\theta$ | Complete-data ML discrepancy fit $\tilde\theta$ to $\hat\beta$ |
| Base statistic | $T_{ML}=2\{\ell(\hat\beta)-\ell(\hat\theta)\}$ | $(n-1)F_{ML}(\tilde\theta)$ |
| Residual projector | $W=A-A\Delta(\Delta'A\Delta)^{-1}\Delta'A$, with the Equation 2 quantities evaluated at the structured model | $U=H-H\Delta(\Delta'H\Delta)^{-1}\Delta'H$, with complete-data observed $H$ at the Stage-2 fit |
| Scale and test | $c_F=\operatorname{tr}(\Omega W)/d$; use $T_{ML}/c_F$ | $c_{TS}=\operatorname{tr}(\Omega U)/d$; use $(n-1)F_{ML}/c_{TS}$ |
| Reference | $\chi^2_d$, reject at $\alpha=.05$ | $\chi^2_d$, reject at $\alpha=.05$ |

This follows Equations 1--6 of the [published article](https://doi.org/10.1080/10705511.2014.882692).
The earlier open [two-stage derivation](https://escholarship.org/uc/item/89f8q6jc)
also makes the order explicit: saturated observed-data ML first, complete-data
discrepancy minimization second, then projection of the Stage-1 covariance
through the Stage-2 model derivatives. Savalei's separate paper on
[expected versus observed information](https://pubmed.ncbi.nlm.nih.gov/20853954/)
is the information-choice source cited by Appendix B.

The means are part of both $H_0$ and $H_1$, but were freely estimated and
therefore cancel from the model degrees of freedom. Thus Model 1 has $p=14$ and
$d=76$; Model 2 has $p=21$ and $d=186$. The paper's statement
$d=p(p+1)/2-q$ counts covariance restrictions, with $q$ the freely estimated
covariance-structure parameters.

### What “observed” does and does not mean

There are two different information objects:

1. **Stage-1/saturated incomplete-data information $A$.** It is the negative
   observed Hessian of the saturated likelihood. Together with the casewise
   score covariance $B$, it produces $\Omega=A^{-1}BA^{-1}$.
2. **The metric used to remove the structured tangent space.** For FIML this is
   the structured-model $A$ in $W$. For two-stage it is the complete-data $H$
   in $U$. The paper selected the observed form here as well.

`SE=EXACT` is therefore not “expected information,” and it is not EQS's
numerical observed-Hessian option: Appendix B says `SE=OBS` was the numerical
approximation, whereas `SE=EXACT` was the analytic observed-information formula.
The later computational review explains why these choices remain distinct in
modern software ([Savalei and Rosseel, 2022](https://doi.org/10.1080/10705511.2021.1877548)).

### The published simulation target

The exact-fit study crossed 2 models, 3 sample sizes, 2 kurtoses, 2 missing
proportions, 2 pattern regimes, and 4 missingness mechanisms: 192
cells, 1,000 replications per cell, with both estimators applied to every sample
(384,000 fits).

| Axis | Levels |
|---|---|
| CFA | 14 variables / 2 factors / 76 df; 21 variables / 3 factors / 186 df |
| $N$ | 200, 400, 600 |
| Margins | skewness 2; kurtosis 7 or 15; Vale--Maurelli/Fleishman in EQS 6.1 |
| Missing proportion | 15% or 30% on each selected variable set |
| Patterns | FP: at most 4; MP: at most 64 |
| Mechanism | MCAR, MAR-L ($x>0$), MAR-L2 ($x<0$), MAR-NL ($|x|>.54$) |
| Outcome for these tests | rejection proportion at $\alpha=.05$ under a true model |

Deletion proceeded set by set, repeatedly sampling rows until each set reached
its target missing proportion. A row could be chosen for multiple sets, creating
combined patterns. The exact conditioning/deletion map is retained in
`results/overview/missingness_sets.csv`:



For MCAR, the conditioning column was irrelevant. For MAR-L deletion was allowed
when the conditioning variable exceeded 0; for MAR-L2 when it was below 0; and
for MAR-NL when its absolute value exceeded .54.

Appendix A supplies all 21 standardized Model 2 loadings; residual variances are
$1-\lambda^2$, all observed means are zero and variances one, and factor
correlations are $.446$, $-.124$, and $-.085$ for F1--F2, F1--F3, and F2--F3.
Model 1 takes the F1 and F3 blocks, relabeling V15--V21 as V8--V14, with factor
correlation $-.124$. The machine-readable loading table is in
`results/overview/published_parameters.csv`; `results/overview/published_design.csv` contains the
complete 192-cell grid.

### Verification plan

1. **Freeze an independent equation-level reference.** Implement Equations 2--6
   directly for one fixed dataset. Export and compare $A$, $B$, $\Omega$,
   $\Delta$, $H$, the residual projector, its numerical rank, the trace, scale,
   corrected statistic, df, and p-value. Use the current lavaan reconstruction
   only as a secondary scalar sentinel.
2. **Add paper-specific routes, without changing current ones.** Give the 2014
   FIML and two-stage configurations explicit names. Preserve current MLR and
   ML2S because they already have different, documented lavaan targets.
3. **Gate identities before simulation.** Require symmetry and rank $d$ of the
   residual projector; direct equality of the independently computed trace;
   correct $(n-1)$ handling; mean rows cancelling from df; and convergence toward
   scale 1 under complete normal data. Require matrix agreement around $10^{-8}$
   and scalar agreement around $10^{-6}$ on the fixed reference case before
   interpreting Monte Carlo output.
4. **Run a focused replication first.** Start with Model 1 at $N=200$, both
   kurtoses and missing proportions, FP/MP, MCAR and the harsh MAR-L mechanism.
   Record achieved marginal moments, missing proportions/pattern counts,
   convergence, inadmissibility, and every excluded replication. Compare
   rejection rates with binomial 95% intervals.
5. **Expand only after parity.** Consider the 192-cell expansion only for a named replication request after
   the same-data gates pass. Keep explicitly labeled paper statistics separate from the current MLR baseline.
