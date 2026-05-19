Yes. There are several “obvious” attacks that are not just “choose another divergence from the zoo.” I would separate them by what story they tell, because the story matters more than the exact convex function.

The current paper’s story is:

[
\text{bad cells should not dominate the polychoric fit.}
]

That leads to Pearson-residual downweighting: compare (\hat f_j) to (p_j(\theta)), and cap the influence of cells where (\hat f_j/p_j(\theta)) is too large. The authors explicitly frame their estimator as divergence-based, note related DPD/IRT work, mention person-fit/removal and mixture models as alternative carelessness strategies, and leave other discrepancy functions / SEM extensions open. So there is room.

The main alternatives I’d consider are these.

First: **density power divergence**. This is the most obvious “different recipe” because it has a clean robustness interpretation and is already known in robust estimation. For cells (j=1,\ldots,m),

[
D_\alpha(\hat f,p_\theta)
=========================

## \sum_j p_j(\theta)^{1+\alpha}

\frac{1+\alpha}{\alpha}\sum_j \hat f_j p_j(\theta)^\alpha
]

up to constants, with (\alpha>0), and ML recovered as (\alpha\to 0). The score is essentially

[
\sum_j
\left{
p_j(\theta)^{1+\alpha}
----------------------

\hat f_j p_j(\theta)^\alpha
\right}
s_j(\theta)=0.
]

The intuitive line: observations in cells with tiny model probability get attenuated by (p_j(\theta)^\alpha). Unlike the Welz hard cap, it does not ask “is this cell overcounted relative to the model?” as directly; it asks “is this cell a low-probability cell under the model?” That may be better for some tail/corner contamination and worse for ordinary diffuse carelessness. It is also a nice paper comparator because their paper explicitly says DPD-style robust estimation has been used in IRT, but not yet studied for polychoric correlation, and notes the usual efficiency cost for DPD-type methods.

Second: **Hellinger distance**. This is the canonical no-tuning-parameter simple divergence:

[
H^2(\hat f,p_\theta)
====================

\sum_j
\left(\sqrt{\hat f_j}-\sqrt{p_j(\theta)}\right)^2.
]

The intuitive line: square-rooting compresses large discrepancies, so grotesque cells matter less than under likelihood. It is symmetric, simple, and reviewer-comprehensible. The downside is that it does not encode the “overcounts are suspicious, undercounts are mechanically induced” logic. It treats excesses and deficits symmetrically. That could be fine as a comparator, but I would not make it the hero.

Third: **Huberized residual fitting**. Define a residual, say

[
r_j(\theta)=\frac{\hat f_j-p_j(\theta)}{\sqrt{p_j(\theta)}}
]

or a signed root-deviance residual, and minimize

[
\sum_j \rho(r_j(\theta)),
]

where (\rho) is Huber, Tukey bisquare, or a one-sided version. This is probably the easiest thing to explain to SEM people: “We robustify the residuals between observed and model-implied contingency tables.” It also connects very naturally to diagnostics. The one-sided version is closer to Welz:

[
\rho(r)=
\begin{cases}
r^2/2, & r\le k,\
kr-k^2/2, & r>k,
\end{cases}
]

or a smooth analogue. The caveat is that the exact likelihood geometry gets messier, and symmetric residual robustification may downweight clean cells that are merely low because another cell was inflated.

Fourth: **explicit contamination mixture**. Instead of robustifying the objective, write

[
p_j^{\mathrm{mix}}(\theta,\varepsilon)
======================================

(1-\varepsilon)p_j(\theta)+\varepsilon q_j,
]

and maximize

[
\sum_j N_j\log p_j^{\mathrm{mix}}(\theta,\varepsilon).
]

Here (q) is a simple bad-response distribution: uniform random responding, midpoint responding, straightlining, independent marginals, or a small finite menu of careless types. This is not assumption-free robustness; it is model-based carelessness. But it is extremely intuitive. The paper contrasts its approach with mixture-model approaches to careless responding, so this is a natural benchmark rather than an alien idea.

For simulations, I would include one mixture competitor only if it is easy. Something like

[
q_{xy}=1/(K_XK_Y)
]

for random responding, maybe also (q_{xy}=q_x q_y) estimated from marginals. Do not build an entire latent class carelessness paper by accident.

Fifth: **case-level weights**, not cell-level weights. This is maybe the most psychologically natural attack. Carelessness is usually a respondent-level phenomenon, not “cell ((5,1)) was contaminated by an evil table goblin.” Estimate ordinary or robust polychorics once, compute a respondent-level badness score from how often the person lands in pairwise high-residual cells, set weights

[
w_i = w(B_i),
]

and then recompute weighted polychorics. This is basically “person-fit but continuous, not delete/keep.” The paper mentions person-fit/removal as an existing alternative and emphasizes that their estimator downweights aberrant observations during estimation rather than removing them; a respondent-weighted version is the obvious halfway house.

This could be very SEM-journal-friendly, but it is a bigger object. Pairwise cell robustness is easy. Respondent-level robustness becomes an iterative multivariate procedure. Good extension, maybe not the short paper unless the simulations make it look amazing.

Sixth: **direct robust pairwise SEM**, skipping the polychoric matrix. This is the more principled SEM attack. Instead of first estimating pairwise polychorics and then fitting SEM, fit SEM parameters directly to all bivariate ordinal margins:

[
\min_\vartheta
\sum_{a<b}
\sum_j
p_{ab,j}(\vartheta)
\Phi!\left(
\frac{\hat f_{ab,j}}{p_{ab,j}(\vartheta)}
\right),
]

where (p_{ab,j}(\vartheta)) is the bivariate table probability implied by the SEM latent correlation matrix and thresholds. That is just robust pairwise likelihood / robust composite likelihood for ordinal SEM.

This has a beautiful selling point: it avoids the positive-definite polychoric-matrix problem, because the SEM model itself imposes a valid correlation structure. It is also “actually SEM” in a deeper sense than robust inputs. The downside is that it is no longer the tiny easy paper; it is a new estimator. I would keep it as a paragraph in the discussion, unless magmaan makes it almost free.

Seventh: **latent distribution alternatives**, e.g. (t)-copula or skew-elliptical latent variables. This attacks distributional misspecification rather than careless-response contamination. The paper explicitly distinguishes partial misspecification from distributional misspecification and mentions heavier-tailed/skewed latent distributions as another route.

This is less aligned with the clean “careless cells contaminate polychorics” story. I would not make it central. Maybe use a Clayton/tail-dependence simulation as a stress test, not as the conceptual core.

For the short paper, I would not run every divergence. I would use this hierarchy:

Report: ML, hard cap, one smooth cap, DPD.

Maybe report in supplement: Hellinger and Huber residual.

Maybe mention but not implement: mixture model, case weights, direct robust pairwise SEM.

That gives you three clear mechanisms:

[
\text{ML: no robustness.}
]

[
\text{Hard/smooth cap: cap overcounted cells.}
]

[
\text{DPD: temper unlikely cells.}
]

[
\text{Hellinger/Huber: robustify table residuals.}
]

The smooth cap remains the hero, because it is closest to the original estimator and therefore easiest to defend as a numerical refinement. DPD is the strongest “equally intuitive but different” competitor. Hellinger is the no-tuning sanity check. Mixtures and case weights are discussion/future-work unless you want the project to metastasize.

My instinct: implement more than you report. Let magmaan have `ml`, `hard_cap`, `smooth_cap`, `dpd`, `hellinger`, maybe `huber_residual`. In the paper, report only the methods that teach a clean lesson. The graveyard of unhelpful divergences belongs in your codebase, not in the manuscript.
