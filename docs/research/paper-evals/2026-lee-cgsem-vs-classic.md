# Classic or Computational Graph? A Comparison of SEM Estimation Frameworks

**Cite.** Lee, C., & Gates, K. M. (2026). *Multivariate Behavioral Research*, 61(3):436-438. DOI: 10.1080/00273171.2026.2673277.
**PDF.** `external/refs/Classic or Computational Graph  A Comparison of SEM Estimation Frameworks.pdf`
**Read.** 2026-06-26  ·  **Verdict.** background

## TL;DR
A three-page SMEP brief that frames itself as "two SEM frameworks" but is really
an optimizer comparison: lavaan's quasi-Newton (nlminb/BFGS, run to convergence)
versus tensorsem's Adam (first-order, ML-style early stopping) on the *identical*
normal-theory ML objective. Its headline ("cgSEM is less biased and more
admissible in hard conditions") is a non-convergence selection artifact, not a
property of the framework. Nothing to build; recorded as a clean design contrast
to magmaan's analytic-derivative bet.

## Contribution
- **Same objective, different stopping rule.** Both arms minimize the same ML
  discrepancy. "classic SEM" = analytic gradients + quasi-Newton to a stationary
  point; "cgSEM" = autodiff gradients + Adam with ML-default settings (learning
  rate, finite iterations). The autodiff/computation-graph layer just recomputes
  gradients lavaan already has in closed form; it is statistically inert to the
  bias results. The novelty over Van Kesteren & Oberski (2022), which tuned Adam
  to *mirror* classic optimization for numerical equivalence, is running Adam as
  ML practice would (adaptive, early-stopped).
- **Three sims** (moderate-complexity base): (1) model complexity, (2)
  misspecification, (3) latent-to-indicator strength, over N. Bias is measured
  **only among admissible solutions** (no Heywood, no latent r > 1).
- **Reported result.** In hard cells (small N, misspec, weak indicators) cgSEM
  shows lower bias and higher admissible-solution rates; as weak indicators
  accumulate and N grows, cgSEM shows *higher* bias within the admissible subset;
  in easy large-N cells classic SEM is fast with comparable bias. "No single
  approach fits all."

## Relevance to magmaan
**None to build; a cautionary read and a design contrast.**

The headline finding does not survive scrutiny, and the reason is worth on
record. Bias is conditioned on admissibility, and the admissible subsets are
*different runs* for the two methods. Adam with a learning rate and finite
iterations under-shoots the ML optimum; when the ML solution sits on or past the
Heywood boundary (exactly the small-N / weak-indicator regime), Adam's
non-convergence keeps it off the boundary. So "higher admissible rate" is
implicit shrinkage by under-optimization, and "lower bias among admissibles"
compares two different estimands (converged ML restricted to its admissibles vs.
a partially-converged Adam iterate restricted to its admissibles). The paper's
own large-N reversal is the tell: once N identifies a proper solution, Adam's
failure to finish optimizing reappears as bias. The brief is also silent on
inference (SEs, chi-square, fit indices), which is fatal for SEM: those rest on
a zero-gradient stationary point, which Adam's "admissible" iterates are not.

This cuts directly against magmaan's contract, so there is no port target.
magmaan converges to the exact lavaan stationary point (`fmin = ½F`, parity to
tolerance) and already carries a rich optimizer layer (PORT, PortNls, NLopt +
backends; see [[optimizer-roadmap]]). Adding Adam is trivial and pointless here:
"regularize by not converging" is the negation of lavaan parity. The
computation-graph architecture (runtime autodiff + torch flexibility) is the
philosophical opposite of magmaan's bet (compile-time free-function templates
over duck-typed `Discrepancy`, hand-derived analytic gradients, no virtuals);
useful as a contrast, not as something to adopt.

The one idea with a pulse, early-stopped first-order optimization as implicit
regularization for Heywood avoidance, has a principled form that is an *explicit*
penalty/prior with writable inference, not a stopping heuristic. That is already
captured: the penalized-ML entry in
[backlog/speculative.md](../../backlog/speculative.md) ("Penalized ML for
small-sample convergence and structure", from the De Jonckere & Rosseel 2023
eval) is the on-target version, and it carries the [[feedback-shortcut-variants]]
hidden-validation caveat. This brief adds no reason to build it and is not a
reference for it.

## Verdict
background — optimizer-vs-framework conflation plus an admissibility-conditioned
selection artifact; no estimator, SE, test, or parser feature. Documents a design
contrast (autodiff graph vs. analytic-derivative templates) and a methodological
cautionary tale. The adjacent buildable idea (explicit penalized ML) is already
tracked in [backlog/speculative.md](../../backlog/speculative.md); this paper does
not move it.
