\name{modindices}
\alias{modindices}
\title{Modification indices and equality releases}
\usage{modindices(fit, test = c("score", "lr"), candidates = NULL, releases = TRUE)}
\arguments{
\item{fit}{A \code{magmaan()} fit.}
\item{test}{Robust one-df score tests (default) or refit-based LR tests.}
\item{candidates}{NULL for all rows, operator codes \code{=~}, \code{~~}, \code{~}, \code{==}, or a data frame with lhs, op, rhs and optional group and kind. Selection preserves the full table's order.}
\item{releases}{Include univariate equality releases.}
}
\description{Reports misspecification-robust one-df score statistics with chi-square(1) limiting p-values. There is no reference-distribution selector.}
\details{ML uses observed sensitivity and empirical likelihood scores; FIML uses observed pattern-score geometry. All-ordinal and mixed DWLS use exact first-stage sampling rows and the estimated-weight channel with observed sensitivity. Shared labels, explicit affine equalities and group.equal restrictions are tested one at a time.

LR refits one augmented model per candidate from the embedded fitted estimate and uses the same nested policy as \code{anova()}: ML/FIML LR with PEBA4 (equal to All at one df), or DWLS fit-function differences with All. LR is optional because these refits are costly. EPC columns retain the one-step score EPC rather than a refitted parameter change.

The generator covers loadings, residual/latent covariances and absent regressions among variables already participating in structural equations, including reverse paths. Unidentified regression directions retain typed unavailable rows. ML/FIML score statistics use observed nuisance sensitivity and an expected-information metric; EPCs use expected information. Singular nuisance sensitivity or nonpositive score variance produces \code{numeric_failure} and NA values; LR may still refit these candidates. Failed refits carry \code{refit_failed} or \code{not_converged}. Penalized, non-converged and unsupported fit regimes keep typed rows.}
\value{A data frame with kind (fixed or release), lhs, op, rhs, group, test, statistic, df, pvalue, epc, sepc.lv, sepc.all and reason. Rows remain in candidate order. Equality rows describe native affine restrictions in parameter-label coordinates.}
\seealso{\code{\link{anova.magmaan}}, \code{\link{magmaan}}}
