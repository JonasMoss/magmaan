\name{policy_modification_indices}
\alias{policy_modification_indices}
\title{Policy modification indices and equality releases}
\usage{policy_modification_indices(fit, data = NULL, releases = TRUE)}
\arguments{
\item{fit}{An estimate-only magmaanlab fit.}
\item{data}{Optional raw observations, otherwise the retained fitting data.}
\item{releases}{Include univariate equality-release score tests.}
}
\description{Thin adapter to the C++ ordinary-policy modification-index composer.}
\details{ML/FIML use observed nuisance sensitivity with expected metric/EPCs and empirical likelihood-score meat (pattern scores for FIML), and all-ordinal/mixed DWLS use exact first-stage sampling rows and estimated-weight influence with observed sensitivity. Each available robust score statistic has a chi-square(1) limiting reference. Singular nuisance sensitivity or nonpositive score variance retains numeric_failure rows. Fits outside the policy return typed unavailable rows. No alternative inference convention is substituted.}
\value{A data frame with kind, lhs, op, rhs, group, test, statistic, df, pvalue, epc, sepc.lv, sepc.all and reason. Internal candidate-row metadata supports native one-restriction augmentation and embedded-start refits.}
