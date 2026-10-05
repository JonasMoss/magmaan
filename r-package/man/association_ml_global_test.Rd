\name{association_ml_global_test}
\alias{association_ml_global_test}
\alias{association_ml_nested_test}
\title{Lab association ML global and nested spectral tests}
\usage{
association_ml_global_test(fit, ordinal_stats)
association_ml_nested_test(fit_alt, fit_null, ordinal_stats)
}
\arguments{
\item{fit, fit_alt, fit_null}{Association ML fitted models; alternative first.}
\item{ordinal_stats}{Common complete all-ordinal Stage-1 statistics with raw rows.}
}
\description{Composes exact empirical first-stage influence with the local correlation ML residual metric for global tests, or observed sensitivity and exact parameter restrictions for nested tests. Reports All, SB and PEBA4 references. Ordinary inference remains unavailable.}
\value{A list with availability and a typed reason, statistic, degrees of freedom, spectrum, three references and reconstruction matrices. Zero degrees of freedom has reason \code{zero_df} and undefined p-values. Moment nesting, penalties, boundaries and unsupported Stage-1 channels are refused.}
