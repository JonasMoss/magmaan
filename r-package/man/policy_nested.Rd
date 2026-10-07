\name{policy_nested}
\alias{policy_nested}
\title{Compose ordinary-user nested inference}
\description{Requests the C++ ordinary-user nested policy for a restricted and unrestricted fit.}
\usage{
policy_nested(fit_H1, fit_H0, data = NULL)
}
\arguments{
\item{fit_H1}{Unrestricted (alternative) fitted model.}
\item{fit_H0}{Restricted (null) fitted model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{A nested inference bundle with availability status and supported test summaries.}

\details{ML/FIML use observed-bread empirical-score sandwich covariance and
score and likelihood-ratio tests with PEBA4 as the default reference.
All-ordinal and complete mixed DWLS use exact empirical first-stage sampling
influence and estimated-weight IJ covariance. Global fit-function tests and
observed-Hessian/IJ nested fit-function differences use the exact weighted
chi-square All reference; SB/PEBA4 comparators remain available. Fitting retains
OPG NACOV weights. Reference-law studies support the DWLS default across
ordinal, mixed, threshold-invariance and textbook designs; finite-sample size
varies across settings. Nested DWLS score tests are unavailable.}
