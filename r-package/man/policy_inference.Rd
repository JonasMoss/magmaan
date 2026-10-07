\name{policy_inference}
\alias{policy_inference}
\title{Compose ordinary-user inference}
\description{Requests the C++ ordinary-user inference policy for the fitted estimator and data regime. Unsupported regimes remain explicit.}
\usage{
policy_inference(fit, data = NULL)
}
\arguments{
\item{fit}{A fitted magmaan model.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
}
\value{An inference bundle with availability status, covariance and test summaries where supported.}

\details{ML/FIML use observed-bread empirical-score sandwich covariance and
score and likelihood-ratio tests with PEBA4 as the default reference.
All-ordinal and complete mixed DWLS use exact empirical first-stage sampling
influence and estimated-weight IJ covariance. Global fit-function tests and
observed-Hessian/IJ nested fit-function differences use the exact weighted
chi-square All reference; SB/PEBA4 comparators remain available. Fitting retains
OPG NACOV weights. Reference-law studies support the DWLS default across
ordinal, mixed, threshold-invariance and textbook designs; finite-sample size
varies across settings. Nested DWLS score tests are unavailable.}
