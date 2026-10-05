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

\details{Plain ordinal and complete mixed DWLS use estimated-weight IJ covariance
and observed-Hessian/IJ nested fit-function differences with SB/PEBA4. The global
fit-function test uses the All tail; mixed spectra use exact empirical first-stage
sampling influence while fitting retains OPG NACOV weights. Mixed validation is
limited and calibration is pending. Nested DWLS score tests are unavailable.}
