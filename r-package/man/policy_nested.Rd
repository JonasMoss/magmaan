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

\details{Plain ordinal and complete mixed DWLS use estimated-weight IJ covariance
and observed-Hessian/IJ nested fit-function differences with SB/PEBA4. The global
fit-function test uses the All tail; mixed spectra use exact empirical first-stage
sampling influence while fitting retains OPG NACOV weights. Mixed validation is
limited and calibration is pending. Nested DWLS score tests are unavailable.}
