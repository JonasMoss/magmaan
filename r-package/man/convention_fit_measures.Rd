\name{convention_fit_measures}
\alias{convention_fit_measures}
\title{Lavaan-compatible fit measures}
\usage{convention_fit_measures(fit, convention)}
\arguments{
\item{fit}{A fitted magmaanlab model retaining observations or ordinal stats.}
\item{convention}{A named lavaan inference bundle.}}
\value{A data frame with index, estimate and reason columns and a lavaan_compat attribute. Unsupported bundles error.}
\description{Composes standard, scaled and robust fit-index families in C++, including a matching independence baseline, noncentral chi-square intervals and close-fit p-values. Estimates are retained.}
