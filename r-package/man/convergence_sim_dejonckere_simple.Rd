\name{convergence_sim_dejonckere_simple}
\alias{convergence_sim_dejonckere_simple}
\title{Convergence simulation design: convergence_sim_dejonckere_simple}
\description{Provides a small reproducible convergence-study design. The catalog lists published design names and defaults; generators return data and model information without running a calibration study.}
\usage{
convergence_sim_dejonckere_simple(n = 20L, variant = c("2025", "2022"), beta = NULL, seed = NULL)
}
\arguments{
\item{n}{Number of observations to generate.}
\item{variant}{Published design version, 2025 or 2022.}
\item{beta}{Latent regression coefficient or coefficient vector; NULL uses the design default.}
\item{seed}{Optional random-number seed.}
}
\value{The design catalog, or generated data and model metadata for the selected design.}
