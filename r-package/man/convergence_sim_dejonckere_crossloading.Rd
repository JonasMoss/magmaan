\name{convergence_sim_dejonckere_crossloading}
\alias{convergence_sim_dejonckere_crossloading}
\title{Convergence simulation design: convergence_sim_dejonckere_crossloading}
\description{Provides a small reproducible convergence-study design. The catalog lists published design names and defaults; generators return data and model information without running a calibration study.}
\usage{
convergence_sim_dejonckere_crossloading(n = 20L, variant = c("2025", "2022"), fit_model = c("correct", "omit_lambda8",
    "omit_lambda8_lambda9", "omit_all_crossloadings"), beta = NULL, seed = NULL)
}
\arguments{
\item{n}{Number of observations to generate.}
\item{variant}{Published design version, 2025 or 2022.}
\item{fit_model}{Crossloading specification to fit: correct or one of the omission variants shown in usage.}
\item{beta}{Latent regression coefficient or coefficient vector; NULL uses the design default.}
\item{seed}{Optional random-number seed.}
}
\value{The design catalog, or generated data and model metadata for the selected design.}
