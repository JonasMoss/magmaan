\name{convergence_sim_dejonckere_msst}
\alias{convergence_sim_dejonckere_msst}
\title{Convergence simulation design: convergence_sim_dejonckere_msst}
\description{Provides a small reproducible convergence-study design. The catalog lists published design names and defaults; generators return data and model information without running a calibration study.}
\usage{
convergence_sim_dejonckere_msst(n = 10L, state_loadings = c(1, 0.5, 1.3), trait_loadings = c(1, 0.5, 1.3),
    measurement_error = c(0.6, 0.25, 0.7), trait_var = 0.6, state_residual_var = c(0.6,
        0.25, 0.7), seed = NULL)
}
\arguments{
\item{n}{Number of observations to generate.}
\item{state_loadings}{State-factor loading vector.}
\item{trait_loadings}{Trait-factor loading vector.}
\item{measurement_error}{Indicator measurement-error variance vector.}
\item{trait_var}{Trait-factor variance.}
\item{state_residual_var}{State-factor residual variance vector.}
\item{seed}{Optional random-number seed.}
}
\value{The design catalog, or generated data and model metadata for the selected design.}
