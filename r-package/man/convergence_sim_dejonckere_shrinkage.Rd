\name{convergence_sim_dejonckere_shrinkage}
\alias{convergence_sim_dejonckere_shrinkage}
\title{Convergence simulation design: convergence_sim_dejonckere_shrinkage}
\description{Provides a small reproducible convergence-study design. The catalog lists published design names and defaults; generators return data and model information without running a calibration study.}
\usage{
convergence_sim_dejonckere_shrinkage(n = 20L, study = c("study1", "study2"), indicators = NULL, beta = NULL,
    seed = NULL)
}
\arguments{
\item{n}{Number of observations to generate.}
\item{study}{Shrinkage design: study1 or study2.}
\item{indicators}{Number of indicators per factor; NULL uses the study default.}
\item{beta}{Latent regression coefficient or coefficient vector; NULL uses the design default.}
\item{seed}{Optional random-number seed.}
}
\value{The design catalog, or generated data and model metadata for the selected design.}
