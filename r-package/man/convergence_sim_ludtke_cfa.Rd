\name{convergence_sim_ludtke_cfa}
\alias{convergence_sim_ludtke_cfa}
\title{Convergence simulation design: convergence_sim_ludtke_cfa}
\description{Provides a small reproducible convergence-study design. The catalog lists published design names and defaults; generators return data and model information without running a calibration study.}
\usage{
convergence_sim_ludtke_cfa(n = 30L, rho = 0.3, loading = 0.5, seed = NULL)
}
\arguments{
\item{n}{Number of observations to generate.}
\item{rho}{Latent factor correlation.}
\item{loading}{Common standardized factor loading.}
\item{seed}{Optional random-number seed.}
}
\value{The design catalog, or generated data and model metadata for the selected design.}
