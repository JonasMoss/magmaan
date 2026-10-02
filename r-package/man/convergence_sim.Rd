\name{convergence_sim}
\alias{convergence_sim}
\title{Convergence simulation design: convergence_sim}
\description{Provides a small reproducible convergence-study design. The catalog lists published design names and defaults; generators return data and model information without running a calibration study.}
\usage{
convergence_sim(design = convergence_sim_catalog()$design, ...)
}
\arguments{
\item{design}{Design name from convergence_sim_catalog.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
}
\value{The design catalog, or generated data and model metadata for the selected design.}
