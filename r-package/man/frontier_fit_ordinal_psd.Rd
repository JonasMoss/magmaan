\name{frontier_fit_ordinal_psd}
\alias{frontier_fit_ordinal_psd}
\title{Experimental fitting route: frontier_fit_ordinal_psd}
\description{Fits the named objective through an explicit experimental covariance-domain route. PSD variants constrain covariance blocks; multiinfo variants apply the selected barrier target. Inference remains a separate operation.}
\usage{
frontier_fit_ordinal_psd(model, data, estimator = c("DWLS", "WLS", "ULS"), optimizer = "nlopt-slsqp",
    control = NULL, bounds = NULL, start_eigen_floor = 1e-06, feasibility_tol = 1e-06)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{estimator}{Estimation objective. Supported objectives depend on the data kind and fitting route.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{bounds}{Lower and upper parameter-bound vectors or a named bounds preset; NULL leaves the route default.}
\item{start_eigen_floor}{Eigenvalue floor used to construct a feasible covariance start.}
\item{feasibility_tol}{Numerical tolerance for covariance feasibility.}
}
\value{A fitted-model list with parameter estimates and route diagnostics.}
