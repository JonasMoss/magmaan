\name{frontier_fit_ml_psd}
\alias{frontier_fit_ml_psd}
\title{Experimental fitting route: frontier_fit_ml_psd}
\description{Fits the named objective through an explicit experimental covariance-domain route. PSD variants constrain covariance blocks; multiinfo variants apply the selected barrier target. Inference remains a separate operation.}
\usage{
frontier_fit_ml_psd(model, data, optimizer = "nlopt-slsqp", control = NULL, start_eigen_floor = 1e-06,
    feasibility_tol = 1e-06, missing = c("listwise", "error"), preconditioning = c("diagonal",
        "none"))
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{start_eigen_floor}{Eigenvalue floor used to construct a feasible covariance start.}
\item{feasibility_tol}{Numerical tolerance for covariance feasibility.}
\item{missing}{Missing-data handling: listwise deletion, rejection, or pairwise moments where supported. FIML retains missing observations.}
\item{preconditioning}{Optimizer-coordinate preconditioning: none or diagonal.}
}
\value{A fitted-model list with parameter estimates and route diagnostics.}
