\name{frontier_fit_sphere}
\alias{frontier_fit_sphere}
\title{Experimental fitting route: frontier_fit_sphere}
\description{Fits in sphere coordinates with an explicit metric and pinning penalty, then optionally polishes in ordinary parameter coordinates. Inference remains a separate operation.}
\usage{
frontier_fit_sphere(model, data, estimator = "ML", groups = NULL, ..., cluster = NULL, ordered = NULL,
    parameterization = "delta", psd = FALSE, missing = c("listwise", "error"), W = NULL,
    optimizer = NULL, control = NULL, bounds = NULL, metric = c("unit_free", "raw"),
    pin_weight = 1, pole_tol = 1e-06, polish = TRUE, start = c("canonical", "user"),
    start_eigen_floor = 1e-06, feasibility_tol = 1e-06, preconditioning = c("none",
        "diagonal"))
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{estimator}{Estimation objective. Supported objectives depend on the data kind and fitting route.}
\item{groups}{Grouping-variable name used when constructing a model from syntax.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
\item{cluster}{Cluster-variable name or cluster identifiers for two-level fitting.}
\item{ordered}{Names of ordinal observed variables.}
\item{parameterization}{Ordinal model parameterization; currently delta.}
\item{psd}{Whether to constrain model covariance blocks to be positive semidefinite.}
\item{missing}{Missing-data handling: listwise deletion, rejection, or pairwise moments where supported. FIML retains missing observations.}
\item{W}{Fixed fitting-weight matrix, or a per-group list in model moment order.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{bounds}{Lower and upper parameter-bound vectors or a named bounds preset; NULL leaves the route default.}
\item{metric}{Sphere-coordinate metric: unit_free or raw.}
\item{pin_weight}{Non-negative weight for the sphere pinning penalty.}
\item{pole_tol}{Tolerance for detecting a sphere-coordinate pole.}
\item{polish}{Whether to polish the sphere solution in ordinary parameter coordinates.}
\item{start}{Start source: canonical construction or user hints.}
\item{start_eigen_floor}{Eigenvalue floor used to construct a feasible covariance start.}
\item{feasibility_tol}{Numerical tolerance for covariance feasibility.}
\item{preconditioning}{Optimizer-coordinate preconditioning: none or diagonal.}
}
\value{A fitted-model list with parameter estimates and route diagnostics.}
