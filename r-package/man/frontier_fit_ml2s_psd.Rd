\name{frontier_fit_ml2s_psd}
\alias{frontier_fit_ml2s_psd}
\title{Experimental fitting route: frontier_fit_ml2s_psd}
\description{Fits the named objective through an explicit experimental covariance-domain route. PSD variants constrain covariance blocks; multiinfo variants apply the selected barrier target. Inference remains a separate operation.}
\usage{
frontier_fit_ml2s_psd(model, data, optimizer = "nlopt-slsqp", control = NULL, h_step = 1e-04,
    stage1 = NULL, stage1_regularization = NULL, stage2_weight = "nt", dls_a = 0.5,
    start_eigen_floor = 1e-06, feasibility_tol = 1e-06)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{h_step}{Finite-difference step for the two-stage derivative calculation.}
\item{stage1}{Optional precomputed Stage-1 fit for ML2S.}
\item{stage1_regularization}{Optional Stage-1 regularization specification for ML2S.}
\item{stage2_weight}{Two-stage fitting weight selector, such as nt or adf.}
\item{dls_a}{DLS mixing scalar in [0, 1]: zero is normal theory and one is ADF.}
\item{start_eigen_floor}{Eigenvalue floor used to construct a feasible covariance start.}
\item{feasibility_tol}{Numerical tolerance for covariance feasibility.}
}
\value{A fitted-model list with parameter estimates and route diagnostics.}
