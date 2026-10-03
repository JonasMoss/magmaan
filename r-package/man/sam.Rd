\name{sam}
\alias{sam}
\title{Structural after measurement estimation}
\description{Composes separate measurement and structural fitting using the selected mapping and covariance convention.}
\usage{
sam(model, data, method = c("local", "global"), mapping = c("ml", "gls", "uls"),
    se = c("twostep", "twostep.robust", "standard", "none"), lambda_correction = TRUE,
    alpha_correction = 0L, meanstructure = FALSE, mm_optimizer = "nlopt-lbfgs", struc_optimizer = mm_optimizer,
    mm_control = NULL, struc_control = NULL, raw_data = NULL, missing = c("listwise",
        "error"), ...)
}
\arguments{
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{data}{Observed data frame, named matrix or sample-moment list compatible with the model. FIML requires raw observations.}
\item{method}{Method selector; see the available choices in the usage signature and details.}
\item{mapping}{SAM measurement-to-structural mapping: ml, gls or uls.}
\item{se}{Standard-error convention. fit_model is estimate-only and accepts only "none"; SAM and est_change expose the choices shown in their signatures.}
\item{lambda_correction}{Apply the SAM loading correction.}
\item{alpha_correction}{SAM alpha-correction mode.}
\item{meanstructure}{Whether to include intercepts and means; model_spec also accepts the default convention.}
\item{mm_optimizer}{Optimizer for the SAM measurement model.}
\item{struc_optimizer}{Optimizer for the SAM structural model.}
\item{mm_control}{Control list for the SAM measurement fit.}
\item{struc_control}{Control list for the SAM structural fit.}
\item{raw_data}{Optional raw observations needed for robust SAM inference.}
\item{missing}{Missing-data handling: listwise deletion, rejection, or pairwise moments where supported. FIML retains missing observations.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
}
\value{A fitted SAM result with measurement and structural components.}

\note{The SE/type defaults standardize SAM or case-change diagnostics. They do not assert a generic misspecification-robust inference law; explicit robust or estimated-weight choices remain available within their supported scope.}
