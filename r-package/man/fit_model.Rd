\name{fit_model}
\alias{fit_model}
\title{Estimate a structural equation model}
\description{Fits the requested objective and returns estimates and diagnostics. Inference is composed explicitly after fitting.}
\usage{
fit_model(model, data, estimator = "ML", groups = NULL, ..., cluster = NULL, ordered = NULL,
    parameterization = "delta", missing = c("listwise", "error", "pairwise"), pd_gamma = c("overlap",
        "nominal"), se = "none", test = "none", W = NULL, optimizer = NULL, control = NULL,
    bounds = NULL, stage2_weight = "nt", dls_a = 0.5, stage1_regularization = NULL,
    psd = FALSE, covariance = NULL, barrier = NULL, weight = NULL, options = NULL)
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
\item{missing}{Missing-data handling: listwise deletion, rejection, or pairwise moments where supported. FIML retains missing observations.}
\item{pd_gamma}{Pairwise fourth-moment covariance convention: overlap or nominal.}
\item{se}{Standard-error convention. fit_model is estimate-only and accepts only "none"; SAM and est_change expose the choices shown in their signatures.}
\item{test}{Test convention. fit_model is estimate-only and accepts only "none".}
\item{W}{Fixed fitting-weight matrix, or a per-group list in model moment order.}
\item{optimizer}{Native optimizer name; NULL uses the route default.}
\item{control}{Named optimizer control list, including iteration limits, tolerances and optional starts.}
\item{bounds}{Lower and upper parameter-bound vectors or a named bounds preset; NULL leaves the route default.}
\item{stage2_weight}{Two-stage fitting weight selector, such as nt or adf.}
\item{dls_a}{DLS mixing scalar in [0, 1]: zero is normal theory and one is ADF.}
\item{stage1_regularization}{Optional Stage-1 regularization specification for ML2S.}
\item{psd}{Whether to constrain model covariance blocks to be positive semidefinite.}
\item{covariance}{Covariance-domain policy: unrestricted, psd or barrier.}
\item{barrier}{Barrier options with target and non-negative weight.}
\item{weight}{Weight selector or explicit fitting weight as required by the route.}
\item{options}{Named fitting-options list: preset, starts, optimizer, convergence and marker.
The lavaan-0.7.2 preset selects all four components. marker = "default" keeps
requested markers; marker = "lavaan-0.7.2" applies the weak-marker rule for
complete continuous ML and retries the original model if the switched fit is
rejected. The fit retains the actual model and fitting$marker_switch (lv, old,
new, signed r_old and r_new rounded to three decimals, reverted). FIML and DWLS
currently resolve the preset marker to "default"; explicit adaptation there is
unsupported. Nested inference and refit_from_null explicitly refuse switched
marker coordinates. Components may also be supplied in control$fitting_options.}
}
\value{A magmaan_fit list with parameter estimates, fitting options, convergence and admissibility diagnostics.}
