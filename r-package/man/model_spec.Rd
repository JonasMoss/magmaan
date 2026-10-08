\name{model_spec}
\alias{model_spec}
\title{Construct a lavaan-syntax model specification}
\description{Constructs the model structure, names and start hints used by the lab fitting interfaces.}
\usage{
model_spec(syntax, ..., auto_var = TRUE, auto_cov_lv_x = TRUE, auto_cov_y = FALSE,
    orthogonal = FALSE, auto_fix_first = TRUE, marker = NULL, auto_fix_single = TRUE, std_lv = FALSE,
    effect_coding = FALSE, fixed_x = TRUE, meanstructure = "default", model_type = c("sem",
        "growth"), ordered = NULL, parameterization = "delta", group = NULL, group_labels = NULL,
    group_equal = NULL, group_partial = NULL)
}
\arguments{
\item{syntax}{A single model-syntax string.}
\item{...}{Additional options forwarded to model construction or the underlying named operation.}
\item{auto_var}{Automatically add variances for model variables.}
\item{auto_cov_lv_x}{Automatically add covariances among exogenous latent variables.}
\item{auto_cov_y}{Automatically add covariances among dependent variables.}
\item{orthogonal}{Fix covariances among latent variables to zero.}
\item{auto_fix_first}{Use a marker loading to identify each latent variable.}
\item{marker}{Named character vector mapping latent names to marker indicators. With auto_fix_first, select the first matching loading or fall back to the first loading; preserve row order and apply user modifiers normally.}
\item{auto_fix_single}{Fix the residual variance of a single indicator to zero.}
\item{std_lv}{Identify latent scales by unit latent variances.}
\item{effect_coding}{Effect-code loadings or intercepts when supported.}
\item{fixed_x}{Treat observed exogenous variables as fixed.}
\item{meanstructure}{Whether to include intercepts and means; model_spec also accepts the default convention.}
\item{model_type}{Model construction mode: sem or growth.}
\item{ordered}{Names of ordinal observed variables.}
\item{parameterization}{Ordinal model parameterization; currently delta.}
\item{group}{Grouping-variable name or group identifiers, as appropriate for the data.}
\item{group_labels}{Explicit group labels in block order.}
\item{group_equal}{Parameter classes constrained equal across groups.}
\item{group_partial}{Named exceptions to the group equality constraints.}
}
\value{A model specification containing the projected parameter table and construction metadata.}
