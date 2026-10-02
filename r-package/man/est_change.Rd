\name{est_change}
\alias{est_change}
\title{Summarize case-rerun parameter changes}
\description{Summarizes parameter changes from case_rerun under the selected standard-error convention.}
\usage{
est_change(rerun_out, parameters = NULL, se = c("standard", "robust.sem", "robust.huber.white"))
}
\arguments{
\item{rerun_out}{Case-rerun result from case_rerun.}
\item{parameters}{Optional parameter indices or names to retain.}
\item{se}{Standard-error convention. fit_model is estimate-only and accepts only "none"; SAM and est_change expose the choices shown in their signatures.}
}
\value{A parameter-change results table.}
