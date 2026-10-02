\name{df_to_fiml_data}
\alias{df_to_fiml_data}
\title{Prepare data: df_to_fiml_data}
\description{Converts observations into native sample moments or raw-data input for the corresponding fitting route.}
\usage{
df_to_fiml_data(x, model, group = NULL)
}
\arguments{
\item{x}{Input data frame; for mi_screen, a modification-index or score-test results table.}
\item{model}{Model syntax or a model specification; prepared objects are accepted only by their corresponding interfaces.}
\item{group}{Grouping-variable name or group identifiers, as appropriate for the data.}
}
\value{A prepared data list with observed-variable and group ordering.}
