\name{mi_screen}
\alias{mi_screen}
\title{Screen modification diagnostics by significance and effect}
\description{Adds multiplicity-adjusted p-values and a verdict based on significance and absolute effect size. Missing inputs produce missing verdicts.}
\usage{
mi_screen(x, p = NULL, effect = NULL, alpha = 0.05, effect_min = 0.1, adjust = "BH")
}
\arguments{
\item{x}{Input data frame; for mi_screen, a modification-index or score-test results table.}
\item{p}{Name of the p-value column; NULL selects a recognized robust column.}
\item{effect}{Name of the effect-size column; NULL selects a recognized robust column.}
\item{alpha}{Significance threshold after multiplicity adjustment.}
\item{effect_min}{Minimum absolute effect size considered substantively large.}
\item{adjust}{Multiplicity method passed to stats::p.adjust.}
}
\value{The input table with adjusted p-values and verdict (free, trivial, underpowered or ok).}
