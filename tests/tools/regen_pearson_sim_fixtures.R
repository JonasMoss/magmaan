#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(PearsonDS)
})

probabilities <- c(0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)

cases <- list(
  list(id = "normal_type0", mean = 0, sd = 1, skewness = 0,
       excess_kurtosis = 0, supported = TRUE),
  list(id = "beta_type1", mean = 0, sd = 1, skewness = 1,
       excess_kurtosis = 1, supported = TRUE),
  list(id = "symmetric_beta_type2", mean = 0, sd = 1, skewness = 0,
       excess_kurtosis = -1, supported = TRUE),
  list(id = "gamma_type3", mean = 0, sd = 1, skewness = 2,
       excess_kurtosis = 6, supported = TRUE),
  list(id = "beta_prime_type6", mean = 0, sd = 1, skewness = 3,
       excess_kurtosis = 17, supported = TRUE),
  list(id = "student_type7", mean = 0, sd = 1, skewness = 0,
       excess_kurtosis = 1, supported = TRUE),
  list(id = "negative_skew_beta_type1", mean = 0, sd = 1, skewness = -1,
       excess_kurtosis = 1, supported = TRUE),
  list(id = "type4_unsupported", mean = 0, sd = 1, skewness = 1,
       excess_kurtosis = 4, supported = FALSE)
)

param_vector <- function(params) {
  type <- params$type
  switch(as.character(type),
    "0" = c(params$mean, params$sd, 0, 0),
    "1" = c(params$a, params$b, params$location, params$scale),
    "2" = c(params$a, params$location, params$scale, 0),
    "3" = c(params$shape, params$location, params$scale, 0),
    "4" = c(params$m, params$nu, params$location, params$scale),
    "5" = c(params$shape, params$location, params$scale, 0),
    "6" = c(params$a, params$b, params$location, params$scale),
    "7" = c(params$df, params$location, params$scale, 0),
    stop("unknown Pearson type")
  )
}

out <- lapply(cases, function(case) {
  raw_kurtosis <- case$excess_kurtosis + 3
  params <- pearsonFitM(
    mean = case$mean,
    variance = case$sd^2,
    skewness = case$skewness,
    kurtosis = raw_kurtosis
  )
  entry <- c(case, list(
    raw_kurtosis = raw_kurtosis,
    pearson_type = unname(params$type),
    pearson_params = unname(param_vector(params))
  ))
  if (isTRUE(case$supported)) {
    entry$probabilities <- probabilities
    entry$quantiles <- unname(qpearson(probabilities, params = params))
  }
  entry
})

fixture <- list(
  `_meta` = list(
    format_version = 1,
    fixture_kind = "sim.pearson_moment_match",
    oracle = "PearsonDS 1.3.2 pearsonFitM/qpearson"
  ),
  cases = out
)

args_file <- sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))][1])
script_dir <- dirname(normalizePath(args_file, mustWork = TRUE))
root <- normalizePath(file.path(script_dir, "..", "fixtures"), mustWork = TRUE)
dir.create(file.path(root, "sim"), showWarnings = FALSE)
write_json(
  fixture,
  file.path(root, "sim", "pearson_moment_match.json"),
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)
