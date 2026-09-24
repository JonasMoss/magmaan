#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(SuppDists)
})

# Oracle strategy. SuppDists::JohnsonFit (Hill AS99/AS100) fits the Johnson
# shape pair (gamma, delta) from (skewness, excess kurtosis); that fit is only
# ~1e-3 accurate and its xi/lambda use a quirky mean offset. So we treat only
# the returned shape (gamma, delta, type) as the oracle, build xi/lambda
# ourselves to hit a chosen (mean, sd), and record the *realized* moments of
# the resulting distribution computed by high-accuracy quadrature. magmaan is
# then asked to match those realized moments, so both describe the identical
# distribution and agree to solver precision. qJohnson supplies the independent
# quantile oracle. magmaan's johnson_type integers: 1 = SL, 2 = SU, 3 = SB.

probabilities <- c(0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)

type_to_int <- c(SL = 1L, SU = 2L, SB = 3L)

# Inverse of the Johnson normalizing transform: x_raw = g^{-1}((z - gamma)/delta).
g_inverse <- function(type, y) {
  switch(type,
    SU = sinh(y),
    SB = 1 / (1 + exp(-y)),
    SL = exp(y),
    stop("unsupported Johnson type for raw transform")
  )
}

# Central moments / standardized shape of the raw transform under z ~ N(0, 1).
raw_summary <- function(type, gamma, delta) {
  f <- function(z, k) g_inverse(type, (z - gamma) / delta)^k * dnorm(z)
  m <- sapply(1:4, function(k)
    integrate(f, -40, 40, k = k, rel.tol = 1e-13, subdivisions = 4000)$value)
  variance <- m[2] - m[1]^2
  sd <- sqrt(variance)
  mu3 <- m[3] - 3 * m[1] * m[2] + 2 * m[1]^3
  mu4 <- m[4] - 4 * m[1] * m[3] + 6 * m[1]^2 * m[2] - 3 * m[1]^4
  list(raw_mean = m[1], raw_sd = sd,
       skewness = mu3 / sd^3, excess_kurtosis = mu4 / variance^2 - 3)
}

cases <- list(
  list(id = "su_positive_skew", skewness = 0.8, excess_kurtosis = 1.5,
       mean = 0, sd = 1),
  list(id = "su_negative_skew", skewness = -0.8, excess_kurtosis = 1.5,
       mean = 2, sd = 3),
  list(id = "su_high_kurtosis", skewness = 0.5, excess_kurtosis = 3.0,
       mean = -1, sd = 0.5),
  list(id = "su_symmetric_heavy_tail", skewness = 0.0, excess_kurtosis = 2.0,
       mean = 0, sd = 1),
  list(id = "sb_symmetric_platykurtic", skewness = 0.0, excess_kurtosis = -0.8,
       mean = 0, sd = 1),
  list(id = "sb_skewed", skewness = 0.3, excess_kurtosis = -0.6,
       mean = 1, sd = 2)
)

out <- lapply(cases, function(case) {
  # Shape fit is invariant to the supplied mean/variance, so request the
  # standardized moments: t = c(mean, variance, mu3, mu4) with var = 1.
  shape <- JohnsonFit(
    c(0, 1, case$skewness, case$excess_kurtosis + 3),
    moment = "use"
  )
  type <- shape$type
  summary <- raw_summary(type, shape$gamma, shape$delta)

  lambda <- case$sd / summary$raw_sd
  xi <- case$mean - lambda * summary$raw_mean
  parms <- list(gamma = shape$gamma, delta = shape$delta,
                xi = xi, lambda = lambda, type = type)

  list(
    id = case$id,
    requested_skewness = case$skewness,
    requested_excess_kurtosis = case$excess_kurtosis,
    mean = case$mean,
    sd = case$sd,
    skewness = summary$skewness,
    excess_kurtosis = summary$excess_kurtosis,
    johnson_type = unname(type_to_int[[type]]),
    johnson_gamma = shape$gamma,
    johnson_delta = shape$delta,
    probabilities = probabilities,
    quantiles = unname(qJohnson(probabilities, parms))
  )
})

suppdists_version <- as.character(utils::packageVersion("SuppDists"))

fixture <- list(
  `_meta` = list(
    format_version = 1,
    fixture_kind = "sim.johnson_moment_match",
    oracle = paste0(
      "SuppDists ", suppdists_version,
      " JohnsonFit/qJohnson (shape + quantiles); realized moments via R integrate"
    )
  ),
  cases = out
)

args_file <- sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))][1])
script_dir <- dirname(normalizePath(args_file, mustWork = TRUE))
root <- normalizePath(file.path(script_dir, "..", "fixtures"), mustWork = TRUE)
dir.create(file.path(root, "sim"), showWarnings = FALSE)
write_json(
  fixture,
  file.path(root, "sim", "johnson_moment_match.json"),
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)
