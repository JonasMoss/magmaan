#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(rvinecopulib)
})

families <- list(
  list(id = "independence", family = "indep", theta = numeric()),
  list(id = "clayton_theta2", family = "clayton", theta = 2.0),
  list(id = "gumbel_theta2", family = "gumbel", theta = 2.0),
  list(id = "frank_theta5", family = "frank", theta = 5.0),
  list(id = "frank_theta_neg5", family = "frank", theta = -5.0),
  list(id = "joe_theta2", family = "joe", theta = 2.0)
)

points <- list(
  list(u = 0.10, v = 0.20, p = 0.30),
  list(u = 0.20, v = 0.70, p = 0.60),
  list(u = 0.75, v = 0.35, p = 0.80),
  list(u = 0.92, v = 0.88, p = 0.15)
)

oracle_case <- function(fam) {
  rows <- lapply(points, function(pt) {
    uv <- matrix(c(pt$u, pt$v), ncol = 2)
    up <- matrix(c(pt$u, pt$p), ncol = 2)
    list(
      u = pt$u,
      v = pt$v,
      p = pt$p,
      conditional_cdf = unname(hbicop(
        uv,
        cond_var = 1,
        family = fam$family,
        rotation = 0,
        parameters = fam$theta
      )),
      conditional_quantile = unname(hbicop(
        up,
        cond_var = 1,
        family = fam$family,
        rotation = 0,
        parameters = fam$theta,
        inverse = TRUE
      ))
    )
  })
  list(
    id = fam$id,
    family = fam$family,
    rotation = 0L,
    parameters = unname(fam$theta),
    points = rows
  )
}

rvinecopulib_version <- as.character(utils::packageVersion("rvinecopulib"))

fixture <- list(
  `_meta` = list(
    format_version = 1,
    fixture_kind = "sim.bivariate_copula_hfunc",
    oracle = paste0(
      "rvinecopulib ", rvinecopulib_version,
      " hbicop(cond_var=1, rotation=0)"
    )
  ),
  cases = lapply(families, oracle_case)
)

args_file <- sub("^--file=", "", commandArgs(FALSE)[grep("^--file=", commandArgs(FALSE))][1])
script_dir <- dirname(normalizePath(args_file, mustWork = TRUE))
root <- normalizePath(file.path(script_dir, "..", "fixtures"), mustWork = TRUE)
dir.create(file.path(root, "sim"), showWarnings = FALSE)
write_json(
  fixture,
  file.path(root, "sim", "bivariate_copula_hfunc.json"),
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)
