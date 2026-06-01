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

cvine3_points <- matrix(
  c(
    0.11, 0.22, 0.33,
    0.41, 0.52, 0.63,
    0.73, 0.14, 0.86,
    0.91, 0.82, 0.17
  ),
  ncol = 3,
  byrow = TRUE
)

cvine3_cases <- list(
  list(
    id = "frank_root0_mixed",
    copula_01 = list(family = "frank", rotation = 0L, parameters = 5.0),
    copula_02 = list(family = "frank", rotation = 0L, parameters = -5.0),
    copula_12_given_0 = list(family = "frank", rotation = 0L, parameters = 3.0)
  ),
  list(
    id = "clayton_gumbel_independent_conditional",
    copula_01 = list(family = "clayton", rotation = 0L, parameters = 2.0),
    copula_02 = list(family = "gumbel", rotation = 0L, parameters = 2.0),
    copula_12_given_0 = list(family = "indep", rotation = 0L, parameters = numeric())
  )
)

bicop_from_spec <- function(spec) {
  bicop_dist(
    family = spec$family,
    rotation = spec$rotation,
    parameters = spec$parameters
  )
}

cvine3_oracle_case <- function(case) {
  # rvinecopulib's root-0 C-vine uses order c(3, 2, 1). In tree 1, the
  # pair-copula order is (0,2), then (0,1); tree 2 is (1,2 | 0).
  model <- vinecop_dist(
    pair_copulas = list(
      list(bicop_from_spec(case$copula_02), bicop_from_spec(case$copula_01)),
      list(bicop_from_spec(case$copula_12_given_0))
    ),
    structure = cvine_structure(c(3, 2, 1))
  )
  out <- inverse_rosenblatt(cvine3_points, model)
  rows <- lapply(seq_len(nrow(cvine3_points)), function(i) {
    list(
      independent_u = unname(cvine3_points[i, ]),
      copula_u = unname(out[i, ])
    )
  })
  list(
    id = case$id,
    structure = "cvine_root0_order_3_2_1",
    copula_01 = case$copula_01,
    copula_02 = case$copula_02,
    copula_12_given_0 = case$copula_12_given_0,
    points = rows
  )
}

cvine3_fixture <- list(
  `_meta` = list(
    format_version = 1,
    fixture_kind = "sim.cvine3_inverse_rosenblatt",
    oracle = paste0(
      "rvinecopulib ", rvinecopulib_version,
      " inverse_rosenblatt(vinecop_dist(..., cvine_structure(c(3,2,1))))"
    )
  ),
  cases = lapply(cvine3_cases, cvine3_oracle_case)
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
write_json(
  cvine3_fixture,
  file.path(root, "sim", "cvine3_inverse_rosenblatt.json"),
  auto_unbox = TRUE,
  digits = NA,
  pretty = TRUE
)
