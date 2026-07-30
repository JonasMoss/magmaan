#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(magmaan)
})

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(normalizePath(sub("^--file=", "", file_arg[[1L]]),
                         mustWork = TRUE))
  }
  normalizePath("tests/tools/regen_psd_ml_corpus_geometries.R",
                mustWork = TRUE)
}

repo_root <- function(start = dirname(script_path())) {
  path <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(path, "CMakeLists.txt")) &&
        file.exists(file.path(path, "r-package", "DESCRIPTION"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) {
      stop("could not find magmaan repository root", call. = FALSE)
    }
    path <- parent
  }
}

root <- repo_root()
output <- file.path(
  root, "tests", "fixtures", "psd_ml", "corpus_geometries.json")

targets <- data.frame(
  set = c("little", "little", "geiser", "geiser"),
  id = c(
    "ch8_fig3a_unconstrained_4wave_negaff",
    "ch8_fig4c_gc_linear_4wave_negaff",
    "cfa_second_order",
    "growth_quadratic"
  ),
  geometry = c(
    "same_fit_reallocation",
    "material_constrained_optimum",
    "negative_structural_disturbance",
    "joint_indefinite_covariance"
  ),
  stringsAsFactors = FALSE
)

source_path <- function(set) {
  switch(
    set,
    little = file.path(
      root, "tests", "fixtures", "little", "continuous_reference.json"),
    geiser = file.path(
      root, "tests", "fixtures", "geiser", "gls_reference.json"),
    stop("unknown source set: ", set, call. = FALSE)
  )
}

source_cases <- lapply(
  unique(targets$set),
  function(set) read_json(source_path(set), simplifyVector = FALSE)$cases)
names(source_cases) <- unique(targets$set)

scalar <- function(x, default = NULL) {
  if (is.null(x) || !length(x)) default else
    unlist(x, use.names = FALSE)[[1L]]
}

matrix_rows <- function(x) {
  if (is.matrix(x)) return(unname(x))
  do.call(rbind, lapply(x, function(row) as.numeric(unlist(row))))
}

find_case <- function(set, id) {
  cases <- source_cases[[set]]
  hit <- which(vapply(
    cases, function(x) identical(as.character(scalar(x$id, "")), id),
    logical(1)))
  if (length(hit) != 1L) {
    stop("expected one source case for ", set, "::", id, call. = FALSE)
  }
  cases[[hit]]
}

with_oracle_start <- function(spec, oracle) {
  if (is.null(oracle) || is.null(oracle$theta) ||
      !is.list(oracle$theta) || !length(oracle$theta) ||
      !is.list(oracle$theta[[1L]])) {
    return(spec)
  }
  out <- spec
  for (z in oracle$theta) {
    if (is.null(z$lhs) || is.null(z$op) || is.null(z$rhs) ||
        is.null(z$est)) {
      next
    }
    hit <- which(
      out$partable$lhs == scalar(z$lhs, "") &
      out$partable$op == scalar(z$op, "") &
      out$partable$rhs == scalar(z$rhs, "") &
      out$partable$group == as.integer(scalar(z$group, 1L)) &
      out$partable$free > 0L
    )
    if (length(hit) == 1L) {
      out$partable$ustart[hit] <- as.numeric(scalar(z$est, NA_real_))
    }
  }
  out
}

with_fit_start <- function(spec, fit) {
  out <- spec
  rows <- which(out$partable$free > 0L)
  out$partable$ustart[rows] <- fit$theta[out$partable$free[rows]]
  out
}

block_min <- function(blocks) {
  if (!length(blocks)) return(NA_real_)
  min(vapply(blocks, function(x) x$min_eigenvalue, numeric(1)))
}

fit_case <- function(set, id, geometry) {
  x <- find_case(set, id)
  S <- matrix_rows(x$sample_cov)
  ov_names <- as.character(unlist(x$ov_names, use.names = FALSE))
  dimnames(S) <- list(ov_names, ov_names)
  mean <- as.numeric(unlist(x$sample_mean, use.names = FALSE))
  if (!length(mean) || all(!is.finite(mean))) {
    mean <- NULL
  } else {
    names(mean) <- ov_names
  }
  stats <- list(
    S = list(S),
    mean = if (is.null(mean)) NULL else list(mean),
    nobs = as.integer(x$n_obs)
  )
  spec <- model_spec(
    as.character(x$model),
    fixed_x = isTRUE(x$fixed_x),
    meanstructure = isTRUE(x$meanstructure)
  )
  oracle_spec <- with_oracle_start(spec, x$lavaan)
  ordinary <- suppressWarnings(magmaan_core$fit_ml(
    oracle_spec, stats, optimizer = "nlopt-lbfgs",
    control = list(max_iter = 10000L, ftol = 1e-12, gtol = 1e-8)
  ))
  psd <- suppressWarnings(frontier_fit_ml_psd(
    with_fit_start(spec, ordinary), stats, optimizer = "nlopt-slsqp",
    control = list(max_iter = 10000L, ftol = 1e-12, gtol = 1e-8)
  ))
  if (!isTRUE(ordinary$converged) || !isTRUE(psd$converged) ||
      isTRUE(ordinary$diagnostics$admissibility$admissible) ||
      !isTRUE(psd$diagnostics$admissibility$admissible)) {
    stop("unexpected fit classification for ", set, "::", id,
         call. = FALSE)
  }

  focus_keys <- switch(
    geometry,
    same_fit_reallocation = list(
      c("NegT1", "~~", "NegT1"),
      c("NegLevl", "~~", "NegLevl")
    ),
    material_constrained_optimum = list(
      c("NegT1", "~~", "NegT1"),
      c("NegLevl", "~~", "NegLevl"),
      c("NegLin", "~~", "NegLin")
    ),
    negative_structural_disturbance = list(
      c("KFT_N", "~~", "KFT_N")
    ),
    joint_indefinite_covariance = list(
      c("a4", "~~", "a4")
    )
  )
  focus <- lapply(focus_keys, function(key) {
    hit <- which(
      ordinary$partable$lhs == key[[1L]] &
      ordinary$partable$op == key[[2L]] &
      ordinary$partable$rhs == key[[3L]] &
      ordinary$partable$group == 1L
    )
    if (length(hit) != 1L) {
      stop("focus row not unique: ", paste(key, collapse = " "),
           call. = FALSE)
    }
    row <- hit[[1L]]
    list(
      lhs = key[[1L]],
      op = key[[2L]],
      rhs = key[[3L]],
      ordinary = unname(ordinary$partable$est[[row]]),
      psd = unname(psd$partable$est[[row]])
    )
  })

  ordinary_audit <- ordinary$diagnostics$admissibility
  psd_audit <- psd$diagnostics$admissibility
  list(
    set = set,
    id = id,
    geometry = geometry,
    model = as.character(x$model),
    fixed_x = isTRUE(x$fixed_x),
    meanstructure = isTRUE(x$meanstructure),
    observed_names = ov_names,
    sample_cov = unname(S),
    sample_mean = if (is.null(mean)) NULL else unname(mean),
    n_obs = as.integer(x$n_obs),
    ordinary_start = unname(ordinary$theta),
    expected = list(
      ordinary_theta = unname(ordinary$theta),
      psd_theta = unname(psd$theta),
      ordinary_fmin = unname(ordinary$fmin),
      psd_fmin = unname(psd$fmin),
      lr_scale_increase =
        2 * as.numeric(x$n_obs) * (psd$fmin - ordinary$fmin),
      ordinary_theta_min = block_min(ordinary_audit$theta),
      ordinary_psi_min = block_min(ordinary_audit$psi),
      psd_theta_min = block_min(psd_audit$theta),
      psd_psi_min = block_min(psd_audit$psi),
      max_abs_parameter_change =
        max(abs(psd$theta - ordinary$theta)),
      focus = focus
    )
  )
}

cases <- lapply(seq_len(nrow(targets)), function(i) {
  message("[", i, "/", nrow(targets), "] ",
          targets$set[[i]], "::", targets$id[[i]])
  fit_case(targets$set[[i]], targets$id[[i]], targets$geometry[[i]])
})

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write_json(
  list(
    `_meta` = list(
      schema = "magmaan.psd_ml_corpus_geometries/1",
      description = paste(
        "Compact deterministic model/data and ordinary/PSD-ML regression",
        "summaries extracted from the advisory 97-case corpus audit."
      ),
      generated_by = "tests/tools/regen_psd_ml_corpus_geometries.R",
      source_fixtures = c(
        "tests/fixtures/little/continuous_reference.json",
        "tests/fixtures/geiser/gls_reference.json"
      ),
      magmaan_version = as.character(packageVersion("magmaan"))
    ),
    cases = cases
  ),
  output,
  auto_unbox = TRUE,
  digits = 16,
  pretty = TRUE,
  na = "null"
)
message("Wrote ", normalizePath(output))
