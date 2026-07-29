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
  normalizePath("psd_ml_corpus_audit.R", mustWork = TRUE)
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

usage <- function() cat(
  "Usage: Rscript psd_ml_corpus_audit.R [--audit-first] [--output-dir PATH]\n\n",
  "Fit ordinary complete-data NTML to the checked-in continuous corpus\n",
  "fixtures, inspect primitive-covariance admissibility, and refit with\n",
  "PSD-ML. By default every case is refitted so interior equivalence is also\n",
  "checked. --audit-first refits only inadmissible ordinary solutions.\n",
  sep = "")

parse_args <- function(args) {
  out <- list(refit_all = TRUE, output_dir = NULL)
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--audit-first") {
      out$refit_all <- FALSE
    } else if (arg == "--output-dir") {
      i <- i + 1L
      if (i > length(args)) {
        stop("missing value after --output-dir", call. = FALSE)
      }
      out$output_dir <- args[[i]]
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  out
}

root <- repo_root()
opts <- parse_args(commandArgs(trailingOnly = TRUE))
output_dir <- opts$output_dir
if (is.null(output_dir)) {
  output_dir <- file.path(dirname(script_path()), "results")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

matrix_rows <- function(x) {
  if (is.matrix(x)) return(unname(x))
  do.call(rbind, lapply(x, function(row) as.numeric(unlist(row))))
}

scalar <- function(x, default = NULL) {
  if (is.null(x) || !length(x)) default else
    unlist(x, use.names = FALSE)[[1L]]
}

single_case <- function(set, x, covariance, mean = NULL, nobs = NULL,
                        ov_names = NULL, oracle = x$lavaan) {
  S <- matrix_rows(covariance)
  ov <- as.character(unlist(ov_names, use.names = FALSE))
  if (length(ov) == ncol(S)) dimnames(S) <- list(ov, ov)
  mu <- if (is.null(mean)) NULL else as.numeric(unlist(mean))
  if (!length(mu) || all(!is.finite(mu))) mu <- NULL
  if (!is.null(mu) && length(ov) == length(mu)) names(mu) <- ov
  list(
    set = set,
    id = as.character(scalar(x$id, scalar(x$case_id, ""))),
    family = as.character(scalar(x$family, scalar(x$model_kind, ""))),
    model = as.character(scalar(x$model, "")),
    fixed_x = isTRUE(scalar(x$fixed_x, FALSE)),
    meanstructure = isTRUE(scalar(x$meanstructure, FALSE)),
    model_type = if (identical(
      tolower(as.character(scalar(x$lavaan_function, ""))), "growth")) {
      "growth"
    } else {
      "sem"
    },
    stats = list(
      S = list(S),
      mean = if (is.null(mu)) NULL else list(mu),
      nobs = as.integer(scalar(nobs, scalar(x$n_obs, NA_integer_)))
    ),
    n_groups = 1L,
    oracle = oracle
  )
}

read_cases <- function(path) {
  read_json(file.path(root, path), simplifyVector = FALSE)$cases
}

cases <- list()
push <- function(x) cases[[length(cases) + 1L]] <<- x

fixture_files <- c(
  "tests/fixtures/little/continuous_reference.json",
  "tests/fixtures/newsom/continuous_reference.json",
  "tests/fixtures/geiser/gls_reference.json",
  "tests/fixtures/mplus_sem/continuous_reference.json",
  "tests/fixtures/paper_corpus/zxqvn_reference.json",
  "tests/fixtures/textbook_corpus/case_exports.json"
)

for (x in read_cases(fixture_files[[1L]])) {
  push(single_case("little", x, x$sample_cov, x$sample_mean,
                   x$n_obs, x$ov_names))
}
for (x in read_cases(fixture_files[[2L]])) {
  push(single_case("newsom", x, x$sample_cov, x$sample_mean,
                   x$n_obs, x$ov_names))
}
for (x in read_cases(fixture_files[[3L]])) {
  push(single_case("geiser", x, x$sample_cov, x$sample_mean,
                   x$n_obs, x$ov_names))
}
for (x in read_cases(fixture_files[[4L]])) {
  ml <- x$fits$ML
  if (is.null(ml)) next
  push(single_case("mplus_sem", x, ml$sample_cov, ml$sample_mean,
                   ml$n_obs, x$ov_names, oracle = ml))
}
for (x in read_cases(fixture_files[[5L]])) {
  ml <- x$fits$ML
  push(single_case("paper_corpus", x, ml$sample_cov, ml$sample_mean,
                   ml$n_obs, x$observed_variables, oracle = ml))
}
for (x in read_cases(fixture_files[[6L]])) {
  ng <- as.integer(scalar(x$data$n_groups, length(x$data$sample_cov)))
  S <- lapply(x$data$sample_cov, matrix_rows)
  means <- if (is.null(x$data$sample_mean)) NULL else
    lapply(x$data$sample_mean, function(z) as.numeric(unlist(z)))
  push(list(
    set = "kline_guo",
    id = as.character(scalar(x$case_id, "")),
    family = "measurement invariance",
    model = as.character(scalar(x$model, "")),
    fixed_x = isTRUE(scalar(x$model_options$fixed_x, FALSE)),
    meanstructure = isTRUE(
      scalar(x$model_options$meanstructure, FALSE)),
    model_type = "sem",
    stats = list(
      S = S,
      mean = means,
      nobs = as.integer(unlist(x$data$n_obs))
    ),
    n_groups = ng,
    oracle = x$lavaan
  ))
}

make_spec <- function(x) {
  args <- list(
    syntax = x$model,
    fixed_x = x$fixed_x,
    meanstructure = x$meanstructure,
    model_type = x$model_type
  )
  if (x$n_groups > 1L) {
    args$group <- ".fixture_group"
    args$group_labels <- paste0("g", seq_len(x$n_groups))
  }
  do.call(model_spec, args)
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
    group <- as.integer(scalar(z$group, 1L))
    hit <- which(
      out$partable$lhs == scalar(z$lhs, "") &
      out$partable$op == scalar(z$op, "") &
      out$partable$rhs == scalar(z$rhs, "") &
      out$partable$group == group &
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
  free <- out$partable$free
  rows <- which(free > 0L)
  out$partable$ustart[rows] <- fit$theta[free[rows]]
  out
}

ordinary_fit <- function(spec, x) {
  primary <- tryCatch(
    suppressWarnings(magmaan_core$fit_ml(
      spec, x$stats, optimizer = "nlopt-lbfgs",
      control = list(max_iter = 5000L, gtol = 1e-8))),
    error = identity
  )
  if (!inherits(primary, "error")) {
    return(list(fit = primary, route = "nlopt-lbfgs"))
  }

  oracle_spec <- with_oracle_start(spec, x$oracle)
  oracle_start <- tryCatch(
    suppressWarnings(magmaan_core$fit_ml(
      oracle_spec, x$stats, optimizer = "nlopt-lbfgs",
      control = list(max_iter = 5000L, gtol = 1e-8))),
    error = identity
  )
  if (!inherits(oracle_start, "error")) {
    return(list(fit = oracle_start, route = "nlopt-lbfgs-oracle-start"))
  }

  retry <- suppressWarnings(magmaan_core$fit_ml(
    oracle_spec, x$stats, optimizer = "nlopt-slsqp",
    control = list(max_iter = 10000L, gtol = 1e-8)))
  list(fit = retry, route = "nlopt-slsqp-retry")
}

block_min <- function(blocks) {
  if (!length(blocks)) return(NA_real_)
  min(vapply(blocks, function(z) z$min_eigenvalue, numeric(1)),
      na.rm = TRUE)
}

bad_kind <- function(audit) {
  kinds <- character()
  if (length(audit$theta) &&
      any(!vapply(audit$theta, `[[`, logical(1), "psd"))) {
    kinds <- c(kinds, "Theta")
  }
  if (length(audit$psi) &&
      any(!vapply(audit$psi, `[[`, logical(1), "psd"))) {
    kinds <- c(kinds, "Psi")
  }
  if (!isTRUE(audit$implied_sigma_pd)) kinds <- c(kinds, "Sigma")
  paste(kinds, collapse = "+")
}

largest_parameter <- function(fit, delta) {
  if (!length(delta) || all(!is.finite(delta))) return("")
  k <- which.max(abs(delta))
  rows <- which(fit$partable$free == k)
  if (!length(rows)) return(paste0("free#", k))
  r <- rows[[1L]]
  paste0(fit$partable$lhs[[r]], fit$partable$op[[r]],
         fit$partable$rhs[[r]], "[g", fit$partable$group[[r]], "]")
}

summary_rows <- vector("list", length(cases))
parameter_rows <- list()
failure_rows <- list()

for (i in seq_along(cases)) {
  x <- cases[[i]]
  message(sprintf("[%d/%d] %s::%s", i, length(cases), x$set, x$id))
  result <- tryCatch({
    spec <- make_spec(x)
    ordinary_result <- ordinary_fit(spec, x)
    ordinary <- ordinary_result$fit
    audit <- ordinary$diagnostics$admissibility
    do_refit <- opts$refit_all || !isTRUE(audit$admissible)
    psd <- if (do_refit) {
      suppressWarnings(frontier_fit_ml_psd(
        with_fit_start(spec, ordinary), x$stats,
        optimizer = "nlopt-slsqp",
        control = list(max_iter = 5000L, gtol = 1e-8)))
    } else {
      NULL
    }
    psd_audit <- if (is.null(psd)) NULL else
      psd$diagnostics$admissibility
    delta <- if (is.null(psd)) numeric() else psd$theta - ordinary$theta

    if (!isTRUE(audit$admissible) && !is.null(psd)) {
      po <- ordinary$partable
      pp <- psd$partable
      keep <- which(po$free > 0L)
      parameter_rows[[length(parameter_rows) + 1L]] <-
        data.frame(
          set = x$set, id = x$id, row = keep,
          lhs = po$lhs[keep], op = po$op[keep], rhs = po$rhs[keep],
          group = po$group[keep], free = po$free[keep],
          ordinary = po$est[keep], psd = pp$est[keep],
          change = pp$est[keep] - po$est[keep],
          stringsAsFactors = FALSE
        )
    }

    data.frame(
      set = x$set,
      id = x$id,
      family = x$family,
      p = ncol(x$stats$S[[1L]]),
      n_groups = x$n_groups,
      n_total = sum(x$stats$nobs),
      ordinary_route = ordinary_result$route,
      ordinary_converged = isTRUE(ordinary$converged),
      ordinary_admissible = isTRUE(audit$admissible),
      bad_kind = bad_kind(audit),
      theta_min_eigenvalue = block_min(audit$theta),
      psi_min_eigenvalue = block_min(audit$psi),
      ordinary_fmin = ordinary$fmin,
      psd_attempted = do_refit,
      psd_converged = if (is.null(psd)) NA else isTRUE(psd$converged),
      psd_admissible = if (is.null(psd)) NA else
        isTRUE(psd_audit$admissible),
      psd_theta_min_eigenvalue = if (is.null(psd)) NA_real_ else
        block_min(psd_audit$theta),
      psd_psi_min_eigenvalue = if (is.null(psd)) NA_real_ else
        block_min(psd_audit$psi),
      psd_fmin = if (is.null(psd)) NA_real_ else psd$fmin,
      fmin_increase = if (is.null(psd)) NA_real_ else
        psd$fmin - ordinary$fmin,
      lr_scale_increase = if (is.null(psd)) NA_real_ else
        2 * sum(x$stats$nobs) * (psd$fmin - ordinary$fmin),
      max_abs_parameter_change = if (!length(delta)) NA_real_ else
        max(abs(delta)),
      largest_changed_parameter = if (!length(delta)) "" else
        largest_parameter(ordinary, delta),
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    failure_rows[[length(failure_rows) + 1L]] <<- data.frame(
      set = x$set, id = x$id, error = conditionMessage(e),
      stringsAsFactors = FALSE)
    NULL
  })
  summary_rows[[i]] <- result
}

summary_rows <- Filter(Negate(is.null), summary_rows)
case_summary <- if (length(summary_rows)) {
  do.call(rbind, summary_rows)
} else {
  data.frame()
}
parameter_changes <- if (length(parameter_rows)) {
  do.call(rbind, parameter_rows)
} else {
  data.frame()
}
failures <- if (length(failure_rows)) {
  do.call(rbind, failure_rows)
} else {
  data.frame(set = character(), id = character(), error = character())
}

corpus_summary <- if (nrow(case_summary)) {
  total <- aggregate(
    rep(1L, nrow(case_summary)),
    list(set = case_summary$set),
    sum
  )
  names(total)[[2L]] <- "cases"
  bad <- aggregate(
    as.integer(!case_summary$ordinary_admissible),
    list(set = case_summary$set),
    sum
  )
  names(bad)[[2L]] <- "ordinary_inadmissible"
  out <- merge(total, bad, by = "set", sort = FALSE)
  out$percent_inadmissible <-
    100 * out$ordinary_inadmissible / out$cases
  out
} else {
  data.frame()
}

write.csv(case_summary, file.path(output_dir, "case_summary.csv"),
          row.names = FALSE, na = "")
write.csv(parameter_changes,
          file.path(output_dir, "parameter_changes.csv"),
          row.names = FALSE, na = "")
write.csv(corpus_summary, file.path(output_dir, "corpus_summary.csv"),
          row.names = FALSE, na = "")
write.csv(failures, file.path(output_dir, "failures.csv"),
          row.names = FALSE, na = "")

git_head <- tryCatch(
  system2("git", c("-C", root, "rev-parse", "HEAD"),
          stdout = TRUE, stderr = FALSE)[[1L]],
  error = function(e) ""
)
metadata <- data.frame(
  key = c("fixture_count", "refit_all", "git_head",
          "magmaan_version", "R_version", "fixture_files"),
  value = c(
    length(cases),
    opts$refit_all,
    git_head,
    as.character(packageVersion("magmaan")),
    R.version.string,
    paste(fixture_files, collapse = ";")
  ),
  stringsAsFactors = FALSE
)
write.csv(metadata, file.path(output_dir, "metadata.csv"),
          row.names = FALSE, na = "")

cat("\nCorpus summary:\n")
print(corpus_summary, row.names = FALSE)
cat("\nInadmissible ordinary fits:\n")
print(
  case_summary[!case_summary$ordinary_admissible,
               c("set", "id", "bad_kind",
                 "theta_min_eigenvalue", "psi_min_eigenvalue",
                 "lr_scale_increase", "max_abs_parameter_change",
                 "largest_changed_parameter")],
  row.names = FALSE
)
cat("\nWrote results to ", normalizePath(output_dir), "\n", sep = "")

valid <- !nrow(failures) &&
  nrow(case_summary) == length(cases) &&
  all(case_summary$ordinary_converged) &&
  all(case_summary$psd_converged[case_summary$psd_attempted]) &&
  all(case_summary$psd_admissible[case_summary$psd_attempted])
if (!valid) quit(save = "no", status = 1L)
