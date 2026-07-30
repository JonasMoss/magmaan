#!/usr/bin/env Rscript

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  script <- if (length(file_arg)) {
    normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    normalizePath("run_experiment.R", mustWork = FALSE)
  }
  file.path(dirname(dirname(script)), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)
set_single_threaded_math()
require_pkg("magmaan")
suppressPackageStartupMessages(library(magmaan))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot|--full] [options]\n\n",
  "Time complete-data normal-theory ML under ordinary, direct-PSD, and\n",
  "audit-then-refit policies. SLSQP is always included; --include-ipopt\n",
  "adds matched direct and audit-triggered IPOPT policies.\n\n",
  "Profiles:\n",
  "  --smoke   4 sentinel cases x 3 repetitions (default)\n",
  "  --pilot   9 cases through p=12 x 30 repetitions\n",
  "  --full    10 cases through p=24 x 100 repetitions\n\n",
  "Options:\n",
  "  --repeats N --warmups N --seed-base N\n",
  "  --cases ID1,ID2 --results-dir PATH --include-ipopt\n",
  sep = "")

parse_args <- function(args) {
  out <- list(
    profile = "smoke",
    repeats = NULL,
    warmups = NULL,
    seed_base = 20260729L,
    cases = NULL,
    results_dir = NULL,
    include_ipopt = FALSE
  )
  i <- 1L
  take <- function() {
    i <<- i + 1L
    if (i > length(args)) {
      stop("missing value after ", args[[i - 1L]], call. = FALSE)
    }
    args[[i]]
  }
  while (i <= length(args)) {
    arg <- args[[i]]
    if (arg %in% c("-h", "--help")) {
      usage()
      quit(save = "no", status = 0L)
    } else if (arg == "--smoke") out$profile <- "smoke"
    else if (arg == "--pilot") out$profile <- "pilot"
    else if (arg == "--full") out$profile <- "full"
    else if (arg == "--repeats") out$repeats <- as.integer(take())
    else if (arg == "--warmups") out$warmups <- as.integer(take())
    else if (arg == "--seed-base") out$seed_base <- as.integer(take())
    else if (arg == "--cases") out$cases <- parse_csv_arg(take())
    else if (arg == "--results-dir") out$results_dir <- take()
    else if (arg == "--include-ipopt") out$include_ipopt <- TRUE
    else stop("unknown argument: ", arg, call. = FALSE)
    i <- i + 1L
  }
  defaults <- switch(
    out$profile,
    smoke = c(repeats = 3L, warmups = 1L),
    pilot = c(repeats = 30L, warmups = 3L),
    full = c(repeats = 100L, warmups = 5L)
  )
  if (is.null(out$repeats)) out$repeats <- defaults[["repeats"]]
  if (is.null(out$warmups)) out$warmups <- defaults[["warmups"]]
  if (any(!is.finite(c(out$repeats, out$warmups, out$seed_base))) ||
      out$repeats < 1L || out$warmups < 0L || out$seed_base < 1L) {
    stop("repeats and seed must be positive; warmups must be nonnegative",
         call. = FALSE)
  }
  out
}

exact_data <- function(S, n = 200L) {
  p <- ncol(S)
  H <- stats::contr.helmert(n)[, seq_len(p), drop = FALSE]
  Q <- sweep(H, 2L, sqrt(colSums(H^2)), "/")
  X <- sqrt(n) * Q %*% chol(S)
  X <- as.data.frame(X)
  names(X) <- paste0("x", seq_len(p))
  X
}

one_factor_fixture <- function(p) {
  lambda <- seq(0.80, 0.60, length.out = p)
  S <- tcrossprod(lambda) + diag(1 - lambda^2)
  list(
    case_id = paste0("interior_cfa_p", p),
    family = "interior scaling",
    geometry = "interior",
    expected_ordinary_admissible = TRUE,
    model = paste0("f =~ ", paste0("x", seq_len(p), collapse = " + ")),
    S = S
  )
}

fixture_catalog <- function() {
  joint_S <- matrix(c(
     1.9,  0.9,  0.9,
     0.9,  1.9, -0.9,
     0.9, -0.9,  1.9
  ), 3L, 3L, byrow = TRUE)
  fixtures <- list(
    one_factor_fixture(3L),
    one_factor_fixture(6L),
    one_factor_fixture(12L),
    one_factor_fixture(24L),
    list(
      case_id = "near_boundary_cfa",
      family = "boundary",
      geometry = "near boundary",
      expected_ordinary_admissible = FALSE,
      model = "f =~ x1 + x2 + x3",
      S = {
        epsilon <- 1e-6
        a <- 0.5
        c23 <- a^2 / (1 + epsilon)
        matrix(c(
          1, a, a,
          a, 1, c23,
          a, c23, 1
        ), 3L, 3L, byrow = TRUE)
      }
    ),
    list(
      case_id = "heywood_residual",
      family = "boundary",
      geometry = "negative residual variance",
      expected_ordinary_admissible = FALSE,
      model = "f =~ x1 + x2 + x3",
      S = matrix(c(
        1.0, 0.7, 0.7,
        0.7, 1.0, 0.3,
        0.7, 0.3, 1.0
      ), 3L, 3L, byrow = TRUE)
    ),
    list(
      case_id = "joint_indefinite_psi",
      family = "joint covariance",
      geometry = "indefinite Psi",
      expected_ordinary_admissible = FALSE,
      model = paste(
        "f1 =~ 1*x1",
        "f2 =~ 1*x2",
        "f3 =~ 1*x3",
        "x1 ~~ 0.9*x1",
        "x2 ~~ 0.9*x2",
        "x3 ~~ 0.9*x3",
        sep = "\n"
      ),
      S = joint_S
    ),
    list(
      case_id = "joint_indefinite_theta",
      family = "joint covariance",
      geometry = "indefinite Theta",
      expected_ordinary_admissible = FALSE,
      model = paste(
        "f1 =~ 1*x1",
        "f2 =~ 1*x2",
        "f3 =~ 1*x3",
        "f1 ~~ 0.9*f1",
        "f2 ~~ 0.9*f2",
        "f3 ~~ 0.9*f3",
        "f1 ~~ 0*f2",
        "f1 ~~ 0*f3",
        "f2 ~~ 0*f3",
        "x1 ~~ x1 + x2 + x3",
        "x2 ~~ x2 + x3",
        "x3 ~~ x3",
        sep = "\n"
      ),
      S = joint_S
    ),
    list(
      case_id = "negative_disturbance",
      family = "structural",
      geometry = "negative disturbance variance",
      expected_ordinary_admissible = FALSE,
      model = paste(
        "eta1 =~ 1*x1",
        "eta2 =~ 1*x2",
        "eta2 ~ eta1",
        "x1 ~~ 0.1*x1",
        "x2 ~~ 0.1*x2",
        sep = "\n"
      ),
      S = matrix(c(
        1.1, 0.9,
        0.9, 0.8
      ), 2L, 2L, byrow = TRUE)
    ),
    list(
      case_id = "shared_residual_control",
      family = "equality control",
      geometry = "interior equality",
      expected_ordinary_admissible = TRUE,
      model = paste(
        "f =~ x1 + x2 + x3",
        "x1 ~~ a*x1",
        "x2 ~~ a*x2",
        sep = "\n"
      ),
      S = matrix(c(
        1.0, 0.8, 0.8,
        0.8, 1.0, 0.3,
        0.8, 0.3, 1.0
      ), 3L, 3L, byrow = TRUE)
    )
  )
  names(fixtures) <- vapply(fixtures, `[[`, character(1), "case_id")
  fixtures
}

profile_cases <- function(profile) {
  switch(
    profile,
    smoke = c(
      "interior_cfa_p3", "heywood_residual",
      "joint_indefinite_psi", "negative_disturbance"
    ),
    pilot = c(
      "interior_cfa_p3", "interior_cfa_p6", "interior_cfa_p12",
      "near_boundary_cfa", "heywood_residual",
      "joint_indefinite_psi", "joint_indefinite_theta",
      "negative_disturbance", "shared_residual_control"
    ),
    full = names(fixture_catalog())
  )
}

prepare_fixture <- function(fixture, n = 200L) {
  model <- model_spec(fixture$model)
  data <- df_to_data(exact_data(fixture$S, n), model)
  c(fixture, list(
    p = ncol(fixture$S),
    n = n,
    model_object = model,
    data_object = data
  ))
}

ordinary_fit <- function(fixture) {
  suppressWarnings(magmaan_core$fit_ml(
    fixture$model_object, fixture$data_object,
    optimizer = "nlopt-lbfgs",
    control = list(max_iter = 5000L, gtol = 1e-8)
  ))
}

psd_fit <- function(fixture, model = fixture$model_object,
                    optimizer = "nlopt-slsqp") {
  suppressWarnings(frontier_fit_ml_psd(
    model, fixture$data_object,
    optimizer = optimizer,
    control = list(max_iter = 5000L, gtol = 1e-8)
  ))
}

with_fit_start <- function(model, fit) {
  out <- model
  free <- out$partable$free
  rows <- which(free > 0L)
  out$partable$ustart[rows] <- fit$theta[free[rows]]
  out
}

run_policy <- function(policy, fixture) {
  fallback_used <- FALSE
  ordinary_admissible <- NA
  ordinary <- NULL
  final <- switch(
    policy,
    ordinary = {
      ordinary <- ordinary_fit(fixture)
      ordinary_admissible <-
        isTRUE(ordinary$diagnostics$admissibility$admissible)
      ordinary
    },
    psd_direct = psd_fit(fixture),
    psd_direct_ipopt = psd_fit(fixture, optimizer = "ipopt"),
    audit_then_psd = {
      ordinary <- ordinary_fit(fixture)
      ordinary_admissible <-
        isTRUE(ordinary$diagnostics$admissibility$admissible)
      if (ordinary_admissible) {
        ordinary
      } else {
        fallback_used <- TRUE
        psd_fit(fixture, with_fit_start(fixture$model_object, ordinary))
      }
    },
    audit_then_psd_ipopt = {
      ordinary <- ordinary_fit(fixture)
      ordinary_admissible <-
        isTRUE(ordinary$diagnostics$admissibility$admissible)
      if (ordinary_admissible) {
        ordinary
      } else {
        fallback_used <- TRUE
        psd_fit(
          fixture, with_fit_start(fixture$model_object, ordinary),
          optimizer = "ipopt")
      }
    },
    stop("unknown policy: ", policy, call. = FALSE)
  )
  list(
    fit = final,
    ordinary = ordinary,
    fallback_used = fallback_used,
    ordinary_admissible = ordinary_admissible
  )
}

time_policy <- function(policy, fixture, batch_size = 1L) {
  gc(FALSE)
  start <- proc.time()[["elapsed"]]
  result <- NULL
  for (batch_index in seq_len(batch_size)) {
    result <- tryCatch(
      run_policy(policy, fixture),
      error = function(e) e
    )
    if (inherits(result, "error")) break
  }
  elapsed <- (proc.time()[["elapsed"]] - start) / batch_size
  if (inherits(result, "error")) {
    return(list(
      elapsed_seconds = elapsed,
      error = conditionMessage(result)
    ))
  }
  fit <- result$fit
  list(
    elapsed_seconds = elapsed,
    error = "",
    converged = isTRUE(fit$converged),
    fallback_used = result$fallback_used,
    ordinary_admissible = result$ordinary_admissible,
    final_admissible =
      isTRUE(fit$diagnostics$admissibility$admissible),
    fmin = fit$fmin,
    iterations = fit$iterations,
    f_evals = fit$f_evals,
    g_evals = fit$g_evals,
    optimizer_status = fit$optimizer_status
  )
}

calibrate_batch <- function(policy, fixture, target_seconds = 0.05,
                            max_batch = 1024L) {
  batch_size <- 1L
  repeat {
    trial <- time_policy(policy, fixture, batch_size)
    if (nzchar(trial$error %||% "")) return(1L)
    total <- trial$elapsed_seconds * batch_size
    if (total >= target_seconds || batch_size >= max_batch) {
      return(batch_size)
    }
    batch_size <- min(max_batch, 2L * batch_size)
  }
}

timing_row <- function(fixture, policy, repetition, order_position, batch_size,
                       result) {
  data.frame(
    case_id = fixture$case_id,
    family = fixture$family,
    geometry = fixture$geometry,
    p = fixture$p,
    n = fixture$n,
    policy = policy,
    repetition = repetition,
    order_position = order_position,
    batch_size = batch_size,
    elapsed_seconds = result$elapsed_seconds %||% NA_real_,
    converged = result$converged %||% FALSE,
    fallback_used = result$fallback_used %||% FALSE,
    ordinary_admissible = result$ordinary_admissible %||% NA,
    final_admissible = result$final_admissible %||% FALSE,
    fmin = result$fmin %||% NA_real_,
    iterations = result$iterations %||% NA_integer_,
    f_evals = result$f_evals %||% NA_integer_,
    g_evals = result$g_evals %||% NA_integer_,
    optimizer_status = result$optimizer_status %||% "",
    error = result$error %||% "",
    stringsAsFactors = FALSE
  )
}

median_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) stats::median(x) else NA_real_
}

summarize_timings <- function(timings, fixtures) {
  keys <- unique(timings[c("case_id", "family", "geometry", "p", "n")])
  policies <- unique(timings$policy)
  rows <- lapply(seq_len(nrow(keys)), function(i) {
    key <- keys[i, ]
    case <- timings[timings$case_id == key$case_id, , drop = FALSE]
    values <- lapply(policies, function(policy) {
      x <- case[case$policy == policy & !nzchar(case$error), , drop = FALSE]
      c(
        median_ms = 1000 * median_or_na(x$elapsed_seconds),
        p25_ms = 1000 * as.numeric(stats::quantile(
          x$elapsed_seconds, .25, na.rm = TRUE, names = FALSE)),
        p75_ms = 1000 * as.numeric(stats::quantile(
          x$elapsed_seconds, .75, na.rm = TRUE, names = FALSE)),
        median_iterations = median_or_na(x$iterations),
        fallback_rate = mean(x$fallback_used),
        success_rate = mean(x$converged & x$final_admissible)
      )
    })
    names(values) <- policies
    metric <- function(policy, name) {
      value <- values[[policy]]
      if (is.null(value)) NA_real_ else unname(value[[name]])
    }
    ordinary_ms <- values$ordinary[["median_ms"]]
    ordinary_rows <- case[
      case$policy == "ordinary" & !nzchar(case$error), , drop = FALSE]
    data.frame(
      key,
      ordinary_admissible = fixtures[[key$case_id]]$
        expected_ordinary_admissible,
      ordinary_median_ms = ordinary_ms,
      psd_median_ms = values$psd_direct[["median_ms"]],
      audit_median_ms = values$audit_then_psd[["median_ms"]],
      psd_over_ordinary =
        values$psd_direct[["median_ms"]] / ordinary_ms,
      audit_over_ordinary =
        values$audit_then_psd[["median_ms"]] / ordinary_ms,
      psd_median_iterations =
        values$psd_direct[["median_iterations"]],
      audit_fallback_rate =
        values$audit_then_psd[["fallback_rate"]],
      psd_ipopt_median_ms =
        metric("psd_direct_ipopt", "median_ms"),
      audit_ipopt_median_ms =
        metric("audit_then_psd_ipopt", "median_ms"),
      psd_ipopt_over_ordinary =
        metric("psd_direct_ipopt", "median_ms") / ordinary_ms,
      audit_ipopt_over_ordinary =
        metric("audit_then_psd_ipopt", "median_ms") / ordinary_ms,
      psd_ipopt_median_iterations =
        metric("psd_direct_ipopt", "median_iterations"),
      audit_ipopt_fallback_rate =
        metric("audit_then_psd_ipopt", "fallback_rate"),
      ordinary_success_rate = mean(ordinary_rows$converged),
      psd_success_rate = values$psd_direct[["success_rate"]],
      audit_success_rate = values$audit_then_psd[["success_rate"]],
      psd_ipopt_success_rate =
        metric("psd_direct_ipopt", "success_rate"),
      audit_ipopt_success_rate =
        metric("audit_then_psd_ipopt", "success_rate"),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

validation_rows <- function(fixtures, include_ipopt = FALSE) {
  do.call(rbind, lapply(fixtures, function(fixture) {
    ordinary <- ordinary_fit(fixture)
    psd <- psd_fit(fixture)
    psd_ipopt <- if (include_ipopt) {
      psd_fit(fixture, optimizer = "ipopt")
    } else {
      NULL
    }
    data.frame(
      case_id = fixture$case_id,
      expected_ordinary_admissible =
        fixture$expected_ordinary_admissible,
      ordinary_converged = ordinary$converged,
      ordinary_admissible =
        ordinary$diagnostics$admissibility$admissible,
      psd_converged = psd$converged,
      psd_admissible = psd$diagnostics$admissibility$admissible,
      ordinary_fmin = ordinary$fmin,
      psd_fmin = psd$fmin,
      psd_ipopt_converged =
        if (include_ipopt) psd_ipopt$converged else NA,
      psd_ipopt_admissible = if (include_ipopt) {
        psd_ipopt$diagnostics$admissibility$admissible
      } else {
        NA
      },
      psd_ipopt_fmin =
        if (include_ipopt) psd_ipopt$fmin else NA_real_,
      fmin_difference = psd$fmin - ordinary$fmin,
      psd_backend_fmin_difference =
        if (include_ipopt) psd_ipopt$fmin - psd$fmin else NA_real_,
      max_theta_difference = if (length(psd$theta) == length(ordinary$theta)) {
        max(abs(psd$theta - ordinary$theta))
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  }))
}

cfg <- parse_args(commandArgs(trailingOnly = TRUE))
catalog <- fixture_catalog()
selected <- cfg$cases %||% profile_cases(cfg$profile)
unknown <- setdiff(selected, names(catalog))
if (length(unknown)) {
  stop("unknown case IDs: ", paste(unknown, collapse = ", "), call. = FALSE)
}
fixtures <- lapply(catalog[selected], prepare_fixture)
names(fixtures) <- selected

results <- cfg$results_dir %||%
  experiment_path("results", cfg$profile)
dir.create(results, recursive = TRUE, showWarnings = FALSE)

design <- do.call(rbind, lapply(fixtures, function(x) data.frame(
  case_id = x$case_id,
  family = x$family,
  geometry = x$geometry,
  p = x$p,
  n = x$n,
  expected_ordinary_admissible = x$expected_ordinary_admissible,
  stringsAsFactors = FALSE
)))
write_csv(design, file.path(results, "design.csv"))
write_metadata(
  file.path(results, "metadata.csv"),
  values = list(
    experiment = "73-psd-ml-timing",
    profile = cfg$profile,
    repeats = cfg$repeats,
    warmups = cfg$warmups,
    seed_base = cfg$seed_base,
    cases = paste(selected, collapse = ","),
    ordinary_optimizer = "nlopt-lbfgs",
    psd_optimizers = if (cfg$include_ipopt) {
      "nlopt-slsqp,ipopt"
    } else {
      "nlopt-slsqp"
    },
    timing_scope = "prebuilt model and sample statistics; fit call only",
    timer_batch_target_seconds = 0.05,
    timer_batch_max = 1024L
  ),
  packages = "magmaan"
)

policies <- c("ordinary", "psd_direct", "audit_then_psd")
if (cfg$include_ipopt) {
  policies <- c(
    policies, "psd_direct_ipopt", "audit_then_psd_ipopt")
}
cat(sprintf(
  "Profile %s: %d cases x %d policies x %d repetitions\n",
  cfg$profile, length(fixtures), length(policies), cfg$repeats))
cat("Warming all case/policy combinations...\n")
for (fixture in fixtures) {
  for (policy in policies) {
    for (warmup in seq_len(cfg$warmups)) {
      invisible(run_policy(policy, fixture))
    }
  }
}

cat("Calibrating timer batches...\n")
batch_sizes <- lapply(fixtures, function(fixture) {
  stats::setNames(
    vapply(
      policies,
      calibrate_batch,
      integer(1),
      fixture = fixture
    ),
    policies
  )
})

rows <- vector("list", length(fixtures) * length(policies) * cfg$repeats)
row_id <- 1L
for (case_index in seq_along(fixtures)) {
  fixture <- fixtures[[case_index]]
  cat(sprintf("[%d/%d] %s\n", case_index, length(fixtures), fixture$case_id))
  for (repetition in seq_len(cfg$repeats)) {
    set.seed(cfg$seed_base + 1000L * case_index + repetition)
    order <- sample(policies)
    for (position in seq_along(order)) {
      policy <- order[[position]]
      batch_size <- batch_sizes[[fixture$case_id]][[policy]]
      result <- time_policy(policy, fixture, batch_size)
      rows[[row_id]] <- timing_row(
        fixture, policy, repetition, position, batch_size, result)
      row_id <- row_id + 1L
    }
  }
}
timings <- do.call(rbind, rows)
summary <- summarize_timings(timings, fixtures)
validation <- validation_rows(fixtures, cfg$include_ipopt)
ipopt_pass <- if (cfg$include_ipopt) {
  with(
    validation,
    psd_ipopt_converged & psd_ipopt_admissible &
      abs(psd_backend_fmin_difference) < 1e-6
  )
} else {
  rep(TRUE, nrow(validation))
}
validation$pass <- with(
  validation,
  ordinary_converged &
    ordinary_admissible == expected_ordinary_admissible &
    psd_converged & psd_admissible &
    (!expected_ordinary_admissible | abs(fmin_difference) < 1e-7)
) & ipopt_pass

write_csv(timings, file.path(results, "timings.csv"))
write_csv(summary, file.path(results, "case_summary.csv"))
write_csv(validation, file.path(results, "validation.csv"))

failed <- timings[
  nzchar(timings$error) | !timings$converged |
    (timings$policy != "ordinary" & !timings$final_admissible),
  , drop = FALSE]
write_csv(failed, file.path(results, "failures.csv"))

cat(sprintf(
  "Wrote %d timings; %d failed or inadmissible final fits.\n",
  nrow(timings), nrow(failed)))
cat("Results: ", normalizePath(results), "\n", sep = "")
if (nrow(failed) || !all(validation$pass)) {
  stop("timing or validation checks failed; inspect results", call. = FALSE)
}
