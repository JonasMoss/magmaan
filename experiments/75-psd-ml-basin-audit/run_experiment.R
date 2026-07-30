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
require_pkg("magmaan", "install the current R package first")
suppressPackageStartupMessages(library(magmaan))

usage <- function() cat(
  "Usage: Rscript run_experiment.R [--smoke|--pilot] [options]\n\n",
  "Audit start sensitivity and competing stationary basins for PSD-ML.\n",
  "The reference is the best attained admissible KKT solution, not a\n",
  "claimed global optimum.\n\n",
  "Profiles:\n",
  "  --smoke   30 screened / 10 random-core datasets per N; 6 starts\n",
  "  --pilot   1000 screened / 100 random-core datasets per N; 13 starts\n\n",
  "Options:\n",
  "  --screen-reps N --core-reps N --extreme-reps N\n",
  "  --n-values 10,20,50 --seed-base N --max-iter N\n",
  "  --results-dir PATH --progress-every N\n",
  sep = "")

parse_int_csv <- function(x) {
  out <- as.integer(parse_csv_arg(x))
  if (!length(out) || anyNA(out)) {
    stop("integer CSV argument is empty or invalid", call. = FALSE)
  }
  out
}

parse_args <- function(args) {
  out <- list(
    profile = "smoke",
    screen_reps = NULL,
    core_reps = NULL,
    extreme_reps = NULL,
    n_values = c(10L, 20L, 50L),
    seed_base = 20260729L,
    max_iter = 5000L,
    results_dir = NULL,
    progress_every = 100L
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
    else if (arg == "--screen-reps") {
      out$screen_reps <- as.integer(take())
    } else if (arg == "--core-reps") {
      out$core_reps <- as.integer(take())
    } else if (arg == "--extreme-reps") {
      out$extreme_reps <- as.integer(take())
    } else if (arg == "--n-values") {
      out$n_values <- parse_int_csv(take())
    } else if (arg == "--seed-base") {
      out$seed_base <- as.integer(take())
    } else if (arg == "--max-iter") {
      out$max_iter <- as.integer(take())
    } else if (arg == "--results-dir") {
      out$results_dir <- take()
    } else if (arg == "--progress-every") {
      out$progress_every <- as.integer(take())
    } else {
      stop("unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }

  defaults <- switch(
    out$profile,
    smoke = list(screen_reps = 30L, core_reps = 10L, extreme_reps = 3L),
    pilot = list(screen_reps = 1000L, core_reps = 100L, extreme_reps = 25L)
  )
  for (name in names(defaults)) {
    if (is.null(out[[name]])) out[[name]] <- defaults[[name]]
  }
  out$n_values <- sort(unique(out$n_values))
  ints <- c(
    out$screen_reps, out$core_reps, out$extreme_reps, out$n_values,
    out$seed_base, out$max_iter, out$progress_every
  )
  if (anyNA(ints) || any(!is.finite(ints)) ||
      out$screen_reps < 1L || out$core_reps < 1L ||
      out$core_reps > out$screen_reps || out$extreme_reps < 0L ||
      any(out$n_values <= 6L) || out$seed_base < 1L ||
      out$max_iter < 1L || out$progress_every < 1L) {
    stop("invalid profile counts, sample sizes, seed, or iteration budget",
         call. = FALSE)
  }
  out
}

model_syntax <- function() {
  paste(
    "Y =~ y1 + y2 + y3",
    "X =~ x1 + x2 + x3",
    "Y ~ X",
    sep = "\n"
  )
}

simulate_data <- function(n, beta = 0.25, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  loadings <- c(1.0, 0.8, 0.6)
  eta_x <- stats::rnorm(n)
  eta_y <- beta * eta_x + stats::rnorm(n)
  epsilon <- matrix(stats::rnorm(n * 6L), nrow = n, ncol = 6L)
  out <- cbind(
    tcrossprod(eta_y, loadings),
    tcrossprod(eta_x, loadings)
  ) + epsilon
  out <- as.data.frame(out)
  names(out) <- c("y1", "y2", "y3", "x1", "x2", "x3")
  out
}

dataset_seed <- function(seed_base, n, rep) {
  seed_base + 1000003L * n + rep
}

audit_converged <- function(fit) {
  is.list(fit) && is.list(fit$audit) &&
    identical(fit$audit$advisory_status, "converged")
}

audit_number <- function(fit, name) {
  if (!is.list(fit) || !is.list(fit$audit)) return(NA_real_)
  value <- fit$audit[[name]]
  if (is.null(value) || length(value) != 1L) return(NA_real_)
  as.numeric(value)
}

admissible <- function(fit) {
  is.list(fit) && is.list(fit$diagnostics) &&
    is.list(fit$diagnostics$admissibility) &&
    isTRUE(fit$diagnostics$admissibility$admissible)
}

minimum_eigenvalue <- function(fit) {
  if (!is.list(fit) || !is.list(fit$diagnostics) ||
      !is.list(fit$diagnostics$admissibility)) {
    return(NA_real_)
  }
  blocks <- c(
    fit$diagnostics$admissibility$theta,
    fit$diagnostics$admissibility$psi
  )
  values <- vapply(
    blocks,
    function(x) {
      value <- x$min_eigenvalue
      if (is.null(value) || !length(value)) NA_real_ else as.numeric(value[[1L]])
    },
    numeric(1)
  )
  values <- values[is.finite(values)]
  if (length(values)) min(values) else NA_real_
}

beta_estimate <- function(fit) {
  if (!is.list(fit) || !is.data.frame(fit$partable)) return(NA_real_)
  pt <- fit$partable
  hit <- which(pt$lhs == "Y" & pt$op == "~" & pt$rhs == "X")
  if (length(hit) == 1L) as.numeric(pt$est[[hit]]) else NA_real_
}

primitive_rank_signature <- function(fit, tol = 1e-6) {
  if (!is.list(fit) || !is.data.frame(fit$partable)) return("")
  pt <- fit$partable
  diagonal <- pt$op == "~~" & pt$lhs == pt$rhs
  observed <- diagonal & grepl("^[xy][1-3]$", pt$lhs)
  latent <- diagonal & pt$lhs %in% c("X", "Y")
  if (!any(observed) || !any(latent)) return("")
  paste0(
    "theta", sum(pt$est[observed] > tol),
    "-psi", sum(pt$est[latent] > tol)
  )
}

implied_vector <- function(fit) {
  implied <- magmaan_core$model_implied(fit)
  sigma <- unlist(lapply(implied$sigma, function(x) x[lower.tri(x, TRUE)]),
                  use.names = FALSE)
  mu <- unlist(implied$mu, use.names = FALSE)
  c(sigma, mu)
}

with_start <- function(model, theta) {
  out <- model
  free <- out$partable$free
  rows <- which(free > 0L)
  if (length(theta) != max(free)) {
    stop("start vector length does not match the free partable", call. = FALSE)
  }
  out$partable$ustart[rows] <- theta[free[rows]]
  out
}

fit_psd <- function(model, data, control) {
  suppressWarnings(frontier_fit_ml_psd(
    model, data, optimizer = "nlopt-slsqp", control = control
  ))
}

fit_ordinary <- function(model, data, control) {
  suppressWarnings(magmaan_core$fit_ml(
    model, data, optimizer = "nlopt-lbfgs", control = control
  ))
}

fit_attempt <- function(model, data, control) {
  started <- proc.time()[["elapsed"]]
  fit <- tryCatch(fit_psd(model, data, control), error = identity)
  elapsed_ms <- 1000 * (proc.time()[["elapsed"]] - started)
  list(
    fit = fit,
    returned = !inherits(fit, "error"),
    elapsed_ms = elapsed_ms
  )
}

fit_payload <- function(attempt, n_theta, n_implied) {
  fit <- attempt$fit
  ok <- isTRUE(attempt$returned)
  stationary <- ok && audit_converged(fit)
  fit_admissible <- ok && admissible(fit)
  theta <- if (ok && length(fit$theta) == n_theta) {
    as.numeric(fit$theta)
  } else {
    rep(NA_real_, n_theta)
  }
  implied_error <- ""
  implied <- if (ok) {
    tryCatch(
      implied_vector(fit),
      error = function(e) {
        implied_error <<- conditionMessage(e)
        rep(NA_real_, n_implied)
      }
    )
  } else {
    rep(NA_real_, n_implied)
  }
  if (length(implied) != n_implied) implied <- rep(NA_real_, n_implied)
  finite <- ok && is.finite(fit$fmin) &&
    all(is.finite(theta)) && all(is.finite(implied))
  min_eigen <- if (ok) minimum_eigenvalue(fit) else NA_real_
  list(
    returned = ok,
    solver_converged = ok && isTRUE(fit$converged),
    audit_converged = stationary,
    admissible = fit_admissible,
    eligible = stationary && fit_admissible && finite,
    covariance_boundary =
      ok && is.finite(min_eigen) && min_eigen <= 1e-6,
    min_covariance_eigenvalue = min_eigen,
    rank_signature = if (ok) primitive_rank_signature(fit) else "",
    beta_hat = if (ok) beta_estimate(fit) else NA_real_,
    fmin = if (ok) as.numeric(fit$fmin) else NA_real_,
    stationarity_inf =
      if (ok) audit_number(fit, "grad_inf_norm") else NA_real_,
    constraint_violation_inf =
      if (ok) audit_number(fit, "constraint_violation_inf") else NA_real_,
    f_evals = if (ok) as.integer(fit$f_evals) else NA_integer_,
    elapsed_ms = attempt$elapsed_ms,
    optimizer_status =
      if (ok) as.character(fit$optimizer_status) else "",
    implied_error = implied_error,
    error = if (ok) "" else conditionMessage(fit),
    theta = theta,
    implied = implied
  )
}

payload_row <- function(payload, prefix, n, rep, seed,
                        random_core = FALSE,
                        failure_enrichment = FALSE,
                        extreme_enrichment = FALSE,
                        start_id = "default",
                        start_family = "default",
                        start_source = "FABIN",
                        amplitude = 0,
                        start_distance = 0) {
  base <- data.frame(
    n = n,
    rep = rep,
    seed = seed,
    random_core = random_core,
    failure_enrichment = failure_enrichment,
    extreme_enrichment = extreme_enrichment,
    start_id = start_id,
    start_family = start_family,
    start_source = start_source,
    amplitude = amplitude,
    start_distance = start_distance,
    returned = payload$returned,
    solver_converged = payload$solver_converged,
    audit_converged = payload$audit_converged,
    admissible = payload$admissible,
    eligible = payload$eligible,
    covariance_boundary = payload$covariance_boundary,
    min_covariance_eigenvalue = payload$min_covariance_eigenvalue,
    rank_signature = payload$rank_signature,
    beta_hat = payload$beta_hat,
    fmin = payload$fmin,
    stationarity_inf = payload$stationarity_inf,
    constraint_violation_inf = payload$constraint_violation_inf,
    f_evals = payload$f_evals,
    elapsed_ms = payload$elapsed_ms,
    optimizer_status = payload$optimizer_status,
    implied_error = payload$implied_error,
    error = payload$error,
    stringsAsFactors = FALSE
  )
  theta <- as.data.frame(as.list(payload$theta), check.names = FALSE)
  names(theta) <- paste0("theta_", seq_along(payload$theta))
  implied <- as.data.frame(as.list(payload$implied), check.names = FALSE)
  names(implied) <- paste0("implied_", seq_along(payload$implied))
  out <- cbind(base, theta, implied)
  names(out)[names(out) == "start_id"] <- paste0(prefix, "start_id")
  out
}

screen_datasets <- function(cfg, model, control, n_theta, n_implied) {
  rows <- vector("list", length(cfg$n_values) * cfg$screen_reps)
  index <- 0L
  total <- length(rows)
  started <- Sys.time()
  for (n in cfg$n_values) {
    for (rep in seq_len(cfg$screen_reps)) {
      seed <- dataset_seed(cfg$seed_base, n, rep)
      x <- simulate_data(n, seed = seed)
      data <- df_to_data(x, model, scaling = "n-1")
      payload <- fit_payload(
        fit_attempt(model, data, control), n_theta, n_implied
      )
      index <- index + 1L
      row <- payload_row(payload, "", n, rep, seed)
      rows[[index]] <- row[, c(
        "n", "rep", "seed", "returned", "solver_converged",
        "audit_converged", "admissible", "eligible",
        "covariance_boundary", "min_covariance_eigenvalue",
        "rank_signature", "beta_hat", "fmin", "stationarity_inf",
        "constraint_violation_inf", "f_evals", "elapsed_ms",
        "optimizer_status", "error"
      )]
      if (index %% cfg$progress_every == 0L || index == total) {
        elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
        eta <- elapsed * (total - index) / max(index, 1L)
        message(sprintf(
          "  screen %d/%d, elapsed %.1fs, ETA %.1fs",
          index, total, elapsed, eta
        ))
      }
    }
  }
  do.call(rbind, rows)
}

select_replays <- function(screen, cfg) {
  rows <- lapply(cfg$n_values, function(n) {
    x <- screen[screen$n == n, ]
    set.seed(as.integer((cfg$seed_base + 7919L * n) %% .Machine$integer.max))
    core_rep <- sample(x$rep, cfg$core_reps, replace = FALSE)
    failure_rep <- x$rep[!x$eligible]
    eligible <- x[x$eligible & is.finite(x$beta_hat), ]
    extreme_rep <- if (cfg$extreme_reps > 0L && nrow(eligible)) {
      ordered <- eligible[order(-abs(eligible$beta_hat), eligible$rep), ]
      head(ordered$rep, cfg$extreme_reps)
    } else {
      integer()
    }
    reps <- sort(unique(c(core_rep, failure_rep, extreme_rep)))
    data.frame(
      n = n,
      rep = reps,
      seed = dataset_seed(cfg$seed_base, n, reps),
      random_core = reps %in% core_rep,
      failure_enrichment = reps %in% failure_rep,
      extreme_enrichment = reps %in% extreme_rep,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

perturb_start <- function(base, amplitude, seed) {
  set.seed(as.integer(seed %% .Machine$integer.max))
  z <- stats::rnorm(length(base))
  base + amplitude * pmax(abs(base), 0.25) * z
}

perturb_seed <- function(seed, family_code, index) {
  as.integer(
    (as.double(seed) + 104729 * family_code + 1009 * index) %%
      .Machine$integer.max
  )
}

add_start_spec <- function(specs, id, family, source, amplitude, theta, fabin) {
  if (is.null(theta) || any(!is.finite(theta))) return(specs)
  specs[[length(specs) + 1L]] <- list(
    id = id,
    family = family,
    source = source,
    amplitude = amplitude,
    theta = theta,
    start_distance = max(abs(theta - fabin) / (1 + abs(fabin)))
  )
  specs
}

start_specs <- function(profile, seed, fabin, ordinary_theta,
                        default_psd_theta) {
  specs <- list()
  specs <- add_start_spec(
    specs, "ordinary_warm", "ordinary_warm", "ordinary_ml",
    0, ordinary_theta, fabin
  )
  if (profile == "pilot") {
    specs <- add_start_spec(
      specs, "psd_restart", "psd_restart", "default_psd",
      0, default_psd_theta, fabin
    )
  }
  counts <- if (profile == "smoke") {
    c(fabin_moderate = 1L, fabin_broad = 1L,
      ordinary_moderate = 1L, ordinary_broad = 1L)
  } else {
    c(fabin_moderate = 3L, fabin_broad = 3L,
      ordinary_moderate = 2L, ordinary_broad = 2L)
  }
  family_index <- c(
    fabin_moderate = 1L, fabin_broad = 2L,
    ordinary_moderate = 3L, ordinary_broad = 4L
  )
  for (family in names(counts)) {
    source <- if (startsWith(family, "fabin")) fabin else ordinary_theta
    if (is.null(source)) next
    amplitude <- if (endsWith(family, "moderate")) 0.10 else 0.50
    for (j in seq_len(counts[[family]])) {
      theta <- perturb_start(
        source, amplitude,
        perturb_seed(seed, family_index[[family]], j)
      )
      specs <- add_start_spec(
        specs, paste0(family, "_", j), family,
        if (startsWith(family, "fabin")) "FABIN" else "ordinary_ml",
        amplitude, theta, fabin
      )
    }
  }
  specs
}

replay_datasets <- function(selection, cfg, model, control,
                            n_theta, n_implied) {
  rows <- list()
  ordinary_rows <- list()
  completed <- 0L
  expected_per_dataset <- if (cfg$profile == "smoke") 6L else 13L
  expected <- nrow(selection) * expected_per_dataset
  started <- Sys.time()
  for (i in seq_len(nrow(selection))) {
    key <- selection[i, ]
    x <- simulate_data(key$n, seed = key$seed)
    data <- df_to_data(x, model, scaling = "n-1")
    fabin <- as.numeric(
      magmaan_core$estimate_start_values(model$partable, data)
    )

    ordinary_started <- proc.time()[["elapsed"]]
    ordinary <- tryCatch(
      fit_ordinary(model, data, control), error = identity
    )
    ordinary_elapsed <- 1000 * (
      proc.time()[["elapsed"]] - ordinary_started
    )
    ordinary_ok <- !inherits(ordinary, "error")
    ordinary_rows[[length(ordinary_rows) + 1L]] <- data.frame(
      n = key$n,
      rep = key$rep,
      seed = key$seed,
      returned = ordinary_ok,
      audit_converged = ordinary_ok && audit_converged(ordinary),
      admissible = ordinary_ok && admissible(ordinary),
      beta_hat = if (ordinary_ok) beta_estimate(ordinary) else NA_real_,
      fmin = if (ordinary_ok) ordinary$fmin else NA_real_,
      elapsed_ms = ordinary_elapsed,
      error = if (ordinary_ok) "" else conditionMessage(ordinary),
      stringsAsFactors = FALSE
    )
    ordinary_theta <- if (ordinary_ok) ordinary$theta else NULL

    default_attempt <- fit_attempt(model, data, control)
    default_payload <- fit_payload(
      default_attempt, n_theta, n_implied
    )
    rows[[length(rows) + 1L]] <- payload_row(
      default_payload, "", key$n, key$rep, key$seed,
      key$random_core, key$failure_enrichment, key$extreme_enrichment,
      "default", "default", "FABIN", 0, 0
    )
    completed <- completed + 1L
    default_theta <- if (default_attempt$returned) {
      default_attempt$fit$theta
    } else {
      NULL
    }

    specs <- start_specs(
      cfg$profile, key$seed, fabin, ordinary_theta, default_theta
    )
    for (spec in specs) {
      attempt <- fit_attempt(with_start(model, spec$theta), data, control)
      payload <- fit_payload(attempt, n_theta, n_implied)
      rows[[length(rows) + 1L]] <- payload_row(
        payload, "", key$n, key$rep, key$seed,
        key$random_core, key$failure_enrichment, key$extreme_enrichment,
        spec$id, spec$family, spec$source, spec$amplitude,
        spec$start_distance
      )
      completed <- completed + 1L
    }
    if (i %% max(1L, cfg$progress_every %/% 10L) == 0L ||
        i == nrow(selection)) {
      elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
      eta <- elapsed * (nrow(selection) - i) / i
      message(sprintf(
        "  replay datasets %d/%d, fits %d/~%d, elapsed %.1fs, ETA %.1fs",
        i, nrow(selection), completed, expected, elapsed, eta
      ))
    }
  }
  list(
    raw = do.call(rbind, rows),
    ordinary = do.call(rbind, ordinary_rows)
  )
}

objective_cluster_ids <- function(f, rel_tol = 1e-6) {
  ids <- integer(length(f))
  order_f <- order(f)
  cluster <- 1L
  representative <- f[order_f[[1L]]]
  ids[order_f[[1L]]] <- cluster
  if (length(f) > 1L) {
    for (position in order_f[-1L]) {
      tol <- rel_tol * max(1, abs(representative))
      if (f[[position]] - representative > tol) {
        cluster <- cluster + 1L
        representative <- f[[position]]
      }
      ids[[position]] <- cluster
    }
  }
  ids
}

relative_parameter_distance <- function(x, ref) {
  max(abs(x - ref) / (1 + abs(ref)))
}

relative_implied_distance <- function(x, ref) {
  sqrt(sum((x - ref)^2)) / max(1, sqrt(sum(ref^2)))
}

classify_dataset <- function(x, theta_columns, implied_columns) {
  eligible <- x[x$eligible, , drop = FALSE]
  base <- x[1L, c(
    "n", "rep", "seed", "random_core",
    "failure_enrichment", "extreme_enrichment"
  )]
  default <- x[x$start_id == "default", , drop = FALSE]
  ordinary <- x[x$start_id == "ordinary_warm", , drop = FALSE]
  if (!nrow(eligible)) {
    return(cbind(base, data.frame(
      n_starts = nrow(x),
      n_eligible = 0L,
      objective_basin_class = "unresolved",
      n_objective_clusters = 0L,
      second_objective_gap = NA_real_,
      best_equivalence_class = "unresolved",
      max_best_parameter_distance = NA_real_,
      max_best_implied_distance = NA_real_,
      start_eligibility_class = "start-sensitive",
      default_best = FALSE,
      default_objective_gap = NA_real_,
      ordinary_warm_available = nrow(ordinary) == 1L,
      ordinary_warm_best = FALSE,
      ordinary_warm_objective_gap = NA_real_,
      best_fmin = NA_real_,
      best_beta = NA_real_,
      best_boundary = NA,
      best_rank_signature = "",
      stringsAsFactors = FALSE
    )))
  }

  eligible$objective_cluster <- objective_cluster_ids(eligible$fmin)
  representatives <- tapply(
    eligible$fmin, eligible$objective_cluster, min
  )
  best <- eligible[eligible$objective_cluster == 1L, , drop = FALSE]
  reference <- best[which.min(best$fmin), , drop = FALSE]
  ref_theta <- as.numeric(reference[1L, theta_columns])
  ref_implied <- as.numeric(reference[1L, implied_columns])
  parameter_distances <- apply(
    best[, theta_columns, drop = FALSE], 1L,
    function(v) relative_parameter_distance(as.numeric(v), ref_theta)
  )
  implied_distances <- apply(
    best[, implied_columns, drop = FALSE], 1L,
    function(v) relative_implied_distance(as.numeric(v), ref_implied)
  )
  parameter_equivalent <- max(parameter_distances) <= 1e-4
  implied_equivalent <- max(implied_distances) <= 1e-6
  best_equivalence <- if (!implied_equivalent) {
    "equal-objective-different-moments"
  } else if (!parameter_equivalent) {
    "same-fit-different-parameters"
  } else {
    "unique-parameters-and-moments"
  }
  fstar <- min(eligible$fmin)
  gap <- function(row) {
    if (!nrow(row) || !isTRUE(row$eligible[[1L]])) return(NA_real_)
    row$fmin[[1L]] - fstar
  }
  is_best <- function(row) {
    value <- gap(row)
    is.finite(value) && value <= 1e-6 * max(1, abs(fstar))
  }
  cbind(base, data.frame(
    n_starts = nrow(x),
    n_eligible = nrow(eligible),
    objective_basin_class =
      if (length(representatives) > 1L) "competing" else "single",
    n_objective_clusters = length(representatives),
    second_objective_gap =
      if (length(representatives) > 1L) {
        unname(representatives[[2L]] - representatives[[1L]])
      } else {
        NA_real_
      },
    best_equivalence_class = best_equivalence,
    max_best_parameter_distance = max(parameter_distances),
    max_best_implied_distance = max(implied_distances),
    start_eligibility_class =
      if (all(x$eligible)) "all-eligible" else "start-sensitive",
    default_best = is_best(default),
    default_objective_gap = gap(default),
    ordinary_warm_available = nrow(ordinary) == 1L,
    ordinary_warm_best = is_best(ordinary),
    ordinary_warm_objective_gap = gap(ordinary),
    best_fmin = reference$fmin,
    best_beta = reference$beta_hat,
    best_boundary = reference$covariance_boundary,
    best_rank_signature = reference$rank_signature,
    stringsAsFactors = FALSE
  ))
}

classify_all <- function(raw) {
  theta_columns <- grep("^theta_[0-9]+$", names(raw), value = TRUE)
  implied_columns <- grep("^implied_[0-9]+$", names(raw), value = TRUE)
  key <- interaction(raw$n, raw$rep, drop = TRUE, lex.order = TRUE)
  rows <- lapply(
    split(raw, key),
    classify_dataset,
    theta_columns = theta_columns,
    implied_columns = implied_columns
  )
  out <- do.call(rbind, rows)
  row.names(out) <- NULL
  out[order(out$n, out$rep), ]
}

mean_or_na <- function(x) {
  if (length(x)) mean(x) else NA_real_
}

median_or_na <- function(x) {
  x <- x[is.finite(x)]
  if (length(x)) stats::median(x) else NA_real_
}

scope_rows <- function(x, scope) {
  switch(
    scope,
    random_core = x[x$random_core, , drop = FALSE],
    failure_enrichment = x[x$failure_enrichment, , drop = FALSE],
    extreme_enrichment = x[x$extreme_enrichment, , drop = FALSE]
  )
}

summarize_datasets <- function(datasets) {
  rows <- list()
  for (scope in c(
    "random_core", "failure_enrichment", "extreme_enrichment"
  )) {
    scoped <- scope_rows(datasets, scope)
    for (n in sort(unique(datasets$n))) {
      x <- scoped[scoped$n == n, , drop = FALSE]
      if (!nrow(x)) next
      rows[[length(rows) + 1L]] <- data.frame(
        scope = scope,
        n = n,
        datasets = nrow(x),
        default_best_rate = mean(x$default_best),
        ordinary_warm_available_rate = mean(x$ordinary_warm_available),
        ordinary_warm_best_rate =
          mean_or_na(x$ordinary_warm_best[x$ordinary_warm_available]),
        competing_basin_rate =
          mean(x$objective_basin_class == "competing"),
        unresolved_rate =
          mean(x$objective_basin_class == "unresolved"),
        different_parameter_rate =
          mean(x$best_equivalence_class == "same-fit-different-parameters"),
        different_moment_rate =
          mean(x$best_equivalence_class ==
                 "equal-objective-different-moments"),
        start_sensitive_rate =
          mean(x$start_eligibility_class == "start-sensitive"),
        best_boundary_rate = mean(x$best_boundary, na.rm = TRUE),
        median_default_gap = median_or_na(x$default_objective_gap),
        max_default_gap =
          if (any(is.finite(x$default_objective_gap))) {
            max(x$default_objective_gap, na.rm = TRUE)
          } else {
            NA_real_
          },
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

summarize_starts <- function(raw) {
  rows <- list()
  for (scope in c(
    "random_core", "failure_enrichment", "extreme_enrichment"
  )) {
    scoped <- scope_rows(raw, scope)
    if (!nrow(scoped)) next
    key <- interaction(
      scoped$n, scoped$start_family, drop = TRUE, lex.order = TRUE
    )
    for (x in split(scoped, key)) {
      rows[[length(rows) + 1L]] <- data.frame(
        scope = scope,
        n = x$n[[1L]],
        start_family = x$start_family[[1L]],
        attempts = nrow(x),
        return_rate = mean(x$returned),
        audit_rate = mean(x$audit_converged),
        eligible_rate = mean(x$eligible),
        best_hit_rate = mean(x$best_hit),
        median_elapsed_ms = median_or_na(x$elapsed_ms),
        median_f_evals = median_or_na(x$f_evals),
        stringsAsFactors = FALSE
      )
    }
  }
  out <- do.call(rbind, rows)
  out[order(out$scope, out$n, out$start_family), ]
}

attach_cluster_telemetry <- function(raw, datasets) {
  best <- datasets[, c("n", "rep", "best_fmin")]
  out <- merge(raw, best, by = c("n", "rep"), all.x = TRUE, sort = FALSE)
  tol <- 1e-6 * pmax(1, abs(out$best_fmin))
  out$best_hit <- out$eligible & is.finite(out$best_fmin) &
    out$fmin - out$best_fmin <= tol
  out$objective_gap <- ifelse(
    out$eligible, out$fmin - out$best_fmin, NA_real_
  )
  out$objective_cluster <- NA_integer_
  out$parameter_distance_to_best <- NA_real_
  out$implied_distance_to_best <- NA_real_
  out$cluster_representative <- FALSE
  theta_columns <- grep("^theta_[0-9]+$", names(out), value = TRUE)
  implied_columns <- grep("^implied_[0-9]+$", names(out), value = TRUE)
  keys <- interaction(out$n, out$rep, drop = TRUE, lex.order = TRUE)
  for (indices in split(seq_len(nrow(out)), keys)) {
    eligible_indices <- indices[out$eligible[indices]]
    if (!length(eligible_indices)) next
    f <- out$fmin[eligible_indices]
    clusters <- objective_cluster_ids(f)
    out$objective_cluster[eligible_indices] <- clusters
    reference_index <- eligible_indices[[which.min(f)]]
    ref_theta <- as.numeric(out[reference_index, theta_columns])
    ref_implied <- as.numeric(out[reference_index, implied_columns])
    for (row in eligible_indices) {
      out$parameter_distance_to_best[[row]] <-
        relative_parameter_distance(
          as.numeric(out[row, theta_columns]), ref_theta
        )
      out$implied_distance_to_best[[row]] <-
        relative_implied_distance(
          as.numeric(out[row, implied_columns]), ref_implied
        )
    }
    for (cluster in unique(clusters)) {
      members <- eligible_indices[clusters == cluster]
      representative <- members[[which.min(out$fmin[members])]]
      out$cluster_representative[[representative]] <- TRUE
    }
  }
  out
}

main <- function() {
  cfg <- parse_args(commandArgs(trailingOnly = TRUE))
  output_dir <- if (is.null(cfg$results_dir)) {
    experiment_path("results", cfg$profile)
  } else {
    normalizePath(cfg$results_dir, mustWork = FALSE)
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  started <- Sys.time()
  model <- model_spec(model_syntax())
  n_theta <- max(model$partable$free)
  n_implied <- (6L * 7L) %/% 2L
  control <- list(
    max_iter = cfg$max_iter,
    gtol = 1e-8,
    ftol = 1e-12
  )

  message(
    "PSD-ML basin audit ", cfg$profile, ": ",
    length(cfg$n_values), " N cells x ", cfg$screen_reps,
    " screened datasets"
  )
  screen <- screen_datasets(
    cfg, model, control, n_theta, n_implied
  )
  selection <- select_replays(screen, cfg)
  message(
    "Selected ", nrow(selection), " unique replay datasets: ",
    sum(selection$random_core), " random-core memberships, ",
    sum(selection$failure_enrichment), " failures, ",
    sum(selection$extreme_enrichment), " extreme-estimate memberships"
  )
  replay <- replay_datasets(
    selection, cfg, model, control, n_theta, n_implied
  )
  datasets <- classify_all(replay$raw)
  raw <- attach_cluster_telemetry(replay$raw, datasets)
  dataset_summary <- summarize_datasets(datasets)
  start_summary <- summarize_starts(raw)
  candidates <- datasets[
    datasets$objective_basin_class == "competing" |
      datasets$best_equivalence_class ==
        "equal-objective-different-moments",
    ,
    drop = FALSE
  ]

  screen_default <- raw[raw$start_id == "default", c(
    "n", "rep", "fmin", "eligible"
  )]
  screen_check <- merge(
    screen[, c("n", "rep", "fmin", "eligible")],
    screen_default,
    by = c("n", "rep"),
    suffixes = c("_screen", "_replay")
  )
  comparable <- screen_check$eligible_screen & screen_check$eligible_replay
  max_replay_gap <- if (any(comparable)) {
    max(abs(
      screen_check$fmin_screen[comparable] -
        screen_check$fmin_replay[comparable]
    ))
  } else {
    NA_real_
  }
  valid <- nrow(screen) ==
      length(cfg$n_values) * cfg$screen_reps &&
    nrow(selection) == nrow(datasets) &&
    !anyDuplicated(raw[c("n", "rep", "start_id")]) &&
    all(raw$admissible[raw$eligible]) &&
    all(raw$audit_converged[raw$eligible]) &&
    (!is.finite(max_replay_gap) || max_replay_gap < 1e-8)

  write_csv(screen, file.path(output_dir, "screen.csv"))
  write_csv(selection, file.path(output_dir, "selection.csv"))
  write_csv(raw, file.path(output_dir, "raw.csv"))
  write_csv(replay$ordinary, file.path(output_dir, "ordinary.csv"))
  write_csv(datasets, file.path(output_dir, "dataset_summary.csv"))
  membership_columns <- c(
    "n", "rep", "seed", "start_id", "start_family",
    "eligible", "fmin", "best_fmin", "objective_gap",
    "objective_cluster", "best_hit", "cluster_representative",
    "parameter_distance_to_best", "implied_distance_to_best",
    "beta_hat", "covariance_boundary", "rank_signature",
    "stationarity_inf", "constraint_violation_inf"
  )
  write_csv(
    raw[, membership_columns],
    file.path(output_dir, "cluster_membership.csv")
  )
  write_csv(
    dataset_summary, file.path(output_dir, "classification_summary.csv")
  )
  write_csv(start_summary, file.path(output_dir, "start_summary.csv"))
  write_csv(
    candidates, file.path(output_dir, "confirmation_candidates.csv")
  )
  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  ref <- magmaan_cache_ref()
  write_metadata(
    file.path(output_dir, "metadata.csv"),
    values = list(
      experiment = "75-psd-ml-basin-audit",
      profile = cfg$profile,
      n_values = cfg$n_values,
      screen_reps = cfg$screen_reps,
      core_reps = cfg$core_reps,
      extreme_reps = cfg$extreme_reps,
      seed_base = cfg$seed_base,
      max_iter = cfg$max_iter,
      objective_tolerance = "1e-6 * max(1, abs(F_best))",
      parameter_tolerance = 1e-4,
      implied_moment_tolerance = 1e-6,
      screen_rows = nrow(screen),
      selected_datasets = nrow(selection),
      replay_fit_rows = nrow(raw),
      confirmation_candidates = nrow(candidates),
      max_screen_replay_fmin_difference = max_replay_gap,
      elapsed_sec = elapsed,
      magmaan_git_head = ref$git_head,
      magmaan_git_dirty = ref$git_dirty
    ),
    packages = "magmaan"
  )

  cat("\nRandom-core basin audit:\n")
  print(
    dataset_summary[
      dataset_summary$scope == "random_core",
      c(
        "n", "datasets", "default_best_rate", "ordinary_warm_best_rate",
        "competing_basin_rate", "different_parameter_rate",
        "different_moment_rate", "start_sensitive_rate",
        "best_boundary_rate", "max_default_gap"
      )
    ],
    row.names = FALSE,
    digits = 4
  )
  cat("\nEnriched confirmation-candidate count: ", nrow(candidates), "\n",
      sep = "")
  message(
    "Wrote ", nrow(raw), " replay fits to ",
    normalizePath(output_dir), " in ", sprintf("%.1fs", elapsed)
  )
  if (!valid) quit(save = "no", status = 1L)
}

if (sys.nframe() == 0L) main()
