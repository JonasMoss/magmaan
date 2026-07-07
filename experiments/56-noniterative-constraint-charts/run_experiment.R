#!/usr/bin/env Rscript
# Experiment 56: chart sanity checks for constrained non-iterative CFA.
#
# The constrained Guttman machinery projects a configural closed-form estimate
# onto the linear equalities carried by the partable. This experiment checks the
# invariance-like property we actually need for measurement-invariance work:
# changing an arbitrary marker chart should change coordinates, but not the
# constrained fitted covariance or the Wald statistic, provided the H estimator
# and the induced covariance metric are transformed with the chart.

suppressWarnings(suppressMessages(library(magmaan)))

source(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])),
  "..", "_support", "R", "helpers.R"
))

set_single_threaded_math()
core <- magmaan::magmaan_core

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Marker-chart sanity check for closed-form metric-invariance projection.\n\n",
    "Options:\n",
    "  --smoke              Quick deterministic + finite-sample run. Default.\n",
    "  --full               Larger finite-sample run.\n",
    "  --reps N             Finite-sample replications. Smoke default: 20; full: 300.\n",
    "  --n LIST             Per-group sample sizes for finite draws. Default: 400.\n",
    "  --estimators LIST    Comma-separated guttman,guttman_gls_aligned. Default: both.\n",
    "  --scenarios LIST     Comma-separated invariant,metric_viol. Default: both.\n",
    "  --seed-base N        Base RNG seed. Default: 20260707.\n",
    "  --results-dir PATH   Output directory. Default: results.\n",
    "  --help               Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    smoke = TRUE,
    reps = 20L,
    n = 400L,
    estimators = c("guttman", "guttman_gls_aligned"),
    scenarios = c("invariant", "metric_viol"),
    seed_base = 20260707L,
    results_dir = experiment_path("results")
  )
  explicit <- character()
  i <- 1L
  take <- function(name) {
    i <<- i + 1L
    if (i > length(args)) stop(name, " needs a value", call. = FALSE)
    args[[i]]
  }
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--help") {
      usage()
      quit(status = 0L)
    } else if (a == "--smoke") {
      opts$smoke <- TRUE
    } else if (a == "--full") {
      opts$smoke <- FALSE
      if (!"reps" %in% explicit) opts$reps <- 300L
      if (!"n" %in% explicit) opts$n <- c(200L, 400L, 1000L)
    } else if (a == "--reps") {
      opts$reps <- as.integer(take(a)); explicit <- c(explicit, "reps")
    } else if (grepl("^--reps=", a)) {
      opts$reps <- as.integer(sub("^--reps=", "", a)); explicit <- c(explicit, "reps")
    } else if (a == "--n") {
      opts$n <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "n")
    } else if (grepl("^--n=", a)) {
      opts$n <- as.integer(parse_csv_arg(sub("^--n=", "", a))); explicit <- c(explicit, "n")
    } else if (a == "--estimators") {
      opts$estimators <- parse_csv_arg(take(a)); explicit <- c(explicit, "estimators")
    } else if (grepl("^--estimators=", a)) {
      opts$estimators <- parse_csv_arg(sub("^--estimators=", "", a))
      explicit <- c(explicit, "estimators")
    } else if (a == "--scenarios") {
      opts$scenarios <- parse_csv_arg(take(a)); explicit <- c(explicit, "scenarios")
    } else if (grepl("^--scenarios=", a)) {
      opts$scenarios <- parse_csv_arg(sub("^--scenarios=", "", a))
      explicit <- c(explicit, "scenarios")
    } else if (a == "--seed-base") {
      opts$seed_base <- as.integer(take(a))
    } else if (grepl("^--seed-base=", a)) {
      opts$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--results-dir") {
      opts$results_dir <- take(a)
    } else if (grepl("^--results-dir=", a)) {
      opts$results_dir <- sub("^--results-dir=", "", a)
    } else {
      stop("unknown option: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  bad <- setdiff(opts$estimators, c("guttman", "guttman_gls_aligned"))
  if (length(bad)) stop("unknown estimators: ", paste(bad, collapse = ","), call. = FALSE)
  bad <- setdiff(opts$scenarios, c("invariant", "metric_viol"))
  if (length(bad)) stop("unknown scenarios: ", paste(bad, collapse = ","), call. = FALSE)
  if (!is.finite(opts$reps) || opts$reps < 1) {
    stop("--reps must be positive", call. = FALSE)
  }
  if (any(!is.finite(opts$n)) || any(opts$n < 20)) {
    stop("--n values must be at least 20", call. = FALSE)
  }
  opts
}

opts <- parse_args(commandArgs(TRUE))
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)

p <- 6L
ov <- paste0("x", seq_len(p))

model_x1 <- paste(
  "f1 =~ x1 + x2 + x3",
  "f2 =~ x4 + x5 + x6",
  "f1 ~~ f2",
  sep = "\n"
)

model_x2 <- paste(
  "f1 =~ NA*x1 + 1*x2 + x3",
  "f2 =~ NA*x4 + 1*x5 + x6",
  "f1 ~~ f2",
  sep = "\n"
)

partables <- list(
  marker_x1x4 = core$lavaan_lavaanify(model_x1, n_groups = 2L,
                                       group_equal = "loadings"),
  marker_x2x5 = core$lavaan_lavaanify(model_x2, n_groups = 2L,
                                       group_equal = "loadings")
)

make_lambda <- function(lam) {
  L <- matrix(0, p, 2L)
  L[1:3, 1] <- lam[1:3]
  L[4:6, 2] <- lam[4:6]
  L
}

make_sigma <- function(lam, Phi, theta) {
  S <- make_lambda(lam) %*% Phi %*% t(make_lambda(lam)) + diag(theta, p)
  dimnames(S) <- list(ov, ov)
  S
}

population <- function(scenario) {
  lam1 <- c(1.00, 0.80, 1.20, 1.00, 0.70, 1.30)
  lam2 <- lam1
  if (scenario == "metric_viol") {
    # Change one alternate marker and one ordinary indicator. This makes the
    # misspecification genuinely off the metric-invariance surface in both
    # marker charts, while the surface itself remains the same model.
    lam2[2] <- 0.95
    lam2[6] <- 1.05
  }
  Phi1 <- matrix(c(1.00, 0.30, 0.30, 1.20), 2L, 2L)
  Phi2 <- matrix(c(1.35, 0.18, 0.18, 0.90), 2L, 2L)
  th1 <- c(0.45, 0.55, 0.50, 0.60, 0.50, 0.40)
  th2 <- c(0.50, 0.45, 0.65, 0.55, 0.70, 0.50)
  S <- list(make_sigma(lam1, Phi1, th1), make_sigma(lam2, Phi2, th2))
  for (b in seq_along(S)) {
    ev <- eigen(S[[b]], symmetric = TRUE, only.values = TRUE)$values
    if (min(ev) <= 1e-8) stop("non-PD population covariance", call. = FALSE)
  }
  S
}

sample_stats_population <- function(S) {
  list(S = S, nobs = as.integer(c(1000000L, 1000000L)))
}

draw_normal <- function(S, n, seed) {
  set.seed(seed)
  out <- matrix(stats::rnorm(n * ncol(S)), n, ncol(S)) %*% chol(S)
  colnames(out) <- colnames(S)
  out
}

sample_stats_finite <- function(S, n, seed) {
  X1 <- draw_normal(S[[1L]], n, seed)
  X2 <- draw_normal(S[[2L]], n, seed + 104729L)
  core$data_sample_stats_from_raw(list(X1, X2))
}

with_theta <- function(fit, theta) {
  out <- fit
  out$theta <- as.numeric(theta)
  out
}

max_moment_diff <- function(a, b) {
  max(vapply(seq_along(a$sigma), function(i) {
    max(abs(as.matrix(a$sigma[[i]]) - as.matrix(b$sigma[[i]])))
  }, numeric(1)))
}

fit_chart <- function(pt, ss, estimator) {
  fit <- fit_noniterative_cfa(pt, ss, estimator = estimator)
  con <- noniterative_cfa_constrained(fit, estimator = estimator,
                                      discrepancy = "ntml", gamma = "nt")
  list(
    fit = fit,
    con = con,
    implied_config = core$model_implied(fit),
    implied_constrained = core$model_implied(with_theta(fit, con$theta_tilde))
  )
}

compare_charts <- function(ss, scenario, estimator, source, rep_id, n) {
  a <- fit_chart(partables$marker_x1x4, ss, estimator)
  b <- fit_chart(partables$marker_x2x5, ss, estimator)
  data.frame(
    source = source,
    rep = rep_id,
    n = n,
    scenario = scenario,
    estimator = estimator,
    k_x1x4 = a$con$k,
    k_x2x5 = b$con$k,
    W_x1x4 = a$con$W,
    W_x2x5 = b$con$W,
    W_abs_diff = abs(a$con$W - b$con$W),
    p_abs_diff = abs(a$con$p_wald - b$con$p_wald),
    config_sigma_max_abs_diff = max_moment_diff(a$implied_config, b$implied_config),
    constrained_sigma_max_abs_diff = max_moment_diff(a$implied_constrained,
                                                     b$implied_constrained),
    raw_theta_max_abs_diff = max(abs(a$fit$theta - b$fit$theta)),
    constrained_theta_max_abs_diff = max(abs(a$con$theta_tilde - b$con$theta_tilde)),
    stringsAsFactors = FALSE
  )
}

project_linear <- function(theta, Omega, R, c) {
  OmRt <- Omega %*% t(R)
  middle <- R %*% OmRt
  sol <- solve(middle, R %*% theta - c)
  theta_tilde <- as.numeric(theta - OmRt %*% sol)
  W <- drop(crossprod(R %*% theta - c, sol))
  list(theta_tilde = theta_tilde, W = W)
}

linear_projection_control <- function(reps, seed_base) {
  rows <- vector("list", reps)
  for (r in seq_len(reps)) {
    set.seed(seed_base + r)
    q <- 10L
    k <- 4L
    theta <- stats::rnorm(q)
    A <- matrix(stats::rnorm(q * q), q, q)
    Omega <- crossprod(A) + diag(seq(0.2, 0.6, length.out = q))
    R <- matrix(stats::rnorm(k * q), k, q)
    c <- stats::rnorm(k)
    B <- qr.Q(qr(matrix(stats::rnorm(q * q), q, q))) %*%
      diag(seq(0.7, 1.6, length.out = q))

    # theta = B eta. The same affine set is R B eta = c, and
    # Cov(eta) = B^{-1} Omega B^{-T}.
    eta <- solve(B, theta)
    Binv <- solve(B)
    Omega_eta <- Binv %*% Omega %*% t(Binv)
    p_theta <- project_linear(theta, Omega, R, c)
    p_eta <- project_linear(eta, Omega_eta, R %*% B, c)
    rows[[r]] <- data.frame(
      rep = r,
      projected_theta_max_abs_diff =
        max(abs(as.numeric(B %*% p_eta$theta_tilde) - p_theta$theta_tilde)),
      W_abs_diff = abs(p_theta$W - p_eta$W)
    )
  }
  do.call(rbind, rows)
}

rows <- list()
row_id <- 0L
for (scenario in opts$scenarios) {
  S <- population(scenario)
  ss_pop <- sample_stats_population(S)
  for (estimator in opts$estimators) {
    row_id <- row_id + 1L
    rows[[row_id]] <- compare_charts(ss_pop, scenario, estimator,
                                     source = "population", rep_id = 0L,
                                     n = ss_pop$nobs[[1L]])
  }
  for (n in opts$n) {
    for (rep_id in seq_len(opts$reps)) {
      ss <- sample_stats_finite(S, n,
                                opts$seed_base +
                                  100000L * match(scenario, opts$scenarios) +
                                  1000L * match(n, opts$n) + rep_id)
      for (estimator in opts$estimators) {
        row_id <- row_id + 1L
        rows[[row_id]] <- compare_charts(ss, scenario, estimator,
                                         source = "finite_sample", rep_id = rep_id,
                                         n = n)
      }
    }
  }
}

raw <- do.call(rbind, rows)
summary <- stats::aggregate(
  raw[c("W_abs_diff", "p_abs_diff", "config_sigma_max_abs_diff",
        "constrained_sigma_max_abs_diff", "raw_theta_max_abs_diff",
        "constrained_theta_max_abs_diff")],
  raw[c("source", "n", "scenario", "estimator")],
  function(x) c(max = max(x), median = stats::median(x))
)

flatten_aggregate <- function(x) {
  out <- x[seq_len(4L)]
  for (nm in names(x)[-(seq_len(4L))]) {
    mat <- if (is.matrix(x[[nm]])) x[[nm]] else do.call(rbind, x[[nm]])
    out[[paste0(nm, "_max")]] <- mat[, "max"]
    out[[paste0(nm, "_median")]] <- mat[, "median"]
  }
  out
}
summary <- flatten_aggregate(summary)

lin <- linear_projection_control(opts$reps, opts$seed_base + 900000L)
lin_summary <- data.frame(
  projected_theta_max_abs_diff_max = max(lin$projected_theta_max_abs_diff),
  projected_theta_max_abs_diff_median = stats::median(lin$projected_theta_max_abs_diff),
  W_abs_diff_max = max(lin$W_abs_diff),
  W_abs_diff_median = stats::median(lin$W_abs_diff)
)

write_csv(raw, file.path(opts$results_dir, "chart_comparison_raw.csv"))
write_csv(summary, file.path(opts$results_dir, "chart_comparison_summary.csv"))
write_csv(lin, file.path(opts$results_dir, "linear_projection_raw.csv"))
write_csv(lin_summary, file.path(opts$results_dir, "linear_projection_summary.csv"))
write_csv(metadata_frame(
  list(
    reps = opts$reps,
    n = opts$n,
    estimators = opts$estimators,
    scenarios = opts$scenarios,
    seed_base = opts$seed_base
  ),
  packages = "magmaan"
), file.path(opts$results_dir, "metadata.csv"))

cat("Wrote:\n")
cat("  ", file.path(opts$results_dir, "chart_comparison_raw.csv"), "\n", sep = "")
cat("  ", file.path(opts$results_dir, "chart_comparison_summary.csv"), "\n", sep = "")
cat("  ", file.path(opts$results_dir, "linear_projection_summary.csv"), "\n", sep = "")
print(summary, row.names = FALSE)
cat("\nLinear projection control:\n")
print(lin_summary, row.names = FALSE)
