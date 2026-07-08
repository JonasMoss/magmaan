#!/usr/bin/env Rscript

# Compare the standardized, GLS-aligned non-iterative CFA map against iterative
# ML and ULS on a Holzinger-Swineford-shaped three-factor CFA.

suppressWarnings(suppressMessages(library(magmaan)))

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])
source(file.path(dirname(script_file), "..", "_support", "R", "helpers.R"))

require_pkg("lavaan", "needed only for the HolzingerSwineford1939 data set")
set_single_threaded_math()
core <- magmaan::magmaan_core

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "Holzinger-Swineford-shaped comparison of standardized augmented Guttman ILS\n",
    "against normal-theory ML and ULS: estimates, Wald intervals, GOF tests,\n",
    "and a nested visual~~speed restriction.\n\n",
    "Options:\n",
    "  --smoke              Quick run. Default.\n",
    "  --full               Larger default grid.\n",
    "  --reps N             Replications per cell. Smoke: 12; full: 500.\n",
    "  --n LIST             Comma-separated sample sizes. Smoke: 150,300;\n",
    "                       full: 150,300,600.\n",
    "  --scenarios LIST     null,alternative. Default: both.\n",
    "  --cores N            mclapply cores. Default: 1.\n",
    "  --alpha X            Test and CI alpha. Default: 0.05.\n",
    "  --seed-base N        Base RNG seed. Default: 20260708.\n",
    "  --ml-max-iter N      ML/ULS optimizer max iterations. Default: 2000.\n",
    "  --results-dir PATH   Output directory. Default: results.\n",
    "  --help               Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    smoke = TRUE,
    reps = 12L,
    n = c(150L, 300L),
    scenarios = c("null", "alternative"),
    cores = 1L,
    alpha = 0.05,
    seed_base = 20260708L,
    ml_max_iter = 2000L,
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
      if (!"reps" %in% explicit) opts$reps <- 500L
      if (!"n" %in% explicit) opts$n <- c(150L, 300L, 600L)
    } else if (a == "--reps") {
      opts$reps <- as.integer(take(a)); explicit <- c(explicit, "reps")
    } else if (grepl("^--reps=", a)) {
      opts$reps <- as.integer(sub("^--reps=", "", a)); explicit <- c(explicit, "reps")
    } else if (a == "--n") {
      opts$n <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "n")
    } else if (grepl("^--n=", a)) {
      opts$n <- as.integer(parse_csv_arg(sub("^--n=", "", a))); explicit <- c(explicit, "n")
    } else if (a == "--scenarios") {
      opts$scenarios <- parse_csv_arg(take(a)); explicit <- c(explicit, "scenarios")
    } else if (grepl("^--scenarios=", a)) {
      opts$scenarios <- parse_csv_arg(sub("^--scenarios=", "", a))
      explicit <- c(explicit, "scenarios")
    } else if (a == "--cores") {
      opts$cores <- as.integer(take(a))
    } else if (grepl("^--cores=", a)) {
      opts$cores <- as.integer(sub("^--cores=", "", a))
    } else if (a == "--alpha") {
      opts$alpha <- as.numeric(take(a))
    } else if (grepl("^--alpha=", a)) {
      opts$alpha <- as.numeric(sub("^--alpha=", "", a))
    } else if (a == "--seed-base") {
      opts$seed_base <- as.integer(take(a))
    } else if (grepl("^--seed-base=", a)) {
      opts$seed_base <- as.integer(sub("^--seed-base=", "", a))
    } else if (a == "--ml-max-iter") {
      opts$ml_max_iter <- as.integer(take(a))
    } else if (grepl("^--ml-max-iter=", a)) {
      opts$ml_max_iter <- as.integer(sub("^--ml-max-iter=", "", a))
    } else if (a == "--results-dir") {
      opts$results_dir <- take(a)
    } else if (grepl("^--results-dir=", a)) {
      opts$results_dir <- sub("^--results-dir=", "", a)
    } else {
      stop("unknown option: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  bad <- setdiff(opts$scenarios, c("null", "alternative"))
  if (length(bad)) stop("unknown scenarios: ", paste(bad, collapse = ","), call. = FALSE)
  if (!is.finite(opts$reps) || opts$reps < 1L) stop("--reps must be positive", call. = FALSE)
  if (any(!is.finite(opts$n)) || any(opts$n < 30L)) stop("--n values must be at least 30", call. = FALSE)
  if (!is.finite(opts$cores) || opts$cores < 1L) stop("--cores must be positive", call. = FALSE)
  opts
}

opts <- parse_args(commandArgs(TRUE))
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
zcrit <- stats::qnorm(1 - opts$alpha / 2)
fit_control <- list(max_iter = opts$ml_max_iter, ftol = 1e-11, gtol = 1e-8)

factors <- c("visual", "textual", "speed")
ov <- paste0("x", 1:9)
model_h1 <- paste(
  "visual  =~ x1 + x2 + x3",
  "textual =~ x4 + x5 + x6",
  "speed   =~ x7 + x8 + x9",
  sep = "\n"
)
model_h0 <- paste(model_h1, "visual ~~ 0*speed", sep = "\n")
pt_h1 <- core$lavaan_lavaanify(model_h1)
pt_h0 <- core$lavaan_lavaanify(model_h0)

data("HolzingerSwineford1939", package = "lavaan", envir = environment())
hs <- HolzingerSwineford1939[, ov]
X_hs <- as.matrix(hs)

safe_eval <- function(expr) {
  tryCatch(
    list(ok = TRUE, value = force(expr), error = ""),
    error = function(e) list(ok = FALSE, value = NULL, error = conditionMessage(e))
  )
}

sample_stats <- function(X) {
  X <- as.matrix(X)
  colnames(X) <- ov
  core$data_sample_stats_from_raw(X)
}

draw_mvn <- function(n, Sigma, seed) {
  set.seed(seed)
  X <- matrix(stats::rnorm(n * ncol(Sigma)), n, ncol(Sigma)) %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  X
}

ml_fit <- function(pt, ss) core$fit_ml(pt, ss, control = fit_control)
uls_fit <- function(pt, ss) core$fit_uls(pt, ss, control = fit_control)
guttman_fit <- function(pt, ss) {
  magmaan::fit_noniterative_cfa(
    pt, ss, estimator = "guttman_gls_aligned", composite = "standardized")
}

ml_se <- function(fit) {
  info <- core$infer_information_expected(fit)
  vc <- core$infer_vcov_partable(info, fit$partable)
  core$infer_se(vc)
}

uls_se <- function(fit, X) {
  core$robust_se_raw_fit(fit, X, bread = "expected")$se
}

guttman_inf <- function(fit, X, discrepancy) {
  magmaan::noniterative_cfa_inference(
    fit, discrepancy = discrepancy, gamma = "empirical", data = X)
}

guttman_lrt <- function(fit_h1, fit_h0, X, discrepancy) {
  magmaan::noniterative_cfa_pseudo_lrt(
    fit_h1, fit_h0, discrepancy = discrepancy, gamma = "empirical", data = X)
}

free_rows_ordered <- function(pt) {
  rows <- which(pt$free > 0)
  rows[order(pt$free[rows])]
}

param_metadata <- function(pt) {
  rows <- free_rows_ordered(pt)
  lhs <- pt$lhs[rows]
  op <- pt$op[rows]
  rhs <- pt$rhs[rows]
  group <- ifelse(op == "=~", "loading",
           ifelse(op == "~~" & lhs %in% factors & rhs %in% factors, "factor_cov",
           ifelse(op == "~~" & lhs == rhs & lhs %in% ov, "residual_var", "other")))
  data.frame(
    free = pt$free[rows],
    lhs = lhs,
    op = op,
    rhs = rhs,
    param = paste(lhs, op, rhs),
    group = group,
    stringsAsFactors = FALSE
  )
}

param_info <- param_metadata(pt_h1)
resid_mask <- param_info$group == "residual_var"

population_from_fit <- function(fit, scenario) {
  pt <- fit$partable
  Lambda <- matrix(0, length(ov), length(factors), dimnames = list(ov, factors))
  Phi <- matrix(0, length(factors), length(factors), dimnames = list(factors, factors))
  Theta <- stats::setNames(rep(NA_real_, length(ov)), ov)
  for (i in seq_len(nrow(pt))) {
    lhs <- pt$lhs[i]; op <- pt$op[i]; rhs <- pt$rhs[i]
    value <- pt$est[i]
    if (op == "=~" && lhs %in% factors && rhs %in% ov) {
      Lambda[rhs, lhs] <- value
    } else if (op == "~~" && lhs %in% factors && rhs %in% factors) {
      Phi[lhs, rhs] <- value
      Phi[rhs, lhs] <- value
    } else if (op == "~~" && lhs == rhs && lhs %in% ov) {
      Theta[lhs] <- value
    }
  }
  if (scenario == "null") {
    Phi["visual", "speed"] <- 0
    Phi["speed", "visual"] <- 0
  } else if (scenario != "alternative") {
    stop("unknown scenario: ", scenario, call. = FALSE)
  }
  Sigma <- Lambda %*% Phi %*% t(Lambda) + diag(Theta, length(Theta))
  dimnames(Sigma) <- list(ov, ov)
  ev <- eigen(Sigma, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 1e-8) stop("population covariance is not positive definite", call. = FALSE)
  list(Lambda = Lambda, Phi = Phi, Theta = Theta, Sigma = Sigma, scenario = scenario)
}

theta_target <- function(pt, pop) {
  rows <- free_rows_ordered(pt)
  out <- numeric(length(rows))
  for (k in seq_along(rows)) {
    i <- rows[[k]]
    lhs <- pt$lhs[i]; op <- pt$op[i]; rhs <- pt$rhs[i]
    out[[k]] <- if (op == "=~") {
      pop$Lambda[rhs, lhs]
    } else if (op == "~~" && lhs %in% factors && rhs %in% factors) {
      pop$Phi[lhs, rhs]
    } else if (op == "~~" && lhs == rhs && lhs %in% ov) {
      pop$Theta[lhs]
    } else {
      NA_real_
    }
  }
  out
}

converged <- function(fit) is.null(fit$converged) || isTRUE(fit$converged)

improper <- function(fit) {
  if (is.null(fit) || is.null(fit$theta) || length(fit$theta) < length(resid_mask)) return(NA)
  any(as.numeric(fit$theta)[resid_mask] < -1e-8, na.rm = TRUE)
}

estimate_rows <- function(rep_id, scenario, n, estimator, theta, se, target) {
  theta <- as.numeric(theta)
  se <- as.numeric(se)
  ok <- is.finite(theta) & is.finite(se) & is.finite(target) & se >= 0
  data.frame(
    rep = rep_id,
    scenario = scenario,
    n = n,
    estimator = estimator,
    free = param_info$free,
    param = param_info$param,
    group = param_info$group,
    estimate = theta,
    se = se,
    target = target,
    error = theta - target,
    ci_lower = theta - zcrit * se,
    ci_upper = theta + zcrit * se,
    covered = ok & (target >= theta - zcrit * se) & (target <= theta + zcrit * se),
    finite = ok,
    stringsAsFactors = FALSE
  )
}

test_row <- function(rep_id, scenario, n, estimator, test, stat, df, p,
                     p_scaled = NA_real_, warning = "") {
  data.frame(
    rep = rep_id,
    scenario = scenario,
    n = n,
    estimator = estimator,
    test = test,
    statistic = as.numeric(stat),
    df = as.numeric(df),
    p = as.numeric(p),
    p_scaled = as.numeric(p_scaled),
    reject = is.finite(p) && p < opts$alpha,
    warning = warning,
    stringsAsFactors = FALSE
  )
}

error_diag <- function(rep_id, scenario, n, stage, estimator, error) {
  data.frame(
    rep = rep_id, scenario = scenario, n = n, stage = stage,
    estimator = estimator, ok = FALSE, converged = FALSE,
    improper = NA, error = error, stringsAsFactors = FALSE
  )
}

ok_diag <- function(rep_id, scenario, n, stage, estimator, fit) {
  data.frame(
    rep = rep_id, scenario = scenario, n = n, stage = stage,
    estimator = estimator, ok = TRUE, converged = converged(fit),
    improper = improper(fit), error = "", stringsAsFactors = FALSE
  )
}

run_rep <- function(rep_id, scenario, n, pop) {
  seed <- opts$seed_base + rep_id + 10000L * match(scenario, c("null", "alternative")) + n
  X <- draw_mvn(n, pop$Sigma, seed)
  ss <- sample_stats(X)
  target <- theta_target(pt_h1, pop)

  est_rows <- list()
  test_rows <- list()
  diag_rows <- list()

  g1 <- safe_eval(guttman_fit(pt_h1, ss))
  g0 <- safe_eval(guttman_fit(pt_h0, ss))
  if (g1$ok) diag_rows <- c(diag_rows, list(ok_diag(rep_id, scenario, n, "H1", "guttman_std", g1$value)))
  else diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "H1", "guttman_std", g1$error)))
  if (g0$ok) diag_rows <- c(diag_rows, list(ok_diag(rep_id, scenario, n, "H0", "guttman_std", g0$value)))
  else diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "H0", "guttman_std", g0$error)))
  if (g1$ok) {
    for (disc in c("ntml", "uls")) {
      gi <- safe_eval(guttman_inf(g1$value, X, disc))
      label <- paste0("guttman_std_", disc)
      if (gi$ok) {
        if (disc == "ntml") {
          est_rows <- c(est_rows, list(estimate_rows(
            rep_id, scenario, n, "guttman_std", g1$value$theta, gi$value$se,
            target)))
        }
        test_rows <- c(test_rows, list(test_row(
          rep_id, scenario, n, label, "gof", gi$value$T, gi$value$df,
          gi$value$p_mixture, gi$value$p_scaled,
          paste(gi$value$warnings, collapse = "; "))))
      } else {
        diag_rows <- c(diag_rows, list(error_diag(
          rep_id, scenario, n, "inference", label, gi$error)))
      }
      if (gi$ok && g0$ok) {
        gd <- safe_eval(guttman_lrt(g1$value, g0$value, X, disc))
        if (gd$ok) {
          test_rows <- c(test_rows, list(test_row(
            rep_id, scenario, n, label, "nested_visual_speed_zero",
            gd$value$T_diff, gd$value$df_diff, gd$value$p_mixture,
            gd$value$p_scaled, paste(gd$value$warnings, collapse = "; "))))
        } else {
          diag_rows <- c(diag_rows, list(error_diag(
            rep_id, scenario, n, "nested", label, gd$error)))
        }
      }
    }
  }

  m1 <- safe_eval(ml_fit(pt_h1, ss))
  m0 <- safe_eval(ml_fit(pt_h0, ss))
  if (m1$ok) diag_rows <- c(diag_rows, list(ok_diag(rep_id, scenario, n, "H1", "ml_nt", m1$value)))
  else diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "H1", "ml_nt", m1$error)))
  if (m0$ok) diag_rows <- c(diag_rows, list(ok_diag(rep_id, scenario, n, "H0", "ml_nt", m0$value)))
  else diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "H0", "ml_nt", m0$error)))
  if (m1$ok && converged(m1$value)) {
    ms <- safe_eval(ml_se(m1$value))
    if (ms$ok) {
      est_rows <- c(est_rows, list(estimate_rows(
        rep_id, scenario, n, "ml_nt", m1$value$theta, ms$value, target)))
    } else {
      diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "se", "ml_nt", ms$error)))
    }
    T1 <- core$infer_chi2_stat(ss, m1$value$fmin)
    df1 <- core$infer_df_stat(m1$value$partable, ss)
    test_rows <- c(test_rows, list(test_row(
      rep_id, scenario, n, "ml_nt", "gof", T1, df1,
      stats::pchisq(T1, df1, lower.tail = FALSE))))
    if (m0$ok && converged(m0$value)) {
      T0 <- core$infer_chi2_stat(ss, m0$value$fmin)
      df0 <- core$infer_df_stat(m0$value$partable, ss)
      Td <- T0 - T1
      dfd <- df0 - df1
      test_rows <- c(test_rows, list(test_row(
        rep_id, scenario, n, "ml_nt", "nested_visual_speed_zero", Td, dfd,
        stats::pchisq(Td, dfd, lower.tail = FALSE))))
    }
  }

  u1 <- safe_eval(uls_fit(pt_h1, ss))
  u0 <- safe_eval(uls_fit(pt_h0, ss))
  if (u1$ok) diag_rows <- c(diag_rows, list(ok_diag(rep_id, scenario, n, "H1", "uls_emp", u1$value)))
  else diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "H1", "uls_emp", u1$error)))
  if (u0$ok) diag_rows <- c(diag_rows, list(ok_diag(rep_id, scenario, n, "H0", "uls_emp", u0$value)))
  else diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "H0", "uls_emp", u0$error)))
  if (u1$ok && converged(u1$value)) {
    us <- safe_eval(uls_se(u1$value, X))
    if (us$ok) {
      est_rows <- c(est_rows, list(estimate_rows(
        rep_id, scenario, n, "uls_emp", u1$value$theta, us$value, target)))
    } else {
      diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "se", "uls_emp", us$error)))
    }
    if (u0$ok && converged(u0$value)) {
      ud <- safe_eval(core$continuous_ls_profile_lrt(u1$value, u0$value, list(X)))
      if (ud$ok) {
        test_rows <- c(test_rows, list(test_row(
          rep_id, scenario, n, "uls_emp", "nested_visual_speed_zero",
          ud$value$T_diff, ud$value$df_diff, ud$value$p_mixture,
          ud$value$p_scaled, paste(ud$value$warnings, collapse = "; "))))
      } else {
        diag_rows <- c(diag_rows, list(error_diag(rep_id, scenario, n, "nested", "uls_emp", ud$error)))
      }
    }
  }

  list(
    estimates = if (length(est_rows)) do.call(rbind, est_rows) else data.frame(),
    tests = if (length(test_rows)) do.call(rbind, test_rows) else data.frame(),
    diagnostics = if (length(diag_rows)) do.call(rbind, diag_rows) else data.frame()
  )
}

summarise_estimates <- function(x) {
  if (!nrow(x)) return(data.frame())
  key <- interaction(x$scenario, x$n, x$estimator, x$group, drop = TRUE, sep = "\r")
  rows <- lapply(split(x, key), function(d) {
    ok <- isTRUE(d$finite)
    data.frame(
      scenario = d$scenario[[1]],
      n = d$n[[1]],
      estimator = d$estimator[[1]],
      group = d$group[[1]],
      n_rows = nrow(d),
      n_finite = sum(d$finite),
      bias = mean(d$error[d$finite], na.rm = TRUE),
      rmse = sqrt(mean(d$error[d$finite]^2, na.rm = TRUE)),
      coverage = mean(d$covered[d$finite], na.rm = TRUE),
      avg_se = mean(d$se[d$finite], na.rm = TRUE),
      emp_sd = stats::sd(d$estimate[d$finite], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

summarise_tests <- function(x) {
  if (!nrow(x)) return(data.frame())
  key <- interaction(x$scenario, x$n, x$estimator, x$test, drop = TRUE, sep = "\r")
  rows <- lapply(split(x, key), function(d) {
    ok <- is.finite(d$p)
    data.frame(
      scenario = d$scenario[[1]],
      n = d$n[[1]],
      estimator = d$estimator[[1]],
      test = d$test[[1]],
      n_rows = nrow(d),
      n_finite = sum(ok),
      rejection_rate = mean(d$reject[ok], na.rm = TRUE),
      mean_p = mean(d$p[ok], na.rm = TRUE),
      negative_stat_rate = mean(d$statistic[ok] < 0, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

summarise_diagnostics <- function(x) {
  if (!nrow(x)) return(data.frame())
  key <- interaction(x$scenario, x$n, x$stage, x$estimator, drop = TRUE, sep = "\r")
  rows <- lapply(split(x, key), function(d) {
    data.frame(
      scenario = d$scenario[[1]],
      n = d$n[[1]],
      stage = d$stage[[1]],
      estimator = d$estimator[[1]],
      attempts = nrow(d),
      ok_rate = mean(d$ok),
      converged_rate = mean(d$converged, na.rm = TRUE),
      improper_rate = mean(d$improper, na.rm = TRUE),
      first_error = paste(unique(d$error[nzchar(d$error)])[1] %||% "", collapse = ""),
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, rows)
}

snapshot_rows <- function(fit, se, estimator, target = rep(NA_real_, length(se))) {
  d <- param_info
  d$estimator <- estimator
  d$estimate <- as.numeric(fit$theta)
  d$se <- as.numeric(se)
  d$ci_lower <- d$estimate - zcrit * d$se
  d$ci_upper <- d$estimate + zcrit * d$se
  d$target <- target
  d
}

cat("Fitting Holzinger-Swineford snapshot and population seed...\n")
hs_stats <- sample_stats(X_hs)
hs_ml <- ml_fit(pt_h1, hs_stats)
hs_uls <- uls_fit(pt_h1, hs_stats)
hs_g <- guttman_fit(pt_h1, hs_stats)
hs_gi <- guttman_inf(hs_g, X_hs, "ntml")
snapshot <- rbind(
  snapshot_rows(hs_ml, ml_se(hs_ml), "ml_nt"),
  snapshot_rows(hs_uls, uls_se(hs_uls, X_hs), "uls_emp"),
  snapshot_rows(hs_g, hs_gi$se, "guttman_std")
)
write_csv(snapshot, file.path(opts$results_dir, "hs_snapshot.csv"))

populations <- lapply(opts$scenarios, function(s) population_from_fit(hs_ml, s))
names(populations) <- opts$scenarios
pop_rows <- do.call(rbind, lapply(populations, function(pop) {
  data.frame(
    scenario = pop$scenario,
    min_eigen = min(eigen(pop$Sigma, symmetric = TRUE, only.values = TRUE)$values),
    visual_speed_cov = pop$Phi["visual", "speed"],
    stringsAsFactors = FALSE
  )
}))
write_csv(pop_rows, file.path(opts$results_dir, "population.csv"))

plan <- expand.grid(
  scenario = opts$scenarios,
  n = opts$n,
  stringsAsFactors = FALSE
)
cat(sprintf("Running %d cells x %d replications (cores=%d)\n",
            nrow(plan), opts$reps, opts$cores))

all_estimates <- list()
all_tests <- list()
all_diagnostics <- list()
for (cell in seq_len(nrow(plan))) {
  scenario <- plan$scenario[[cell]]
  n <- plan$n[[cell]]
  pop <- populations[[scenario]]
  cat(sprintf("[%d/%d] scenario=%s n=%d reps=%d\n",
              cell, nrow(plan), scenario, n, opts$reps))
  reps <- seq_len(opts$reps)
  cell_out <- if (opts$cores > 1L) {
    parallel::mclapply(reps, run_rep, scenario = scenario, n = n, pop = pop,
                       mc.cores = opts$cores, mc.preschedule = TRUE)
  } else {
    lapply(reps, run_rep, scenario = scenario, n = n, pop = pop)
  }
  all_estimates <- c(all_estimates, lapply(cell_out, `[[`, "estimates"))
  all_tests <- c(all_tests, lapply(cell_out, `[[`, "tests"))
  all_diagnostics <- c(all_diagnostics, lapply(cell_out, `[[`, "diagnostics"))
}

estimate_draws <- if (length(all_estimates)) do.call(rbind, all_estimates) else data.frame()
test_draws <- if (length(all_tests)) do.call(rbind, all_tests) else data.frame()
diagnostics <- if (length(all_diagnostics)) do.call(rbind, all_diagnostics) else data.frame()

estimate_summary <- summarise_estimates(estimate_draws)
test_summary <- summarise_tests(test_draws)
diagnostic_summary <- summarise_diagnostics(diagnostics)

write_csv(estimate_draws, file.path(opts$results_dir, "estimate_draws.csv"))
write_csv(estimate_summary, file.path(opts$results_dir, "estimate_summary.csv"))
write_csv(test_draws, file.path(opts$results_dir, "test_draws.csv"))
write_csv(test_summary, file.path(opts$results_dir, "test_summary.csv"))
write_csv(diagnostics, file.path(opts$results_dir, "diagnostics.csv"))
write_csv(diagnostic_summary, file.path(opts$results_dir, "diagnostic_summary.csv"))
write_metadata(
  file.path(opts$results_dir, "metadata.csv"),
  values = list(
    reps = opts$reps,
    n = opts$n,
    scenarios = opts$scenarios,
    alpha = opts$alpha,
    seed_base = opts$seed_base,
    cores = opts$cores,
    ml_max_iter = opts$ml_max_iter,
    population_source = "magmaan ML fit to lavaan::HolzingerSwineford1939"
  ),
  packages = c("magmaan", "lavaan")
)

cat("Wrote:\n")
for (nm in c("hs_snapshot.csv", "population.csv", "estimate_draws.csv",
             "estimate_summary.csv", "test_draws.csv", "test_summary.csv",
             "diagnostics.csv", "diagnostic_summary.csv", "metadata.csv")) {
  cat("  ", file.path(opts$results_dir, nm), "\n", sep = "")
}
