#!/usr/bin/env Rscript

# Overnight-style RMSE/coverage study for closed-form Guttman CFA maps.
# Categorical cells use magmaan's ordinal Pearson-code simulator and then fit
# the code covariance directly; no polychoric statistics enter the Guttman,
# NTML, or ULS arms.

suppressWarnings(suppressMessages(library(magmaan)))

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[[1L]])
source(file.path(dirname(script_file), "..", "_support", "R", "helpers.R"))

set_single_threaded_math()
core <- magmaan::magmaan_core

usage <- function() {
  cat(
    "Usage: Rscript run_experiment.R [options]\n\n",
    "RMSE/coverage comparison for standardized/aligned Guttman CFA maps,\n",
    "continuous NTML, and ULS. Ordinal cells simulate category-score data with\n",
    "magmaan::sim_ordcorr_calibrate(metric = 'pearson_codes') and fit the\n",
    "observed code covariance directly; no polychoric inputs are used.\n\n",
    "Options:\n",
    "  --probe              Tiny timing probe. Default if neither --smoke nor --full.\n",
    "  --smoke              Small validation run.\n",
    "  --full               Larger overnight grid.\n",
    "  --reps N             Replications per cell. Probe: 2; smoke: 5; full: 50.\n",
    "  --n LIST             Comma-separated sample sizes.\n",
    "  --generators LIST    normal,ig,ordinal. Default depends on mode.\n",
    "  --factors LIST       Comma-separated factor counts.\n",
    "  --indicators LIST    Comma-separated indicators per factor.\n",
    "  --rho LIST           Comma-separated latent correlations.\n",
    "  --scales LIST        equal,unequal. Continuous scales or ordinal marginals.\n",
    "  --loadings LIST      mixed,tau. Configural loading patterns.\n",
    "  --no-restricted      Drop tau/residual-equality constrained cells.\n",
    "  --max-cells N        Keep only first N cells after grid construction.\n",
    "  --cores N            mclapply cores. Default: 1.\n",
    "  --alpha X            CI alpha. Default: 0.05.\n",
    "  --seed-base N        Base RNG seed. Default: 20260708.\n",
    "  --ml-max-iter N      ML/ULS optimizer max iterations. Default: 1500.\n",
    "  --keep-draws         Write raw estimate/whole draw rows as CSV.\n",
    "  --results-dir PATH   Output directory. Default: results.\n",
    "  --help               Show this help.\n",
    sep = ""
  )
}

parse_args <- function(args) {
  opts <- list(
    mode = "probe",
    reps = 2L,
    n = c(120L),
    generators = c("normal", "ig", "ordinal"),
    factors = c(2L, 3L),
    indicators = c(3L),
    rho = c(0, 0.35),
    scales = c("equal"),
    loadings = c("mixed"),
    include_restricted = TRUE,
    max_cells = NA_integer_,
    cores = 1L,
    alpha = 0.05,
    seed_base = 20260708L,
    ml_max_iter = 1500L,
    keep_draws = FALSE,
    results_dir = experiment_path("results")
  )
  explicit <- character()
  apply_mode <- function(mode) {
    opts$mode <<- mode
    if (mode == "probe") {
      if (!"reps" %in% explicit) opts$reps <<- 2L
      if (!"n" %in% explicit) opts$n <<- c(120L)
      if (!"generators" %in% explicit) opts$generators <<- c("normal", "ig", "ordinal")
      if (!"factors" %in% explicit) opts$factors <<- c(2L, 3L)
      if (!"indicators" %in% explicit) opts$indicators <<- c(3L)
      if (!"rho" %in% explicit) opts$rho <<- c(0, 0.35)
      if (!"scales" %in% explicit) opts$scales <<- c("equal")
      if (!"loadings" %in% explicit) opts$loadings <<- c("mixed")
    } else if (mode == "smoke") {
      if (!"reps" %in% explicit) opts$reps <<- 5L
      if (!"n" %in% explicit) opts$n <<- c(120L, 300L)
      if (!"generators" %in% explicit) opts$generators <<- c("normal", "ordinal")
      if (!"factors" %in% explicit) opts$factors <<- c(2L, 3L)
      if (!"indicators" %in% explicit) opts$indicators <<- c(3L)
      if (!"rho" %in% explicit) opts$rho <<- c(0, 0.35)
      if (!"scales" %in% explicit) opts$scales <<- c("equal", "unequal")
      if (!"loadings" %in% explicit) opts$loadings <<- c("mixed")
    } else if (mode == "full") {
      if (!"reps" %in% explicit) opts$reps <<- 50L
      if (!"n" %in% explicit) opts$n <<- c(100L, 300L, 800L)
      if (!"generators" %in% explicit) opts$generators <<- c("normal", "ig", "ordinal")
      if (!"factors" %in% explicit) opts$factors <<- c(2L, 3L, 5L)
      if (!"indicators" %in% explicit) opts$indicators <<- c(3L, 5L)
      if (!"rho" %in% explicit) opts$rho <<- c(0, 0.35)
      if (!"scales" %in% explicit) opts$scales <<- c("equal", "unequal")
      if (!"loadings" %in% explicit) opts$loadings <<- c("mixed")
    }
  }
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
    } else if (a == "--probe") {
      apply_mode("probe")
    } else if (a == "--smoke") {
      apply_mode("smoke")
    } else if (a == "--full") {
      apply_mode("full")
    } else if (a == "--reps") {
      opts$reps <- as.integer(take(a)); explicit <- c(explicit, "reps")
    } else if (grepl("^--reps=", a)) {
      opts$reps <- as.integer(sub("^--reps=", "", a)); explicit <- c(explicit, "reps")
    } else if (a == "--n") {
      opts$n <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "n")
    } else if (grepl("^--n=", a)) {
      opts$n <- as.integer(parse_csv_arg(sub("^--n=", "", a))); explicit <- c(explicit, "n")
    } else if (a == "--generators") {
      opts$generators <- parse_csv_arg(take(a)); explicit <- c(explicit, "generators")
    } else if (grepl("^--generators=", a)) {
      opts$generators <- parse_csv_arg(sub("^--generators=", "", a)); explicit <- c(explicit, "generators")
    } else if (a == "--factors") {
      opts$factors <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "factors")
    } else if (grepl("^--factors=", a)) {
      opts$factors <- as.integer(parse_csv_arg(sub("^--factors=", "", a))); explicit <- c(explicit, "factors")
    } else if (a == "--indicators") {
      opts$indicators <- as.integer(parse_csv_arg(take(a))); explicit <- c(explicit, "indicators")
    } else if (grepl("^--indicators=", a)) {
      opts$indicators <- as.integer(parse_csv_arg(sub("^--indicators=", "", a))); explicit <- c(explicit, "indicators")
    } else if (a == "--rho") {
      opts$rho <- parse_csv_numeric(take(a)); explicit <- c(explicit, "rho")
    } else if (grepl("^--rho=", a)) {
      opts$rho <- parse_csv_numeric(sub("^--rho=", "", a)); explicit <- c(explicit, "rho")
    } else if (a == "--scales") {
      opts$scales <- parse_csv_arg(take(a)); explicit <- c(explicit, "scales")
    } else if (grepl("^--scales=", a)) {
      opts$scales <- parse_csv_arg(sub("^--scales=", "", a)); explicit <- c(explicit, "scales")
    } else if (a == "--loadings") {
      opts$loadings <- parse_csv_arg(take(a)); explicit <- c(explicit, "loadings")
    } else if (grepl("^--loadings=", a)) {
      opts$loadings <- parse_csv_arg(sub("^--loadings=", "", a)); explicit <- c(explicit, "loadings")
    } else if (a == "--no-restricted") {
      opts$include_restricted <- FALSE
    } else if (a == "--max-cells") {
      opts$max_cells <- as.integer(take(a))
    } else if (grepl("^--max-cells=", a)) {
      opts$max_cells <- as.integer(sub("^--max-cells=", "", a))
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
    } else if (a == "--keep-draws") {
      opts$keep_draws <- TRUE
    } else if (a == "--results-dir") {
      opts$results_dir <- take(a)
    } else if (grepl("^--results-dir=", a)) {
      opts$results_dir <- sub("^--results-dir=", "", a)
    } else {
      stop("unknown option: ", a, call. = FALSE)
    }
    i <- i + 1L
  }
  bad_gen <- setdiff(opts$generators, c("normal", "ig", "ordinal"))
  bad_scales <- setdiff(opts$scales, c("equal", "unequal"))
  bad_loadings <- setdiff(opts$loadings, c("mixed", "tau"))
  if (length(bad_gen)) stop("unknown generators: ", paste(bad_gen, collapse = ","), call. = FALSE)
  if (length(bad_scales)) stop("unknown scales: ", paste(bad_scales, collapse = ","), call. = FALSE)
  if (length(bad_loadings)) stop("unknown loadings: ", paste(bad_loadings, collapse = ","), call. = FALSE)
  if (!is.finite(opts$reps) || opts$reps < 1L) stop("--reps must be positive", call. = FALSE)
  if (any(!is.finite(opts$n)) || any(opts$n < 30L)) stop("--n values must be at least 30", call. = FALSE)
  if (any(opts$factors < 1L) || any(opts$indicators < 3L)) stop("need at least 3 indicators per factor", call. = FALSE)
  if (!is.finite(opts$cores) || opts$cores < 1L) stop("--cores must be positive", call. = FALSE)
  opts
}

opts <- parse_args(commandArgs(TRUE))
dir.create(opts$results_dir, recursive = TRUE, showWarnings = FALSE)
zcrit <- stats::qnorm(1 - opts$alpha / 2)
fit_control <- list(max_iter = opts$ml_max_iter, ftol = 1e-10, gtol = 1e-7)

condition_cols <- c("generator", "factors", "indicators", "rho", "scale",
                    "loading", "constraint", "n")

interaction_key <- function(df, cols) {
  do.call(interaction, c(df[cols], list(drop = TRUE, sep = "\r")))
}

bind_rows <- function(xs) {
  xs <- xs[vapply(xs, function(x) is.data.frame(x) && nrow(x) > 0L, logical(1))]
  if (!length(xs)) return(data.frame())
  do.call(rbind, xs)
}

safe_eval <- function(expr) {
  tryCatch(
    list(ok = TRUE, value = force(expr), error = ""),
    error = function(e) list(ok = FALSE, value = NULL, error = conditionMessage(e))
  )
}

timed_eval <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  out <- safe_eval(expr)
  out$ms <- 1000 * (proc.time()[["elapsed"]] - t0)
  out
}

vars_for <- function(q, m) paste0("x", seq_len(q * m))
factors_for <- function(q) paste0("f", seq_len(q))

block_index <- function(q, m) rep(seq_len(q), each = m)

model_syntax <- function(q, m, restricted = FALSE) {
  vars <- vars_for(q, m)
  fac <- factors_for(q)
  rows <- character()
  for (f in seq_len(q)) {
    idx <- ((f - 1L) * m + 1L):(f * m)
    rhs <- vars[idx]
    if (restricted) {
      rhs[-1L] <- paste0("l", f, "*", rhs[-1L])
    }
    rows <- c(rows, paste0(fac[[f]], " =~ ", paste(rhs, collapse = " + ")))
  }
  if (restricted) {
    for (f in seq_len(q)) {
      idx <- ((f - 1L) * m + 1L):(f * m)
      for (v in vars[idx]) rows <- c(rows, paste0(v, " ~~ e", f, "*", v))
    }
  }
  paste(rows, collapse = "\n")
}

compound_phi <- function(q, rho) {
  Phi <- matrix(rho, q, q)
  diag(Phi) <- 1
  Phi
}

loading_values <- function(q, m, loading) {
  if (loading == "tau") return(rep(0.70, q * m))
  base <- rep(c(0.55, 0.65, 0.75, 0.60, 0.70), length.out = m)
  rep(base, q)
}

continuous_scales <- function(q, m, scale) {
  if (scale == "equal") return(rep(1, q * m))
  base <- rep(c(0.70, 1.00, 1.45, 0.85, 1.25), length.out = m)
  rep(base, q)
}

ordinal_probs_for <- function(q, m, scale) {
  base <- c(.08, .17, .35, .25, .15)
  if (scale == "equal") return(rep(list(base), q * m))
  pats <- list(
    c(.03, .12, .30, .35, .20),
    c(.15, .30, .30, .18, .07),
    c(.05, .20, .50, .20, .05),
    c(.20, .25, .30, .17, .08),
    c(.04, .16, .28, .32, .20)
  )
  rep(pats, length.out = q * m)
}

code_variance <- function(prob) {
  x <- seq_along(prob)
  mu <- sum(x * prob)
  sum((x - mu)^2 * prob)
}

make_population <- function(q, m, rho, loading, scale, generator) {
  vars <- vars_for(q, m)
  fac <- factors_for(q)
  blocks <- block_index(q, m)
  lambda_std <- loading_values(q, m, loading)
  theta_std <- 1 - lambda_std^2
  Phi_cor <- compound_phi(q, rho)
  Lstd <- matrix(0, q * m, q, dimnames = list(vars, fac))
  for (j in seq_along(vars)) Lstd[j, blocks[[j]]] <- lambda_std[[j]]
  R <- Lstd %*% Phi_cor %*% t(Lstd) + diag(theta_std, q * m)
  dimnames(R) <- list(vars, vars)
  marginals <- NULL
  scales <- continuous_scales(q, m, scale)
  if (generator == "ordinal") {
    marginals <- ordinal_probs_for(q, m, scale)
    scales <- sqrt(vapply(marginals, code_variance, numeric(1)))
  }
  D <- diag(scales, q * m)
  Sigma <- D %*% R %*% D
  Lambda <- matrix(0, q * m, q, dimnames = list(vars, fac))
  Phi <- matrix(0, q, q, dimnames = list(fac, fac))
  Theta <- stats::setNames(scales^2 * theta_std, vars)
  markers <- integer(q)
  for (f in seq_len(q)) {
    markers[[f]] <- (f - 1L) * m + 1L
  }
  marker_scale_loading <- scales[markers] * lambda_std[markers]
  for (j in seq_along(vars)) {
    f <- blocks[[j]]
    Lambda[j, f] <- scales[[j]] * lambda_std[[j]] / marker_scale_loading[[f]]
  }
  for (f in seq_len(q)) {
    for (g in seq_len(q)) {
      Phi[f, g] <- Phi_cor[f, g] * marker_scale_loading[[f]] * marker_scale_loading[[g]]
    }
  }
  common <- Lambda %*% Phi %*% t(Lambda)
  ev <- eigen(Sigma, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) <= 1e-8) stop("population covariance is not positive definite", call. = FALSE)
  list(
    vars = vars, factors = fac, blocks = blocks, markers = markers,
    lambda_std = lambda_std, Phi_cor = Phi_cor, R = R, scales = scales,
    marginals = marginals, Lambda = Lambda, Phi = Phi, Theta = Theta,
    common = common, sigma = Sigma, min_eigen = min(ev)
  )
}

sample_stats <- function(X, vars) {
  X <- as.matrix(X)
  colnames(X) <- vars
  core$data_sample_stats_from_raw(X)
}

draw_matrix <- function(d) {
  if (is.matrix(d) || is.data.frame(d)) return(as.matrix(d))
  if (is.list(d) && !is.null(d$X)) return(as.matrix(d$X))
  stop("unsupported simulation draw shape", call. = FALSE)
}

sim_mvn <- function(n, Sigma, seed) {
  set.seed(seed)
  X <- matrix(stats::rnorm(n * ncol(Sigma)), n, ncol(Sigma)) %*% chol(Sigma)
  colnames(X) <- colnames(Sigma)
  X
}

calibration_cache <- new.env(parent = emptyenv())

cache_key <- function(prefix, pop, extra = "") {
  paste(prefix, nrow(pop$sigma), paste(round(pop$sigma[upper.tri(pop$sigma, diag = TRUE)], 8), collapse = ","),
        extra, sep = "|")
}

draws_for_condition <- function(pop, generator, n, reps, seed_base) {
  if (generator == "normal") {
    return(lapply(seq_len(reps), function(r) sim_mvn(n, pop$sigma, seed_base + r)))
  }
  if (generator == "ig") {
    key <- cache_key("ig", pop, "skew0.2-kurt0.5")
    if (!exists(key, envir = calibration_cache, inherits = FALSE)) {
      cal <- core$sim_ig_calibrate(
        pop$sigma,
        target_skewness = rep(0.2, nrow(pop$sigma)),
        target_excess_kurtosis = rep(0.5, nrow(pop$sigma)),
        generator_family = "tukey_gh"
      )
      assign(key, cal, envir = calibration_cache)
    }
    sim <- core$sim_ig_draw(get(key, envir = calibration_cache), n = n, reps = reps,
                            seed_base = seed_base)
    return(lapply(sim$draws, function(d) {
      x <- draw_matrix(d)
      colnames(x) <- pop$vars
      x
    }))
  }
  if (generator == "ordinal") {
    key <- cache_key("ordinal", pop, paste(vapply(pop$marginals, paste, "", collapse = ":"), collapse = ";"))
    if (!exists(key, envir = calibration_cache, inherits = FALSE)) {
      marginals <- stats::setNames(pop$marginals, pop$vars)
      cal <- magmaan::sim_ordcorr_calibrate(
        pop$R, marginals, metric = "pearson_codes", matrix_repair = "none")
      assign(key, cal, envir = calibration_cache)
    }
    sim <- magmaan::sim_ordcorr_draw(get(key, envir = calibration_cache),
                                     n = n, reps = reps, seed_base = seed_base)
    return(lapply(sim$draws, function(d) {
      x <- draw_matrix(d)
      colnames(x) <- pop$vars
      x
    }))
  }
  stop("unknown generator: ", generator, call. = FALSE)
}

free_param_info <- function(pt, vars, fac) {
  rows <- which(pt$free > 0)
  rows <- rows[!duplicated(pt$free[rows])]
  rows <- rows[order(pt$free[rows])]
  lhs <- pt$lhs[rows]
  op <- pt$op[rows]
  rhs <- pt$rhs[rows]
  group <- ifelse(op == "=~", "loading",
           ifelse(op == "~~" & lhs %in% fac & rhs %in% fac & lhs == rhs, "factor_var",
           ifelse(op == "~~" & lhs %in% fac & rhs %in% fac, "factor_cov",
           ifelse(op == "~~" & lhs == rhs & lhs %in% vars, "residual_var", "other"))))
  data.frame(
    free = pt$free[rows],
    row = rows,
    lhs = lhs,
    op = op,
    rhs = rhs,
    param = paste(lhs, op, rhs),
    group = group,
    stringsAsFactors = FALSE
  )
}

theta_target <- function(info, pop) {
  out <- numeric(nrow(info))
  for (i in seq_len(nrow(info))) {
    lhs <- info$lhs[[i]]
    op <- info$op[[i]]
    rhs <- info$rhs[[i]]
    out[[i]] <- if (op == "=~") {
      pop$Lambda[rhs, lhs]
    } else if (op == "~~" && lhs %in% pop$factors && rhs %in% pop$factors) {
      pop$Phi[lhs, rhs]
    } else if (op == "~~" && lhs == rhs && lhs %in% pop$vars) {
      pop$Theta[[lhs]]
    } else {
      NA_real_
    }
  }
  out
}

matrices_from_partable <- function(pt, pop) {
  Lambda <- matrix(0, length(pop$vars), length(pop$factors),
                   dimnames = list(pop$vars, pop$factors))
  Phi <- matrix(NA_real_, length(pop$factors), length(pop$factors),
                dimnames = list(pop$factors, pop$factors))
  Theta <- stats::setNames(rep(NA_real_, length(pop$vars)), pop$vars)
  est <- pt$est %||% pt$ustart
  for (i in seq_len(nrow(pt))) {
    lhs <- pt$lhs[[i]]
    op <- pt$op[[i]]
    rhs <- pt$rhs[[i]]
    value <- est[[i]]
    if (!is.finite(value)) next
    if (op == "=~" && lhs %in% pop$factors && rhs %in% pop$vars) {
      Lambda[rhs, lhs] <- value
    } else if (op == "~~" && lhs %in% pop$factors && rhs %in% pop$factors) {
      Phi[lhs, rhs] <- value
      Phi[rhs, lhs] <- value
    } else if (op == "~~" && lhs == rhs && lhs %in% pop$vars) {
      Theta[lhs] <- value
    }
  }
  if (anyNA(Phi) || anyNA(Theta)) return(NULL)
  common <- Lambda %*% Phi %*% t(Lambda)
  sigma <- common + diag(Theta, length(Theta))
  list(Lambda = Lambda, Phi = Phi, Theta = Theta, common = common, sigma = sigma)
}

fit_whole_row <- function(cond, rep_id, estimator, fit, pop) {
  mats <- matrices_from_partable(fit$partable, pop)
  ok <- !is.null(mats) && all(is.finite(mats$sigma))
  data.frame(
    cond,
    rep = rep_id,
    estimator = estimator,
    finite = ok,
    common_mse = if (ok) mean((mats$common - pop$common)^2) else NA_real_,
    sigma_mse = if (ok) mean((mats$sigma - pop$sigma)^2) else NA_real_,
    common_diag_mse = if (ok) mean((diag(mats$common) - diag(pop$common))^2) else NA_real_,
    residual_diag_mse = if (ok) mean((mats$Theta - pop$Theta)^2) else NA_real_,
    min_resid = if (ok) min(mats$Theta) else NA_real_,
    sigma_min_eig = if (ok) min(eigen(mats$sigma, symmetric = TRUE, only.values = TRUE)$values) else NA_real_,
    stringsAsFactors = FALSE
  )
}

estimate_rows <- function(cond, rep_id, estimator, theta, se, target, info) {
  theta <- as.numeric(theta)
  se <- as.numeric(se)
  target <- as.numeric(target)
  len <- min(length(theta), length(se), length(target), nrow(info))
  theta <- theta[seq_len(len)]
  se <- se[seq_len(len)]
  target <- target[seq_len(len)]
  d <- info[seq_len(len), , drop = FALSE]
  finite <- is.finite(theta) & is.finite(se) & is.finite(target) & se >= 0
  base <- cond[rep(1L, len), , drop = FALSE]
  rownames(base) <- NULL
  data.frame(
    base,
    rep = rep_id,
    estimator = estimator,
    free = d$free,
    lhs = d$lhs,
    op = d$op,
    rhs = d$rhs,
    param = d$param,
    group = d$group,
    estimate = theta,
    se = se,
    target = target,
    error = theta - target,
    ci_lower = theta - zcrit * se,
    ci_upper = theta + zcrit * se,
    covered = finite & (target >= theta - zcrit * se) & (target <= theta + zcrit * se),
    finite = finite,
    stringsAsFactors = FALSE
  )
}

converged <- function(fit) is.null(fit$converged) || isTRUE(fit$converged)

fit_diagnostics <- function(cond, rep_id, stage, estimator, fit_ok, fit = NULL,
                            fit_ms = NA_real_, se_ok = NA, se_ms = NA_real_,
                            error = "") {
  min_resid <- NA_real_
  sigma_min_eig <- NA_real_
  improper <- NA
  if (isTRUE(fit_ok) && !is.null(fit)) {
    mats <- matrices_from_partable(fit$partable, attr(cond, "pop"))
    if (!is.null(mats)) {
      min_resid <- min(mats$Theta)
      sigma_min_eig <- min(eigen(mats$sigma, symmetric = TRUE, only.values = TRUE)$values)
      improper <- min_resid < -1e-8 || sigma_min_eig < -1e-8
    }
  }
  data.frame(
    cond,
    rep = rep_id,
    stage = stage,
    estimator = estimator,
    fit_ok = isTRUE(fit_ok),
    converged = if (isTRUE(fit_ok) && !is.null(fit)) converged(fit) else FALSE,
    se_ok = if (is.na(se_ok)) NA else isTRUE(se_ok),
    improper = improper,
    min_resid = min_resid,
    sigma_min_eig = sigma_min_eig,
    fit_ms = fit_ms,
    se_ms = se_ms,
    total_ms = fit_ms + ifelse(is.finite(se_ms), se_ms, 0),
    error = error,
    stringsAsFactors = FALSE
  )
}

guttman_fit <- function(pt, ss, composite, restricted = FALSE) {
  if (restricted) {
    magmaan::fit_noniterative_cfa_restricted(
      pt, ss, estimator = "guttman_gls_aligned",
      communality = "gmm_block", composite = composite)
  } else {
    magmaan::fit_noniterative_cfa(
      pt, ss, estimator = "guttman_gls_aligned", composite = composite)
  }
}

legacy_guttman_fit <- function(pt, ss) {
  magmaan::fit_noniterative_cfa(pt, ss, estimator = "guttman", composite = "auto")
}

guttman_se <- function(fit, X) {
  magmaan::noniterative_cfa_inference(
    fit, discrepancy = "ntml", gamma = "empirical", data = X)$se
}

ml_fit <- function(pt, ss) core$fit_ml(pt, ss, control = fit_control)
uls_fit <- function(pt, ss) core$fit_uls(pt, ss, control = fit_control)

empirical_se <- function(fit, X) {
  core$robust_se_raw_fit(fit, X, bread = "expected")$se
}

run_arm <- function(cond, rep_id, arm, pt, ss, X, pop, info, target) {
  attr(cond, "pop") <- pop
  fit_expr <- switch(
    arm,
    guttman_std = quote(guttman_fit(pt, ss, "standardized", FALSE)),
    guttman_gls = quote(guttman_fit(pt, ss, "gls_aligned", FALSE)),
    guttman_unit = quote(guttman_fit(pt, ss, "unit", FALSE)),
    guttman_legacy = quote(legacy_guttman_fit(pt, ss)),
    guttman_std_restricted = quote(guttman_fit(pt, ss, "standardized", TRUE)),
    guttman_gls_restricted = quote(guttman_fit(pt, ss, "gls_aligned", TRUE)),
    ml_emp = quote(ml_fit(pt, ss)),
    ml_emp_restricted = quote(ml_fit(pt, ss)),
    uls_emp = quote(uls_fit(pt, ss)),
    uls_emp_restricted = quote(uls_fit(pt, ss)),
    stop("unknown arm: ", arm, call. = FALSE)
  )
  fit_res <- timed_eval(eval(fit_expr))
  if (!fit_res$ok) {
    return(list(
      estimates = data.frame(),
      whole = data.frame(),
      diagnostics = fit_diagnostics(cond, rep_id, "fit", arm, FALSE,
                                    fit_ms = fit_res$ms, error = fit_res$error)
    ))
  }
  fit <- fit_res$value
  whole <- fit_whole_row(cond, rep_id, arm, fit, pop)
  se_res <- timed_eval({
    if (grepl("^guttman", arm)) guttman_se(fit, X) else empirical_se(fit, X)
  })
  estimates <- data.frame()
  if (se_res$ok && converged(fit)) {
    estimates <- estimate_rows(cond, rep_id, arm, fit$theta, se_res$value, target, info)
  }
  diag <- fit_diagnostics(cond, rep_id, "fit_se", arm, TRUE, fit = fit,
                          fit_ms = fit_res$ms, se_ok = se_res$ok,
                          se_ms = se_res$ms,
                          error = if (se_res$ok) "" else se_res$error)
  list(estimates = estimates, whole = whole, diagnostics = diag)
}

run_rep <- function(rep_id, cond, pop, pt, arms, info, target, X) {
  ss <- sample_stats(X, pop$vars)
  outs <- lapply(arms, function(arm) {
    run_arm(cond = cond, rep_id = rep_id, arm = arm, pt = pt,
            ss = ss, X = X, pop = pop, info = info, target = target)
  })
  list(
    estimates = bind_rows(lapply(outs, `[[`, "estimates")),
    whole = bind_rows(lapply(outs, `[[`, "whole")),
    diagnostics = bind_rows(lapply(outs, `[[`, "diagnostics"))
  )
}

summarise_estimates <- function(x) {
  if (!nrow(x)) return(data.frame())
  by <- c(condition_cols, "estimator", "group")
  key <- interaction_key(x, by)
  rows <- lapply(split(x, key), function(d) {
    ok <- d$finite
    data.frame(
      d[1, by, drop = FALSE],
      n_rows = nrow(d),
      n_finite = sum(ok),
      n_reps = length(unique(d$rep)),
      n_finite_reps = length(unique(d$rep[ok])),
      bias = mean(d$error[ok], na.rm = TRUE),
      rmse = sqrt(mean(d$error[ok]^2, na.rm = TRUE)),
      coverage = mean(d$covered[ok], na.rm = TRUE),
      avg_se = mean(d$se[ok], na.rm = TRUE),
      emp_sd = stats::sd(d$estimate[ok], na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  bind_rows(rows)
}

summarise_estimates_joint <- function(x, baseline) {
  if (!nrow(x)) return(data.frame())
  base <- x[x$estimator == baseline,
            c(condition_cols, "rep", "free", "param", "group", "error", "finite")]
  if (!nrow(base)) return(data.frame())
  names(base)[names(base) == "error"] <- "baseline_error"
  names(base)[names(base) == "finite"] <- "baseline_finite"
  j <- merge(x, base, by = c(condition_cols, "rep", "free", "param", "group"))
  if (!nrow(j)) return(data.frame())
  by <- c(condition_cols, "estimator", "group")
  key <- interaction_key(j, by)
  rows <- lapply(split(j, key), function(d) {
    ok <- d$finite & d$baseline_finite
    rmse <- sqrt(mean(d$error[ok]^2, na.rm = TRUE))
    base_rmse <- sqrt(mean(d$baseline_error[ok]^2, na.rm = TRUE))
    data.frame(
      d[1, by, drop = FALSE],
      baseline = baseline,
      n_joint_rows = sum(ok),
      n_joint_reps = length(unique(d$rep[ok])),
      rmse = rmse,
      baseline_rmse = base_rmse,
      rmse_ratio = rmse / base_rmse,
      stringsAsFactors = FALSE
    )
  })
  bind_rows(rows)
}

summarise_whole <- function(x) {
  if (!nrow(x)) return(data.frame())
  by <- c(condition_cols, "estimator")
  key <- interaction_key(x, by)
  rows <- lapply(split(x, key), function(d) {
    ok <- d$finite
    data.frame(
      d[1, by, drop = FALSE],
      n_reps = nrow(d),
      n_finite = sum(ok),
      common_rmse = sqrt(mean(d$common_mse[ok], na.rm = TRUE)),
      sigma_rmse = sqrt(mean(d$sigma_mse[ok], na.rm = TRUE)),
      common_diag_rmse = sqrt(mean(d$common_diag_mse[ok], na.rm = TRUE)),
      residual_diag_rmse = sqrt(mean(d$residual_diag_mse[ok], na.rm = TRUE)),
      min_resid_p05 = stats::quantile(d$min_resid[ok], .05, na.rm = TRUE, names = FALSE),
      sigma_min_eig_p05 = stats::quantile(d$sigma_min_eig[ok], .05, na.rm = TRUE, names = FALSE),
      stringsAsFactors = FALSE
    )
  })
  bind_rows(rows)
}

summarise_whole_joint <- function(x, baseline) {
  if (!nrow(x)) return(data.frame())
  base <- x[x$estimator == baseline,
            c(condition_cols, "rep", "finite", "common_mse", "sigma_mse",
              "common_diag_mse", "residual_diag_mse")]
  if (!nrow(base)) return(data.frame())
  names(base)[names(base) == "finite"] <- "baseline_finite"
  names(base)[names(base) == "common_mse"] <- "baseline_common_mse"
  names(base)[names(base) == "sigma_mse"] <- "baseline_sigma_mse"
  names(base)[names(base) == "common_diag_mse"] <- "baseline_common_diag_mse"
  names(base)[names(base) == "residual_diag_mse"] <- "baseline_residual_diag_mse"
  j <- merge(x, base, by = c(condition_cols, "rep"))
  if (!nrow(j)) return(data.frame())
  by <- c(condition_cols, "estimator")
  key <- interaction_key(j, by)
  rows <- lapply(split(j, key), function(d) {
    ok <- d$finite & d$baseline_finite
    common_rmse <- sqrt(mean(d$common_mse[ok], na.rm = TRUE))
    sigma_rmse <- sqrt(mean(d$sigma_mse[ok], na.rm = TRUE))
    common_diag_rmse <- sqrt(mean(d$common_diag_mse[ok], na.rm = TRUE))
    residual_diag_rmse <- sqrt(mean(d$residual_diag_mse[ok], na.rm = TRUE))
    base_common_rmse <- sqrt(mean(d$baseline_common_mse[ok], na.rm = TRUE))
    base_sigma_rmse <- sqrt(mean(d$baseline_sigma_mse[ok], na.rm = TRUE))
    base_common_diag_rmse <- sqrt(mean(d$baseline_common_diag_mse[ok], na.rm = TRUE))
    base_resid_rmse <- sqrt(mean(d$baseline_residual_diag_mse[ok], na.rm = TRUE))
    data.frame(
      d[1, by, drop = FALSE],
      baseline = baseline,
      n_joint_reps = sum(ok),
      common_rmse = common_rmse,
      common_rmse_ratio = common_rmse / base_common_rmse,
      sigma_rmse = sigma_rmse,
      sigma_rmse_ratio = sigma_rmse / base_sigma_rmse,
      common_diag_rmse = common_diag_rmse,
      common_diag_rmse_ratio = common_diag_rmse / base_common_diag_rmse,
      residual_diag_rmse = residual_diag_rmse,
      residual_diag_rmse_ratio = residual_diag_rmse / base_resid_rmse,
      stringsAsFactors = FALSE
    )
  })
  bind_rows(rows)
}

summarise_diagnostics <- function(x) {
  if (!nrow(x)) return(data.frame())
  by <- c(condition_cols, "estimator")
  key <- interaction_key(x, by)
  rows <- lapply(split(x, key), function(d) {
    data.frame(
      d[1, by, drop = FALSE],
      attempts = nrow(d),
      fit_ok_rate = mean(d$fit_ok),
      converged_rate = mean(d$converged, na.rm = TRUE),
      se_ok_rate = mean(d$se_ok, na.rm = TRUE),
      improper_rate = mean(d$improper, na.rm = TRUE),
      min_resid_p05 = stats::quantile(d$min_resid, .05, na.rm = TRUE, names = FALSE),
      sigma_min_eig_p05 = stats::quantile(d$sigma_min_eig, .05, na.rm = TRUE, names = FALSE),
      first_error = paste(unique(d$error[nzchar(d$error)])[1] %||% "", collapse = ""),
      stringsAsFactors = FALSE
    )
  })
  bind_rows(rows)
}

summarise_timing <- function(x) {
  if (!nrow(x)) return(data.frame())
  by <- c(condition_cols, "estimator")
  key <- interaction_key(x, by)
  rows <- lapply(split(x, key), function(d) {
    ok <- d$fit_ok
    data.frame(
      d[1, by, drop = FALSE],
      attempts = nrow(d),
      fit_median_ms = stats::median(d$fit_ms[ok], na.rm = TRUE),
      fit_mean_ms = mean(d$fit_ms[ok], na.rm = TRUE),
      se_median_ms = stats::median(d$se_ms[ok], na.rm = TRUE),
      total_median_ms = stats::median(d$total_ms[ok], na.rm = TRUE),
      total_p90_ms = stats::quantile(d$total_ms[ok], .90, na.rm = TRUE, names = FALSE),
      stringsAsFactors = FALSE
    )
  })
  bind_rows(rows)
}

condition_grid <- function(opts) {
  config <- expand.grid(
    generator = opts$generators,
    factors = opts$factors,
    indicators = opts$indicators,
    rho = opts$rho,
    scale = opts$scales,
    loading = opts$loadings,
    constraint = "configural",
    n = opts$n,
    stringsAsFactors = FALSE
  )
  restricted <- data.frame()
  if (opts$include_restricted) {
    restricted <- expand.grid(
      generator = opts$generators,
      factors = opts$factors,
      indicators = opts$indicators,
      rho = opts$rho,
      scale = "equal",
      loading = "tau",
      constraint = "tau_resid",
      n = opts$n,
      stringsAsFactors = FALSE
    )
  }
  grid <- rbind(config, restricted)
  grid <- grid[order(grid$constraint, grid$generator, grid$factors,
                     grid$indicators, grid$rho, grid$scale, grid$loading,
                     grid$n), ]
  rownames(grid) <- NULL
  if (is.finite(opts$max_cells)) grid <- head(grid, opts$max_cells)
  grid
}

population_rows <- function(grid) {
  pops <- lapply(seq_len(nrow(grid)), function(i) {
    g <- grid[i, ]
    pop <- make_population(g$factors, g$indicators, g$rho, g$loading,
                           g$scale, g$generator)
    data.frame(
      g,
      p = length(pop$vars),
      min_eigen = pop$min_eigen,
      min_loading_std = min(pop$lambda_std),
      max_loading_std = max(pop$lambda_std),
      min_scale = min(pop$scales),
      max_scale = max(pop$scales),
      stringsAsFactors = FALSE
    )
  })
  bind_rows(pops)
}

write_current <- function(outputs, opts) {
  write_csv(bind_rows(outputs$estimate_summary), file.path(opts$results_dir, "estimate_summary.csv"))
  write_csv(bind_rows(outputs$estimate_joint_summary), file.path(opts$results_dir, "estimate_joint_summary.csv"))
  write_csv(bind_rows(outputs$whole_summary), file.path(opts$results_dir, "whole_summary.csv"))
  write_csv(bind_rows(outputs$whole_joint_summary), file.path(opts$results_dir, "whole_joint_summary.csv"))
  write_csv(bind_rows(outputs$diagnostic_summary), file.path(opts$results_dir, "diagnostic_summary.csv"))
  write_csv(bind_rows(outputs$timing_summary), file.path(opts$results_dir, "timing_summary.csv"))
  write_csv(bind_rows(outputs$diagnostics), file.path(opts$results_dir, "diagnostics.csv"))
  write_csv(bind_rows(outputs$progress), file.path(opts$results_dir, "progress.csv"))
  if (isTRUE(opts$keep_draws)) {
    write_csv(bind_rows(outputs$estimate_draws), file.path(opts$results_dir, "estimate_draws.csv"))
    write_csv(bind_rows(outputs$whole_draws), file.path(opts$results_dir, "whole_draws.csv"))
  }
}

grid <- condition_grid(opts)
write_csv(grid, file.path(opts$results_dir, "design.csv"))
write_csv(population_rows(unique(grid[setdiff(condition_cols, "n")])),
          file.path(opts$results_dir, "population.csv"))
write_metadata(
  file.path(opts$results_dir, "metadata.csv"),
  values = list(
    mode = opts$mode,
    reps = opts$reps,
    n = opts$n,
    generators = opts$generators,
    factors = opts$factors,
    indicators = opts$indicators,
    rho = opts$rho,
    scales = opts$scales,
    loadings = opts$loadings,
    include_restricted = opts$include_restricted,
    max_cells = opts$max_cells,
    cores = opts$cores,
    alpha = opts$alpha,
    seed_base = opts$seed_base,
    ml_max_iter = opts$ml_max_iter,
    categorical_metric = "observed Pearson-code covariance; no polychoric inputs"
  ),
  packages = c("magmaan")
)

cat(sprintf("Running %d cells x %d reps (mode=%s, cores=%d)\n",
            nrow(grid), opts$reps, opts$mode, opts$cores))

outputs <- list(
  estimate_summary = list(), estimate_joint_summary = list(),
  whole_summary = list(), whole_joint_summary = list(),
  diagnostic_summary = list(), timing_summary = list(),
  diagnostics = list(), progress = list(),
  estimate_draws = list(), whole_draws = list()
)

for (cell in seq_len(nrow(grid))) {
  cond <- grid[cell, condition_cols, drop = FALSE]
  restricted <- identical(cond$constraint[[1]], "tau_resid")
  pop <- make_population(cond$factors[[1]], cond$indicators[[1]], cond$rho[[1]],
                         cond$loading[[1]], cond$scale[[1]], cond$generator[[1]])
  pt <- core$lavaan_lavaanify(model_syntax(cond$factors[[1]], cond$indicators[[1]], restricted))
  info <- free_param_info(pt, pop$vars, pop$factors)
  target <- theta_target(info, pop)
  arms <- if (restricted) {
    c("guttman_std_restricted", "guttman_gls_restricted",
      "ml_emp_restricted", "uls_emp_restricted")
  } else {
    c("guttman_std", "guttman_gls", "guttman_unit", "guttman_legacy",
      "ml_emp", "uls_emp")
  }
  baseline <- if (restricted) "ml_emp_restricted" else "ml_emp"
  seed <- opts$seed_base + 100000L * cell
  cat(sprintf("[%d/%d] %s q=%d m=%d rho=%.2f scale=%s loading=%s constraint=%s n=%d reps=%d\n",
              cell, nrow(grid), cond$generator, cond$factors, cond$indicators,
              cond$rho, cond$scale, cond$loading, cond$constraint, cond$n, opts$reps))
  draws_res <- timed_eval(draws_for_condition(pop, cond$generator[[1]], cond$n[[1]],
                                              opts$reps, seed))
  if (!draws_res$ok) {
    outputs$progress[[length(outputs$progress) + 1L]] <- data.frame(
      cond, cell = cell, reps = opts$reps, draw_ok = FALSE,
      draw_ms = draws_res$ms, elapsed_sec = NA_real_,
      error = draws_res$error, stringsAsFactors = FALSE
    )
    write_current(outputs, opts)
    next
  }
  cell_t0 <- proc.time()[["elapsed"]]
  reps <- seq_len(opts$reps)
  cell_out <- if (opts$cores > 1L) {
    parallel::mclapply(
      reps,
      function(r) run_rep(r, cond, pop, pt, arms, info, target, draws_res$value[[r]]),
      mc.cores = opts$cores, mc.preschedule = TRUE
    )
  } else {
    lapply(reps, function(r) run_rep(r, cond, pop, pt, arms, info, target, draws_res$value[[r]]))
  }
  estimate_draws <- bind_rows(lapply(cell_out, `[[`, "estimates"))
  whole_draws <- bind_rows(lapply(cell_out, `[[`, "whole"))
  diagnostics <- bind_rows(lapply(cell_out, `[[`, "diagnostics"))
  outputs$estimate_summary[[length(outputs$estimate_summary) + 1L]] <- summarise_estimates(estimate_draws)
  outputs$estimate_joint_summary[[length(outputs$estimate_joint_summary) + 1L]] <- summarise_estimates_joint(estimate_draws, baseline)
  outputs$whole_summary[[length(outputs$whole_summary) + 1L]] <- summarise_whole(whole_draws)
  outputs$whole_joint_summary[[length(outputs$whole_joint_summary) + 1L]] <- summarise_whole_joint(whole_draws, baseline)
  outputs$diagnostic_summary[[length(outputs$diagnostic_summary) + 1L]] <- summarise_diagnostics(diagnostics)
  outputs$timing_summary[[length(outputs$timing_summary) + 1L]] <- summarise_timing(diagnostics)
  outputs$diagnostics[[length(outputs$diagnostics) + 1L]] <- diagnostics
  if (isTRUE(opts$keep_draws)) {
    outputs$estimate_draws[[length(outputs$estimate_draws) + 1L]] <- estimate_draws
    outputs$whole_draws[[length(outputs$whole_draws) + 1L]] <- whole_draws
  }
  outputs$progress[[length(outputs$progress) + 1L]] <- data.frame(
    cond, cell = cell, reps = opts$reps, draw_ok = TRUE,
    draw_ms = draws_res$ms,
    elapsed_sec = proc.time()[["elapsed"]] - cell_t0,
    error = "", stringsAsFactors = FALSE
  )
  write_current(outputs, opts)
}

write_current(outputs, opts)

cat("Wrote:\n")
for (nm in c("metadata.csv", "design.csv", "population.csv",
             "estimate_summary.csv", "estimate_joint_summary.csv",
             "whole_summary.csv", "whole_joint_summary.csv",
             "diagnostic_summary.csv", "timing_summary.csv",
             "diagnostics.csv", "progress.csv")) {
  cat("  ", file.path(opts$results_dir, nm), "\n", sep = "")
}
