#!/usr/bin/env Rscript

suppressWarnings(suppressMessages(library(magmaanlab)))

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1L]])))
} else normalizePath(".")
source(file.path(script_dir, "..", "..", "_support", "R", "helpers.R"))
source(file.path(script_dir, "..", "..", "_support", "R", "missingness.R"))
source(file.path(script_dir, "R", "sem_models.R"))
set_single_threaded_math()

usage <- function() cat(
  "Usage: Rscript estimate_pseudotrue_shadow.R [options]\n\n",
  "Approximates the Gaussian-FIML H0 and saturated H1 pseudo-true targets\n",
  "under nonnormal MAR, then runs a deliberately artificial Gaussian shadow\n",
  "experiment generated at the H0 pseudo-true parameters.\n\n",
  "  --population-n N     Large draw for pseudo-targets (default 200000).\n",
  "  --reps N             Monte Carlo reps per arm (default 500).\n",
  "  --n N                Monte Carlo sample size (default 500).\n",
  "  --cores N            Parallel workers (default up to 4).\n",
  "  --distributions CSV  Default vm2,ig2.\n",
  "  --seed-base N        Deterministic seed base.\n",
  "  --results-dir P      Output directory.\n",
  "  --help               Show this help.\n", sep = "")

opts <- list(
  population_n = 200000L,
  reps = 500L,
  n = 500L,
  cores = min(4L, max(1L, parallel::detectCores() - 2L)),
  distributions = c("vm2", "ig2"),
  seed_base = 20260903L,
  results_dir = NULL)
args <- commandArgs(TRUE)
i <- 1L
take <- function() {
  i <<- i + 1L
  if (i > length(args)) stop("missing option value", call. = FALSE)
  args[[i]]
}
while (i <= length(args)) {
  arg <- args[[i]]
  if (arg %in% c("-h", "--help")) { usage(); quit(save = "no", status = 0L) }
  else if (arg == "--population-n") opts$population_n <- as.integer(take())
  else if (arg == "--reps") opts$reps <- as.integer(take())
  else if (arg == "--n") opts$n <- as.integer(take())
  else if (arg == "--cores") opts$cores <- as.integer(take())
  else if (arg == "--distributions") {
    opts$distributions <- parse_csv_arg(take())
  } else if (arg == "--seed-base") opts$seed_base <- as.integer(take())
  else if (arg == "--results-dir") opts$results_dir <- take()
  else stop("unknown argument: ", arg, call. = FALSE)
  i <- i + 1L
}
stopifnot(
  opts$population_n >= 10000L,
  opts$reps > 0L,
  opts$n >= 80L,
  opts$cores > 0L,
  all(opts$distributions %in% c("vm1", "ig1", "vm2", "ig2")))

results <- opts$results_dir %||% file.path(
  script_dir, "results", "pseudotrue-shadow")
dir.create(results, recursive = TRUE, showWarnings = FALSE)
model <- sem_model_catalog()[["one_factor_6"]]
control <- list(max_iter = 8000L, ftol = 1e-11, gtol = 1e-8)

fit_fiml_with_stage1 <- function(X) {
  fd <- magmaanlab::df_to_fiml_data(as.data.frame(X), model$spec)
  em <- magmaanlab::magmaan_core$estimate_saturated_em_moments(
    fd, control = control)
  fit <- magmaanlab::magmaan_core$fit_fiml(
    model$spec, fd,
    optimizer = "nlopt-lbfgs-slsqp-fallback",
    control = control)
  fit$stage1 <- em
  if (!isTRUE(fit$converged)) stop("FIML fit did not converge", call. = FALSE)
  list(fit = fit, em = em)
}

true_value <- function(lhs, op, rhs) {
  if (op == "=~") return(c(x1 = 1.00, x2 = 0.80, x3 = 0.90,
                            x4 = 0.70, x5 = 1.10, x6 = 0.85)[[rhs]])
  if (op == "~~" && lhs == rhs && lhs == "f") return(1)
  if (op == "~~" && lhs == rhs) {
    return(c(x1 = 0.50, x2 = 0.60, x3 = 0.55,
             x4 = 0.65, x5 = 0.50, x6 = 0.60)[[lhs]])
  }
  if (op == "~1") return(0)
  NA_real_
}

targets <- list()
parameter_rows <- list()
moment_rows <- list()
shadow_residual_rows <- list()
tetrad_rows <- list()
target_summary <- list()
for (index in seq_along(opts$distributions)) {
  distribution <- opts$distributions[[index]]
  message("pseudo-target: ", distribution, " / mar_30")
  sampler <- sem_calibrate_sampler(model, distribution)
  seed <- sem_seed(opts$seed_base + index * 100003L)
  X <- sem_draw(model, sampler, opts$population_n, seed)
  set.seed(seed + 700001L)
  X <- sem_apply_missingness(X, "mar_30")
  fitted <- fit_fiml_with_stage1(X)
  fit <- fitted$fit
  em <- fitted$em
  implied <- magmaanlab:::model_implied(fit)
  fmg <- magmaanlab::fmg_tests(fit, tests = c("SB", "pEBA4", "all"))
  target <- list(
    distribution = distribution,
    theta = fit$theta,
    mu_h0 = implied$mu[[1L]],
    Sigma_h0 = implied$sigma[[1L]],
    mu_h1 = em$mean[[1L]],
    Sigma_h1 = em$cov[[1L]],
    discrepancy_per_case = fmg$base_statistic[[1L]] / opts$population_n,
    lrt = fmg$base_statistic[[1L]],
    df = as.integer(fmg$df[[1L]]),
    eigenvalues = fmg$eigenvalues[[1L]],
    realized_missing = mean(is.na(X[, -(1:2), drop = FALSE])))
  targets[[distribution]] <- target

  pt <- fit$partable
  pt <- pt[pt$free > 0L, c("lhs", "op", "rhs", "free", "est")]
  pt$generating_value <- mapply(true_value, pt$lhs, pt$op, pt$rhs)
  pt$drift <- pt$est - pt$generating_value
  pt$distribution <- distribution
  parameter_rows[[distribution]] <- pt[c(
    "distribution", "lhs", "op", "rhs", "free", "generating_value",
    "est", "drift")]

  moment_index <- expand.grid(
    row = seq_len(model$p), col = seq_len(model$p),
    KEEP.OUT.ATTRS = FALSE)
  moment_index <- moment_index[moment_index$row >= moment_index$col, ]
  moment_index$distribution <- distribution
  moment_index$variable_row <- model$ov[moment_index$row]
  moment_index$variable_col <- model$ov[moment_index$col]
  moment_index$generating <- model$Sigma[cbind(
    moment_index$row, moment_index$col)]
  moment_index$h1_pseudotrue <- target$Sigma_h1[cbind(
    moment_index$row, moment_index$col)]
  moment_index$h0_pseudotrue <- target$Sigma_h0[cbind(
    moment_index$row, moment_index$col)]
  moment_index$h1_minus_generating <-
    moment_index$h1_pseudotrue - moment_index$generating
  moment_index$h0_minus_h1 <-
    moment_index$h0_pseudotrue - moment_index$h1_pseudotrue
  moment_rows[[distribution]] <- moment_index

  offdiag <- which(lower.tri(target$Sigma_h1), arr.ind = TRUE)
  shadow_residual <- target$Sigma_h1 - target$Sigma_h0
  standardized_residual <- shadow_residual / sqrt(outer(
    diag(target$Sigma_h1), diag(target$Sigma_h1)))
  shadow_residual_rows[[distribution]] <- data.frame(
    distribution = distribution,
    lhs = model$ov[offdiag[, 1L]],
    op = "~~",
    rhs = model$ov[offdiag[, 2L]],
    h0_value = 0,
    h1_shadow_value = shadow_residual[offdiag],
    standardized_h1_shadow_value = standardized_residual[offdiag],
    stringsAsFactors = FALSE)

  h1_cor <- stats::cov2cor(target$Sigma_h1)
  generating_cor <- stats::cov2cor(model$Sigma)
  tetrad_combinations <- utils::combn(seq_len(model$p), 4L)
  tetrads <- do.call(rbind, lapply(seq_len(ncol(tetrad_combinations)),
                                  function(k) {
    j <- tetrad_combinations[, k]
    tetrad_values <- function(R) c(
      R[j[1L], j[2L]] * R[j[3L], j[4L]] -
        R[j[1L], j[3L]] * R[j[2L], j[4L]],
      R[j[1L], j[2L]] * R[j[3L], j[4L]] -
        R[j[1L], j[4L]] * R[j[2L], j[3L]])
    data.frame(
      distribution = distribution,
      variables = paste(model$ov[j], collapse = ","),
      tetrad = c("ab_cd_minus_ac_bd", "ab_cd_minus_ad_bc"),
      generating_value = tetrad_values(generating_cor),
      h1_pseudotrue_value = tetrad_values(h1_cor),
      stringsAsFactors = FALSE)
  }))
  tetrad_rows[[distribution]] <- tetrads

  target_summary[[distribution]] <- data.frame(
    distribution = distribution,
    population_n = opts$population_n,
    realized_missing = target$realized_missing,
    df = target$df,
    lrt = target$lrt,
    discrepancy_per_case = target$discrepancy_per_case,
    n_times_discrepancy = opts$n * target$discrepancy_per_case,
    max_abs_parameter_drift = max(abs(pt$drift)),
    rms_parameter_drift = sqrt(mean(pt$drift^2)),
    max_abs_h1_moment_drift = max(abs(c(
      target$mu_h1 - model$mu,
      target$Sigma_h1 - model$Sigma))),
    max_abs_h0_h1_moment_gap = max(abs(c(
      target$mu_h0 - target$mu_h1,
      target$Sigma_h0 - target$Sigma_h1))),
    max_abs_h1_shadow_residual_correlation = max(abs(
      standardized_residual[offdiag])),
    max_abs_generating_tetrad = max(abs(tetrads$generating_value)),
    max_abs_h1_pseudotrue_tetrad = max(abs(tetrads$h1_pseudotrue_value)),
    mean_eigenvalue = mean(target$eigenvalues),
    stringsAsFactors = FALSE)
}

parameter_table <- do.call(rbind, parameter_rows)
row.names(parameter_table) <- NULL
moment_table <- do.call(rbind, moment_rows)
row.names(moment_table) <- NULL
shadow_residual_table <- do.call(rbind, shadow_residual_rows)
shadow_residual_table <- shadow_residual_table[order(
  shadow_residual_table$distribution,
  -abs(shadow_residual_table$standardized_h1_shadow_value)), ]
row.names(shadow_residual_table) <- NULL
tetrad_table <- do.call(rbind, tetrad_rows)
tetrad_table <- tetrad_table[order(
  tetrad_table$distribution,
  -abs(tetrad_table$h1_pseudotrue_value)), ]
row.names(tetrad_table) <- NULL
target_summary <- do.call(rbind, target_summary)
row.names(target_summary) <- NULL
write_csv(parameter_table, file.path(results, "pseudotrue_parameters.csv"))
write_csv(moment_table, file.path(results, "pseudotrue_covariances.csv"))
write_csv(shadow_residual_table, file.path(
  results, "h1_shadow_residual_covariances.csv"))
write_csv(tetrad_table, file.path(results, "h1_pseudotrue_tetrads.csv"))
write_csv(target_summary, file.path(results, "pseudotrue_summary.csv"))

arms <- do.call(rbind, lapply(opts$distributions, function(distribution) {
  data.frame(
    distribution = distribution,
    arm = c("original_nonnormal_mar", "gaussian_h0_shadow_mar"),
    stringsAsFactors = FALSE)
}))
arms$cell_id <- seq_len(nrow(arms))

arm_samplers <- list()
for (index in seq_len(nrow(arms))) {
  cell <- arms[index, ]
  target <- targets[[cell$distribution]]
  key <- paste(cell$distribution, cell$arm, sep = "::")
  arm_samplers[[key]] <- if (cell$arm == "original_nonnormal_mar") {
    sem_calibrate_sampler(model, cell$distribution)
  } else {
    sem_calibrate_sampler(
      model, "normal", Sigma = target$Sigma_h0, mu = target$mu_h0)
  }
}

one_rep <- function(cell, rep_id) {
  distribution <- cell$distribution
  seed <- sem_seed(opts$seed_base + 10000019L + cell$cell_id * 100003L + rep_id)
  begin <- proc.time()[["elapsed"]]
  ans <- tryCatch({
    sampler <- arm_samplers[[paste(distribution, cell$arm, sep = "::")]]
    X <- sem_draw(model, sampler, opts$n, seed)
    set.seed(seed + 700001L)
    X <- sem_apply_missingness(X, "mar_30")
    fitted <- fit_fiml_with_stage1(X)
    fit <- fitted$fit
    fmg <- magmaanlab::fmg_tests(fit, tests = c("SB", "pEBA4", "all"))
    key <- sub("_ml$", "", fmg$label)
    p <- stats::setNames(fmg$p_value, key)
    list(
      ok = TRUE,
      error = "",
      realized_missing = mean(is.na(X[, -(1:2), drop = FALSE])),
      lrt = fmg$base_statistic[[1L]],
      p_sb = unname(p[["sb"]]),
      p_peba4 = unname(p[["peba4"]]),
      p_all = unname(p[["all"]]))
  }, error = function(e) list(
    ok = FALSE,
    error = conditionMessage(e),
    realized_missing = NA_real_,
    lrt = NA_real_,
    p_sb = NA_real_,
    p_peba4 = NA_real_,
    p_all = NA_real_))
  data.frame(
    distribution = distribution,
    arm = cell$arm,
    n = opts$n,
    rep = rep_id,
    seed = seed,
    ok = ans$ok,
    error = ans$error,
    realized_missing = ans$realized_missing,
    lrt = ans$lrt,
    p_sb = ans$p_sb,
    p_peba4 = ans$p_peba4,
    p_all = ans$p_all,
    seconds = proc.time()[["elapsed"]] - begin,
    stringsAsFactors = FALSE)
}

one_cell <- function(index) {
  cell <- as.list(arms[index, , drop = FALSE])
  message(sprintf("shadow cell %d/%d: %s / %s", index, nrow(arms),
                  cell$distribution, cell$arm))
  do.call(rbind, lapply(seq_len(opts$reps), function(rep_id) {
    one_rep(cell, rep_id)
  }))
}

begin <- proc.time()[["elapsed"]]
pieces <- if (.Platform$OS.type != "windows" && opts$cores > 1L) {
  parallel::mclapply(
    seq_len(nrow(arms)), one_cell,
    mc.cores = min(opts$cores, nrow(arms)), mc.preschedule = FALSE)
} else lapply(seq_len(nrow(arms)), one_cell)
raw <- do.call(rbind, pieces)
row.names(raw) <- NULL
write_csv(raw, file.path(results, "shadow_replications.csv"))

groups <- split(seq_len(nrow(raw)), interaction(
  raw[c("distribution", "arm")], drop = TRUE, lex.order = TRUE))
summary <- do.call(rbind, lapply(groups, function(ii) {
  z <- raw[ii, , drop = FALSE]
  usable <- z$ok & is.finite(z$p_peba4)
  data.frame(
    distribution = z$distribution[[1L]],
    arm = z$arm[[1L]],
    n = opts$n,
    attempted = nrow(z),
    usable = sum(usable),
    mean_lrt = mean(z$lrt[usable]),
    rejection_sb = mean(z$p_sb[usable] <= .05),
    rejection_peba4 = mean(z$p_peba4[usable] <= .05),
    rejection_all = mean(z$p_all[usable] <= .05),
    mean_missing = mean(z$realized_missing[usable]),
    stringsAsFactors = FALSE)
}))
row.names(summary) <- NULL
write_csv(summary, file.path(results, "shadow_summary.csv"))
write_metadata(file.path(results, "metadata.csv"), list(
  population_n = opts$population_n,
  reps = opts$reps,
  n = opts$n,
  distributions = opts$distributions,
  seed_base = opts$seed_base,
  failures = sum(!raw$ok),
  shadow_runtime_wall_seconds = proc.time()[["elapsed"]] - begin),
  packages = "magmaanlab")

cat("\nPseudo-target summary\n")
print(target_summary, row.names = FALSE, digits = 5)
cat("\nShadow experiment\n")
print(summary, row.names = FALSE, digits = 4)
