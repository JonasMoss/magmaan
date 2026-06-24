# Welz et al. (2026) Design 4: copula distributional-misspecification stress test.
#
# Replicates the simulation in Section 8.2 of
#   Welz, Mair & Alfons (2026), "Robust Estimation of Polychoric Correlation",
#   Psychometrika 91, 247-278,
# and runs the full magmaan robust-ordinal estimator family on it (not just the
# WMA hard cap / ML pair the paper reports). The question this answers: on pure
# distributional misspecification (no contamination), do the *portable* robust
# recipes (Huber/Tukey on the cell residual, DPD) track the WMA hard cap (the
# robcat estimator the paper proposes) and ML, or do they diverge? WMA is
# polychoric-only; if a portable recipe matches it here, that recipe is the
# candidate for the mixed/polyserial extension.
#
# Faithful to the paper:
#   - latent (xi, eta) ~ Clayton copula (main text) with standard normal margins,
#     calibrated to population (Pearson) correlation rho_G; Gumbel in supplement.
#   - five-category discretization with thresholds
#       a = (-1.5, 0, 0.5, 1),  b = (-1, 1, 1.5, 2);
#   - rho_G in {0.9, 0.3}; N = 1000; truth = rho_G.
# Deviations: we add Frank (no tail dependence) as a contrast and sweep the full
# 8-estimator family; the t-copula supplement is not run here (no calibrated
# bivariate t wrapper on the R surface yet, though magmaan::sim has the C++ path).
#
# The latent copula DGP uses magmaan's own bivariate-copula generator
# (sim_bicop_*), which for two variables is equivalent to the VITA / covsim
# construction the paper used: calibrate the copula parameter so the
# normal-margined Pearson correlation equals rho_G, then discretize.

roc_find_repo_root <- function(start = getwd()) {
  path <- normalizePath(start, mustWork = TRUE)
  repeat {
    marker <- file.path(path, "docs", "research", "sims", "r", "common.R")
    if (file.exists(marker) && file.exists(file.path(path, "AGENTS.md"))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) {
      stop("Could not locate magmaan repository root", call. = FALSE)
    }
    path <- parent
  }
}

roc_repo_root <- roc_find_repo_root()
# Reuse the paper runner's estimator specs, per-rep evaluation, and tidy-row
# builder. Sourcing does not auto-run its main(): the `sys.nframe() == 0L` guard
# is false under source().
source(file.path(roc_repo_root, "docs", "research", "sims", "r",
                 "robust_ordinal_sem_paper.R"))

# Welz Section 8.2 discretization thresholds (five categories each).
roc_copula_thresholds <- function() {
  list(x1 = c(-1.5, 0, 0.5, 1),
       x2 = c(-1, 1, 1.5, 2))
}

# Estimator family. Same eight recipes as the contamination designs, but with
# the smooth cap re-centered to (1.3, 1.9) so its plateau h_inf = (a+b)/2 = 1.6
# matches the WMA hard cap at k = 1.6 (= Welz tuning c = 0.6). See
# docs/research/notes/robust_ordinal_gamma.tex.
roc_method_specs <- function() {
  specs <- ros_method_specs()
  for (i in seq_along(specs)) {
    if (identical(specs[[i]]$method, "smooth_cap")) {
      specs[[i]] <- list(
        method = "smooth_cap", tuning = "h_a=1.3;h_b=1.9",
        args = list(robust = "h_weighted", h_kind = "smooth_cap",
                    h_a = 1.3, h_b = 1.9, h_lambda = 0.2))
    }
  }
  specs
}

# Calibrate the copula once per (family, rho_G) cell, then draw `reps` latent
# samples. Returns a list of n-by-2 continuous matrices with N(0,1) margins and
# the requested copula dependence at population correlation rho_G.
roc_draw_latent <- function(core, family, rho_G, n, reps, seed_base,
                            quadrature_points = 64L,
                            max_bisection_iter = 200L) {
  marginals <- list(list(kind = "standard_normal"),
                    list(kind = "standard_normal"))
  cal <- tryCatch(
    core$sim_bicop_calibrate(target_corr = rho_G, marginals = marginals,
                             family = family,
                             quadrature_points = quadrature_points,
                             max_bisection_iter = max_bisection_iter),
    error = function(e) e)
  if (inherits(cal, "error")) {
    return(structure(list(error = conditionMessage(cal)), class = "roc_cal_fail"))
  }
  drawn <- core$sim_bicop_draw(cal, n = n, reps = reps, seed_base = seed_base,
                               quadrature_points = quadrature_points)
  list(calibration = cal,
       copula_param = cal$copula$parameters[[1]],
       draws = drawn$draws)
}

roc_make_sim <- function(z, family, rho_G) {
  colnames(z) <- c("x1", "x2")
  truth <- matrix(c(1, rho_G, rho_G, 1), 2, 2,
                  dimnames = list(c("x1", "x2"), c("x1", "x2")))
  list(
    design = paste0("welz_copula_", family),
    X = ros_ordinalize_matrix(z, roc_copula_thresholds()),
    truth_R = truth,
    n_contaminated = 0L
  )
}

robust_ordinal_copula_run <- function(
    families = c("clayton", "gumbel"),
    rhos = c(0.3, 0.9),
    reps = 1000L,
    n = 1000L,
    seed = 20260624L,
    methods = roc_method_specs()) {
  core <- ros_require_magmaan()
  rows <- list()
  cursor <- 0L
  fam_index <- 0L
  for (family in families) {
    fam_index <- fam_index + 1L
    rho_index <- 0L
    for (rho_G in rhos) {
      rho_index <- rho_index + 1L
      cell_seed <- seed + 1000L * fam_index + 100000L * rho_index
      latent <- roc_draw_latent(core, family, rho_G, n, reps, cell_seed)
      if (inherits(latent, "roc_cal_fail")) {
        message("calibration failed for ", family, " rho_G=", rho_G, ": ",
                latent$error)
        next
      }
      copula_param <- latent$copula_param
      for (rep in seq_len(reps)) {
        sim <- roc_make_sim(latent$draws[[rep]], family, rho_G)
        for (method in methods) {
          cursor <- cursor + 1L
          out <- ros_eval_method(sim, method, rep, n, contamination = 0)
          out$family <- family
          out$rho_target <- rho_G
          out$copula_param <- copula_param
          rows[[cursor]] <- out
        }
      }
    }
  }
  do.call(rbind, rows)
}

# Welz Table 5 style summary: per (family, rho_target, method), point-estimate
# bias and RMSE, mean estimated SE vs the across-rep SD of the estimate (SE
# bias), and 95% CI coverage / length. Failures (non-converged moment builds)
# are reported as a rate and excluded from the moment-summary statistics.
roc_summary <- function(rows) {
  rows <- rows[rows$parameter == "rho" | rows$parameter == "__moment_builder__", ]
  split_key <- with(rows, paste(family, rho_target, method, tuning, sep = "\r"))
  parts <- split(rows, split_key)
  out <- lapply(parts, function(df) {
    ok <- df[isTRUE_vec(df$converged) & is.finite(df$estimate), , drop = FALSE]
    n_total <- nrow(df)
    n_ok <- nrow(ok)
    truth <- ok$truth[1]
    est <- ok$estimate
    se <- ok$se
    covered <- truth >= ok$ci_low & truth <= ok$ci_high
    data.frame(
      family = df$family[1],
      rho_target = df$rho_target[1],
      method = df$method[1],
      tuning = df$tuning[1],
      reps = n_total,
      fail_rate = (n_total - n_ok) / n_total,
      mean_est = mean(est),
      bias = mean(est) - truth,
      rmse = sqrt(mean((est - truth)^2)),
      emp_sd = stats::sd(est),
      mean_se = mean(se[is.finite(se)]),
      se_bias = mean(se[is.finite(se)]) - stats::sd(est),
      coverage = mean(covered[is.finite(covered)]),
      ci_length = 2 * stats::qnorm(.975) * mean(se[is.finite(se)]),
      median_ms = stats::median(ok$elapsed_ms),
      min_eigen_r = min(ok$min_eigen_r, na.rm = TRUE),
      stringsAsFactors = FALSE
    )
  })
  res <- do.call(rbind, out)
  res <- res[order(res$family, res$rho_target,
                   factor(res$method, levels = vapply(roc_method_specs(),
                                                      `[[`, "", "method"))), ]
  rownames(res) <- NULL
  res
}

isTRUE_vec <- function(x) !is.na(x) & x

roc_parse_arg <- function(args, name, default = NULL) {
  prefix <- paste0("--", name, "=")
  hit <- grep(paste0("^", prefix), args, value = TRUE)
  if (!length(hit)) return(default)
  sub(prefix, "", hit[[length(hit)]], fixed = TRUE)
}

roc_parse_csv <- function(x, default) {
  if (is.null(x) || !nzchar(x)) return(default)
  strsplit(x, ",", fixed = TRUE)[[1]]
}

roc_main <- function(args = commandArgs(trailingOnly = TRUE)) {
  reps <- as.integer(roc_parse_arg(args, "reps", "1000"))
  n <- as.integer(roc_parse_arg(args, "n", "1000"))
  seed <- as.integer(roc_parse_arg(args, "seed", "20260624"))
  families <- roc_parse_csv(roc_parse_arg(args, "families", NULL),
                            c("clayton", "gumbel"))
  rhos <- as.numeric(roc_parse_csv(roc_parse_arg(args, "rhos", NULL),
                                   c("0.3", "0.9")))
  out <- roc_parse_arg(
    args, "out",
    file.path(roc_repo_root, "docs", "research", "sims", "results",
              "robust_ordinal_copula.csv"))

  result <- robust_ordinal_copula_run(families = families, rhos = rhos,
                                      reps = reps, n = n, seed = seed)
  dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(result, out, row.names = FALSE)
  summary <- roc_summary(result)
  summary_path <- sub("\\.csv$", "_summary.csv", out)
  utils::write.csv(summary, summary_path, row.names = FALSE)
  message("wrote ", nrow(result), " rows to ", out)
  message("wrote ", nrow(summary), " summary rows to ", summary_path)
  invisible(list(rows = result, summary = summary))
}

if (sys.nframe() == 0L) {
  roc_main()
}
