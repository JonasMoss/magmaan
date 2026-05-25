#!/usr/bin/env Rscript

# Probe the muthen_2017_ch2_ex2_1__adf audit-parity side finding:
# saturated ADF, near-zero residual, and a highly ill-conditioned empirical
# NACOV that amplifies the projected gradient.

.support_helpers <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    script <- normalizePath(sub("^--file=", "", file_arg[[1L]]), mustWork = TRUE)
  } else {
    ofile <- tryCatch(sys.frames()[[1L]]$ofile, error = function(e) NULL)
    script <- normalizePath(if (is.null(ofile)) "diagnose_adf_conditioning.R" else ofile,
                            mustWork = FALSE)
  }
  file.path(dirname(dirname(dirname(script))), "_support", "R", "helpers.R")
}
source(.support_helpers())
rm(.support_helpers)

experiment_results_dir <- function(create = FALSE) {
  out <- file.path(dirname(dirname(script_path())), "results")
  if (isTRUE(create)) dir.create(out, recursive = TRUE, showWarnings = FALSE)
  out
}

paper_dir <- file.path(repo_root(), "papers", "snlls-constrained")
pkg_dir <- file.path(paper_dir, "r-package")
require_pkg("pkgload")
require_pkg("lavaan")
pkgload::load_all(pkg_dir, quiet = TRUE)
suppressPackageStartupMessages(library(lavaan))

canon_keys <- function(df) {
  swap <- df$op == "~~" & df$lhs > df$rhs
  if (any(swap)) {
    lhs <- df$lhs[swap]
    df$lhs[swap] <- df$rhs[swap]
    df$rhs[swap] <- lhs
  }
  df
}

corpus_root <- snlls_corpus_root(paper_dir)
cases <- corpus_cases(corpus_root, weights = "ADF", books = "muthen_2017")
hit <- grep("muthen_2017_ch2_ex2_1__adf$", names(cases), value = TRUE)[[1L]]
case <- cases[[hit]]

fit_lav <- lavaan::sem(
  model = case$model, data = case$data, estimator = "WLS",
  meanstructure = isTRUE(case$model_spec_args$meanstructure),
  fixed.x = isTRUE(case$model_spec_args$fixed_x),
  check.gradient = TRUE, se = "none", test = "none",
  baseline = FALSE, h1 = FALSE)

X <- as.matrix(case$data)
Gamma <- magmaan::magmaan_core$robust_empirical_gamma_with_means(X)
gamma_eigen <- eigen(Gamma, symmetric = TRUE, only.values = TRUE)$values
gamma_condition <- max(abs(gamma_eigen)) / max(min(abs(gamma_eigen)), 1e-300)

prob <- snlls_make_problem(case)
pt <- as.data.frame(prob$spec$partable)
pt$row_idx <- seq_len(nrow(pt))
free_pt <- pt[pt$free > 0L, , drop = FALSE]
if (!"group" %in% names(free_pt)) free_pt$group <- 1L
free_pt <- canon_keys(free_pt)
pt_lav <- lavaan::parameterTable(fit_lav)
est_lav <- canon_keys(pt_lav[pt_lav$free > 0L,
                             c("lhs", "op", "rhs", "group", "est")])
matched <- merge(free_pt[, c("row_idx", "lhs", "op", "rhs", "group")],
                 est_lav, by = c("lhs", "op", "rhs", "group"), sort = FALSE)
prob$spec$partable$ustart[matched$row_idx] <- matched$est

ss <- magmaan:::sample_stats_arg(prob$dat)
pt_for_starts <- magmaan:::partable_arg(prob$spec)
theta_full <- magmaan::magmaan_core$fit_start_values(pt_for_starts, ss)
ev <- magmaan::magmaan_core$evaluate_at(
  prob$spec, prob$dat, as.numeric(theta_full),
  estimator = "WLS", W = prob$W)

im <- magmaan::magmaan_core$model_implied(ev)
sigma_mag <- as.matrix(im$sigma[[1]])
mu_mag <- as.numeric(im$mu[[1]])
imp_lav <- lavaan::lavInspect(fit_lav, "implied")
sigma_lav <- as.matrix(imp_lav$cov)
mu_lav <- as.numeric(imp_lav$mean)
ov_lav <- rownames(sigma_lav)
ov_mag <- rownames(sigma_mag)
if (!is.null(ov_mag) && !identical(ov_mag, ov_lav) && all(ov_lav %in% ov_mag)) {
  idx <- match(ov_lav, ov_mag)
  sigma_mag <- sigma_mag[idx, idx, drop = FALSE]
  mu_mag <- mu_mag[idx]
}
S_mag <- if (is.list(ss$S)) ss$S[[1L]] else ss$S
mb_mag <- if (is.list(ss$mean)) ss$mean[[1L]] else ss$mean

summary <- data.frame(
  case = hit,
  n = nrow(case$data),
  p = ncol(case$data),
  moment_dim = ncol(case$data) + ncol(case$data) * (ncol(case$data) + 1L) / 2L,
  n_free = length(lavaan::coef(fit_lav)),
  lavaan_converged = lavaan::lavInspect(fit_lav, "converged"),
  lavaan_fx = lavaan::lavInspect(fit_lav, "optim")$fx,
  gamma_min_eigen = min(gamma_eigen),
  gamma_max_eigen = max(gamma_eigen),
  gamma_condition = gamma_condition,
  gamma_rcond = rcond(Gamma),
  magmaan_fmin = ev$fmin,
  audit_grad_inf = ev$audit$grad_inf_norm,
  audit_stationary = ev$audit$stationary,
  residual_weighted_norm = sqrt(max(2 * ev$fmin, 0)),
  max_sigma_diff_vs_lavaan = max(abs(sigma_mag - sigma_lav)),
  max_mu_diff_vs_lavaan = max(abs(mu_mag - mu_lav)),
  max_sample_sigma_residual = max(abs(S_mag - sigma_mag)),
  max_sample_mean_residual = max(abs(mb_mag - mu_mag)),
  stringsAsFactors = FALSE
)

eig <- data.frame(index = seq_along(gamma_eigen),
                  eigenvalue = gamma_eigen,
                  stringsAsFactors = FALSE)

out_dir <- file.path(experiment_results_dir(create = TRUE), "adf-conditioning")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(summary, file.path(out_dir, "muthen_adf_conditioning_summary.csv"))
write_csv(eig, file.path(out_dir, "muthen_adf_gamma_eigenvalues.csv"))
print(summary)
