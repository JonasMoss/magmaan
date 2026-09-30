# One replicate: fit H1 (and H0), then every base statistic under every
# spectrum and calibration.
#
# Base statistics
#   lr    : 2N F_ML (global) or its difference (nested).
#   score : magmaan's expected-information score statistic at the null fit
#           (the ordinary-user policy's `score`).
#   rls   : lavaan's browne.residual.nt.model (global) or the difference of the
#           two models' RLS statistics (nested; lavaan's single-group nested FMG
#           base).
# Spectra (eigenvalues of U Gamma for the tested restrictions)
#   own      : the policy's spectrum for that statistic (score rows at the null
#              fit for score; centred rows for lr). Only for lr and score.
#   biased   : empirical Gamma at the fitted H1 (global) or Satorra's (2000)
#              restriction map at H1 (nested), shared by all three bases.
#   unbiased : the same with Du and Bentler's unbiased Gamma.
#   rows     : no spectrum; the corrected MV below estimates the two moments it
#              needs directly from the policy score quadratic's casewise rows.
# Calibrations: std (chi-square df), sb (mean scaling), mv (Satterthwaite
# mean-and-variance adjusted), peba4 (penalised eigenvalue block averaging, 4
# blocks), and cmv (experiment research/49's corrected MV, rows spectrum only).
calibrations <- c(std = "standard", sb = "sb", mv = "mean_var_adjusted", peba4 = "peba")

calibrate_all <- function(statistic, df, eig) {
  vapply(names(calibrations), function(m) {
    if (!is.finite(statistic) || is.null(eig) || length(eig) != df || any(!is.finite(eig))) return(NA_real_)
    if (m == "std") return(pchisq(max(statistic, 0), df, lower.tail = FALSE))
    if (statistic < 0) return(1)
    magmaanlab:::infer_fmg_test(statistic, df, eig, method = calibrations[[m]],
                                            param = if (m == "peba4") 4 else 0)$p_value
  }, numeric(1))
}

# Corrected MV (research/49): the Satterthwaite reference for T ~ sum_k l_k chi2_1
# needs tau1 = tr(V) and tau2 = tr(V^2), V the covariance of s = sum_i y_i. With
# independent groups V = sum_g n_g C_g, C_g the within-group row covariance, so
# tau1 = sum_g n_g tr(C_g) and
# tau2 = sum_g n_g^2 tr(C_g^2) + sum_{g != h} n_g n_h tr(C_g C_h).
# tr(C_g) and tr(C_g C_h) use the unbiased sample covariances (independent across
# groups); tr(C_g^2) uses the translation-invariant all-distinct U-statistic of
# research/49 (kernel ((y_i - y_j)'(y_k - y_l))^2 / 4). Unbiased for iid rows
# under a fixed transform; the SEM projection is estimated, as in 49.
u_trace_square <- function(Y) {
  n <- nrow(Y)
  Z <- sweep(Y, 2, colMeans(Y), "-")
  norm2 <- rowSums(Z^2); T <- sum(norm2); D <- sum(norm2^2)
  C <- if (ncol(Z) <= n) sum(crossprod(Z)^2) else sum(tcrossprod(Z)^2)
  (C - D) / (n * (n - 1)) - 2 * (2 * D - C) / (n * (n - 1) * (n - 2)) +
    (T^2 + 2 * C - 6 * D) / (n * (n - 1) * (n - 2) * (n - 3))
}

cmv_moments <- function(rows, sizes) {
  stopifnot(nrow(rows) == sum(sizes), all(sizes >= 4L))
  blocks <- split(seq_len(nrow(rows)), rep(seq_along(sizes), sizes))
  C <- lapply(blocks, function(i) stats::cov(rows[i, , drop = FALSE]))
  tau1 <- sum(sizes * vapply(C, function(c) sum(diag(c)), numeric(1)))
  tau2 <- sum(sizes^2 * vapply(blocks, function(i) u_trace_square(rows[i, , drop = FALSE]), numeric(1)))
  for (g in seq_along(sizes)) for (h in seq_along(sizes)) if (g != h)
    tau2 <- tau2 + sizes[g] * sizes[h] * sum(C[[g]] * C[[h]])
  c(tau1 = tau1, tau2 = tau2)
}

cmv_p <- function(statistic, m) {
  if (!is.finite(statistic) || !all(is.finite(m)) || m[["tau1"]] <= 0 || m[["tau2"]] <= 0) return(NA_real_)
  pchisq(max(statistic, 0) * m[["tau1"]] / m[["tau2"]], df = m[["tau1"]]^2 / m[["tau2"]], lower.tail = FALSE)
}

block_sizes <- function(pop, d) {
  if (length(pop$groups) == 1L) return(nrow(d))
  vapply(pop$groups, function(g) sum(d$group == g$label), integer(1))
}

model_type <- function(pop) if (identical(pop$lavaan_function, "growth")) "growth" else "sem"

fit_one <- function(pop, syntax, d) {
  groups <- if (length(pop$groups) > 1L) "group" else NULL
  magmaanlab::fit_model(syntax, d, estimator = "ML", groups = groups,
                        meanstructure = pop$meanstructure, model_type = model_type(pop))
}

raw_blocks <- function(pop, d) {
  ov <- pop$groups[[1]]$ov
  if (length(pop$groups) == 1L) return(as.matrix(d[, ov]))
  lapply(pop$groups, function(g) as.matrix(d[d$group == g$label, ov]))
}

rls_stat <- function(fit) magmaanlab:::infer_rls_chi2_fit(fit, NULL)$statistic

# Wide per-replicate record: the three statistics, the degrees of freedom, and
# one p-value column per base x spectrum x calibration.
arm_names <- function() {
  arms <- c(outer(c("lr_own", "score_own"), names(calibrations), paste, sep = "_"))
  c(arms, c(outer(outer(c("lr", "score", "rls"), c("biased", "unbiased"), paste, sep = "_"),
                  names(calibrations), paste, sep = "_")),
    paste0(c("lr", "score", "rls"), "_rows_cmv"))
}

p_columns <- function(stats, df, own, shared, moments) {
  out <- setNames(rep(NA_real_, length(arm_names())), paste0("p_", arm_names()))
  for (b in names(own)) {
    p <- calibrate_all(stats[[b]], df, own[[b]])
    out[paste0("p_", b, "_own_", names(p))] <- p
  }
  for (b in names(stats)) for (s in names(shared)) {
    p <- calibrate_all(stats[[b]], df, shared[[s]])
    out[paste0("p_", b, "_", s, "_", names(p))] <- p
  }
  for (b in names(stats)) out[[paste0("p_", b, "_rows_cmv")]] <- cmv_p(stats[[b]], moments)
  out
}

# Rows of the policy's own score quadratic and the corrected-MV moments; the
# check is the statistic's reconstruction from its rows (must be ~0).
score_rows_moments <- function(quadratic, statistic, sizes) {
  rows <- magmaanlab::inference_rows(quadratic)
  m <- cmv_moments(rows, sizes)
  c(m, rows_check = abs(sum(colSums(rows)^2) - statistic) / max(1, statistic))
}

global_record <- function(f1, d, X, pol, sizes) {
  sp <- magmaanlab:::infer_fmg_ugamma_spectra(f1, X, need_unbiased = TRUE)
  st <- c(lr = pol$lr$statistic, score = pol$score$statistic, rls = rls_stat(f1))
  df <- pol$lr$df
  shared <- magmaanlab::prepare_inference_data(f1, d, storage = "casewise")
  q <- magmaanlab::inference_quadratic(magmaanlab::prepare_inference(f1, shared), "score")
  m <- score_rows_moments(q, st[["score"]], sizes)
  c(df = df, stat_lr = st[["lr"]], stat_score = st[["score"]], stat_rls = st[["rls"]],
    check = max(abs(sort(pol$lr$eigenvalues) - sort(sp$biased))), m,
    p_columns(st, df, list(lr = pol$lr$eigenvalues, score = pol$score$eigenvalues),
              list(biased = sp$biased, unbiased = sp$unbiased), m))
}

nested_record <- function(f1, f0, d, X, sizes) {
  pn <- magmaanlab::policy_nested(f1, f0, d)
  rn <- magmaanlab::robust_nested_lrt(f1, f0, data = X, gamma = "both", method = "restriction_map")
  df <- rn$df_diff
  st <- c(lr = pn$lr$statistic, score = pn$score$statistic, rls = rls_stat(f0) - rls_stat(f1))
  shared <- magmaanlab::prepare_inference_data(f1, d, storage = "casewise")
  h <- magmaanlab::prepare_hypothesis(magmaanlab::prepare_inference(f0, shared),
                                      magmaanlab::prepare_inference(f1, shared))
  m <- score_rows_moments(magmaanlab::inference_quadratic(h, "score"), st[["score"]], sizes)
  c(df = df, stat_lr = st[["lr"]], stat_score = st[["score"]], stat_rls = st[["rls"]],
    check = abs(rn$T_diff - pn$lr$statistic), m,
    p_columns(st, df, list(lr = pn$lr$eigenvalues, score = pn$score$eigenvalues),
              list(biased = rn$eigenvalues, unbiased = rn$eigenvalues_unbiased), m))
}

quietly <- function(expr) tryCatch(
  withCallingHandlers(expr, warning = function(w) invokeRestart("muffleWarning")),
  error = function(e) e)

fit_status <- function(fit) {
  if (inherits(fit, "error")) return(paste("fit error:", conditionMessage(fit)))
  if (!isTRUE(fit$converged)) return("not converged")
  ""
}

record_row <- function(meta, test, status, admissible, values = NULL) {
  empty <- c(df = NA_real_, stat_lr = NA_real_, stat_score = NA_real_, stat_rls = NA_real_,
             check = NA_real_, tau1 = NA_real_, tau2 = NA_real_, rows_check = NA_real_, setNames(rep(NA_real_, length(arm_names())), paste0("p_", arm_names())))
  if (!is.null(values)) empty[names(values)] <- values
  cbind(meta, test = test, status = status, admissible = admissible,
        as.data.frame(as.list(empty)), stringsAsFactors = FALSE)
}

one_rep <- function(pop, cals, cell, rep_id, seed) {
  t0 <- proc.time()[["elapsed"]]
  meta <- data.frame(case = pop$id, dgp = cell$dgp, n = cell$n, rep = rep_id, seed = seed,
                     stringsAsFactors = FALSE)
  tests <- c("gof", if (!is.null(pop$h0)) "nested")
  finish <- function(rows) {
    out <- do.call(rbind, rows)
    out$seconds <- proc.time()[["elapsed"]] - t0
    rownames(out) <- NULL
    out
  }
  d <- quietly(draw_sample(pop, cals, cell$n, cell$dgp, seed))
  if (inherits(d, "error"))
    return(finish(lapply(tests, record_row, meta = meta, status = paste("draw:", conditionMessage(d)), admissible = NA)))
  X <- raw_blocks(pop, d)
  sizes <- block_sizes(pop, d)
  f1 <- quietly(fit_one(pop, pop$h1, d))
  st1 <- fit_status(f1)
  adm1 <- if (nzchar(st1)) NA else isTRUE(f1$diagnostics$admissibility$admissible)
  rows <- list()
  rows$gof <- if (nzchar(st1)) record_row(meta, "gof", paste("H1", st1), NA) else {
    v <- quietly(global_record(f1, d, X, magmaanlab::policy_inference(f1, d), sizes))
    if (inherits(v, "error")) record_row(meta, "gof", conditionMessage(v), adm1) else
      record_row(meta, "gof", "", adm1, v)
  }
  if (!is.null(pop$h0)) {
    f0 <- quietly(fit_one(pop, pop$h0, d))
    st0 <- fit_status(f0)
    status <- paste(c(if (nzchar(st1)) paste("H1", st1), if (nzchar(st0)) paste("H0", st0)), collapse = "; ")
    rows$nested <- if (nzchar(status)) record_row(meta, "nested", status, NA) else {
      adm <- adm1 && isTRUE(f0$diagnostics$admissibility$admissible)
      v <- quietly(nested_record(f1, f0, d, X, sizes))
      if (inherits(v, "error")) record_row(meta, "nested", conditionMessage(v), adm) else
        record_row(meta, "nested", "", adm, v)
    }
  }
  finish(rows)
}
