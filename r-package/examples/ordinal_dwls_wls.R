suppressMessages(requireNamespace("lavaan"))
core <- magmaan::magmaan_core

make_ord_df <- function(n, cuts_by_var, seed = 1L) {
  set.seed(seed)
  p <- length(cuts_by_var)
  z <- matrix(rnorm(n * p), n, p)
  z[, 2] <- 0.55 * z[, 1] + sqrt(1 - 0.55^2) * z[, 2]
  z[, 3] <- 0.40 * z[, 1] + 0.25 * z[, 2] + sqrt(0.78) * z[, 3]
  z[, 4] <- 0.30 * z[, 1] + 0.20 * z[, 2] + sqrt(0.87) * z[, 4]
  out <- data.frame(row.names = seq_len(n))
  for (j in seq_len(p)) {
    out[[paste0("x", j)]] <- ordered(cut(z[, j],
                                         c(-Inf, cuts_by_var[[j]], Inf),
                                         labels = FALSE))
  }
  out
}

model <- "f =~ x1 + x2 + x3 + x4"
ordered <- paste0("x", 1:4)
df <- make_ord_df(360, list(c(-0.70, 0.35), c(-0.55, 0.60),
                            c(-0.85, 0.20), c(-0.45, 0.75)),
                  seed = 11L)

m <- magmaan::model_spec(model, ordered = ordered, parameterization = "delta")
d <- core$data_ordinal_stats_from_df(df, m)
d_h <- core$data_ordinal_stats_from_df(
  df, m, robust = "h_weighted", h_kind = "wma_hard_cap", h_k = 1.25)
d_dpd <- core$data_ordinal_stats_from_df(df, m, robust = "dpd", alpha = 0.25)
stopifnot(identical(d_h$robust_method, "h_weighted"))
stopifnot(identical(d_dpd$robust_method, "dpd"))
stopifnot(length(d_h$diagnostics) == 1L, length(d_dpd$diagnostics) == 1L)
stopifnot(nrow(d_h$NACOV[[1]]) == nrow(d$NACOV[[1]]))
stopifnot(nrow(d_dpd$NACOV[[1]]) == nrow(d$NACOV[[1]]))

lavaan_wls <- lavaan::cfa(model, data = df, ordered = ordered,
                          estimator = "WLS", parameterization = "delta")
lavaan_samp <- lavaan::lavInspect(lavaan_wls, "sampstat")

stopifnot(max(abs(d$thresholds[[1]] - as.numeric(lavaan_samp$th))) < 5e-8)
stopifnot(max(abs(d$R[[1]] - unname(as.matrix(lavaan_samp$cov)))) < 5e-4)
stopifnot(max(abs(d$W_wls[[1]] %*% d$NACOV[[1]] - diag(nrow(d$NACOV[[1]])))) < 1e-8)
stopifnot(identical(dimnames(d$R[[1]]), list(ordered, ordered)))
stopifnot(identical(d$ov_names[[1]], ordered))
stopifnot(identical(d$ordered, ordered))
stopifnot(identical(attr(m$partable, "magmaan.parameterization"), "delta"))
stopifnot(identical(d$threshold_ov[[1]], rep(seq_along(ordered), each = 2L)))
stopifnot(identical(d$threshold_level[[1]], rep(1:2, times = length(ordered))))
moments <- c(d$thresholds[[1]], d$R[[1]][lower.tri(d$R[[1]])])
stopifnot(is.list(d$moments), max(abs(d$moments[[1]] - moments)) < 1e-12)
stopifnot(length(moments) == nrow(d$NACOV[[1]]))
stopifnot(length(moments) == nrow(d$W_dwls[[1]]))
stopifnot(length(moments) == nrow(d$W_wls[[1]]))

fit_dwls <- core$fit_dwls_ordinal(
  m, d, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
fit_wls <- core$fit_wls_ordinal(
  m, d, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
fit_dwls_h <- core$fit_dwls_ordinal(
  m, d_h, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
fit_dwls_dpd <- core$fit_dwls_ordinal(
  m, d_dpd, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))

stopifnot(isTRUE(fit_dwls$ordinal), identical(fit_dwls$estimator, "DWLS"))
stopifnot(isTRUE(fit_wls$ordinal), identical(fit_wls$estimator, "WLS"))
stopifnot(isTRUE(fit_dwls_h$ordinal), isTRUE(fit_dwls_dpd$ordinal))
stopifnot(length(fit_dwls$theta) == sum(lavaan::parTable(lavaan_wls)$free > 0L))

lavaan_dwls <- lavaan::cfa(model, data = df, ordered = ordered,
                           estimator = "DWLS", parameterization = "delta")
# fit$fmin = ½F (uniform objective scale); χ² = 2·N·fmin = N·F.
chisq_dwls <- 2 * fit_dwls$ntotal * fit_dwls$fmin
chisq_wls <- 2 * fit_wls$ntotal * fit_wls$fmin
stopifnot(abs(chisq_dwls - lavaan::fitMeasures(lavaan_dwls, "chisq")) < 0.08)
stopifnot(abs(chisq_wls - lavaan::fitMeasures(lavaan_wls, "chisq")) < 0.08)

rob_dwls <- core$robust_ordinal(fit_dwls, d)
rob_wls <- core$robust_ordinal(fit_wls, d, weight = "WLS")
stopifnot(nrow(rob_dwls$vcov) == length(fit_dwls$theta))
stopifnot(length(rob_dwls$se) == length(fit_dwls$theta))
stopifnot(all(is.finite(rob_dwls$se)))
stopifnot(identical(rob_dwls$df, 2L))
stopifnot(length(rob_dwls$eigvals) == rob_dwls$df)
stopifnot(is.finite(rob_dwls$satorra_bentler$scale_c))
stopifnot(is.finite(rob_dwls$mean_var_adjusted$df_adj))
stopifnot(is.finite(rob_dwls$scaled_shifted$scale_a))
stopifnot(all(is.finite(rob_wls$se)))
stopifnot(identical(rob_wls$df, 2L))

# Regression: standardized solutions are exposed for ordinal/mixed fits and
# match lavaan std.all. Under the delta parameterization a categorical
# indicator's latent response is unit-variance, so std.all standardizes its
# loading by the latent SD only — recovering the true loading. The earlier
# guard refused these calls because the generic path divided by the assembled
# σ_rr (= λ²ψ + 1), shrinking a true .6 loading to ~.52; see
# docs/backlog/todo.md.
lv_std_dwls <- lavaan::standardizedSolution(lavaan_dwls)
lv_load <- lv_std_dwls[lv_std_dwls$op == "=~", ]
ld_rows <- which(fit_dwls$partable$op == "=~" & fit_dwls$partable$free > 0L)
ld_names <- fit_dwls$partable$rhs[ld_rows]
lv_match <- lv_load$est.std[match(ld_names, lv_load$rhs)]
for (call in c("measures_standardize_all", "measures_standardize_lv")) {
  sol <- core[[call]](fit_dwls, rob_dwls$vcov)
  mg_load <- sol$theta[fit_dwls$partable$free[ld_rows]]
  stopifnot(length(mg_load) == length(lv_match), all(is.finite(mg_load)),
            max(abs(mg_load - lv_match)) < 5e-3)
}
# Factor scores remain unsupported for ordinal fits.
fs_msg <- tryCatch(
  core$measures_factor_scores(fit_dwls, as.matrix(df), method = "regression"),
  error = function(e) conditionMessage(e))
stopifnot(is.character(fs_msg), grepl("not exposed for ordinal", fs_msg))

df_mixed <- df
set.seed(22L)
eta <- rnorm(nrow(df_mixed))
df_mixed$x3 <- 0.8 * eta + 0.6 * rnorm(nrow(df_mixed))
df_mixed$x4 <- 0.7 * eta + 0.7 * rnorm(nrow(df_mixed))
m_mixed <- magmaan::model_spec(model, ordered = c("x1", "x2"),
                               parameterization = "delta", meanstructure = TRUE)
d_mixed <- core$data_mixed_ordinal_stats_from_df(df_mixed, m_mixed)
d_mixed_dpd <- core$data_mixed_ordinal_stats_from_df(
  df_mixed, m_mixed, polyserial = "dpd", alpha = 0.35)
stopifnot(identical(d_mixed_dpd$robust_method, "polyserial_dpd"))
stopifnot(length(d_mixed_dpd$diagnostics) == 1L)
stopifnot(nrow(d_mixed_dpd$NACOV[[1]]) == nrow(d_mixed$NACOV[[1]]))
stopifnot(identical(d_mixed_dpd$thresholds, d_mixed$thresholds))
fit_mixed_dpd <- core$fit_dwls_mixed_ordinal(
  m_mixed, d_mixed_dpd,
  control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
stopifnot(isTRUE(fit_mixed_dpd$mixed_ordinal))
stopifnot(all(is.finite(fit_mixed_dpd$theta)))

bad <- df
bad$x1 <- ordered(as.integer(bad$x1), levels = 1:4)
bad <- bad[as.integer(bad$x1) != 4L, , drop = FALSE]
err <- tryCatch(core$data_ordinal_stats_from_df(bad, m),
                error = function(e) conditionMessage(e))
stopifnot(grepl("empty ordinal category", err, fixed = TRUE))

cat("ordinal magmaan_core$data_ordinal_stats_from_df()/DWLS/WLS workflow: ok\n")
