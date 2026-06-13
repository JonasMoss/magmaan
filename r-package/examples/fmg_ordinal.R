# Foldnes-Moss-Gronneberg goodness-of-fit tests for ordinal / mixed-ordinal
# (polychoric least-squares) fits -- the entry point that unblocks the polychoric
# FMG paper (Paper 2).
#
# fmg_tests_ordinal() / fmg_tests_mixed_ordinal() apply the estimator-agnostic
# FMG eigenvalue-tail transforms to the same polychoric-NACOV UGamma spectrum
# that robust_ordinal() / robust_mixed_ordinal() report for single- and
# multi-group fits. The categorical sample statistics are supplied explicitly,
# exactly like robust_ordinal(fit, stats).
#
# Validation here pins the wiring: the SB row reproduces robust_ordinal()'s
# Satorra-Bentler scaling to numerical tolerance. The pEBA/pOLS/PALL transforms
# are magmaan constructions with no external oracle (as on the complete-data and
# FIML paths), so they are only checked for being proper p-values.

core <- magmaan::magmaan_core

make_ord_df <- function(n, cuts_by_var, seed = 1L) {
  set.seed(seed)
  p <- length(cuts_by_var)
  z <- matrix(rnorm(n * p), n, p)
  z[, 2] <- 0.55 * z[, 1] + sqrt(1 - 0.55^2) * z[, 2]
  z[, 3] <- 0.45 * z[, 1] + sqrt(1 - 0.45^2) * z[, 3]
  z[, 4] <- 0.40 * z[, 1] + sqrt(1 - 0.40^2) * z[, 4]
  out <- data.frame(row.names = seq_len(n))
  for (j in seq_len(p)) {
    out[[paste0("x", j)]] <- ordered(cut(z[, j], c(-Inf, cuts_by_var[[j]], Inf),
                                         labels = FALSE))
  }
  out
}

model <- "f =~ x1 + x2 + x3 + x4"
ordered <- paste0("x", 1:4)
df <- make_ord_df(500, list(c(-0.70, 0.40), c(-0.50, 0.60),
                            c(-0.85, 0.20), c(-0.45, 0.75)),
                  seed = 13L)
m <- magmaan::model_spec(model, ordered = ordered, parameterization = "delta")
d <- core$data_ordinal_stats_from_df(df, m)
fit_dwls <- core$fit_dwls_ordinal(
  m, d, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
stopifnot(isTRUE(fit_dwls$ordinal), identical(fit_dwls$estimator, "DWLS"))

## --- default ordinal FMG tests run end to end -------------------------------
tab <- magmaan::fmg_tests_ordinal(fit_dwls, d)
stopifnot(inherits(tab, "magmaan_fmg_tests"))
stopifnot(nrow(tab) == 5L)            # SB, pEBA2, pEBA4, pEBA6, pOLS
stopifnot(all(tab$base == "ls"))      # single least-squares base statistic
stopifnot(all(grepl("_ls$", tab$label)))
stopifnot(all(tab$df == tab$df[[1]]), tab$df[[1]] > 0L)
stopifnot(all(is.finite(tab$p_value)),
          all(tab$p_value >= 0), all(tab$p_value <= 1))

## --- SB row reproduces robust_ordinal()'s Satorra-Bentler scaling -----------
rob <- core$robust_ordinal(fit_dwls, d)
sb_direct <- pchisq(rob$satorra_bentler$chi2_scaled, rob$satorra_bentler$df,
                    lower.tail = FALSE)
sb_row <- tab[tab$input == "SB", , drop = FALSE]
stopifnot(nrow(sb_row) == 1L)
stopifnot(abs(sb_row$p_value - sb_direct) < 1e-10)

## --- WLS weight also works (resolved explicitly) ----------------------------
fit_wls <- core$fit_wls_ordinal(
  m, d, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
tab_wls <- magmaan::fmg_tests_ordinal(fit_wls, d, weight = "WLS")
stopifnot(all(is.finite(tab_wls$p_value)))

## --- _ug and _ml are rejected for an ordinal least-squares fit --------------
err_ug <- tryCatch(magmaan::fmg_tests_ordinal(fit_dwls, d, tests = "SB_UG"),
                   error = function(e) conditionMessage(e))
stopifnot(grepl("unbiased Du-Bentler Gamma", err_ug, fixed = TRUE))
err_ml <- tryCatch(magmaan::fmg_tests_ordinal(fit_dwls, d, tests = "pEBA4_ML"),
                   error = function(e) conditionMessage(e))
stopifnot(grepl("no ML", err_ml))

## --- two-group ordinal GOF reuses the pooled robust_ordinal spectrum --------
g1 <- make_ord_df(220, list(c(-0.70, 0.40), c(-0.50, 0.60),
                            c(-0.85, 0.20), c(-0.45, 0.75)),
                  seed = 31L)
g2 <- make_ord_df(260, list(c(-0.60, 0.55), c(-0.40, 0.70),
                            c(-0.75, 0.30), c(-0.35, 0.85)),
                  seed = 32L)
g1$grp <- "g1"
g2$grp <- "g2"
df_group <- rbind(g1, g2)
m_group <- magmaan::model_spec(model, ordered = ordered, group = "grp",
                               group_labels = c("g1", "g2"),
                               parameterization = "delta")
d_group <- core$data_ordinal_stats_from_df(df_group, m_group)
fit_group <- core$fit_dwls_ordinal(
  m_group, d_group, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
stopifnot(isTRUE(fit_group$ordinal), fit_group$ngroups == 2L)
tab_group <- magmaan::fmg_tests_ordinal(fit_group, d_group)
rob_group <- core$robust_ordinal(fit_group, d_group)
sb_group_direct <- pchisq(rob_group$satorra_bentler$chi2_scaled,
                          rob_group$satorra_bentler$df, lower.tail = FALSE)
sb_group_row <- tab_group[tab_group$input == "SB", , drop = FALSE]
stopifnot(nrow(sb_group_row) == 1L)
stopifnot(abs(sb_group_row$p_value - sb_group_direct) < 1e-10)

## --- mixed continuous/ordinal smoke -----------------------------------------
df_mixed <- df
set.seed(24L)
eta <- rnorm(nrow(df_mixed))
df_mixed$x3 <- 0.8 * eta + 0.6 * rnorm(nrow(df_mixed))
df_mixed$x4 <- 0.7 * eta + 0.7 * rnorm(nrow(df_mixed))
m_mixed <- magmaan::model_spec(model, ordered = c("x1", "x2"),
                               parameterization = "delta", meanstructure = TRUE)
d_mixed <- core$data_mixed_ordinal_stats_from_df(df_mixed, m_mixed)
fit_mixed <- core$fit_dwls_mixed_ordinal(
  m_mixed, d_mixed, control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
stopifnot(isTRUE(fit_mixed$mixed_ordinal))
tab_mixed <- magmaan::fmg_tests_mixed_ordinal(fit_mixed, d_mixed)
stopifnot(all(is.finite(tab_mixed$p_value)))
rob_mixed <- core$robust_mixed_ordinal(fit_mixed, d_mixed)
sb_mixed_direct <- pchisq(rob_mixed$satorra_bentler$chi2_scaled,
                          rob_mixed$satorra_bentler$df, lower.tail = FALSE)
sb_mixed_row <- tab_mixed[tab_mixed$input == "SB", , drop = FALSE]
stopifnot(abs(sb_mixed_row$p_value - sb_mixed_direct) < 1e-10)

## --- two-group mixed continuous/ordinal smoke -------------------------------
df_mixed_group <- df_group
set.seed(43L)
eta_group <- rnorm(nrow(df_mixed_group))
df_mixed_group$x3 <- 0.8 * eta_group + 0.6 * rnorm(nrow(df_mixed_group))
df_mixed_group$x4 <- 0.7 * eta_group + 0.7 * rnorm(nrow(df_mixed_group))
m_mixed_group <- magmaan::model_spec(
  model, ordered = c("x1", "x2"), group = "grp",
  group_labels = c("g1", "g2"), parameterization = "delta",
  meanstructure = TRUE)
d_mixed_group <- core$data_mixed_ordinal_stats_from_df(df_mixed_group,
                                                       m_mixed_group)
fit_mixed_group <- core$fit_dwls_mixed_ordinal(
  m_mixed_group, d_mixed_group,
  control = list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8))
stopifnot(isTRUE(fit_mixed_group$mixed_ordinal),
          fit_mixed_group$ngroups == 2L)
tab_mixed_group <- magmaan::fmg_tests_mixed_ordinal(fit_mixed_group,
                                                    d_mixed_group)
rob_mixed_group <- core$robust_mixed_ordinal(fit_mixed_group, d_mixed_group)
sb_mixed_group_direct <- pchisq(
  rob_mixed_group$satorra_bentler$chi2_scaled,
  rob_mixed_group$satorra_bentler$df, lower.tail = FALSE)
sb_mixed_group_row <- tab_mixed_group[tab_mixed_group$input == "SB", ,
                                      drop = FALSE]
stopifnot(nrow(sb_mixed_group_row) == 1L)
stopifnot(abs(sb_mixed_group_row$p_value - sb_mixed_group_direct) < 1e-10)

cat("ordinal/mixed fmg_tests_ordinal() workflow: ok\n")
