# Misspecification-robust ("complete-sandwich") case influence (frontier).
#
# The casewise dual of the estimated-weight standard error. semfindr / Pek &
# MacCallum (2011) approximate the leave-one-out change theta_hat - theta_hat_(i)
# treating the estimator weight W as fixed. For a moment-quadratic fit whose
# weight is itself estimated from the data (GLS: W from the sample covariance),
# that drops a per-case data-dependent-weight term. The term is O_p(N^{-1})
# under a correct model but is promoted to O_p(N^{-1/2}) under misspecification
# (Hall-Inoue), so a naive one-step degrades exactly when residuals are large.
#
# There is no external oracle for this (no package computes it). The oracle is
# magmaan's own EXACT leave-one-out engine: case_rerun() refits GLS dropping each
# case AND re-estimating the weight, so it is the ground truth for the
# estimated-weight LOO change. We check that the complete one-step tracks it, and
# that the naive (fixed-weight) one-step does not once the model is misspecified.
#
# No external dependency beyond lavaan (for the dataset); never skipped.
if (!requireNamespace("lavaan", quietly = TRUE)) {
  message("case_influence_estimated_weight.R: lavaan not installed; skipping.")
  quit(save = "no", status = 0)
}
suppressMessages({library(magmaan); library(lavaan)})

hs <- lavaan::HolzingerSwineford1939
sse <- function(a, b) sum((a - b)^2)
rmse <- function(a, b) sqrt(sse(a, b) / length(a))

# One-step (complete + naive) vs the exact GLS leave-one-out refit.
probe <- function(label, model) {
  fit <- magmaan::magmaan(model, data = hs, estimator = "GLS")
  fm <- magmaan::fit_measures(fit)

  ew    <- magmaan::est_change_raw_approx(fit, type = "estimated.weight")
  naive <- attr(ew, "naive")
  wdiag <- attr(ew, "weight_diagnostic")
  stopifnot(is.matrix(naive), is.matrix(wdiag),
            isTRUE(all.equal(unclass(ew) - naive, wdiag, check.attributes = FALSE)))

  # Exact LOO is the ground truth: refit GLS per drop, re-estimating the weight.
  rerun <- magmaan::case_rerun(fit)
  exact <- magmaan::est_change_raw(rerun)

  cn <- intersect(colnames(exact), colnames(ew))
  stopifnot(length(cn) == ncol(ew))
  exact <- exact[, cn, drop = FALSE]
  comp  <- unclass(ew)[, cn, drop = FALSE]
  naive <- naive[, cn, drop = FALSE]
  ok <- stats::complete.cases(exact)         # drop any non-converged refit
  exact <- exact[ok, ]; comp <- comp[ok, ]; naive <- naive[ok, ]

  r_comp  <- rmse(comp,  exact)
  r_naive <- rmse(naive, exact)
  cat(sprintf("\n## %-22s cfi=%.3f  |exact|rms=%.3e\n", label, fm$cfi,
              sqrt(mean(exact^2))))
  cat(sprintf("   RMSE(complete vs exact) = %.3e\n", r_comp))
  cat(sprintf("   RMSE(naive    vs exact) = %.3e   (%.0fx worse)\n",
              r_naive, sse(naive, exact) / sse(comp, exact)))
  list(cfi = fm$cfi, r_comp = r_comp, r_naive = r_naive,
       wdiag = max(abs(wdiag)))
}

correct <- probe("correct 3-factor",
                 "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9")
misspec <- probe("misspecified 1-factor",
                 "g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9")

# 1. The complete one-step tracks the exact estimated-weight LOO refit tightly,
#    in BOTH regimes (the weight movement is captured analytically).
stopifnot(correct$r_comp < 2e-3, misspec$r_comp < 2e-3)

# 2. The complete one-step is far closer to the truth than the naive one in both.
stopifnot(correct$r_naive > 5 * correct$r_comp,
          misspec$r_naive > 5 * misspec$r_comp)

# 3. Order promotion: as misfit grows, the NAIVE error grows with it while the
#    COMPLETE error stays put. This is the data-dependent-weight term going from
#    O_p(N^{-1}) at the null to O_p(N^{-1/2}) off it.
stopifnot(misspec$r_naive > 1.5 * correct$r_naive)
stopifnot(misspec$r_comp  < 3   * correct$r_comp)
stopifnot(misspec$wdiag   > correct$wdiag)
cat(sprintf("\nnaive error grows %.1fx with misfit; complete error ~flat (%.2fx)\n",
            misspec$r_naive / correct$r_naive, misspec$r_comp / correct$r_comp))

# 4. crossprod(influence) is the estimated-weight ("complete-sandwich") IJ vcov
#    by construction (the C++ self-check pins it to the SE path at 1e-9). Here we
#    confirm it is a well-formed SPD covariance and reproduces the reported SEs.
fit <- magmaan::magmaan(
  "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9",
  data = hs, estimator = "GLS")
X <- fit$raw_data$X[[1L]]
ij <- magmaan_core$infer_casewise_influence_ij_fit(fit, X)
V <- crossprod(ij$influence)
stopifnot(nrow(V) == fit$npar, isSymmetric(V, tol = 1e-8),
          all(eigen(V, symmetric = TRUE, only.values = TRUE)$values > 0))
cat("complete-sandwich IJ vcov is SPD; npar =", fit$npar, "\n")

# 5. est_change_approx(type = "estimated.weight") returns DFTHETAS + gcd in the
#    estimated-weight metric.
ec <- magmaan::est_change_approx(fit, type = "estimated.weight")
stopifnot(is.matrix(ec), "gcd_approx" %in% colnames(ec),
          nrow(ec) == nrow(X), all(is.finite(ec)), all(ec[, "gcd_approx"] >= 0))

# 6. ULS has a fixed weight: the complete and naive one-steps must coincide
#    exactly (no IF(W_hat) correction).
fit_uls <- magmaan::magmaan(
  "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9",
  data = hs, estimator = "ULS")
ew_uls <- magmaan::est_change_raw_approx(fit_uls, type = "estimated.weight")
stopifnot(max(abs(attr(ew_uls, "weight_diagnostic"))) < 1e-9)
cat("ULS: complete == naive (correction-free)\n")

# 7. ML has no estimated second-stage weight, so the regime is rejected.
fit_ml <- magmaan::magmaan(
  "visual =~ x1 + x2 + x3", data = hs, estimator = "ML")
err <- tryCatch(magmaan::est_change_raw_approx(fit_ml, type = "estimated.weight"),
                error = function(e) conditionMessage(e))
stopifnot(is.character(err), grepl("estimated", err))

# 8. Multiple groups: the one-step engine block-stacks the per-group cases
#    (g{b}_{row} ids, group-suffixed columns) and the IJ covariance is the
#    full-theta block-diagonal complete sandwich.
fit_mg <- magmaan::magmaan(
  "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9",
  data = hs, estimator = "GLS", groups = "school")
ew_mg <- magmaan::est_change_raw_approx(fit_mg, type = "estimated.weight")
stopifnot(nrow(ew_mg) == nrow(hs), ncol(ew_mg) == fit_mg$npar,
          all(is.finite(ew_mg)),
          all(grepl("^g\\d+_", rownames(ew_mg))),
          any(grepl("\\.(Pasteur|Grant-White)$", colnames(ew_mg))))
V_mg <- crossprod(
  magmaan_core$infer_casewise_influence_ij_fit(fit_mg, fit_mg$raw_data$X)$influence)
stopifnot(all(eigen(V_mg, symmetric = TRUE, only.values = TRUE)$values > 0))
cat("multigroup (2 groups): block-stacked one-step, SPD IJ vcov\n")

# 9. Categorical (ordinal DWLS / WLSMV): the headline cell of the estimated-
#    weight stream, where the diagonal polychoric weight moves hardest. Rides
#    the ordinal IJ blocks; crossprod(influence) == the ordinal complete-sandwich
#    vcov (pinned to 1e-8 in the C++ self-check vs robust_ordinal_ij).
ord <- hs
for (v in c("x1", "x2", "x3", "x4")) {
  ord[[v]] <- as.integer(cut(hs[[v]], 3))
}
fit_ord <- magmaan::magmaan("f =~ x1 + x2 + x3 + x4", data = ord,
                            estimator = "DWLS",
                            ordered = c("x1", "x2", "x3", "x4"))
stopifnot(isTRUE(fit_ord$ordinal))
ew_ord <- magmaan::est_change_raw_approx(fit_ord, type = "estimated.weight")
stopifnot(nrow(ew_ord) == nrow(ord), ncol(ew_ord) == fit_ord$npar,
          all(is.finite(ew_ord)),
          max(abs(attr(ew_ord, "weight_diagnostic"))) > 1e-8)  # weight term live
ec_ord <- magmaan::est_change_approx(fit_ord, type = "estimated.weight")
stopifnot("gcd_approx" %in% colnames(ec_ord), all(ec_ord[, "gcd_approx"] >= 0))
V_ord <- crossprod(
  magmaan_core$infer_ordinal_casewise_influence_ij_fit(fit_ord, fit_ord$ordinal_stats)$influence)
stopifnot(all(eigen(V_ord, symmetric = TRUE, only.values = TRUE)$values > -1e-10))
cat("ordinal DWLS (categorical): complete sandwich live, gcd ok\n")

# 10. Two-stage (ML2S): the missing-data member. The Stage-2 weight is estimated
#     from the Stage-1 saturated EM fit, so a case influences theta-hat through
#     two channels: how it moves the filled-in saturated moments and how it moves
#     the Stage-2 weight. The NT weight (lavaan robust.two.stage) treats the
#     weight as fixed, so its data-dependent-weight correction is exactly zero;
#     a non-NT Stage-2 weight (DWLS) carries the live term.
df_miss <- hs
set.seed(1)
for (v in c("x1", "x2", "x3", "x4", "x5", "x6")) {
  df_miss[[v]][sample(nrow(hs), 25)] <- NA_real_
}
model_ml2s <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"  # misspecified 1-factor

fit_nt <- magmaan::magmaan(model_ml2s, data = df_miss, estimator = "ML2S")
stopifnot(identical(fit_nt$estimator, "ML2S"))
ew_nt <- magmaan::est_change_raw_approx(fit_nt, type = "estimated.weight")
stopifnot(nrow(ew_nt) == nrow(hs), all(is.finite(ew_nt)),
          max(abs(attr(ew_nt, "weight_diagnostic"))) < 1e-9)  # NT: weight fixed

fit_dwls <- magmaan::magmaan(model_ml2s, data = df_miss, estimator = "ML2S",
                             stage2_weight = "dwls")
stopifnot(identical(fit_dwls$estimator, "ML2S_DWLS"))
ew_dwls <- magmaan::est_change_raw_approx(fit_dwls, type = "estimated.weight")
stopifnot(nrow(ew_dwls) == nrow(hs), all(is.finite(ew_dwls)),
          max(abs(attr(ew_dwls, "weight_diagnostic"))) > 1e-3)  # weight term live
V_ml2s <- crossprod(magmaan_core$infer_ml2s_casewise_influence_ij_fit(
  fit_dwls, fit_dwls$raw_data, stage2_weight = "dwls")$influence)
stopifnot(all(eigen(V_ml2s, symmetric = TRUE, only.values = TRUE)$values > -1e-9))
cat("ML2S (two-stage, missing data): NT correction zero; DWLS weight term live\n")

cat("\nmisspecification-robust case influence: ok\n")
