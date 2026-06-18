## magmaan R bindings -- ordinal measurement invariance via the `group.equal`
## keyword (Wu-Estabrook theta release), cross-checked against lavaan.
##
## Run from the repo root after installing the package:
##     Rscript r-package/examples/group_equal_ordinal.R
##
## This exercises the R `model_spec(group_equal = ...)` surface end to end. The
## keyword ties the requested families across groups (shared labels, the same
## merge as explicit `equal(...)` labels) and, for ordered-categorical THETA
## models with thresholds equated, frees the group-2 latent-response scale
## (free residual variance `~~`, binary-vetoed). The metric rung also frees
## group-2 indicator intercepts (`~1`); the scalar/intercept rung fixes those
## indicator intercepts back to zero and frees the group-2 latent mean, matching
## lavaan's `cfa(group.equal=, parameterization = "theta")`. The C++ core +
## goldens 0017 (thresholds+loadings), 0019 (thresholds only), and 0020
## (thresholds+loadings+intercepts) already gate the fit; here we confirm the R
## keyword reaches both `lavaan_lavaanify`
## (the cross-group ties) and the ordinal fit (the fit-time release, which the
## partable round-trip cannot carry on its own).
##
## Parity is asserted the way the goldens gate it: npar, the LS chi-square, and
## the estimates matched by op|lhs|rhs|group (robust to plabel numbering), plus
## the structural release count. The data mirror the gated fixtures.

suppressMessages({ library(magmaan); library(lavaan) })
core <- magmaan::magmaan_core
ctrl <- list(max_iter = 4000, ftol = 1e-13, gtol = 1e-8)

make_ord_df_scaled <- function(n, cuts_by_var, seed = 1L,
                               sd_scale = 1, mean_shift = 0) {
  set.seed(seed)
  p <- length(cuts_by_var)
  z <- matrix(rnorm(n * p), n, p)
  z[, 2] <- 0.55 * z[, 1] + sqrt(1 - 0.55^2) * z[, 2]
  z[, 3] <- 0.40 * z[, 1] + 0.25 * z[, 2] + sqrt(0.78) * z[, 3]
  z[, 4] <- 0.30 * z[, 1] + 0.20 * z[, 2] + sqrt(0.87) * z[, 4]
  z <- z * sd_scale + mean_shift
  out <- data.frame(row.names = seq_len(n))
  for (j in seq_len(p))
    out[[paste0("x", j)]] <- ordered(cut(z[, j], c(-Inf, cuts_by_var[[j]], Inf),
                                         labels = FALSE),
                                     levels = seq_len(length(cuts_by_var[[j]]) + 1L))
  out
}

model   <- "f =~ x1 + x2 + x3 + x4"
ordered <- paste0("x", 1:4)
cuts    <- list(c(-0.70, 0.35), c(-0.55, 0.60), c(-0.85, 0.20), c(-0.45, 0.75))
d1 <- make_ord_df_scaled(420, cuts, seed = 211L, sd_scale = 1.0, mean_shift = 0.00)
d2 <- make_ord_df_scaled(360, cuts, seed = 223L, sd_scale = 1.5, mean_shift = 0.45)
d1$school <- "Pasteur"; d2$school <- "Grant-White"
df       <- rbind(d1, d2)
glabels  <- c("Pasteur", "Grant-White")

## Estimates matched on the structural key; constraint/defined rows carry no
## estimate and lavaan/magmaan number their plabels independently, so the key
## (not the plabel) is what we align on. `~*~` is excluded: under theta the
## latent-response scaling is a fixed row, and lavaan reports its derived
## implied value (a function of the fit) while magmaan keeps the nominal 1.0 and
## carries the scale on the released `~~`. The free parameters -- the gated
## quantity -- agree; the C++ goldens compare theta_hat the same way.
est_by_key <- function(pt) {
  estimable <- pt$op %in% c("=~", "|", "~~", "~1", "~")
  d <- pt[estimable, c("op", "lhs", "rhs", "group", "est"), drop = FALSE]
  d$key <- paste(d$op, d$lhs, d$rhs, d$group, sep = "|")
  d
}

fit_and_check <- function(label, ge, est_tol = 5e-3, chisq_tol = 5e-2) {
  spec  <- magmaan::model_spec(model, ordered = ordered,
                               parameterization = "theta",
                               group = "school", group_labels = glabels,
                               group_equal = ge)
  # The keyword survives onto the partable as an integer-index attribute that
  # the ordinal fit reads back to apply the release.
  stopifnot(!is.null(attr(spec$partable, "magmaan.group_equal", exact = TRUE)))

  stats <- core$data_ordinal_stats_from_df(df, spec)
  fit   <- core$fit_dwls_ordinal(spec, stats, control = ctrl)
  lav   <- lavaan::cfa(model, data = df, ordered = ordered, group = "school",
                       estimator = "DWLS", parameterization = "theta",
                       group.equal = ge)

  ## npar
  lav_pt   <- lavaan::parTable(lav)
  lav_npar <- sum(lav_pt$free > 0L)
  stopifnot(length(fit$theta) == lav_npar)

  ## LS chi-square: fit$fmin = 1/2 F, so chi-square = 2 N fmin = N F.
  mg_chisq  <- 2 * fit$ntotal * fit$fmin
  lav_chisq <- as.numeric(lavaan::fitMeasures(lav, "chisq"))

  ## Wu-Estabrook release: group-2 indicator residual `~~` rows are free
  ## (binary-vetoed). Indicator intercepts `~1` are free for threshold/metric
  ## but fixed again for scalar; count the free group-2 indicator `~~`/`~1`
  ## rows in both implementations.
  released <- function(pt) {
    rel <- (pt$op %in% c("~~", "~1")) & pt$lhs %in% ordered &
      pt$group == 2L & pt$free > 0L
    sum(rel)
  }
  mg_rel  <- released(fit$partable)
  lav_rel <- released(lav_pt)

  ## estimate parity, matched by structural key
  mg <- est_by_key(fit$partable)
  lv <- est_by_key(lavaan::parameterEstimates(lav))
  stopifnot(setequal(mg$key, lv$key))
  m <- merge(mg[c("key", "est")], lv[c("key", "est")], by = "key",
             suffixes = c(".mg", ".lav"))
  est_max <- max(abs(m$est.mg - m$est.lav))

  cat(sprintf("\n== %s :: group.equal = c(%s) ==\n", label,
              paste(sprintf('"%s"', ge), collapse = ", ")))
  cat(sprintf("  npar:          magmaan = %d, lavaan = %d\n",
              length(fit$theta), lav_npar))
  cat(sprintf("  LS chi-square: magmaan = %.4f, lavaan = %.4f  (|diff| = %.2e)\n",
              mg_chisq, lav_chisq, abs(mg_chisq - lav_chisq)))
  cat(sprintf("  released grp-2 ~~/~1 rows: magmaan = %d, lavaan = %d\n",
              mg_rel, lav_rel))
  cat(sprintf("  max |est diff| over %d matched rows: %.2e\n",
              nrow(m), est_max))

  stopifnot(identical(as.integer(mg_rel), as.integer(lav_rel)), mg_rel > 0L)
  stopifnot(abs(mg_chisq - lav_chisq) < chisq_tol)
  stopifnot(est_max < est_tol)
  invisible(fit)
}

## 0017 analogue: thresholds + loadings tied (the standard ordinal metric step).
fit_and_check("thresholds + loadings", c("thresholds", "loadings"))
## 0019 analogue: thresholds only -- the scale/intercept release is keyed on the
## threshold tie, loadings stay free per group.
fit_and_check("thresholds only", c("thresholds"))
## 0020 analogue: scalar/intercept rung -- group-2 indicator intercepts are
## pinned back to zero while the group-2 latent mean is freed.
fit_and_check("thresholds + loadings + intercepts",
              c("thresholds", "loadings", "intercepts"))

cat("\nordinal group.equal (theta) R surface vs lavaan: ok\n")
