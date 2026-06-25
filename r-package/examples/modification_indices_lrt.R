## LRT (refit) modification indices: the lavaan modindices() table, by refit.
##
## modification_indices_lrt(fit, data) is the refit counterpart of
## modification_indices(): for each parameter ABSENT from the model (cross
## loadings, residual covariances) it refits the augmented model and reports the
## nested chi-square difference (lrt), the model-misspecification-robust
## observed-bread reference law (lrt_p_obs), and the EXACT refitted change
## (epc_lrt, plus its std.all standardization sepc_lrt). mi_screen() then adds the
## "modern MI" decision layer: BH multiplicity + a significant x substantial
## verdict. See docs/research/notes/lrt_modification_indices.tex.

suppressMessages(library(magmaan))

## Single-factor CFA with an omitted x1-x2 residual doublet (shared nuisance u):
## the 1-factor model leaves a strong, unambiguous x1~~x2 modification index.
set.seed(1)
n <- 1000
f <- rnorm(n); u <- rnorm(n)
mk <- function(load, extra = 0)
  load * f + extra * u + rnorm(n, 0, sqrt(pmax(1 - load^2 - extra^2, 1e-6)))
dat <- data.frame(x1 = mk(.7, .5), x2 = mk(.7, .5), x3 = mk(.7),
                  x4 = mk(.7), x5 = mk(.7), x6 = mk(.7))
syntax <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"
fit <- magmaan(syntax, dat, estimator = "ML")

mi <- modification_indices_lrt(fit, dat)
stopifnot(inherits(mi, "magmaan_mi_lrt"),
          all(c("mi", "mi_p", "lrt", "lrt_p", "lrt_p_obs", "epc", "epc_lrt",
                "sepc_lrt") %in% names(mi)),
          nrow(mi) > 1L)
## The dominant candidate is the omitted doublet x1~~x2.
stopifnot(mi$op[1] == "~~", setequal(c(mi$lhs[1], mi$rhs[1]), c("x1", "x2")))

## lrt_p_obs is the model-misspecification-robust reference law (observed-Hessian
## profile-LRT exact mixture); it matches a direct ml_profile_lrt() call.
top <- mi[1, ]
rel <- magmaan(paste0(syntax, "\n", top$lhs, " ", top$op, " ", top$rhs),
               dat, estimator = "ML")
ov <- if (is.list(fit$ov_names)) fit$ov_names[[1]] else fit$ov_names
pr <- magmaan_core$ml_profile_lrt(rel, fit, list(as.matrix(dat[, ov])))
stopifnot(is.finite(mi$lrt_p_obs[1]), abs(mi$lrt_p_obs[1] - pr$p_mixture) < 1e-9)

## The nested chi-square difference reproduces a manual augmented refit, and
## sepc_lrt matches a direct standardized() call on the released fit.
chi2 <- function(g) magmaan_core$inference_chi2_stat(
  list(S = g$S, nobs = g$nobs, mean = g$sample_mean), g$fmin)
stopifnot(abs(top$lrt - (chi2(fit) - chi2(rel))) < 1e-6)
sel <- rel$partable$op == top$op & rel$partable$lhs == top$lhs &
  rel$partable$rhs == top$rhs & rel$partable$free > 0
fk  <- rel$partable$free[sel][1]; np <- max(rel$partable$free)
stopifnot(abs(top$sepc_lrt -
              standardized(rel, matrix(0, np, np), "all")$theta[fk]) < 1e-9)

cat("LRT vs one-step modification indices (top 5):\n")
print(head(mi[, c("lhs", "op", "rhs", "mi", "lrt", "lrt_p_obs", "epc_lrt",
                  "sepc_lrt")], 5),
      row.names = FALSE)

## The "modern MI" decision layer: mi_screen() adds BH-FDR multiplicity control on
## the observed-bread robust p, plus a Saris-Satorra-Sorbom verdict combining
## corrected significance with substantive effect size (|sepc_lrt|).
scr <- mi_screen(mi)  # auto-resolves p = lrt_p_obs, effect = sepc_lrt
stopifnot(inherits(scr, "magmaan_mi_lrt"),
          all(c("lrt_p_obs_bh", "verdict") %in% names(scr)),
          all(scr$verdict %in% c("free", "trivial", "underpowered", "ok", NA)),
          all(scr$lrt_p_obs_bh >= scr$lrt_p_obs - 1e-12, na.rm = TRUE))
## The omitted doublet survives BH + the effect-size gate as a real release.
stopifnot(scr$verdict[1] == "free")

cat("\nverdict counts (BH alpha=0.05, |sepc|>=0.10):\n")
print(table(scr$verdict, useNA = "ifany"))
cat("modern MI: ok\n")

## --- Ordinal DWLS: the same table, dispatched to the categorical profile-LRT ---
## For an all-ordinal DWLS fit, lrt_p_obs routes to the ordinal estimated-weight
## profile-LRT (the ML observed-Hessian path is ML-only). Everything else -- the
## refit, the nested chi-square, the EPC standardization, mi_screen() -- is shared.
set.seed(2)
no <- 1500
fo <- rnorm(no); uo <- rnorm(no)
mko <- function(load, extra = 0)
  load * fo + extra * uo + rnorm(no, 0, sqrt(pmax(1 - load^2 - extra^2, 1e-6)))
cut3 <- function(z) as.integer(cut(z, c(-Inf, -0.6, 0.6, Inf)))
odat <- data.frame(x1 = cut3(mko(.7, .5)), x2 = cut3(mko(.7, .5)),
                   x3 = cut3(mko(.7)), x4 = cut3(mko(.7)),
                   x5 = cut3(mko(.7)), x6 = cut3(mko(.7)))
ovo  <- paste0("x", 1:6)
ofit <- magmaan(syntax, odat, estimator = "DWLS", ordered = ovo)

omi <- modification_indices_lrt(ofit, odat)
stopifnot(inherits(omi, "magmaan_mi_lrt"),
          all(c("lrt", "lrt_p", "lrt_p_obs", "epc_lrt", "sepc_lrt") %in%
                names(omi)),
          nrow(omi) > 1L)
## The omitted doublet x1~~x2 is the dominant candidate and carries a finite,
## dispatched robust p (NOT the ML reference law).
odoub <- which(omi$op == "~~" &
                 mapply(function(l, r) setequal(c(l, r), c("x1", "x2")),
                        omi$lhs, omi$rhs))
stopifnot(length(odoub) == 1L, odoub == 1L, is.finite(omi$lrt_p_obs[1]))

## lrt_p_obs matches a direct ordinal_profile_lrt() call on the released fit,
## sharing the anchor's polychoric stage (fit$ordinal_stats).
otop <- omi[1, ]
orel <- magmaan(paste0(syntax, "\n", otop$lhs, " ", otop$op, " ", otop$rhs),
                odat, estimator = "DWLS", ordered = ovo)
opr  <- magmaan_core$ordinal_profile_lrt(orel, ofit, ofit$ordinal_stats)
stopifnot(abs(omi$lrt_p_obs[1] - opr$p_mixture) < 1e-9)

oscr <- mi_screen(omi)  # same decision layer, p = lrt_p_obs, effect = sepc_lrt
stopifnot("verdict" %in% names(oscr), oscr$verdict[1] == "free")

cat("\nordinal DWLS LRT modification indices (top 3):\n")
print(head(omi[, c("lhs", "op", "rhs", "lrt", "lrt_p_obs", "sepc_lrt")], 3),
      row.names = FALSE)
cat("ordinal modern MI: ok\n")

## --- Mixed (ordinal + continuous) DWLS: the mixed-ordinal profile-LRT branch ---
## x1-x3 ordinal, x4-x6 continuous; the doublet x1~~x2 sits in the ordinal block.
mdat <- data.frame(x1 = cut3(mko(.7, .5)), x2 = cut3(mko(.7, .5)),
                   x3 = cut3(mko(.7)), x4 = mko(.7), x5 = mko(.7), x6 = mko(.7))
mfit <- magmaan(syntax, mdat, estimator = "DWLS", ordered = c("x1", "x2", "x3"),
                meanstructure = TRUE)
stopifnot(isTRUE(mfit$mixed_ordinal))
mmi <- modification_indices_lrt(mfit, mdat)
mrow <- which(mmi$op == "~~" &
                mapply(function(l, r) setequal(c(l, r), c("x1", "x2")),
                       mmi$lhs, mmi$rhs))
stopifnot(length(mrow) == 1L, is.finite(mmi$lrt_p_obs[mrow]))
## Matches a direct mixed_ordinal_profile_lrt() call on the released fit.
mtop <- mmi[mrow, ]
mrel <- magmaan(paste0(syntax, "\n", mtop$lhs, " ", mtop$op, " ", mtop$rhs),
                mdat, estimator = "DWLS", ordered = c("x1", "x2", "x3"),
                meanstructure = TRUE)
mpr  <- magmaan_core$mixed_ordinal_profile_lrt(mrel, mfit, mfit$mixed_ordinal_stats)
stopifnot(abs(mmi$lrt_p_obs[mrow] - mpr$p_mixture) < 1e-9)
cat("\nmixed-ordinal DWLS: x1~~x2 lrt_p_obs = ",
    format(mmi$lrt_p_obs[mrow], digits = 3), " (dispatched)\n", sep = "")
cat("mixed modern MI: ok\n")
