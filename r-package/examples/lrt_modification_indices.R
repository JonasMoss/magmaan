## LRT-based (refit) modification indices for measurement-invariance releases.
##
## modification_indices_lrt(fit, data) takes a constrained group.equal anchor and,
## for each cross-group tied parameter family, refits the released model (that one
## family freed via group_partial) and runs the nested Satorra-2000 difference
## test. Unlike the one-step score_tests() modification index, the refit isolates
## each release exactly even under structural misspecification elsewhere, and its
## "EPC" is the exact refitted parameter change. It composes magmaan() and
## robust_nested_lrt() and adds no SEM logic of its own. See
## docs/research/notes/lrt_modification_indices.tex.

suppressMessages(library(magmaan))

## Two-group one-factor CFA, loadings 0.6, with x3 non-invariant in the smaller
## group so there is a real release for the sweep to localize.
set.seed(1)
gen <- function(n, load) {
  p <- length(load); f <- rnorm(n); rs <- sqrt(pmax(1 - load^2, 1e-6))
  X <- sapply(seq_len(p), function(j) load[j] * f + rnorm(n, 0, rs[j]))
  colnames(X) <- paste0("x", seq_len(p)); as.data.frame(X)
}
dA <- gen(200, c(.6, .6, .9, .6, .6, .6)); dA$g <- "A"
dB <- gen(800, rep(.6, 6));                dB$g <- "B"
dat <- rbind(dA, dB)
syntax <- "f =~ x1 + x2 + x3 + x4 + x5 + x6"

anchor <- magmaan(syntax, dat, estimator = "ML", groups = "g",
                  group_equal = "loadings")
mi <- modification_indices_lrt(anchor, dat)

stopifnot(inherits(mi, "magmaan_mi_lrt"), nrow(mi) == 5L)
## x3 is the largest release and is significant; reference-law columns present.
stopifnot(mi$rhs[1] == "x3",
          all(c("lrt", "p_unscaled", "p_scaled", "p_mixture") %in% names(mi)),
          mi$p_mixture[1] < 0.01)

## The LRT and its robust p-value match a hand-built two-fit nested test exactly.
ov <- paste0("x", 1:6)
data_list <- lapply(c("A", "B"),
                    function(g) as.matrix(dat[dat$g == g, ov, drop = FALSE]))
rel <- magmaan(syntax, dat, estimator = "ML", groups = "g",
               group_equal = "loadings", group_partial = "f=~x3")
ref <- robust_nested_lrt(rel, anchor, data = data_list, method = "restriction_map")
stopifnot(abs(mi$lrt[mi$rhs == "x3"] - ref$T_diff) < 1e-8,
          abs(mi$p_mixture[mi$rhs == "x3"] - ref$p_mixture) < 1e-10)

## Exact EPC: the released x3 loading differs across groups (the non-invariance).
est_x3 <- attr(mi, "released_estimates")[[1]]$est
stopifnot(length(est_x3) == 2L, abs(diff(est_x3)) > 0.3)

cat("LRT-based modification indices (ML):\n")
print(mi[, c("lhs", "op", "rhs", "df", "lrt", "p_mixture", "epc_range")],
      row.names = FALSE)

## Conditional localization: an anchor that already frees x5 (a group_partial
## baseline) is carried into every released refit, so x5 drops out of the
## candidate set and the remaining releases are tested conditional on it.
anchor_c <- magmaan(syntax, dat, estimator = "ML", groups = "g",
                    group_equal = "loadings", group_partial = "f=~x5")
mi_c <- modification_indices_lrt(anchor_c, dat)
stopifnot(!("x5" %in% mi_c$rhs), identical(attr(mi_c, "conditioned_on"), "f=~x5"),
          mi_c$rhs[1] == "x3")

cat("conditional (x5 freed) localization still flags x3; x5 excluded: ok\n")
