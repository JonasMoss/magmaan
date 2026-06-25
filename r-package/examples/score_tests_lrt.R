## LRT (refit) equality-release score tests for measurement invariance.
##
## score_tests_lrt(fit, data) is the refit counterpart of score_tests() /
## lavaan lavTestScore(): for each cross-group tied parameter family in a
## constrained group.equal anchor, it refits the released model (that family
## freed via group_partial) and runs the nested Satorra-2000 difference test,
## returning the robust reference-law family plus the exact refitted per-group
## estimates. Unlike the one-step score test, the refit isolates each release
## exactly even under structural misspecification elsewhere, and an existing
## group_partial baseline on the anchor gives conditional localization for free.
## See docs/research/notes/lrt_modification_indices.tex.

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
st <- score_tests_lrt(anchor, dat)

stopifnot(inherits(st, "magmaan_score_lrt"), nrow(st) == 5L)
stopifnot(st$rhs[1] == "x3",
          all(c("lrt", "p_unscaled", "p_scaled", "p_mixture") %in% names(st)),
          st$p_mixture[1] < 0.01)

## The release statistic matches a hand-built two-fit nested test exactly.
ov <- paste0("x", 1:6)
data_list <- lapply(c("A", "B"),
                    function(g) as.matrix(dat[dat$g == g, ov, drop = FALSE]))
rel <- magmaan(syntax, dat, estimator = "ML", groups = "g",
               group_equal = "loadings", group_partial = "f=~x3")
ref <- robust_nested_lrt(rel, anchor, data = data_list, method = "restriction_map")
stopifnot(abs(st$lrt[st$rhs == "x3"] - ref$T_diff) < 1e-7,
          abs(st$p_mixture[st$rhs == "x3"] - ref$p_mixture) < 1e-9)

## Exact EPC: the released x3 loading differs across groups (the non-invariance).
est_x3 <- attr(st, "released_estimates")[[1]]$est
stopifnot(length(est_x3) == 2L, abs(diff(est_x3)) > 0.3)

cat("LRT equality-release score tests (ML):\n")
print(st[, c("lhs", "op", "rhs", "df", "lrt", "p_mixture", "epc_range")],
      row.names = FALSE)

## Conditional localization: an anchor that already frees x5 (a group_partial
## baseline) carries that into every refit, so x5 drops out of the candidate set
## and the rest are tested conditional on it.
anchor_c <- magmaan(syntax, dat, estimator = "ML", groups = "g",
                    group_equal = "loadings", group_partial = "f=~x5")
st_c <- score_tests_lrt(anchor_c, dat)
stopifnot(!("x5" %in% st_c$rhs),
          identical(attr(st_c, "conditioned_on"), "f=~x5"),
          st_c$rhs[1] == "x3")

cat("conditional (x5 freed) localization still flags x3; x5 excluded: ok\n")
