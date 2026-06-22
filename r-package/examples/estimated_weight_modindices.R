## Estimated-weight (complete-sandwich) robust modification indices.
##
## modification_indices_robust(..., estimated_weight = TRUE) routes the
## per-direction robust scaling through the complete Hall-Inoue infinitesimal-
## jackknife sandwich, which carries the data-dependent-weight IF(W-hat) meat
## term. lavaan never builds a per-parameter robust MI denominator (it scales MI
## only by the global Satorra-Bentler scalar), so this is a frontier extension.
## It applies to estimated second-stage weights (categorical DWLS/WLS, continuous
## GLS/WLS) and is leading-order only under misspecification.

suppressMessages(requireNamespace("lavaan"))
core <- magmaan::magmaan_core

## Misspecified ordinal data: one factor plus an omitted x1-x2 residual
## association, so the single-factor model leaves real misfit for the MI to find.
set.seed(20260619)
n <- 1200
eta <- rnorm(n); nuis <- rnorm(n)
mk <- function(j) {
  extra <- if (j <= 2) 0.40 * nuis else 0
  rsd   <- if (j <= 2) sqrt(0.35) else sqrt(0.51)
  y <- 0.70 * eta + extra + rsd * rnorm(n)
  ordered(1 + (y > -0.5) + (y > 0.45))
}
df <- data.frame(x1 = mk(1), x2 = mk(2), x3 = mk(3), x4 = mk(4))

model <- "f =~ x1 + x2 + x3 + x4"
ordered <- paste0("x", 1:4)
m <- magmaan::model_spec(model, ordered = ordered, parameterization = "delta")
d <- core$data_ordinal_stats_from_df(df, m)
fit <- core$fit_dwls_ordinal(m, d, control = list(max_iter = 4000, ftol = 1e-13))
stopifnot(isTRUE(fit$ordinal), identical(fit$estimator, "DWLS"), fit$converged)

## Fixed-weight robust MI (the existing Satorra-Bentler-style sandwich) ...
fixed <- magmaan::modification_indices_robust(fit, candidates = "all",
                                              estimated_weight = FALSE)
## ... vs the complete (estimated-weight) sandwich.
ew <- magmaan::modification_indices_robust(fit, candidates = "all",
                                           estimated_weight = TRUE)

stopifnot(nrow(fixed) == nrow(ew), nrow(ew) > 1L)
## The ordinary statistic is unchanged; only the robust denominator moves.
stopifnot(max(abs(fixed$mi - ew$mi)) < 1e-6)
shift <- max(abs(fixed$scaling.factor - ew$scaling.factor))
stopifnot(is.finite(shift), shift > 0.01)

cat(sprintf("max |c_fixed - c_estweight| = %.4f over %d candidates\n",
            shift, nrow(ew)))
cat("estimated-weight ordinal modification indices (top 4 by mi.scaled):\n")
print(head(ew[order(ew$mi.scaled, decreasing = TRUE),
              c("lhs", "op", "rhs", "mi", "scaling.factor", "mi.scaled")], 4),
      row.names = FALSE)

## ML carries no estimated second-stage weight: the flag is rejected, not ignored.
hs <- lavaan::HolzingerSwineford1939
fit_ml <- magmaan::magmaan("f =~ x1 + x2 + x3", hs, estimator = "ML",
                           se = "none", test = "none")
err <- tryCatch(
  magmaan::modification_indices_robust(fit_ml, data = hs,
                                       estimated_weight = TRUE),
  error = function(e) conditionMessage(e))
stopifnot(is.character(err), grepl("estimated_weight", err))

cat("estimated-weight modification indices: ok\n")
