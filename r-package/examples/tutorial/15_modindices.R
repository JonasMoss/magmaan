## lavaan tutorial — Modification indices   https://lavaan.ugent.be/tutorial/modindices.html
##
## The C++ core and `magmaan::api` compute modification indices / score tests
## with EPCs, and the R scaffold exposes them through explicit post-fit
## primitives.

suppressMessages(requireNamespace("lavaan"))
core <- magmaan::magmaan_core

ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}

model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"
utils::data("HolzingerSwineford1939", package = "lavaan")
hs <- HolzingerSwineford1939
fit <- magmaan::magmaan(model, hs, estimator = "ML", se = "none", test = "none")

cat("=== modification indices ===\n")
ok(fit$converged, "magmaan fits the model (prerequisite for modindices)")

mmi <- core$inference_modification_indices(fit, candidates = "all")
ok(is.data.frame(mmi) && nrow(mmi) > 0L, "magmaan_core modification indices")

cat("  lavaan modindices() reference (top 3 by MI):\n")
mi <- lavaan::modindices(lavaan::cfa(model, data = hs), sort = TRUE)
print(utils::head(mi[, c("lhs", "op", "rhs", "mi", "epc")], 3), row.names = FALSE)
cat("  magmaan_core$inference_modification_indices() (top 3 by MI):\n")
print(utils::head(mmi[order(mmi$mi, decreasing = TRUE),
                      c("lhs", "op", "rhs", "mi", "epc")], 3),
      row.names = FALSE)

cat("modification indices: ok\n")
