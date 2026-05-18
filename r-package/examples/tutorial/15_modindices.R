## lavaan tutorial — Modification indices   https://lavaan.ugent.be/tutorial/modindices.html
##
## KNOWN GAP. The magmaan C++ core and `magmaan::api` compute modification
## indices / score tests (with EPC), and the C++ suite gates them against
## lavaan — but the R package's `magmaan_core` does not yet bind them. This
## script documents the gap: it fits the model magmaan-side (the prerequisite)
## and shows lavaan's modindices() as the reference output. See
## docs/lavaan_tutorial_parity.md (section 15) and docs/todo.md section 4.

suppressMessages({ library(magmaan); library(lavaan) })

ok <- function(cond, what) {
  cat(sprintf("  %-44s %s\n", what, if (isTRUE(cond)) "ok" else "MISMATCH"))
  stopifnot(isTRUE(cond))
}

model <- "
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"
hs  <- HolzingerSwineford1939
fit <- magmaan(model, hs, estimator = "ML", se = "none", test = "none")

cat("=== modification indices ===\n")
ok(fit$converged, "magmaan fits the model (prerequisite for modindices)")

has_mi <- exists("modification_indices", envir = magmaan_core) ||
          "modification_indices" %in% ls(magmaan_core)
cat(sprintf("  R binding for modification indices present: %s\n",
            if (has_mi) "yes" else "no (C++ api only — tracked gap)"))

cat("  lavaan modindices() reference (top 3 by MI):\n")
mi <- modindices(cfa(model, data = hs), sort = TRUE)
print(utils::head(mi[, c("lhs", "op", "rhs", "mi", "epc")], 3), row.names = FALSE)

cat("modification indices: ok (R-binding gap documented)\n")
