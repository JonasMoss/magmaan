utils::data("HolzingerSwineford1939", package = "lavaan")

core <- magmaan::magmaan_core
df <- HolzingerSwineford1939
X <- as.matrix(df[paste0("x", 1:9)])

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"

fit_ml <- magmaan::magmaan(model, df, estimator = "ML")
rbm_ml <- core$frontier_rbm(fit_ml, X, method = "explicit")

fit_uls <- magmaan::magmaan(model, df, estimator = "ULS")
rbm_uls <- core$frontier_rbm(fit_uls, X, method = "explicit")

stopifnot(is.list(rbm_ml$rbm))
stopifnot(is.list(rbm_uls$rbm))
stopifnot(is.finite(rbm_ml$rbm$trace_term))
stopifnot(is.finite(rbm_uls$rbm$trace_term))
stopifnot(length(rbm_ml$theta) == length(fit_ml$theta))
stopifnot(length(rbm_uls$theta) == length(fit_uls$theta))

cat("frontier RBM estimator dispatch: ok\n")
