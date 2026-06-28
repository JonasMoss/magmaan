library(magmaan)
library(lavaan)

match_est <- function(mag_pt, lav_pt) {
  key <- function(pt) paste(pt$group, pt$lhs, pt$op, pt$rhs, sep = "\r")
  mag_free <- mag_pt[mag_pt$free > 0L, ]
  lav_free <- lav_pt[lav_pt$free > 0L, ]
  m <- match(key(mag_free), key(lav_free))
  stopifnot(!anyNA(m))
  cbind(magmaan = mag_free$est, lavaan = lav_free$est[m])
}

hs <- lavaan::HolzingerSwineford1939
model <- "visual =~ x1 + x2 + x3"

df <- hs
df$x2[seq(7L, nrow(df), by = 11L)] <- NA_real_
df$x3[seq(5L, nrow(df), by = 13L)] <- NA_real_

m <- model_spec(model, meanstructure = TRUE)
d <- df_to_fiml_data(df, m)
stopifnot(inherits(d, "magmaan_fiml_data"))
stopifnot(anyNA(d$X[[1L]]))
stopifnot(!all(d$mask[[1L]]))
stopifnot(identical(colnames(d$X[[1L]]), d$ov_names[[1L]]))
stopifnot(identical(d$nobs[[1L]], nrow(df)))

fit <- magmaan_core$fit_fiml(m, d, control = list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8))
lav <- cfa(model, data = df, missing = "fiml", meanstructure = TRUE)
est <- match_est(fit$partable, parTable(lav))
stopifnot(max(abs(est[, "magmaan"] - est[, "lavaan"])) < 2e-4)
stopifnot(identical(fit$estimator, "FIML"))
stopifnot(isTRUE(fit$fiml))
stopifnot(inherits(fit$raw_data, "magmaan_fiml_data"))
stopifnot(identical(fit$raw_data$nobs[[1L]], nrow(df)))
stopifnot(typeof(fit$fiml_pack) == "externalptr")
stopifnot(typeof(fit$fiml_h1) == "externalptr")
stopifnot(inherits(fit$fiml_pack, "magmaan_fiml_pack"))
stopifnot(inherits(fit$fiml_h1, "magmaan_fiml_h1"))

fit_s3 <- magmaan(
  model, data = df, estimator = "FIML",
  control = list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8)
)
fiml_model <- magmaan_core$fiml_observed_vcov(fit_s3)
fiml_robust <- magmaan_core$estimate_fiml_robust_mlr(fit_s3)
V_model <- vcov(fit_s3, regime = "model")
V_robust <- vcov(fit_s3, regime = "robust")
stopifnot(is.matrix(V_model), is.matrix(V_robust))
stopifnot(nrow(V_model) == fit_s3$npar, ncol(V_model) == fit_s3$npar)
stopifnot(nrow(V_robust) == fit_s3$npar, ncol(V_robust) == fit_s3$npar)
stopifnot(max(abs(V_model - fiml_model$vcov)) < 1e-10)
stopifnot(max(abs(V_robust - fiml_robust$vcov)) < 1e-10)
stopifnot(all(is.finite(fiml_model$se)))
stopifnot(all(is.finite(fiml_robust$se)))
stopifnot(is.nan(fiml_robust$scaling_factor))

model_ml2s <- "
visual =~ x1 + x2 + x3
textual =~ x4 + x5 + x6
"
fit_fiml_ns <- magmaan(
  model_ml2s, data = df, estimator = "FIML",
  control = list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8)
)
fiml_model_ns <- magmaan_core$fiml_observed_vcov(fit_fiml_ns)
fiml_robust_ns <- magmaan_core$estimate_fiml_robust_mlr(fit_fiml_ns)
V_model_ns <- vcov(fit_fiml_ns, regime = "model")
V_robust_ns <- vcov(fit_fiml_ns, regime = "robust")
stopifnot(is.matrix(V_model_ns), is.matrix(V_robust_ns))
stopifnot(nrow(V_model_ns) == fit_fiml_ns$npar,
          ncol(V_model_ns) == fit_fiml_ns$npar)
stopifnot(nrow(V_robust_ns) == fit_fiml_ns$npar,
          ncol(V_robust_ns) == fit_fiml_ns$npar)
stopifnot(max(abs(V_model_ns - fiml_model_ns$vcov)) < 1e-10)
stopifnot(max(abs(V_robust_ns - fiml_robust_ns$vcov)) < 1e-10)
stopifnot(all(is.finite(fiml_model_ns$se)))
stopifnot(all(is.finite(fiml_robust_ns$se)))
fm_fiml_ns <- fit_measures(fit_fiml_ns, robust = TRUE)
lav_fiml_ns <- cfa(model_ml2s, data = df, missing = "fiml",
                   estimator = "MLR", meanstructure = TRUE)
fm_keys_fiml <- c("cfi.robust", "tli.robust", "rmsea.robust")
fm_diff_fiml <- unlist(fm_fiml_ns[fm_keys_fiml]) -
  as.numeric(fitMeasures(lav_fiml_ns)[fm_keys_fiml])
stopifnot(max(abs(fm_diff_fiml), na.rm = TRUE) < 5e-4)

fit_ml2s <- magmaan(
  model_ml2s, data = df, estimator = "ML2S",
  control = list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8)
)
stopifnot(identical(fit_ml2s$estimator, "ML2S"))
stopifnot(inherits(fit_ml2s, "magmaan_fit"))
stopifnot(inherits(fit_ml2s$raw_data, "magmaan_fiml_data"))
stopifnot(is.list(fit_ml2s$stage1), is.list(fit_ml2s$ml2s))
stopifnot(length(fit_ml2s$se) == fit_ml2s$npar)
stopifnot(all(is.finite(fit_ml2s$se)))
stopifnot(is.matrix(fit_ml2s$vcov))
stopifnot(nrow(fit_ml2s$vcov) == fit_ml2s$npar)
stopifnot(ncol(fit_ml2s$vcov) == fit_ml2s$npar)
stopifnot(is.finite(fit_ml2s$chisq))
stopifnot(is.finite(fit_ml2s$chisq_scaled))
stopifnot(is.finite(fit_ml2s$scaling_factor), fit_ml2s$scaling_factor > 0)
stopifnot(identical(fit_ml2s$df, fit_ml2s$ml2s$df))
fm_ml2s <- fit_measures(fit_ml2s, robust = fit_ml2s$ml2s)
lav_ml2s <- cfa(model_ml2s, data = df, missing = "robust.two.stage",
                meanstructure = TRUE)
fm_keys_ml2s <- c("baseline.chisq.scaled",
                  "baseline.chisq.scaling.factor",
                  "cfi.scaled", "tli.scaled", "rmsea.scaled",
                  "cfi.robust", "tli.robust", "rmsea.robust")
fm_diff_ml2s <- unlist(fm_ml2s[fm_keys_ml2s]) -
  as.numeric(fitMeasures(lav_ml2s)[fm_keys_ml2s])
stopifnot(max(abs(fm_diff_ml2s), na.rm = TRUE) < 5e-4)

model_eq <- "visual =~ x1 + a*x2 + a*x3"
m_eq <- model_spec(model_eq, meanstructure = TRUE)
fit_eq <- magmaan_core$fit_fiml(
  m_eq, df_to_fiml_data(df, m_eq),
  control = list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8)
)
nested_emp <- nestedTest(fit, fit_eq, method = "restriction_map")
stopifnot(inherits(nested_emp, "magmaan_nested_test"))
stopifnot(nested_emp$df_diff == 1L)
stopifnot(length(nested_emp$eigenvalues) == nested_emp$df_diff)
stopifnot(all(is.finite(nested_emp$eigenvalues)))
nested_nt <- nestedTest(fit, fit_eq, method = "restriction_map", gamma = "NT")
stopifnot(max(abs(nested_nt$eigenvalues - 1)) < 1e-6)

nested_sb2001 <- nestedTest(fit, fit_eq, method = "satorra.bentler.2001")
stopifnot(inherits(nested_sb2001, "magmaan_nested_test"))
stopifnot(nested_sb2001$df_diff == 1L)
stopifnot(is.finite(nested_sb2001$stat), is.finite(nested_sb2001$scale_c))
err_data_arg <- tryCatch(
  nestedTest(fit, fit_eq, data = df, method = "restriction_map"),
  error = function(e) conditionMessage(e)
)
stopifnot(grepl("fit_H1$raw_data", err_data_arg, fixed = TRUE))
fit_ml_complete <- magmaan_core$fit_ml(
  m, df_to_data(lavaan::HolzingerSwineford1939, m),
  control = list(max_iter = 1000)
)
err_mixed <- tryCatch(
  nestedTest(fit, fit_ml_complete, method = "restriction_map"),
  error = function(e) conditionMessage(e)
)
stopifnot(grepl("mixed FIML/complete-data", err_mixed, fixed = TRUE))

mg <- df_to_fiml_data(df, model_spec(model, meanstructure = TRUE), group = "school")
stopifnot(length(mg$X) == 2L)
# Group order follows lavaan: data-appearance order, not factor levels.
stopifnot(identical(mg$group_labels, unique(as.character(df$school))))
stopifnot(identical(names(mg$X), mg$group_labels))
stopifnot(any(!mg$mask[[1L]]) || any(!mg$mask[[2L]]))

path_model <- model_spec("x1 ~ x2 + x3", meanstructure = TRUE)
path_df <- df
path_df$x2[1L] <- NA_real_
path_d <- df_to_fiml_data(path_df, path_model)
err <- tryCatch(magmaan_core$fit_fiml(path_model, path_d),
                error = function(e) conditionMessage(e))
stopifnot(grepl("fixed.x", err, fixed = TRUE))

cat("FIML df_to_fiml_data()/fit_fiml() workflow: ok\n")
