core <- magmaan::magmaan_core

set.seed(20260515)
Sigma <- matrix(c(2.6, 1.7, 1.4,
                  1.7, 1.945, 1.19,
                  1.4, 1.19, 1.78), 3, 3, byrow = TRUE)
X <- matrix(rnorm(360 * 3), 360, 3) %*% chol(Sigma)
df <- as.data.frame(X)
names(df) <- c("x3", "x1", "x2")
df$school <- rep(c("Pasteur", "Grant-White"), each = 180)

model <- "f =~ x1 + x2 + x3"

fit_ml_high <- magmaan::magmaan(model, df, estimator = "ML", se = "none", test = "none")
fit_uls_high <- magmaan::magmaan(
  model, df, estimator = "ULS",
  control = list(max_iter = 2000, ftol = 1e-12, gtol = 1e-8)
)
fit_group_high <- magmaan::magmaan(model, df, estimator = "ML", groups = "school")
fit_group_spec_high <- magmaan::magmaan(magmaan::model_spec(model, group = "school"), df,
                                        estimator = "ML", groups = "school")

df_missing <- df
df_missing$x2[seq(1L, nrow(df_missing), by = 17L)] <- NA_real_
fit_fiml_high <- magmaan::magmaan(model, df_missing, estimator = "FIML")
fit_fiml_spec_high <- magmaan::magmaan(magmaan::model_spec(model), df_missing,
                                       estimator = "FIML")

stopifnot(inherits(fit_ml_high, "magmaan_fit"))
stopifnot(inherits(fit_ml_high$model, "magmaan_model_spec"))
stopifnot(identical(fit_ml_high$estimator, "ML"))
stopifnot(identical(fit_uls_high$estimator, "ULS"))
stopifnot(identical(fit_group_high$ngroups, 2L))
stopifnot(identical(fit_group_high$group_labels, c("Pasteur", "Grant-White")))
stopifnot(identical(fit_group_spec_high$ngroups, 2L))
stopifnot(identical(fit_fiml_high$estimator, "FIML"))
stopifnot(isTRUE(fit_fiml_high$fiml))
stopifnot(isTRUE(fit_fiml_high$options$model_options$meanstructure))
stopifnot(isTRUE(fit_fiml_spec_high$options$model_options$meanstructure))
stopifnot(any(as.character(fit_fiml_high$partable$op) == "~1"))
stopifnot(identical(fit_ml_high$syntax, model))
stopifnot(identical(fit_ml_high$options$se, "none"))
stopifnot(identical(fit_ml_high$options$test, "none"))
stopifnot(identical(fit_ml_high$options$parameterization, "delta"))
stopifnot(!"vcov" %in% names(fit_ml_high))
stopifnot(!"se" %in% names(fit_ml_high))
printed <- utils::capture.output(print(fit_ml_high))
stopifnot(any(grepl("estimate only", printed, fixed = TRUE)))

ord <- data.frame(
  x1 = ordered(cut(df$x1, c(-Inf, -0.5, 0.4, Inf), labels = FALSE)),
  x2 = ordered(cut(df$x2, c(-Inf, -0.4, 0.5, Inf), labels = FALSE)),
  x3 = ordered(cut(df$x3, c(-Inf, -0.6, 0.3, Inf), labels = FALSE))
)
fit_dwls_high <- magmaan::magmaan(
  model, ord, estimator = "DWLS", ordered = c("x1", "x2", "x3"),
  control = list(max_iter = 3000, ftol = 1e-12, gtol = 1e-8)
)
stopifnot(isTRUE(fit_dwls_high$ordinal))
stopifnot(identical(fit_dwls_high$estimator, "DWLS"))
stopifnot(identical(fit_dwls_high$ordered, c("x1", "x2", "x3")))

info <- core$inference_information_expected(fit_ml_high)
vc <- core$inference_vcov_partable(info, fit_ml_high$partable)
stopifnot(length(core$inference_se(vc)) == fit_ml_high$npar)

err <- tryCatch(magmaan::magmaan(model, df, estimator = "MLM"),
                error = function(e) conditionMessage(e))
stopifnot(grepl("estimate-only", err, fixed = TRUE))
err <- tryCatch(magmaan::magmaan(model, df, estimator = "ML", se = "standard"),
                error = function(e) conditionMessage(e))
stopifnot(grepl("explicit post-fit", err, fixed = TRUE))
err <- tryCatch(magmaan::magmaan(model, df, estimator = "ML", test = "standard"),
                error = function(e) conditionMessage(e))
stopifnot(grepl("explicit post-fit", err, fixed = TRUE))
err <- tryCatch(magmaan::magmaan(model, df_missing, estimator = "FIML",
                                 meanstructure = FALSE),
                error = function(e) conditionMessage(e))
stopifnot(grepl("requires a mean structure", err, fixed = TRUE))

cat("magmaan() high-level estimate-only workflow: ok\n")
