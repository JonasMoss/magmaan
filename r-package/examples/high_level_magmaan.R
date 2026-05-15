library(magmaan)

set.seed(20260515)
Sigma <- matrix(c(2.6, 1.7, 1.4,
                  1.7, 1.945, 1.19,
                  1.4, 1.19, 1.78), 3, 3, byrow = TRUE)
X <- matrix(rnorm(360 * 3), 360, 3) %*% chol(Sigma)
df <- as.data.frame(X)
names(df) <- c("x3", "x1", "x2")
df$school <- rep(c("Pasteur", "Grant-White"), each = 180)

model <- "f =~ x1 + x2 + x3"

fit_ml_high <- magmaan(model, df, estimator = "ML", se = "none", test = "none")
fit_uls_high <- magmaan(
  model, df, estimator = "ULS",
  lbfgsb = list(max_iter = 2000, ftol = 1e-12, gtol = 1e-8)
)
fit_group_high <- magmaan(model, df, estimator = "ML", groups = "school")
fit_group_spec_high <- magmaan(model_spec(model, group = "school"), df,
                               estimator = "ML", groups = "school")

df_missing <- df
df_missing$x2[seq(1L, nrow(df_missing), by = 17L)] <- NA_real_
fit_fiml_high <- magmaan(model, df_missing, estimator = "FIML",
                         meanstructure = TRUE)

stopifnot(identical(fit_ml_high$estimator, "ML"))
stopifnot(identical(fit_uls_high$estimator, "ULS"))
stopifnot(identical(fit_group_high$ngroups, 2L))
stopifnot(identical(fit_group_high$group_labels, c("Pasteur", "Grant-White")))
stopifnot(identical(fit_group_spec_high$ngroups, 2L))
stopifnot(identical(fit_fiml_high$estimator, "FIML"))
stopifnot(isTRUE(fit_fiml_high$fiml))
stopifnot(!"vcov" %in% names(fit_ml_high))
stopifnot(!"se" %in% names(fit_ml_high))

ord <- data.frame(
  x1 = ordered(cut(df$x1, c(-Inf, -0.5, 0.4, Inf), labels = FALSE)),
  x2 = ordered(cut(df$x2, c(-Inf, -0.4, 0.5, Inf), labels = FALSE)),
  x3 = ordered(cut(df$x3, c(-Inf, -0.6, 0.3, Inf), labels = FALSE))
)
fit_dwls_high <- magmaan(
  model, ord, estimator = "DWLS", ordered = c("x1", "x2", "x3"),
  lbfgsb = list(max_iter = 3000, ftol = 1e-12, gtol = 1e-8)
)
stopifnot(isTRUE(fit_dwls_high$ordinal))
stopifnot(identical(fit_dwls_high$estimator, "DWLS"))

err <- tryCatch(magmaan(model, df, estimator = "MLM"),
                error = function(e) conditionMessage(e))
stopifnot(grepl("estimate-only", err, fixed = TRUE))
err <- tryCatch(magmaan(model, df, estimator = "ML", se = "standard"),
                error = function(e) conditionMessage(e))
stopifnot(grepl("explicit post-fit", err, fixed = TRUE))
err <- tryCatch(magmaan(model, df, estimator = "ML", test = "standard"),
                error = function(e) conditionMessage(e))
stopifnot(grepl("explicit post-fit", err, fixed = TRUE))

cat("magmaan() high-level estimate-only workflow: ok\n")
