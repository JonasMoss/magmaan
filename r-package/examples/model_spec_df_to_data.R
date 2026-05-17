library(magmaan)

set.seed(2026)
Sigma <- matrix(c(2.6, 1.7, 1.4,
                  1.7, 1.945, 1.19,
                  1.4, 1.19, 1.78), 3, 3, byrow = TRUE)
X <- matrix(rnorm(600 * 3), 600, 3) %*% chol(Sigma)
df <- as.data.frame(X)
names(df) <- c("x3", "x1", "x2")
df$school <- rep(c("Pasteur", "Grant-White"), each = 300)

m <- model_spec("f =~ x1 + x2 + x3")
d <- df_to_data(df, m)

manual <- magmaan_core$data_sample_stats_from_raw(list(as.matrix(df[c("x1", "x2", "x3")])))
stopifnot(max(abs(d$S[[1]] - manual$S[[1]])) < 1e-12)
stopifnot(max(abs(d$mean[[1]] - manual$mean[[1]])) < 1e-12)

fit_ml_ <- fit_ml(m, d)
fit_uls_ <- fit_uls(m, d, lbfgsb = list(max_iter = 2000, ftol = 1e-12, gtol = 1e-8))
fit_gls_ <- fit_gls(m, d, lbfgsb = list(max_iter = 2000, ftol = 1e-12, gtol = 1e-8))
fit_wls_ <- fit_wls(m, d, diag(6), lbfgsb = list(max_iter = 2000, ftol = 1e-12, gtol = 1e-8))

stopifnot(fit_ml_$estimator == "ML")
stopifnot(fit_uls_$estimator == "ULS")
stopifnot(fit_gls_$estimator == "GLS")
stopifnot(fit_wls_$estimator == "WLS")

dg <- df_to_data(df, m, group = "school")
stopifnot(length(dg$S) == 2L)
stopifnot(identical(dg$group_labels, c("Pasteur", "Grant-White")))

cat("model_spec()/df_to_data() helper workflow: ok\n")
