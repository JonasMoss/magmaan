library(magmaan)
library(lavaan)

data("HolzingerSwineford1939", package = "lavaan")
df <- HolzingerSwineford1939
X <- as.matrix(df[paste0("x", 1:9)])

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"

fit <- magmaan(model, df, estimator = "ML")
ss <- fit_sample_stats(fit)
info <- infer_information_expected(fit)
implied <- model_implied(fit)

close <- function(a, b, tol = 1e-8) isTRUE(all.equal(a, b, tolerance = tol))
same_list <- function(a, b, fields, tol = 1e-8) {
  all(vapply(fields, function(nm) close(a[[nm]], b[[nm]], tol), logical(1)))
}

vc_fit <- infer_vcov_fit(info, fit)
vc_parts <- infer_vcov_partable(info, fit$partable)
stopifnot(close(vc_parts, vc_fit))

se <- infer_se(vc_parts)
z_fit <- infer_z_test_fit(fit, se)
z_theta <- infer_z_test_theta(fit$theta, se)
stopifnot(same_list(z_theta, z_fit, c("z", "pvalue")))

R <- matrix(0, nrow = 1L, ncol = length(fit$theta))
R[1L, 1L] <- 1
wald_fit <- infer_wald_test_fit(fit, R, vc_parts)
wald_theta <- infer_wald_test_theta(fit$theta, R, vc_parts)
stopifnot(same_list(wald_theta, wald_fit, c("chi2", "df", "pvalue")))

rls_fit <- infer_rls_chi2_fit(fit, implied)
rls_sample <- infer_rls_chi2_sample(ss, implied)
stopifnot(same_list(rls_sample, rls_fit, "statistic"))

uf_fit <- infer_build_u_factor_fit(fit)
uf_parts <- infer_build_u_factor_parts(fit$partable, ss, fit$theta)
stopifnot(identical(uf_parts$kind, uf_fit$kind))
stopifnot(identical(uf_parts$df, uf_fit$df))
stopifnot(close(uf_parts$B, uf_fit$B))

rse_raw_fit <- infer_robust_se_raw_fit(fit, X)
rse_raw_parts <- infer_robust_se_raw_parts(fit$partable, ss, fit$theta, X)
stopifnot(same_list(rse_raw_parts, rse_raw_fit, c("vcov", "se"), tol = 1e-7))

Zc <- infer_casewise_contributions(fit$partable, X)
gamma_hat <- crossprod(Zc) / nrow(Zc)
rse_gamma <- infer_robust_se_parts(fit$partable, ss, fit$theta, gamma_hat)
stopifnot(same_list(rse_gamma, rse_raw_parts, c("vcov", "se"), tol = 1e-7))

cat("post-fit primitive/adaptor workflow: ok\n")
