utils::data("HolzingerSwineford1939", package = "lavaan")
core <- magmaan::magmaan_core
df <- HolzingerSwineford1939
X <- as.matrix(df[paste0("x", 1:9)])

model <- "visual  =~ x1 + x2 + x3
          textual =~ x4 + x5 + x6
          speed   =~ x7 + x8 + x9"

fit <- magmaan::magmaan(model, df, estimator = "ML")
ss <- core$fit_sample_stats(fit)
info <- core$inference_information_expected(fit)
implied <- core$model_implied(fit)

close <- function(a, b, tol = 1e-8) isTRUE(all.equal(a, b, tolerance = tol))
same_list <- function(a, b, fields, tol = 1e-8) {
  all(vapply(fields, function(nm) close(a[[nm]], b[[nm]], tol), logical(1)))
}

vc_fit <- core$inference_vcov_fit(info, fit)
vc_parts <- core$inference_vcov_partable(info, fit$partable)
stopifnot(close(vc_parts, vc_fit))

se <- core$inference_se(vc_parts)
z_fit <- core$inference_z_test_fit(fit, se)
z_theta <- core$inference_z_test_theta(fit$theta, se)
stopifnot(same_list(z_theta, z_fit, c("z", "pvalue")))

R <- matrix(0, nrow = 1L, ncol = length(fit$theta))
R[1L, 1L] <- 1
wald_fit <- core$inference_wald_test_fit(fit, R, vc_parts)
wald_theta <- core$inference_wald_test_theta(fit$theta, R, vc_parts)
stopifnot(same_list(wald_theta, wald_fit, c("chi2", "df", "pvalue")))

rls_fit <- core$inference_rls_chi2_fit(fit, implied)
rls_sample <- core$inference_rls_chi2_sample(ss, implied)
stopifnot(same_list(rls_sample, rls_fit, "statistic"))

res <- core$measures_residuals(fit)
std_res <- core$measures_standardized_residuals(fit)
stopifnot(close(res$cov[[1]], ss$S[[1]] - implied$sigma[[1]]))
stopifnot(close(std_res$cov_raw[[1]], res$cov[[1]]))
stopifnot(is.finite(std_res$srmr))
res_friendly <- stats::residuals(fit)
std_res_friendly <- stats::residuals(fit, standardized = TRUE)
stopifnot(close(res_friendly$cov[[1]], res$cov[[1]]))
stopifnot(close(std_res_friendly$cov_raw[[1]], std_res$cov_raw[[1]]))

fs_reg <- core$measures_factor_scores(fit, X, method = "regression")
fs_bar <- core$measures_factor_scores(fit, X, method = "bartlett")
stopifnot(nrow(fs_reg$scores[[1]]) == nrow(X))
stopifnot(ncol(fs_reg$scores[[1]]) == 3L)
stopifnot(identical(dim(fs_bar$scores[[1]]), dim(fs_reg$scores[[1]])))
fs_friendly <- magmaan::factor_scores(fit, df, method = "regression")
stopifnot(close(fs_friendly$scores[[1]], fs_reg$scores[[1]]))

std_friendly <- magmaan::standardized(fit, vc_parts, type = "all")
stopifnot(length(std_friendly$theta) == length(fit$theta))
mi_friendly <- magmaan::modification_indices(fit)
stopifnot(is.data.frame(mi_friendly))

uf_fit <- core$robust_build_u_factor_fit(fit)
uf_parts <- core$robust_build_u_factor_parts(fit$partable, ss, fit$theta)
stopifnot(identical(uf_parts$kind, uf_fit$kind))
stopifnot(identical(uf_parts$df, uf_fit$df))
stopifnot(close(uf_parts$B, uf_fit$B))

rse_raw_fit <- core$robust_se_raw_fit(fit, X)
rse_raw_parts <- core$robust_se_raw_parts(fit$partable, ss, fit$theta, X)
stopifnot(same_list(rse_raw_parts, rse_raw_fit, c("vcov", "se"), tol = 1e-7))

Zc <- core$robust_casewise_contributions(fit$partable, X)
gamma_hat <- crossprod(Zc) / nrow(Zc)
rse_gamma <- core$robust_se_parts(fit$partable, ss, fit$theta, gamma_hat)
stopifnot(same_list(rse_gamma, rse_raw_parts, c("vcov", "se"), tol = 1e-7))

cat("post-fit primitive/adaptor workflow: ok\n")
