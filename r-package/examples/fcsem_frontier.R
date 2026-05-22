library(magmaan)

set.seed(20260522)
n <- 300L
z <- rnorm(n)
df <- data.frame(
  x1 = z + rnorm(n, sd = 0.5),
  x2 = 0.8 * z + rnorm(n, sd = 0.6),
  x3 = -0.7 * z + rnorm(n, sd = 0.7),
  x4 = 0.6 * z + rnorm(n, sd = 0.8),
  x5 = -0.4 * z + rnorm(n, sd = 0.9)
)

model <- "
  C <~ x1 + x2 + x3
  x4 ~ C
  x5 ~ C
"

spec <- fcsem_model_spec(model)
dat <- df_to_fcsem_data(df, spec)
fit <- magmaan_fcsem(spec, dat, control = list(max_iter = 4000L))
se <- fcsem_standard_errors(fit)
fm <- fcsem_fit_measures(fit)
std <- fcsem_standardized_rows(fit, se$vcov)

stopifnot(inherits(spec, "magmaan_fcsem_model_spec"))
stopifnot(inherits(dat, "magmaan_fcsem_data"))
stopifnot(inherits(fit, "magmaan_fcsem_fit"))
stopifnot(isTRUE(fit$converged))
stopifnot(identical(fit$estimator, "FCSEM-ML"))
stopifnot(identical(fit$ngroups, 1L))
stopifnot(identical(fit$ov_names, c("x1", "x2", "x3", "x4", "x5")))
stopifnot(all(dim(se$vcov) == length(fit$theta)))
stopifnot(is.finite(fm$chisq), is.finite(fm$srmr))
stopifnot(nrow(std) > 0L)
stopifnot(all(std$row %in% seq_len(nrow(fit$partable))))
stopifnot(all(c("std.lv", "std.all") %in% names(std)))
stopifnot("frontier_fit_ml_fcsem" %in% attr(magmaan_core, "groups")$frontier)

cat("native FC-SEM R frontier workflow: ok\n")
