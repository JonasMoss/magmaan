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

fit <- fit_fiml(m, d, lbfgs = list(max_iter = 4000, ftol = 1e-12, gtol = 1e-8))
lav <- cfa(model, data = df, missing = "fiml", meanstructure = TRUE)
est <- match_est(fit$partable, parTable(lav))
stopifnot(max(abs(est[, "magmaan"] - est[, "lavaan"])) < 2e-4)
stopifnot(identical(fit$estimator, "FIML"))
stopifnot(isTRUE(fit$fiml))
stopifnot(inherits(fit$raw_data, "magmaan_fiml_data"))
stopifnot(identical(fit$raw_data$nobs[[1L]], nrow(df)))

mg <- df_to_fiml_data(df, model_spec(model, meanstructure = TRUE), group = "school")
stopifnot(length(mg$X) == 2L)
stopifnot(identical(mg$group_labels, levels(df$school)))
stopifnot(identical(names(mg$X), mg$group_labels))
stopifnot(any(!mg$mask[[1L]]) || any(!mg$mask[[2L]]))

path_model <- model_spec("x1 ~ x2 + x3", meanstructure = TRUE)
path_df <- df
path_df$x2[1L] <- NA_real_
path_d <- df_to_fiml_data(path_df, path_model)
err <- tryCatch(fit_fiml(path_model, path_d),
                error = function(e) conditionMessage(e))
stopifnot(grepl("fixed.x", err, fixed = TRUE))

cat("FIML df_to_fiml_data()/fit_fiml() workflow: ok\n")
