suppressPackageStartupMessages(library(magmaan))

model <- model_spec('f =~ x1 + x2 + x3')
interior <- list(S = list(matrix(c(1, .5, .4, .5, 1, .32, .4, .32, 1), 3)),
                 nobs = 300L)
boundary <- list(S = list(matrix(c(1, .7, .7, .7, 1, .3, .7, .3, 1), 3)),
                 nobs = 300L)

# Ordinary success: no PSD attempt or change to the returned ordinary fit.
a <- frontier_fit_ml_psd_fallback(model, interior)
ordinary <- magmaan_core$fit_ml(model, interior)
stopifnot(a$converged, !a$fallback_used, !a$warm_start_used, is.null(a$psd),
          identical(a$fallback_reason, 'none'),
          identical(a$fit, a$ordinary$fit), identical(a$fit$theta, ordinary$theta))

# A converged improper ordinary result triggers a new constrained fit.
b <- frontier_fit_ml_psd_fallback(model, boundary)
stopifnot(b$converged, b$fallback_used, b$warm_start_used,
          b$ordinary$fit$converged, !b$ordinary$fit$diagnostics$admissibility$admissible,
          identical(b$fallback_reason, 'ordinary-inadmissible'),
          identical(b$fit, b$psd$fit), b$fit$diagnostics$admissibility$admissible,
          identical(b$fit$ml_start_policy, 'ordinary-estimates'),
          identical(b$fit$partable$free, ordinary$partable$free))

# Solver errors remain inspectable; PSD uses the original initializer.
c <- frontier_fit_ml_psd_fallback(model, interior,
    ordinary_control = list(max_iter = 1L))
stopifnot(c$converged, c$fallback_used, !c$warm_start_used,
          identical(c$fallback_reason, 'ordinary-error'),
          is.null(c$ordinary$fit), nzchar(c$ordinary$error$detail),
          is.null(c$psd$error))

d <- frontier_fit_ml_psd_fallback(model, interior,
    ordinary_control = list(max_iter = 1L), psd_control = list(max_iter = 1L))
stopifnot(!d$converged, is.null(d$fit),
          nzchar(d$ordinary$error$detail), nzchar(d$psd$error$detail))

# A returned estimate with an unidentified scale must not be selected.
unidentified <- model_spec('f =~ x1 + x2 + x3', auto_fix_first = FALSE)
e <- frontier_fit_ml_psd_fallback(unidentified, interior,
    ordinary_control = list(max_iter = 1L))
stopifnot(!e$converged, is.null(e$fit), !is.null(e$psd$fit),
          !isTRUE(e$psd$fit$converged), is.null(e$psd$error))

# Raw-data metadata remains available to downstream explicit post-fit calls.
set.seed(20260925)
x <- as.data.frame(matrix(rnorm(900), 300, 3) %*% chol(interior$S[[1]]))
names(x) <- paste0('x', 1:3)
raw <- frontier_fit_ml_psd_fallback(model, x)
stopifnot(raw$converged, !is.null(raw$fit$raw_data),
          identical(raw$fit$raw_data, raw$ordinary$fit$raw_data))
cat('Ordinary-first PSD fallback checks passed.\n')
