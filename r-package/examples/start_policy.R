# Shared constructor/transport contract over continuous and missing-data fitters.
suppressPackageStartupMessages(library(magmaanlab))
set.seed(20260925)
x <- as.data.frame(matrix(rnorm(1200), 300, 4) + rnorm(300))
names(x) <- paste0('x', 1:4)
m <- model_spec('f =~ x1 + x2 + x3 + x4')
s <- df_to_data(x, m, scaling='n')
ctl <- list(start='fabin2', start_transport='auto', max_iter=3000)
expected <- magmaan_core$estimate_start_values(m$partable, s, start='fabin2', transport='auto')
check <- function(fit, expected_theta = expected) {
  stopifnot(identical(fit$start$method, 'fabin2'),
            identical(fit$start$requested_transport, 'auto'),
            identical(fit$start$transport, 'std-lv-to-marker'),
            identical(fit$start$fallback_reason, 'none'),
            max(abs(fit$start$theta - expected_theta)) < 1e-12)
  invisible(fit)
}
fitters <- list(
  ml=function(control) magmaan_core$fit_ml(m,s,control=control),
  psd=function(control) frontier_fit_ml_psd(m,s,control=control),
  penalized=function(control) frontier_fit_ml_multiinfo(m,s,eta=1.1,control=control),
  uls=function(control) magmaan_core$fit_uls(m,s,control=control),
  gls=function(control) magmaan_core$fit_gls(m,s,control=control),
  wls=function(control) magmaan_core$fit_wls(m,s,diag(10),control=control),
  uls_snlls=function(control) magmaan_core$fit_uls_snlls(m,s,control=control),
  gls_snlls=function(control) magmaan_core$fit_gls_snlls(m,s,control=control),
  wls_snlls=function(control) magmaan_core$fit_wls_snlls(m,s,diag(10),control=control))
for (name in names(fitters)) {
  f <- fitters[[name]]
  fit <- check(f(ctl))
  default <- f(list(max_iter=3000))
  expected_default <- magmaan_core$estimate_start_values(m$partable,s,
      start=if (name %in% c('ml','psd','penalized')) 'scaled-fabin' else 'fabin3')
  stopifnot(max(abs(default$start$theta - expected_default)) < 1e-12)
  explicit <- f(list(start=fit$theta, max_iter=3000))
  stopifnot(explicit$start$method == 'explicit',
            identical(as.numeric(explicit$start$theta), as.numeric(fit$theta)))
  stopifnot(inherits(try(f(list(start=NaN)), silent=TRUE), 'try-error'))
}
# Marker-only constructors cannot be silently replaced by transported simple.
for (method in c('guttman', 'bentler1982', 'jamesstein')) {
  a <- magmaan_core$estimate_start_values(m$partable,s,start=method,transport='auto')
  b <- magmaan_core$estimate_start_values(m$partable,s,start=method,transport='native')
  stopifnot(identical(as.numeric(a),as.numeric(b)),
            attr(a,'start_fallback_reason') == 'constructor-requires-marker',
            inherits(try(magmaan_core$estimate_start_values(m$partable,s,
              start=method,transport='required'),silent=TRUE),'try-error'))
}
reject <- try(magmaan_core$fit_uls(m,s,
    control=list(start='guttman',start_transport='required')),silent=TRUE)
stopifnot(inherits(reject,'try-error'))
g <- magmaan_core$fit_gls(m,s,control=list(start='guttman',start_transport='auto'))
stopifnot(g$start$method == 'guttman',
          g$start$fallback_reason == 'constructor-requires-marker')
# Missing-data start summaries differ from complete-data summaries. Compare the
# direct, PSD and penalized paths on the same missingness pattern instead.
x[seq(3,300,7),2] <- NA
mf <- model_spec('f =~ x1 + x2 + x3 + x4', meanstructure=TRUE)
d <- df_to_fiml_data(x,mf)
a <- magmaan_core$fit_fiml(mf,d,control=ctl)
check(a,a$start$theta)
check(frontier_fit_fiml_psd(mf,d,control=ctl),a$start$theta)
check(frontier_fit_fiml_multiinfo(mf,d,eta=1.1,control=ctl),a$start$theta)
# Prepared entry points use the same policy and evidence schema.
pm <- prepare_model('f =~ x1 + x2 + x3 + x4',meanstructure=TRUE)
pd <- prepare_data(pm,x,'raw')
check(estimate(pm,pd,control=ctl),a$start$theta)
cat('Uniform continuous/FIML start policy checks passed.\n')
