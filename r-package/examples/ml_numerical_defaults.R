suppressPackageStartupMessages(library(magmaanlab))
set.seed(20260922)
f <- rnorm(300)
x <- as.data.frame(sapply(c(1, .8, 1.2, .7), function(a) a*f+rnorm(300)))
names(x) <- paste0('x', 1:4)
m <- model_spec('f =~ x1 + x2 + x3 + x4')
s <- df_to_data(x, m, scaling='n')
a <- magmaan_core$fit_ml(m,s)
b <- magmaan_core$fit_ml(m,s, control=list(
  nlopt=list(ftol_rel=1e-12,xtol_rel=1e-10,max_eval=5000)))
stopifnot(a$ml_sample_scaling, identical(a$theta,b$theta),
          identical(a$ml_start_policy,'transported-std-lv-fabin'))
legacy <- magmaan_core$fit_ml(m,s,control=list(
  start='fabin3',ml_sample_scaling=FALSE,ftol=1e-12,gtol=1e-10,max_iter=5000))
stopifnot(!legacy$ml_sample_scaling,identical(legacy$ml_start_policy,'fabin3'),
          max(abs(a$theta-legacy$theta)) < 1e-4)
# Legacy fields override profile values, explicit backend fields override both.
c <- magmaan_core$fit_ml(m,s,control=list(ftol=.1,gtol=.1,max_iter=2,
  nlopt=list(ftol_rel=1e-12,xtol_rel=1e-10,max_eval=5000)))
stopifnot(identical(a$theta,c$theta))
limited <- try(magmaan_core$fit_ml(m,s,control=list(max_iter=1L)),silent=TRUE)
stopifnot(inherits(limited,'try-error') || limited$f_evals <= 1L)
p <- frontier_fit_ml_psd(m,s)
stopifnot(identical(p$psd_preconditioning,'diagonal'))
u <- frontier_fit_ml_psd(m,s,preconditioning='none')
stopifnot(identical(u$psd_preconditioning,'none'),max(abs(p$theta-u$theta))<1e-4)
cat('ML numerical default checks passed.\n')

# Start construction, transport and optimizer scaling are independent.
x0 <- magmaan_core$estimate_start_values(m$partable, s)
xd <- magmaan_core$estimate_start_values(m$partable, s, start="default")
stopifnot(identical(x0, xd), attr(x0, "start_transport") == "std-lv-to-marker",
          attr(x0, "start_fallback_reason") == "none")
d <- magmaan_core$fit_ml(m, s, control=list(start="default"))
stopifnot(identical(a$theta, d$theta), d$ml_start_fallback_reason == "none")
xs <- magmaan_core$estimate_start_values(m$partable, s, start="simple", transport="auto")
xn <- magmaan_core$estimate_start_values(m$partable, s, start="simple", transport="native")
stopifnot(attr(xs, "start_transport") == "std-lv-to-marker",
          attr(xn, "start_transport") == "native", max(abs(xs-xn)) > .01)
fixed <- model_spec('f =~ x1 + 2*x2 + x3 + x4')
xf <- magmaan_core$estimate_start_values(fixed$partable, s)
xfn <- magmaan_core$estimate_start_values(fixed$partable, s, start="fabin3")
stopifnot(identical(as.numeric(xf), as.numeric(xfn)),
          attr(xf, "start_fallback_reason") == "fixed-values-unsupported")
required <- try(magmaan_core$estimate_start_values(fixed$partable, s, transport="required"), silent=TRUE)
stopifnot(inherits(required, "try-error"))
fallback_fit <- magmaan_core$fit_ml(fixed, s, optimizer="nlopt-slsqp")
stopifnot(fallback_fit$ml_start_fallback_reason == "fixed-values-unsupported",
          fallback_fit$ml_sample_scaling)
cat('Composable start policy checks passed.\n')
pd <- frontier_fit_ml_psd(m, s, control=list(start="default"))
stopifnot(identical(p$theta, pd$theta), pd$ml_start_fallback_reason == "none")
fb <- frontier_fit_ml_psd_fallback(m, s, ordinary_control=list(start="default"))
stopifnot(fb$ordinary$ml_start_policy == a$ml_start_policy,
          fb$ordinary$ml_start_fallback_reason == "none")
