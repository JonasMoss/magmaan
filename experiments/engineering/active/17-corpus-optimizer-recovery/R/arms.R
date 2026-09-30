optimizer_arms <- function() list(
  lbfgs_default=list(optimizer="nlopt-lbfgs",control=list()),
  lbfgs_budget=list(optimizer="nlopt-lbfgs",control=list(nlopt=list(max_eval=20000L))),
  lbfgs_tight=list(optimizer="nlopt-lbfgs",control=list(nlopt=list(max_eval=5000L,ftol_rel=1e-14,xtol_rel=1e-12,tolg=1e-10))),
  lbfgs_memory50=list(optimizer="nlopt-lbfgs",control=list(nlopt=list(vector_storage=50L))),
  port_default=list(optimizer="port",control=list()),
  port_tight=list(optimizer="port",control=list(max_iter=5000L,port=list(max_eval=20000L,rel_f_tol=1e-14,x_tol=1e-10))),
  slsqp_default=list(optimizer="nlopt-slsqp",control=list()),
  slsqp_tight=list(optimizer="nlopt-slsqp",control=list(nlopt=list(max_eval=5000L,ftol_rel=1e-14,xtol_rel=1e-12))),
  lbfgs_slsqp=list(optimizer="nlopt-lbfgs-slsqp-fallback",control=list()))

one_fit <- function(case, estimator, arm, start) {
  control <- arm$control; control$start<-as.numeric(start)
  fitter <- if(estimator=="ML") magmaanlab::magmaan_core$fit_ml else magmaanlab::magmaan_core$fit_gls
  fitter(case$model,case$sample,optimizer=arm$optimizer,control=control)
}
fit_fields <- function(fit) list(
  returned=TRUE,converged=isTRUE(fit$converged),f=fit$fmin,
  status=fit$optimizer_status,stationary=isTRUE(fit$audit$stationary),
  gradient=fit$audit$grad_inf_norm,raw_gradient=fit$audit$raw_grad_inf_norm,
  geometry_checked=isTRUE(fit$diagnostics$geometric_stationarity$checked),
  geometry_l2=fit$diagnostics$geometric_stationarity$ambient_residual_l2,
  verdict=fit$verdict$status %||% "",
  verdict_reason=fit$verdict$reason %||% "",
  audit_rhs=fit$audit$stationarity_rhs,
  f_consistent=isTRUE(fit$audit$f_consistent),
  admissible=isTRUE(fit$diagnostics$admissibility$admissible),
  f_evals=fit$f_evals,g_evals=fit$g_evals,iterations=fit$iterations)
