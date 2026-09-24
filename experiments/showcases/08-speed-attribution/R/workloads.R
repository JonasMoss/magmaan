workloads <- c('fit', 'wald_nt', 'sb_wald', 'peba4_wald', 'sb_peba4_wald')
methods_for <- function(w) c(if (w %in% c('sb_wald','sb_peba4_wald')) 'sb',
                             if (w %in% c('peba4_wald','sb_peba4_wald')) 'peba4')
robust_for <- function(w) !w %in% c('fit','wald_nt')
row_key <- function(p) paste(p$group, p$lhs, p$op, p$rhs, sep=':')
lav_options <- function(w) list(estimator='ML', meanstructure=FALSE,
  fixed.x=FALSE, information='expected', baseline=FALSE,
  h1=robust_for(w), se=if(w=='fit') 'none' else if(w=='wald_nt') 'standard' else 'robust.sem',
  test=if(w %in% c('fit','wald_nt')) 'none' else
    c(if('sb' %in% methods_for(w)) 'satorra.bentler',
      if('peba4' %in% methods_for(w)) 'peba4_ml'))
lav_raw <- function(case,w) do.call(lavaan::sem,c(list(model=case$syntax,data=case$data),lav_options(w)))
mag_raw <- function(case) magmaan::magmaan(case$syntax,case$data,estimator='ML',
  meanstructure=FALSE,fixed_x=FALSE,auto_cov_y=TRUE)

# Single-group, unconstrained complete-data pilot only. Native parameter ordering
# is retained for covariance alignment; fixed rows are excluded from intervals.
extract_mag <- function(f,w, data, details=FALSE) {
  p <- f$partable; ii <- which(p$free>0L); ii <- ii[order(p$free[ii])]
  out <- list(theta=setNames(f$theta,row_key(p)[ii]), objective=f$fmin,
              covariance=NULL,intervals=NULL,tests=numeric(),spectrum=numeric(),fit=f)
  if (w=='fit') return(out)
  g <- magmaan::prepare_inference(f,data)
  V <- magmaan::inference_covariance(g,robust_for(w))
  out$covariance <- V
  out$intervals <- cbind(f$theta-qnorm(.975)*sqrt(diag(V)),f$theta+qnorm(.975)*sqrt(diag(V)))
  methods <- methods_for(w)
  if(length(methods)) {
    q <- magmaan::inference_quadratic(g,'lr')
    # The trace-only reference avoids an eigensolve for SB-only.
    ref <- if ('peba4' %in% methods) magmaan::score_spectrum(q) else
      getFromNamespace('.score_object','magmaan')(
        getFromNamespace('ntml_reference_impl','magmaan')(q$native,FALSE),
        class='magmaan_quadratic_reference')
    z <- magmaan::calibrate_quadratic(ref,methods)
    out$tests <- c(base=q$statistic,df=q$df,setNames(z$p_value,z$method))
    if('sb' %in% methods) {
      scale <- if('peba4' %in% methods) mean(ref$eigenvalues) else ref$mean_scale
      out$tests <- c(out$tests,sb_scale=scale,sb_stat=q$statistic/scale)
    }
    if('peba4' %in% methods) out$spectrum <- sort(ref$eigenvalues)
  }
  out
}
extract_lav <- function(f,w,V=NULL,tests=NULL) {
  p <- f@ParTable; ii <- which(p$free>0L); ii <- ii[order(p$free[ii])]
  th <- p$est[ii]
  out <- list(theta=setNames(th,row_key(p)[ii]),objective=f@optim$fx,
              covariance=NULL,intervals=NULL,tests=numeric(),spectrum=numeric(),fit=f)
  if(w=='fit') return(out)
  if(is.null(V)) V <- lavaan::vcov(f)
  out$covariance <- V
  out$intervals <- cbind(th-qnorm(.975)*sqrt(diag(V)),th+qnorm(.975)*sqrt(diag(V)))
  if(is.null(tests)) tests <- f@test
  methods <- methods_for(w)
  if(length(methods)) {
    out$tests <- c(base=tests$standard$stat,df=tests$standard$df,
      if('sb' %in% methods) c(sb=tests$satorra.bentler$pvalue),
      if('peba4' %in% methods) c(peba4=tests$peba4_ml$pvalue))
    if('sb' %in% methods) out$tests<-c(out$tests,
      sb_scale=tests$satorra.bentler$scaling.factor,sb_stat=tests$satorra.bentler$stat)
    if('peba4' %in% methods) out$spectrum <- sort(tests$peba4_ml$UGamma.eigenvalues)
  }
  out
}

# Pin internals explicitly. Fail rather than silently timing a different route.
check_lavaan_adapter <- function() {
  if(as.character(packageVersion('lavaan'))!='0.7.2')
    stop('This adapter is validated against lavaan 0.7-2 only.')
  ns <- asNamespace('lavaan')
  required <- list(lav_model_vcov=c('lavmodel','lavsamplestats','lavoptions'),
    lav_model_test=c('x','vcov_1','lavloglik'),
    lav_samplestats_from_data=c('lavdata','lavoptions'),
    lav_step06_h1=c('lavoptions','lavsamplestats'),
    lav_model_objective=c('lavmodel','glist'),lav_model_grad=c('lavmodel','glist'))
  for(n in names(required)) stopifnot(all(required[[n]] %in% names(formals(get(n,ns)))))
}
li <- function(name) get(name,asNamespace('lavaan'))

# Begins with a fit containing no vcov/tests. Rebuild sample ingredients when
# inference requests them: do not donate an already computed empirical Gamma.
lav_post <- function(f,w,options) {
  ss <- if(robust_for(w)) li('lav_samplestats_from_data')(f@Data,options) else f@SampleStats
  h1 <- li('lav_step06_h1')(lavoptions=options,lavsamplestats=ss,
                          lavdata=f@Data,lavpartable=f@ParTable)
  V <- li('lav_model_vcov')(lavmodel=f@Model,lavsamplestats=ss,lavoptions=options,
    lavdata=f@Data,lavpartable=f@ParTable,lavcache=f@Cache,lavimplied=f@implied,lavh1=h1)
  tt <- NULL
  if(length(methods_for(w))) {
    x <- f@optim$x; fx <- f@optim$fx
    attr(fx,'fx.group') <- f@optim$fx.group; attr(x,'fx') <- fx
    ll <- li('lav_step12_loglik')(lavoptions=options,lavdata=f@Data,lavsamplestats=ss,
      lavh1=h1,lavimplied=f@implied,lavmodel=f@Model,lavcache=f@Cache)
    tt <- li('lav_model_test')(lavmodel=f@Model,lavpartable=f@ParTable,
      lavsamplestats=ss,lavimplied=f@implied,lavh1=h1,lavoptions=options,
      x=x,vcov_1=V,lavcache=f@Cache,lavdata=f@Data,lavloglik=ll)
  }
  extract_lav(f,w,V,tt)
}

# Fit-only preparation for every workload. Robust data preparation is charged
# in post-fit work rather than hidden in a workload-specific unfitted object.
lav_prepared_fit <- function(u) {
  o <- u@Options; o$do.fit <- TRUE
  lavaan::lavaan(slot_options=o,slot_par_table=u@ParTable,
    slot_sample_stats=u@SampleStats,slot_data=u@Data,slot_model=u@Model,slot_cache=u@Cache)
}

compare_outputs <- function(a,b,case,boundary,workload,comparison='cross_engine') {
  rows <- list()
  check <- function(metric,x,y,atol,rtol) {
    same <- length(x)==length(y) && length(x)>0L && all(is.finite(c(x,y)))
    delta <- if(same) max(abs(x-y)) else Inf
    scaled <- if(same) max(abs(x-y)/(atol+rtol*abs(y))) else Inf
    rows[[length(rows)+1L]] <<- data.frame(case=case,boundary=boundary,workload=workload,
      comparison=comparison,metric=metric,max_abs=delta,atol=atol,rtol=rtol,passed=scaled<=1)
  }
  j <- match(names(a$theta),names(b$theta))
  check('parameter_keys',as.numeric(!anyNA(j) && setequal(names(a$theta),names(b$theta))),1,0,1e-12)
  check('estimates',a$theta,b$theta[j],1e-4,1e-5)
  check('half_discrepancy',a$objective,b$objective,1e-8,1e-6)
  if(workload!='fit') {
    check('covariance',a$covariance,b$covariance[j,j],1e-6,1e-4)
    check('wald_endpoints',a$intervals,b$intervals[j,,drop=FALSE],1e-4,1e-5)
  }
  if(length(a$tests)) {
    check('base_statistic',a$tests['base'],b$tests['base'],1e-5,1e-6)
    check('df',a$tests['df'],b$tests['df'],0,1e-12)
    if('sb' %in% methods_for(workload)) for(metric in c('sb_scale','sb_stat'))
      check(metric,a$tests[metric],b$tests[metric],1e-5,1e-5)
    for(method in methods_for(workload))
      check(paste0(method,'_log_p'),log(a$tests[method]),log(b$tests[method]),1e-3,0)
  }
  if(length(a$spectrum)) check('spectrum',a$spectrum,b$spectrum,1e-5,1e-5)
  do.call(rbind,rows)
}
