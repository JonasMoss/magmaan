#!/usr/bin/env Rscript
# Frozen fitting components; synthetic ordinal CFA, installed lavaan only.
stopifnot(as.character(utils::packageVersion("lavaan")) == "0.7.2")
stopifnot(gsub("-", ".", trimws(readLines("cpp/tests/fixtures/lavaan_version.txt"))) == "0.7.2")
.fitting_attempts <- list()
trace("lav_model_est", where=asNamespace("lavaan"), print=FALSE, exit=quote({
  out <- returnValue()
  .GlobalEnv$.fitting_attempts[[length(.GlobalEnv$.fitting_attempts)+1L]] <- list(
    simple=start=="simple", standardized=lavoptions$optim.parscale!="none",
    # Failed nlminb calls omit attributes; pin their actual driven inputs.
    start=if(length(attr(out,"start"))) as.numeric(attr(out,"start")) else as.numeric(start_x),
    parameter_scale=if(length(attr(out,"parscale"))) as.numeric(attr(out,"parscale")) else as.numeric(parscale),
    port_scale=if(exists("scale_1",inherits=FALSE)) as.numeric(scale_1) else numeric(),
    gradient=as.numeric(attr(out,"dx")), theta=as.numeric(out),
    accepted=isTRUE(attr(out,"converged")), iterations=attr(out,"iterations"))
}))
set.seed(11072)
x <- outer(rnorm(400),c(1,.8,.6,.9)) + matrix(rnorm(1600,sd=.8),400,4)
d <- as.data.frame(apply(x,2,function(z) as.integer(cut(z,c(-Inf,-.5,.6,Inf)))))
names(d) <- paste0("x",1:4)
d$g <- rep(c("a","b"),c(230,170))
cases <- list()
for(p in c("delta","theta")) for(g in 1:2) {
  cases[[length(cases)+1L]] <- list(parameterization=p,groups=g)
  if(g==2) cases[[length(cases)+1L]] <- list(parameterization=p,groups=g,group_equal="loadings")
}
cases[[length(cases)+1L]] <- list(parameterization="theta",groups=2,group_equal=c("loadings","thresholds"))
cases[[length(cases)+1L]] <- list(parameterization="theta",groups=1,invalid_start=TRUE)
for(i in seq_along(cases)) {
  c <- cases[[i]]
  args <- list(model="f =~ x1+x2+x3+x4",data=d,ordered=paste0("x",1:4),
    estimator="WLSMV",parameterization=c$parameterization,se="none",test="none")
  if(c$groups==2) args$group <- "g"
  if(!is.null(c$group_equal)) args$group.equal <- c$group_equal
  if(isTRUE(c$invalid_start)) {
    initial <- do.call(lavaan::cfa,c(args,list(do.fit=FALSE)))
    args$start <- rep(0,max(lavaan::parTable(initial)$free))
    # A negative latent variance makes theta's implied response variances
    # negative. DWLS has no ML positive-definiteness preflight.
    args$start[ lavaan::parTable(initial)$free[lavaan::parTable(initial)$lhs=="f" &
      lavaan::parTable(initial)$op=="~~"] ] <- -10
  }
  .fitting_attempts <- list()
  lv <- suppressWarnings(do.call(lavaan::cfa,args))
  pt <- lavaan::parTable(lv)
  cases[[i]] <- c(c,list(model=args$model,partable=pt[,c("id","lhs","op","rhs","user","block","group","free","exo","ustart","label","plabel")],
    parameters=pt[pt$free>0,c("lhs","op","rhs","group","start","est")],
    R=lapply(lv@SampleStats@cov,unname),thresholds=lapply(lv@SampleStats@th,as.numeric),
    weight=lapply(lv@SampleStats@WLS.VD,as.numeric),n_obs=as.integer(unlist(lv@SampleStats@nobs)),
    nacov=lapply(lv@SampleStats@NACOV,unname),
    explicit_start=if(isTRUE(c$invalid_start)) args$start else numeric(),
    fmin=as.numeric(lv@optim$fx),converged=isTRUE(lv@optim$converged),
    basis=unname(lv@Model@eq.constraints.K),offset=as.numeric(lv@Model@eq.constraints.k0),attempts=.fitting_attempts))
}
untrace("lav_model_est",where=asNamespace("lavaan"))
jsonlite::write_json(list(source="regen_ordinal_fitting_options.R; seed 11072; synthetic three-category CFA",lavaan_version="0.7.2",cases=cases),
  "cpp/tests/fixtures/fitting/lavaan_ordinal_0_7_2.json",auto_unbox=TRUE,digits=17,pretty=TRUE,na="null")
