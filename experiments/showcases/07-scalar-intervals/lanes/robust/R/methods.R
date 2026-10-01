# API mapping (fixed before simulation): scores() supplies exact casewise log
# likelihood scores and their SUM; inference_information() supplies TOTAL H.
# project_scores(matrix adapter, center=FALSE) supplies nuisance elimination,
# scalar metric and uncentered meat. score_sandwich() supplies robust score;
# score_spectrum() supplies c=j/h for LR. parameter_covariance(H,crossprod(rows))
# is used at H1 for Wald and validated against policy_inference() for Wald-O.
# The landed frontier profile corrections instead center contributions at S.
# They are validation comparators, NOT aliases for the uncentered LR-E/LR-O.
# All SEM derivatives, projections and covariances remain library primitives.
syntax <- paste('f1 =~ x1 + b*x2 + x3 + x4',
                'f2 =~ x5 + x6 + x7 + x8', 'f1 ~~ rho*f2',sep='\n')
truth <- c(loading=.7,correlation=.4)
parameter_label <- c(loading='b',correlation='rho')
methods <- c('wald_O','score_E','lr_E','wald_E','score_O','lr_O',
             'wald_naive','score_naive','lr_naive')
primary <- c('wald_O','score_E','lr_E')
distributions <- c('normal','t10','heterogeneous_gamma')
critical <- qchisq(.95,1)
core <- magmaanlab::magmaan_core

population <- function() {
  L <- matrix(0,8,2); L[1:4,1] <- L[5:8,2] <- c(.8,.7,.9,.6)
  L %*% matrix(c(1,.4,.4,1),2) %*% t(L)+diag(1-rowSums(L^2))
}
generate <- function(n,distribution,seed) {
  set.seed(seed)
  if(distribution=='heterogeneous_gamma') {
    shape <- rep(c(2,.5),5)
    Z <- sapply(shape,function(k) (rgamma(n,shape=k)-k)/sqrt(k))
  } else {
    Z <- matrix(rnorm(n*10),n,10)
    if(distribution=='t10') Z <- Z*sqrt(8/rchisq(n,10))
  }
  F <- cbind(Z[,1],.4*Z[,1]+sqrt(.84)*Z[,2])
  l <- rep(c(.8,.7,.9,.6),2)
  X <- sweep(F[,rep(1:2,each=4)],2,l,'*')+sweep(Z[,3:10],2,sqrt(1-l*l),'*')
  colnames(X) <- paste0('x',1:8)
  unclass(scale(X,scale=FALSE))
}
index_of <- function(fit,target) {
  p <- fit$partable
  id <- p$free[p$label==parameter_label[[target]] & p$op %in% c('=~','~~')]
  if(length(id)!=1L || id<1L) stop('target mapping failed')
  id
}
check_fit <- function(fit,target=NULL,b=NULL) {
  if(!isTRUE(fit$converged)) stop('fit verdict: ',fit$verdict$status)
  if(!is.finite(fit$fmin) || any(!is.finite(fit$theta))) stop('nonfinite fit')
  p <- fit$partable
  anchors <- p$free[p$op=='=~' & p$rhs %in% c('x1','x5')]
  if(any(fit$theta[anchors]<=0)) stop('factor orientation changed')
  if(!is.null(target) && abs(fit$theta[index_of(fit,target)]-b)>1e-6)
    stop('constraint residual exceeds 1e-6')
  fit
}
fit_ml <- function(X,target=NULL,b=NULL,start=NULL) {
  model <- if(is.null(target)) syntax else
    paste(syntax,sprintf('%s == %.17g',parameter_label[[target]],b),sep='\n')
  errors <- character()
  for(attempt in 1:2) {
    control <- list(max_iter=2000L,ftol=1e-12,gtol=1e-8)
    if(!is.null(start)) control$start <- as.numeric(start)
    if(attempt==2) {control$start <- 'scaled-fabin';control$max_iter <- 5000L}
    ans <- tryCatch(suppressWarnings(magmaanlab::fit_model(model,as.data.frame(X),
      std_lv=TRUE,meanstructure=FALSE,optimizer='nlopt-slsqp',control=control,
      bounds=NULL,psd=FALSE)),error=identity)
    if(!inherits(ans,'error')) ans <- tryCatch(check_fit(ans,target,b),error=identity)
    if(!inherits(ans,'error')) {attr(ans,'attempts')<-attempt;return(ans)}
    errors <- c(errors,conditionMessage(ans))
  }
  stop(paste(errors,collapse='; fallback: '),call.=FALSE)
}
is_admissible <- function(fit) isTRUE(fit$diagnostics$admissibility$admissible)
checked_information <- function(context,kind) {
  H <- magmaanlab::inference_information(context,kind)
  if(any(!is.finite(H))) stop('nonfinite sensitivity')
  # No indefinite or nearly singular bread is repaired or replaced.
  tryCatch(chol(H),error=function(e) stop('nonpositive sensitivity',call.=FALSE))
  condition <- rcond(H)
  if(!is.finite(condition) || condition<1e-12) stop('singular sensitivity')
  attr(H,'study_rcond') <- condition
  H
}
project_parameter <- function(context,id,kind) {
  s <- magmaanlab::scores(context)
  H <- checked_information(context,kind)
  q <- ncol(s$rows)
  if(nrow(H)!=q) stop('full-model parameter dimension mismatch')
  basis <- diag(q)
  component <- magmaanlab::score_components_from_matrices(
    score=s$score,rows=s$rows,sensitivity=H,metric=H,
    nuisance=basis[,-id,drop=FALSE],directions=basis[,id,drop=FALSE])
  p <- magmaanlab::project_scores(component,retain_rows=FALSE,center=FALSE)
  if(p$df!=1L || p$meat[1,1]<=0 || p$metric[1,1]<=0) stop('invalid scalar geometry')
  scaled <- magmaanlab::score_sandwich(p)$statistic
  scale <- magmaanlab::score_spectrum(p)$mean_scale
  list(score=scaled,ordinary=p$statistic,scale=scale,
       rcond=attr(H,'study_rcond'),projected=p,H=H,rows=s$rows,total_score=s$score)
}
wald_covariance <- function(u,X,method) {
  context <- magmaanlab::prepare_inference(u,X)
  H <- checked_information(context,if(method=='wald_O') 'observed' else 'expected')
  meat <- if(method=='wald_naive') NULL else crossprod(magmaanlab::scores(context)$rows)
  magmaanlab::parameter_covariance(context,H,meat)
}
lr_statistic <- function(u,r,n) {
  value <- 2*n*(r$fmin-u$fmin)
  if(!is.finite(value) || value < -1e-6) stop('negative/nonfinite LR: ',value)
  max(0,value) # Roundoff only, within the explicit 1e-6 allowance.
}
make_profile <- function(u,X,target,method) {
  state <- new.env(parent=emptyenv())
  state$cache <- new.env(parent=emptyenv());state$log <- list();state$refits <- 0L
  state$pointwise <- NULL
  eval <- function(b,fresh=FALSE,purpose='search') {
    key <- sprintf('%.17g',b)
    if(!fresh && exists(key,state$cache,inherits=FALSE)) return(get(key,state$cache))
    time <- proc.time()[['elapsed']]
    r <- NULL;lr <- NA_real_
    value <- tryCatch({
      r <- tryCatch(fit_ml(X,target,b,u$theta),error=identity)
      state$refits <- state$refits+if(inherits(r,'error')) 2L else attr(r,'attempts')
      if(inherits(r,'error')) stop(conditionMessage(r),call.=FALSE)
      lr <- lr_statistic(u,r,nrow(X));scale<-1;rc<-NA_real_;raw_score<-NA_real_
      if(method=='lr_naive') statistic <- lr else {
        context <- magmaanlab::prepare_inference(r,X)
        g <- project_parameter(context,index_of(r,target),
                               if(method %in% c('score_O','lr_O')) 'observed' else 'expected')
        scale <- g$scale;rc<-g$rcond;raw_score<-g$ordinary
        statistic <- if(method=='score_naive') g$ordinary else if(grepl('^score',method)) g$score else lr/scale
      }
      if(!is.finite(statistic)||statistic<0) stop('invalid test statistic')
      list(statistic=statistic,lr=lr,ordinary_score=raw_score,scale=scale,rcond=rc,
           constraint=abs(r$theta[index_of(r,target)]-b),admissible=is_admissible(r))
    },error=identity)
    row <- data.frame(candidate=b,purpose=purpose,fresh=fresh,statistic=NA_real_,
      raw_lr=NA_real_,raw_score=NA_real_,scale=NA_real_,rcond=NA_real_,
      constraint_residual=NA_real_,converged=FALSE,admissible=NA,error='',seconds=proc.time()[['elapsed']]-time)
    if(!is.null(r) && !inherits(r,'error')) {
      row$converged<-isTRUE(r$converged);row$raw_lr<-lr
      row$constraint_residual<-abs(r$theta[index_of(r,target)]-b)
      row$admissible<-is_admissible(r)
    }
    if(inherits(value,'error')) row$error<-conditionMessage(value) else {
      row$statistic<-value$statistic;row$raw_lr<-value$lr;row$raw_score<-value$ordinary_score
      row$scale<-value$scale;row$rcond<-value$rcond;row$constraint_residual<-value$constraint
      row$admissible<-value$admissible;row$converged<-TRUE
    }
    state$log[[length(state$log)+1L]]<-row
    if(inherits(value,'error')) stop(conditionMessage(value),call.=FALSE)
    assign(key,value,state$cache);value
  }
  list(evaluate=eval,state=state)
}
invert_profile <- function(profile,estimate,step) {
  f <- function(b) profile$evaluate(b)$statistic-critical
  if(f(estimate)>0) stop('estimate rejected by its own test')
  endpoints <- numeric(2)
  for(side in 1:2) {
    delta<-step;direction<-c(-1,1)[side];outer<-estimate+direction*delta
    for(k in seq_len(20)) {
      if(f(outer)>0) break
      delta<-1.5*delta;outer<-estimate+direction*delta
    }
    if(f(outer)<=0) stop('unresolved bracket; not evidence of an unbounded interval')
    endpoints[side]<-uniroot(f,sort(c(estimate,outer)),tol=1e-7,maxiter=100L,check.conv=TRUE)$root
    test<-profile$evaluate(endpoints[side],fresh=TRUE,purpose='endpoint')$statistic
    if(abs(test-critical)>1e-4) stop('endpoint statistic tolerance failed')
  }
  width<-diff(endpoints)
  grid<-sort(unique(c(seq(endpoints[1]-.5*width,endpoints[2]+.5*width,length.out=9),estimate)))
  accept<-vapply(grid,function(b) profile$evaluate(b,purpose='audit')$statistic<=critical,logical(1))
  # Store every candidate. Never replace a disconnected set by its convex hull.
  if(sum(abs(diff(as.integer(accept))))!=2L || accept[1] || tail(accept,1))
    stop('disconnected or unresolved acceptance set on wider grid')
  list(lower=endpoints[1],upper=endpoints[2],audit_lower=min(grid),audit_upper=max(grid))
}
run_replicate <- function(n,distribution,replicate,seed_base) {
  cell <- match(distribution,distributions)
  seed <- as.integer(seed_base+cell*1000000+n*1000+replicate)
  X <- generate(n,distribution,seed)
  begin <- proc.time()[['elapsed']]
  u <- tryCatch(fit_ml(X),error=identity)
  shared <- proc.time()[['elapsed']]-begin
  intervals<-list();candidates<-list()
  # Method-specific preparation, including Wald covariance, is timed below.
  # Every profile uses the same deterministic unrestricted start; no shared
  # candidate cache can give the later method an artificial timing advantage.
  for(target in names(truth)) for(method in methods) {
    start<-proc.time()[['elapsed']];profile<-NULL
    row<-data.frame(target=target,method=method,estimate=NA_real_,truth=truth[[target]],
      lower=NA_real_,upper=NA_real_,valid=FALSE,error='',covered=NA,
      lower_miss=NA,upper_miss=NA,width=NA_real_,asymmetry=NA_real_,
      truth_statistic=NA_real_,truth_agreement=NA,near_cutoff=NA,
      endpoint_gap=NA_real_,audit_lower=NA_real_,audit_upper=NA_real_,
      unrestricted_admissible=NA,candidate_inadmissible=0L,out_of_domain=NA,
      refits=0L,seconds=NA_real_)
    result<-tryCatch({
      if(inherits(u,'error')) stop(conditionMessage(u),call.=FALSE)
      id<-index_of(u,target);est<-u$theta[id];row$estimate<-est
      row$unrestricted_admissible<-is_admissible(u)
      if(grepl('^wald',method)) {
        V<-wald_covariance(u,X,method);se<-sqrt(V[id,id])
        if(!is.finite(se)||se<=0) stop('invalid Wald SE')
        row$lower<-est-qnorm(.975)*se;row$upper<-est+qnorm(.975)*se
        row$truth_statistic<-((est-truth[[target]])/se)^2
      } else {
        # This step is a starting bracket only, not the interval formula.
        V<-wald_covariance(u,X,'wald_naive');se<-sqrt(V[id,id])
        profile<-make_profile(u,X,target,method)
        ci<-invert_profile(profile,est,1.96*se)
        row$lower<-ci$lower;row$upper<-ci$upper
        row$audit_lower<-ci$audit_lower;row$audit_upper<-ci$audit_upper
        row$endpoint_gap<-max(abs(c(profile$evaluate(ci$lower)$statistic,
                                   profile$evaluate(ci$upper)$statistic)-critical))
        # Truth is used only after inversion, never for starts or brackets.
        row$truth_statistic<-profile$evaluate(truth[[target]],purpose='truth')$statistic
      }
      row$covered<-row$lower<=truth[[target]] && row$upper>=truth[[target]]
      row$lower_miss<-truth[[target]]<row$lower;row$upper_miss<-truth[[target]]>row$upper
      row$width<-row$upper-row$lower
      row$asymmetry<-(row$upper+row$lower-2*est)/row$width
      row$near_cutoff<-abs(row$truth_statistic-critical)<=1e-4
      row$truth_agreement<-row$covered==(row$truth_statistic<=critical)
      if(!row$truth_agreement && !row$near_cutoff) stop('truth test/inversion disagreement')
      row$out_of_domain<-target=='correlation' && (row$lower < -1 || row$upper > 1)
      row$valid<-TRUE
      TRUE
    },error=identity)
    if(inherits(result,'error')) row$error<-conditionMessage(result)
    row$seconds<-proc.time()[['elapsed']]-start
    if(!is.null(profile)) {
      row$refits<-profile$state$refits
      if(length(profile$state$log)) {
        z<-do.call(rbind,profile$state$log)
        row$candidate_inadmissible<-sum(!z$admissible,na.rm=TRUE)
        z$method<-method;z$target<-target;candidates[[length(candidates)+1L]]<-z
      }
    }
    intervals[[length(intervals)+1L]]<-row
  }
  identify<-function(list) {
    if(!length(list)) return(data.frame())
    x<-do.call(rbind,list);x$n<-n;x$distribution<-distribution
    x$replicate<-replicate;x$seed<-seed;x
  }
  list(intervals=identify(intervals),candidates=identify(candidates),
       timing=data.frame(n=n,distribution=distribution,replicate=replicate,
          shared_fit_seconds=shared,total_seconds=proc.time()[['elapsed']]-begin))
}
