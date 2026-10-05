association_cells <- function() {
  x <- expand.grid(factors=1:2,categories=c(2L,5L),imbalance=c('symmetric','skewed'),
    generator=c('gaussian','chisq2'),misspecified=c(FALSE,TRUE),n=c(300L,1000L,4000L),
    groups=1:2,stringsAsFactors=FALSE)
  x$cell_id <- seq_len(nrow(x)); x$production_reps <- 2000L; x
}
association_key <- function(x) paste(x$factors,x$categories,x$imbalance,x$generator,x$misspecified,sep='_')
association_draw <- function(cell,seed,n=cell$n) {
  set.seed(seed); k <- cell$factors; p <- 6*k
  L <- matrix(0,p,k); L[cbind(seq_len(p),rep(seq_len(k),each=6))] <- rep(c(.65,.70,.75,.80,.70,.75),k)
  if(cell$misspecified && k==2) L[7,1] <- .3
  phi <- matrix(.3,k,k); diag(phi) <- 1
  v <- 1-diag(L%*%phi%*%t(L)); residual <- diag(p)
  if(cell$misspecified && k==1) residual[5,6] <- residual[6,5] <- .3
  innovation <- function(m) if(cell$generator=='gaussian') rnorm(m) else (rchisq(m,2)-2)/2
  cuts <- qnorm(if(cell$categories==2) if(cell$imbalance=='skewed') .85 else .5 else
    if(cell$imbalance=='skewed') c(.45,.70,.85,.95) else c(.2,.4,.6,.8))
  do.call(rbind,lapply(seq_len(cell$groups),function(g) {
    ng <- as.integer(n/cell$groups)
    f <- matrix(innovation(ng*k),ncol=k)%*%chol(phi)
    e <- matrix(innovation(ng*p),ncol=p)%*%chol(residual)
    z <- f%*%t(L)+sweep(e,2,sqrt(v),'*')
    d <- as.data.frame(lapply(seq_len(p),function(j) as.integer(findInterval(z[,j],cuts)+1L)))
    names(d) <- paste0('x',seq_len(p)); if(cell$groups==2) d$group <- c('a','b')[g]; d
  }))
}
association_fit <- function(cell,data,estimator='ML',equal=NULL,fixed=NULL) {
  lines <- vapply(seq_len(cell$factors),function(f) {
    indicators <- paste0('x',((f-1)*6+1):(f*6))
    if(f==1 && !is.null(fixed)) indicators[2] <- paste0(sprintf('%.17g',fixed),'*x2')
    paste0('f',f,' =~ ',paste(indicators,collapse=' + '))
  },'')
  model <- magmaanlab::model_spec(paste(lines,collapse='\n'),ordered=paste0('x',seq_len(6*cell$factors)),
    parameterization='delta',group=if(cell$groups==2) 'group' else NULL,
    group_labels=if(cell$groups==2) c('a','b') else NULL,group_equal=equal)
  fit <- magmaanlab::fit_model(model,data,estimator=estimator)
  if(!isTRUE(fit$converged)) stop('library_nonconvergence')
  fit
}
association_targets <- function(fit,covariance=NULL) {
  pt <- fit$partable
  index <- function(lhs,op,rhs) {
    i <- which(pt$lhs==lhs & pt$op==op & pt$rhs==rhs & pt$group==1)
    if(length(i)!=1 || pt$free[i]<1) stop('Missing/free target: ',lhs,op,rhs)
    pt$free[i]
  }
  target <- function(i) list(value=fit$theta[i],gradient=replace(numeric(length(fit$theta)),i,1))
  z <- list(loading=target(index('f1','=~','x2')),threshold=target(index('x2','|','t1')))
  if(any(pt$lhs=='f2')) {
    # Partable covariance orientation is fixed by the parser, but accept either order.
    i <- which(pt$op=='~~' & ((pt$lhs=='f1' & pt$rhs=='f2') | (pt$lhs=='f2' & pt$rhs=='f1')) & pt$group==1)
    if(length(i)!=1) stop('Missing factor covariance')
    a <- index('f1','~~','f1'); b <- index('f2','~~','f2'); i <- pt$free[i]
    value <- fit$theta[i]/sqrt(fit$theta[a]*fit$theta[b]); grad <- numeric(length(fit$theta))
    grad[i] <- 1/sqrt(fit$theta[a]*fit$theta[b]); grad[a] <- -value/(2*fit$theta[a]); grad[b] <- -value/(2*fit$theta[b])
    z$correlation <- list(value=value,gradient=grad)
  }
  do.call(rbind,lapply(names(z),function(name) data.frame(target=name,estimate=z[[name]]$value,
    se=if(is.null(covariance)) NA_real_ else sqrt(as.numeric(crossprod(z[[name]]$gradient,covariance%*%z[[name]]$gradient))))))
}
# Count streaming bounds memory; Stage 1 itself remains the library estimator.
association_population_stats <- function(cell,seed,n) {
  cell$groups <- 1L; p <- 6*cell$factors; c <- cell$categories
  pairs <- which(lower.tri(matrix(0,p,p)),arr.ind=TRUE)
  tables <- lapply(seq_len(nrow(pairs)),function(i) matrix(0L,c,c))
  for(first in seq(1L,n,by=10000L)) {
    d <- as.matrix(association_draw(cell,seed+as.integer((first-1L)/10000L),min(10000L,n-first+1L)))
    for(i in seq_len(nrow(pairs))) tables[[i]] <- tables[[i]]+
      matrix(tabulate(d[,pairs[i,1]]+c*(d[,pairs[i,2]]-1L),nbins=c*c),c,c)
  }
  stats <- magmaanlab::magmaan_core$data_ordinal_stats_from_raw(as.matrix(association_draw(cell,seed,2000L)),full_wls_weight=FALSE)
  q <- p*(c-1)+nrow(pairs); thresholds <- numeric(p*(c-1)); W <- numeric(q); R <- diag(p)
  for(i in seq_len(nrow(pairs))) {
    idx <- rep(seq_len(c*c),as.vector(tables[[i]])); d <- cbind((idx-1L)%%c+1L,(idx-1L)%/%c+1L)
    pair <- magmaanlab::magmaan_core$data_ordinal_stats_from_raw(d,full_wls_weight=FALSE)
    a <- pairs[i,1]; b <- pairs[i,2]; ia <- ((a-1L)*(c-1L)+1L):(a*(c-1L)); ib <- ((b-1L)*(c-1L)+1L):(b*(c-1L))
    thresholds[ia] <- pair$thresholds[[1]][seq_len(c-1L)]; thresholds[ib] <- pair$thresholds[[1]][c:(2*(c-1L))]
    wp <- diag(pair$W_dwls[[1]]); W[ia] <- wp[seq_len(c-1L)]; W[ib] <- wp[c:(2*(c-1L))]
    W[p*(c-1L)+i] <- tail(wp,1); R[a,b] <- R[b,a] <- pair$R[[1]][2,1]
    rm(pair,d,idx); gc(FALSE)
  }
  stats$ov_names <- list(paste0('x',seq_len(p))); stats$ordered <- stats$ov_names[[1]]; dimnames(R) <- list(stats$ordered,stats$ordered)
  stats$R <- list(R); stats$thresholds <- list(thresholds); stats$moments <- list(c(thresholds,R[lower.tri(R)]))
  stats$W_dwls <- list(diag(W)); stats$NACOV <- list(diag(1/W)); stats$W_wls <- list(matrix(numeric(),0,0)); stats$nobs <- as.integer(n)
  stats$moment_influence <- list(matrix(numeric(),0,q)); stats$int_data <- list(matrix(integer(),0,p)); stats$moment_bread <- list(matrix(numeric(),0,0))
  stats
}
association_population <- function(cells,out,mode) {
  full <- association_cells(); keys <- unique(vapply(seq_len(nrow(full)),function(i) association_key(full[i,]),''))
  need <- unique(vapply(seq_len(nrow(cells)),function(i) association_key(cells[i,]),''))
  targets <- stage1 <- timings <- list(); n <- if(mode=='smoke') 20000L else 1000000L
  for(key in need) for(draw in 1:2) {
    cell <- cells[which(vapply(seq_len(nrow(cells)),function(i) association_key(cells[i,]),'')==key)[1],]; cell$groups <- 1L
    seed <- c(617000001,1117000001)[draw]+10000*match(key,keys)
    cat('Population',key,'draw',draw,'N',n,'\n'); flush.console(); start <- proc.time()
    stats <- association_population_stats(cell,seed,n)
    stage1[[length(stage1)+1L]] <- data.frame(key,draw,seed,n,moment=seq_along(stats$moments[[1]]),value=stats$moments[[1]],opg_weight=diag(stats$W_dwls[[1]]))
    for(estimator in c('ML','DWLS')) {
      fit <- association_fit(cell,stats,estimator); z <- association_targets(fit)
      z$key <- key; z$draw <- draw; z$seed <- seed; z$n <- n; z$estimator <- estimator; z$population_fmin <- fit$fmin
      targets[[length(targets)+1L]] <- z
    }
    timings[[length(timings)+1L]] <- data.frame(key,draw,n,elapsed_seconds=unname((proc.time()-start)['elapsed']))
    write_csv(do.call(rbind,targets),file.path(out,'population_targets.csv'))
    write_csv(do.call(rbind,stage1),file.path(out,'population_stage1.csv')); write_csv(do.call(rbind,timings),file.path(out,'population_timing.csv'))
  }
  pop <- do.call(rbind,targets); a <- pop[pop$draw==1,]; b <- pop[pop$draw==2,]
  delta <- merge(a,b,by=c('key','target','estimator'),suffixes=c('_first','_second')); delta$target_difference <- delta$estimate_second-delta$estimate_first
  write_csv(delta,file.path(out,'population_uncertainty.csv')); pop
}
association_replicate <- function(cell,replicate,base,population) {
  seed <- base+10000*cell$cell_id+replicate; data <- association_draw(cell,seed); rows <- list()
  truth <- function(estimator,target) {
    z <- population[population$key==association_key(cell) & population$draw==1 & population$estimator==estimator & population$target==target,'estimate']
    if(length(z)!=1 || !is.finite(z)) stop('Missing population truth'); z
  }
  add <- function(arm,target,success,value=NA_real_,estimate=NA_real_,se=NA_real_,error='',seconds=0) {
    rows[[length(rows)+1L]] <<- data.frame(cell_id=cell$cell_id,replicate,seed,arm,target,success,value,estimate,se,error,seconds)
  }
  attempt <- function(arms,targets,fun) {
    start <- proc.time(); before <- length(rows)
    tryCatch(fun(),error=function(e) {
      # Each unavailable result stays in the expected panel denominator.
      rows <<- head(rows,before)
      for(arm in arms) for(target in targets) add(arm,target,FALSE,error=conditionMessage(e))
    })
    elapsed <- unname((proc.time()-start)['elapsed'])
    for(i in seq.int(before+1L,length(rows))) rows[[i]]$seconds <<- elapsed/(length(rows)-before)
  }
  available <- function(z) { if(!isTRUE(z$available)) stop(paste(z$reason,z$detail)); z }
  coverage_targets <- c('loading','threshold',if(cell$factors==2) 'correlation')
  for(estimator in c('ML','DWLS')) {
    prefix <- if(estimator=='ML') 'association' else 'dwls'
    fit <- NULL; fit_error <- ''
    fit_start <- proc.time()
    tryCatch(fit <- association_fit(cell,data,estimator),error=function(e) fit_error <<- conditionMessage(e))
    require_fit <- function() if(is.null(fit)) stop(fit_error)
    if(cell$groups==1) {
      attempt(paste0(prefix,'_ij'),coverage_targets,function() {
        require_fit()
        cov <- if(estimator=='ML') available(magmaanlab::association_ml_ij(fit,fit$ordinal_stats))$vcov else {
          z <- magmaanlab::policy_inference(fit); if(!isTRUE(z$covariance_available)) stop(z$covariance_reason); z$covariance
        }
        z <- association_targets(fit,cov)
        for(i in seq_len(nrow(z))) {
          if(!is.finite(z$se[i]) || z$se[i]<=0) stop('invalid_standard_error')
          add(paste0(prefix,'_ij'),z$target[i],TRUE,as.numeric(abs(z$estimate[i]-truth(estimator,z$target[i]))<=qnorm(.975)*z$se[i]),z$estimate[i],z$se[i])
        }
      })
      arms <- if(estimator=='ML') paste0(prefix,'_global_',c('All','SB','PEBA4')) else 'dwls_global_All'
      attempt(arms,'global',function() {
        require_fit()
        if(estimator=='ML') {
          test <- available(magmaanlab::association_ml_global_test(fit,fit$ordinal_stats))
          for(ref in c('All','SB','PEBA4')) {
            p <- test[[ref]]$p_value; if(!is.finite(p)) stop('nonfinite_p_value'); add(paste0(prefix,'_global_',ref),'global',TRUE,as.numeric(p<.05),estimate=p)
          }
        } else {
          test <- available(magmaanlab::policy_inference(fit)$score); p <- test$p_all
          if(!is.finite(p)) stop('nonfinite_p_value'); add('dwls_global_All','global',TRUE,as.numeric(p<.05),estimate=p)
        }
      })
      if(estimator=='ML') for(role in c('size','power')) attempt(paste0('association_mi_',role),'mi',function() {
        null <- association_fit(cell,data,fixed=truth('ML','loading')+if(role=='power') .15 else 0)
        mi <- available(magmaanlab::association_ml_modification_indices(null,null$ordinal_stats,absent=FALSE))
        z <- Filter(function(x) x$lhs=='f1' && x$op=='=~' && x$rhs=='x2' && x$group==1,mi$rows)
        if(length(z)!=1) stop('MI fixed row missing'); z <- available(z[[1]])
        if(!is.finite(z$p_value)) stop('nonfinite_p_value'); add(paste0('association_mi_',role),'mi',TRUE,as.numeric(z$p_value<.05),estimate=z$p_value)
      })
    } else {
      arms <- paste0(prefix,'_nested_',c('All','SB','PEBA4'))
      attempt(arms,'nested',function() {
        require_fit(); null <- association_fit(cell,data,estimator,equal='loadings')
        test <- if(estimator=='ML') available(magmaanlab::association_ml_nested_test(fit,null,fit$ordinal_stats)) else available(magmaanlab::policy_nested(fit,null)$lr)
        for(ref in c('All','SB','PEBA4')) {
          p <- if(estimator=='ML') test[[ref]]$p_value else if(ref=='All')
            magmaanlab::magmaan_core$robust_fmg_test(test$statistic,test$df,test$eigenvalues,'all',0,truncate_negative=TRUE)$p_value else
            test[[switch(ref,SB='p_sb',PEBA4='p_peba4')]]
          if(length(p)!=1 || !is.finite(p)) stop('nonfinite_p_value'); add(paste0(prefix,'_nested_',ref),'nested',TRUE,as.numeric(p<.05),estimate=p)
        }
      })
    }
    # Fit cost is counted once across this estimator's panel rows.
    ids <- which(vapply(rows,function(z) startsWith(z$arm,prefix),FALSE))
    fit_seconds <- unname((proc.time()-fit_start)['elapsed'])-sum(vapply(rows[ids],function(z) z$seconds,0))
    for(i in ids) rows[[i]]$seconds <- rows[[i]]$seconds+max(0,fit_seconds)/length(ids)
  }
  do.call(rbind,rows)
}
association_summarize <- function(raw,cells,out) {
  wilson <- function(k,n) {
    if(n==0) return(c(NA_real_,NA_real_)); z <- qnorm(.975); p <- k/n
    a <- (p+z*z/(2*n))/(1+z*z/n); b <- z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n); c(a-b,a+b)
  }
  groups <- split(raw,interaction(raw$cell_id,raw$arm,raw$target,drop=TRUE))
  summary <- do.call(rbind,lapply(groups,function(z) {
    attempted <- nrow(z); successful <- sum(z$success); hits <- sum(z$value[z$success]); ci <- wilson(hits,successful)
    cell <- cells[match(z$cell_id[1],cells$cell_id),]; coverage <- z$target[1] %in% c('loading','threshold','correlation')
    descriptive <- grepl('global',z$arm[1]) && (cell$generator!='gaussian' || cell$misspecified)
    power <- grepl('power',z$arm[1]); rate <- if(successful) hits/successful else NA_real_
    data.frame(cell_id=z$cell_id[1],arm=z$arm[1],target=z$target[1],attempted,successful,failures=attempted-successful,
      hits,rate,attempted_rate=hits/attempted,lower=ci[1],upper=ci[2],
      interpretation=if(power) 'power' else if(descriptive) 'descriptive_global' else if(coverage) 'pseudo_true_coverage' else 'null_size',
      mechanical_flag=if(power || descriptive || is.na(rate)) FALSE else if(coverage) rate<.93 else cell$n>=1000 && (rate<.03 || rate>.07),
      mean_seconds=sum(z$seconds)/attempted)
  }))
  summary <- summary[order(summary$cell_id,summary$arm,summary$target),]; rownames(summary) <- NULL
  write_csv(summary,file.path(out,'summary.csv')); write_csv(raw[!raw$success,],file.path(out,'failures.csv'))
  timings <- aggregate(seconds~cell_id+replicate+seed,raw,sum); write_csv(timings,file.path(out,'timing.csv'))
  costs <- association_cells(); costs$priced <- costs$cell_id %in% cells$cell_id
  means <- aggregate(seconds~cell_id,timings,mean); costs$mean_draw_seconds <- means$seconds[match(costs$cell_id,means$cell_id)]
  costs$production_cpu_hours <- costs$mean_draw_seconds*costs$production_reps/3600
  write_csv(costs,file.path(out,'cost.csv'))
}
