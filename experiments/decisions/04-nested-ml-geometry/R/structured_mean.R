structured_cells <- function() {
  x <- expand.grid(n=c(100L,300L,1000L), role=c('null','power'),
    distribution=c('normal','skewed'), larger=c('correct','mild','strong'),
    factors=c(1L,2L), stringsAsFactors=FALSE)
  x$cell_id <- seq_len(nrow(x)); x$production_reps <- ifelse(x$role=='null',2000L,1000L)
  x
}
structured_syntax <- function(factors) paste(c(vapply(seq_len(factors),function(f)
  paste0('f',f,' =~ ',paste0('x',6*(f-1)+1:6,collapse=' + ')),character(1)),
  paste0('f',seq_len(factors),' ~ 0*1')),collapse='\n')
structured_population <- function(cell,group) {
  p <- 6*cell$factors; loading <- rep(c(.7,.8,.75,.65,.7,.8),cell$factors)
  if(cell$role=='power' && group=='b') loading[3] <- loading[3]-.15
  mu <- numeric(p)
  if(cell$larger!='correct') mu[if(cell$larger=='mild') 1 else c(1,2)] <-
    if(cell$larger=='mild') .1 else .2
  if(group=='a') mu <- -mu
  paste(c(vapply(seq_len(cell$factors),function(f) {
    at <- 6*(f-1)+1:6
    paste0('f',f,' =~ ',paste0(loading[at],'*x',at,collapse=' + '))
  },character(1)),paste0('f',seq_len(cell$factors),' ~~ 1*f',seq_len(cell$factors)),
  if(cell$factors==2) 'f1 ~~ .4*f2',
  paste0('x',seq_len(p),' ~~ ',1-loading^2,'*x',seq_len(p)),
  paste0('x',seq_len(p),' ~ ',mu,'*1')),collapse='\n')
}
# Collapse only the alternative's equal-intercept coordinates; loadings stay free.
structured_maps <- function(fit) {
  pt <- fit$partable; q <- length(fit$theta); ids <- seq_len(q)
  for(label in unique(pt$label[pt$op=='~1' & nzchar(pt$label)])) {
    at <- pt$free[pt$label==label & pt$free>0]; ids[at] <- min(at)
  }
  K <- diag(length(unique(ids)))[match(ids,unique(ids)),,drop=FALSE]
  a <- pt[pt$op=='=~' & pt$group==1 & pt$free>0,]
  A <- matrix(0,nrow(a),q)
  for(i in seq_len(nrow(a))) {
    other <- pt$free[pt$op=='=~' & pt$group==2 & pt$lhs==a$lhs[i] & pt$rhs==a$rhs[i]]
    A[i,a$free[i]] <- 1; A[i,other] <- -1
  }
  list(K=K,A=A%*%K)
}
# Pre-66 rows: centered sample covariance/mean influence transported by the
# fitted normal-theory weight and model Jacobian. No fitted-mean likelihood
# linear or constant correction is included. Central differences are a lab
# reconstruction; the exact-row reconstruction below checks the same projection.
structured_lr_reference <- function(fit,context,data,statistic,df) {
  maps <- structured_maps(fit); K <- maps$K; A <- maps$A
  implied <- magmaanlab:::model_implied(fit)
  q <- length(fit$theta); old <- exact <- matrix(0,nrow(data),q)
  for(j in seq_len(q)) {
    step <- 1e-5*max(1,abs(fit$theta[j])); plus <- minus <- fit
    plus$theta[j] <- plus$theta[j]+step; minus$theta[j] <- minus$theta[j]-step
    up <- magmaanlab:::model_implied(plus); down <- magmaanlab:::model_implied(minus)
    for(g in 1:2) {
      at <- which(data$group==c('a','b')[g]); X <- as.matrix(data[at,fit$ov_names,drop=FALSE])
      z <- sweep(X,2,colMeans(X)); e <- sweep(X,2,implied$mu[[g]])
      V <- solve(implied$sigma[[g]])
      dS <- (up$sigma[[g]]-down$sigma[[g]])/(2*step)
      dm <- (up$mu[[g]]-down$mu[[g]])/(2*step)
      B <- V%*%dS%*%V
      old[at,j] <- .5*(rowSums((z%*%B)*z)-sum(B*(crossprod(z)/nrow(z))))+as.numeric(z%*%V%*%dm)
      exact[at,j] <- .5*(rowSums((e%*%B)*e)-sum(V*t(dS)))+as.numeric(e%*%V%*%dm)
    }
  }
  H <- crossprod(K,magmaanlab::inference_information(context,'observed')%*%K)
  R <- solve(H,t(A)); C <- chol(A%*%R)
  spectrum <- function(rows) {
    reduced <- rows%*%K%*%R%*%solve(C)
    eigen(crossprod(reduced),symmetric=TRUE,only.values=TRUE)$values
  }
  list(old=magmaanlab::quadratic_reference(statistic,df,spectrum(old)),exact=spectrum(exact))
}
structured_draw <- function(cell,rep,seed) {
  started <- proc.time()[['elapsed']]
  rows <- cbind(cell[rep(1,3),],rep=rep,seed=seed,
    arm=c('policy_lr','policy_score','pre66_lr'),test=c('lr','score','lr'))
  rows$converged_h0 <- rows$converged_h1 <- FALSE
  rows$statistic <- rows$df <- rows$p_sb <- rows$p_peba4 <- rows$reference_gap <- NA_real_
  rows$error <- rows$error_call <- rows$error_stage <- ''
  stage <- 'fit_and_prepare'
  record_error <- function(e,i=seq_len(nrow(rows))) {
    rows$error[i] <<- conditionMessage(e)
    rows$error_call[i] <<- paste(deparse(conditionCall(e)),collapse=' ')
    rows$error_stage[i] <<- stage
  }
  tryCatch({
    set.seed(seed); p <- 6*cell$factors
    data <- do.call(rbind,lapply(c('a','b'),function(g) {
      x <- lavaan::simulateData(structured_population(cell,g),sample.nobs=cell$n,
        skewness=if(cell$distribution=='skewed') rep(2,p) else NULL,
        kurtosis=if(cell$distribution=='skewed') rep(7,p) else NULL)
      x$group <- g; x
    }))
    fit <- function(eq) magmaan::magmaan(magmaan::magmaan_model(structured_syntax(cell$factors),
      prototype=data,group='group',group.equal=eq),data,estimator='ML')
    h1 <- fit('intercepts'); h0 <- fit(c('intercepts','loadings'))
    f1 <- magmaan::as_lab_fit(h1); f0 <- magmaan::as_lab_fit(h0)
    rows$converged_h0 <- isTRUE(f0$converged); rows$converged_h1 <- isTRUE(f1$converged)
    if(!all(rows$converged_h0 & rows$converged_h1)) stop('library convergence verdict failed')
    shared <- magmaanlab::prepare_inference_data(f1)
    i1 <- magmaanlab::prepare_inference(f1,shared)
    hyp <- magmaanlab::prepare_hypothesis(magmaanlab::prepare_inference(f0,shared),i1)
    for(i in 1:3) {
      stage <- paste0(rows$arm[i],':quadratic')
      tryCatch({
        q <- magmaanlab::inference_quadratic(hyp,rows$test[i])
        if(q$df!=5*cell$factors) stop('unexpected restriction df')
        if(i!=2) {
          stage <- paste0(rows$arm[i],':reference')
          ref <- structured_lr_reference(f1,i1,data,q$statistic,q$df)
          actual <- eigen(crossprod(magmaanlab::inference_rows(q)),symmetric=TRUE,only.values=TRUE)$values
          gap <- max(abs(sort(actual)-sort(ref$exact)))/max(1,max(abs(actual)))
          if(!is.finite(gap) || gap>1e-6) stop('independent exact-row LR reconstruction differs from policy')
          rows$reference_gap[i] <- gap
          if(i==3) q <- ref$old
        }
        stage <- paste0(rows$arm[i],':calibrate')
        z <- magmaanlab::calibrate_quadratic(q,c('sb','peba4'))
        rows$statistic[i] <- q$statistic; rows$df[i] <- q$df
        rows$p_sb[i] <- z$p_value[1]; rows$p_peba4[i] <- z$p_value[2]
        if(any(!is.finite(c(rows$p_sb[i],rows$p_peba4[i])))) stop('nonfinite calibration')
      },error=function(e) record_error(e,i))
    }
  },error=record_error)
  rows$elapsed_s <- proc.time()[['elapsed']]-started; rows
}
structured_summaries <- function(raw,cells,mode) {
  rates <- paired <- list()
  for(id in cells$cell_id) for(method in c('sb','peba4')) {
    cell <- cells[cells$cell_id==id,]; col <- paste0('p_',method)
    for(arm in unique(raw$arm)) {
      x <- raw[raw$cell_id==id & raw$arm==arm,]; ok <- !nzchar(x$error)&is.finite(x[[col]])
      p <- x[[col]][ok]; ci <- wilson(sum(p<.05),length(p)); critical <- adjusted <- NA_real_
      if(cell$role=='power') {
        null <- cells$cell_id[cells$n==cell$n & cells$distribution==cell$distribution &
          cells$larger==cell$larger & cells$factors==cell$factors & cells$role=='null']
        y <- raw[[col]][raw$cell_id==null & raw$arm==arm & !nzchar(raw$error)]
        y <- y[is.finite(y)]
        if(length(y)) critical <- unname(quantile(y,.05,type=1))
        if(length(p)&&is.finite(critical)) adjusted <- mean(p<critical)
      }
      rates[[length(rates)+1]] <- cbind(cell,arm=arm,method=method,draws=nrow(x),available=length(p),
        failures=sum(!ok),rejection=if(length(p)) mean(p<.05) else NA_real_,lower=ci[1],upper=ci[2],
        null_p_critical=critical,size_adjusted_power=adjusted)
    }
    if(cell$role=='null') {
      x <- raw[raw$cell_id==id & raw$arm=='pre66_lr',]; y <- raw[raw$cell_id==id & raw$arm=='policy_lr',]
      y <- y[match(x$rep,y$rep),]; ok <- !nzchar(x$error)&!nzchar(y$error)&is.finite(x[[col]])&is.finite(y[[col]])
      a <- x[[col]][ok]<.05; b <- y[[col]][ok]<.05; delta <- lo <- hi <- NA_real_
      if(length(a)>1) {
        delta <- abs(mean(b)-.05)-abs(mean(a)-.05)
        set.seed(1926100061L+100*id+match(method,c('sb','peba4')))
        boot <- replicate(10000,{at <- sample.int(length(a),replace=TRUE);abs(mean(b[at])-.05)-abs(mean(a[at])-.05)})
        ci <- quantile(boot,c(.025,.975)); lo <- ci[1]; hi <- ci[2]
      }
      paired[[length(paired)+1]] <- cbind(cell,method=method,pairs=length(a),size_error_difference=delta,
        lower=lo,upper=hi,qualifying_loss=is.finite(delta)&&delta>.01&&is.finite(lo)&&lo>0)
    }
  }
  draws <- raw[!duplicated(paste(raw$cell_id,raw$rep)),]
  timing <- do.call(rbind,lapply(cells$cell_id,function(id) {
    t <- draws$elapsed_s[draws$cell_id==id]; cell <- cells[cells$cell_id==id,]
    cbind(cell,draws=length(t),mean_seconds=mean(t),production_cpu_hours=mean(t)*cell$production_reps/3600)
  }))
  paired <- do.call(rbind,paired)
  list(rates=do.call(rbind,rates),paired=paired,timing=timing,
    failures=raw[nzchar(raw$error),c('cell_id','rep','seed','arm','error','error_call','error_stage')],
    checks=data.frame(draws=nrow(draws),failed_arms=sum(nzchar(raw$error)),
      max_reference_gap=max(c(0,raw$reference_gap),na.rm=TRUE)),
    decisions=data.frame(status=if(mode!='production') 'open_development_only' else
      if(any(paired$qualifying_loss[paired$larger!='correct'])) 'finite_sample_follow_up' else 'no_flag'))
}

structured_population_checks <- function() {
  cells <- subset(structured_cells(),n==100 & distribution=='normal')
  do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
    cell <- cells[i,]; p <- 6*cell$factors
    moments <- lapply(c('a','b'),function(g) {
      set.seed(1)
      x <- lavaan::simulateData(structured_population(cell,g),sample.nobs=1000,empirical=TRUE)
      list(S=cov(x),m=colMeans(x))
    })
    fit <- function(eq) lavaan::cfa(structured_syntax(cell$factors),
      sample.cov=lapply(moments,`[[`,'S'),sample.mean=lapply(moments,`[[`,'m'),
      sample.nobs=c(1e7,1e7),sample.cov.rescale=FALSE,meanstructure=TRUE,group.equal=eq)
    h1 <- fit('intercepts'); h0 <- fit(c('intercepts','loadings'))
    if(!lavaan::lavInspect(h1,'converged') || !lavaan::lavInspect(h0,'converged')) stop('Population fit failed')
    pt <- lavaan::parameterEstimates(h1); a <- subset(pt,op=='=~' & group==1)$est
    b <- subset(pt,op=='=~' & group==2)$est
    gap <- max(abs(a-b)); fgap <- unname(lavaan::fitMeasures(h0,'fmin')-lavaan::fitMeasures(h1,'fmin'))
    curvature <- min(eigen(lavaan::lavInspect(h1,'hessian'),symmetric=TRUE,only.values=TRUE)$values)
    if(cell$role=='null' && (gap>1e-5 || abs(fgap)>1e-8 || curvature<=0)) stop('Population null/curvature check failed')
    data.frame(factors=cell$factors,role=cell$role,larger=cell$larger,
      loading_gap=gap,restriction_discrepancy=fgap,min_observed_curvature=curvature,
      population_check=if(cell$role=='null') 'symmetric_local_null_verified' else 'power_departure')
  }))
}
