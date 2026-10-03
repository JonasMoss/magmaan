dwls_cells <- function() {
  base <- function(family, factors, categories, skew, n, groups=1L,
                   parameterization='delta', cross=0, nesting='none', role='null', model='cfa') {
    x <- expand.grid(factors=factors,categories=categories,skew=skew,n=n,
      groups=groups,parameterization=parameterization,cross=cross,nesting=nesting,
      role=role,model=model,stringsAsFactors=FALSE)
    x$family <- family; x
  }
  g <- base('global',1:3,c(2,5),c('symmetric','skewed'),c(200,500,1000))
  mg <- base('global',2:3,c(2,5),'symmetric',c(400,1000),2,c('delta','theta'))
  nested <- base('nested',2:3,c(2,5),'symmetric',c(400,1000),2,'theta',c(0,.3),c('metric','thresholds'))
  coverage <- base('coverage',2,c(2,5),'symmetric',c(150,300,1000),1,'delta',c(0,.2,.4),model=c('cfa','sem'))
  subset <- rbind(base('coverage',2,c(2,5),'symmetric',c(300,1000),2,'theta',.2),
                  base('coverage',2,c(2,5),'symmetric',300,2,'theta',.4))
  power_g <- base('global',2:3,c(2,5),'symmetric',c(200,500,1000),cross=.3,role='power')
  power_n <- base('nested',2:3,c(2,5),'symmetric',c(400,1000),2,'theta',nesting=c('metric','thresholds'),role='power')
  x <- rbind(g,mg,nested,coverage,subset,power_g,power_n)
  x$cell_id <- seq_len(nrow(x)); x$production_reps <- ifelse(x$role=='power',1000L,2000L)
  x
}

dwls_syntax <- function(cell) {
  p <- 6*cell$factors
  lines <- vapply(seq_len(cell$factors),function(f)
    paste0('f',f,' =~ ',paste0('x',((f-1)*6+1):(f*6),collapse=' + ')),character(1))
  if (cell$model=='sem') lines <- c(lines,'f2 ~ f1')
  paste(lines,collapse='\n')
}

dwls_draw_data <- function(cell, seed, n=cell$n) {
  set.seed(seed)
  k <- cell$factors; p <- 6*k
  loading <- rep(c(.65,.7,.75,.8,.7,.75),k)
  phi <- matrix(.3,k,k); diag(phi) <- 1
  cuts <- if (cell$categories==2) qnorm(if(cell$skew=='skewed') .85 else .5) else
    qnorm(if(cell$skew=='skewed') c(.45,.70,.85,.95) else c(.2,.4,.6,.8))
  do.call(rbind,lapply(seq_len(cell$groups),function(g) {
    lambda <- matrix(0,p,k); lambda[cbind(seq_len(p),rep(seq_len(k),each=6))] <- loading
    if (cell$cross>0) lambda[7,1] <- cell$cross
    if (cell$role=='power' && cell$family=='nested' && cell$nesting=='metric' && g==2) lambda[2,1] <- lambda[2,1]+.15
    common <- lambda%*%phi%*%t(lambda)
    sigma <- common+diag(1-diag(common),p)
    stopifnot(min(eigen(sigma,symmetric=TRUE,only.values=TRUE)$values)>0)
    z <- matrix(rnorm(as.integer(n/cell$groups)*p),ncol=p)%*%chol(sigma)
    d <- as.data.frame(lapply(seq_len(p),function(j) {
      threshold <- cuts
      if (cell$role=='power' && cell$family=='nested' && cell$nesting=='thresholds' && g==2 && j==2) threshold[1] <- threshold[1]+.3
      as.integer(cut(z[,j],c(-Inf,threshold,Inf)))
    }))
    names(d) <- paste0('x',seq_len(p)); if(cell$groups==2) d$group <- c('a','b')[g]
    d
  }))
}

dwls_fit <- function(cell, data, equal=NULL) {
  model <- magmaanlab::model_spec(dwls_syntax(cell),ordered=paste0('x',seq_len(6*cell$factors)),
    parameterization=cell$parameterization,group=if(cell$groups==2) 'group' else NULL,
    group_labels=if(cell$groups==2) c('a','b') else NULL,group_equal=equal)
  magmaanlab::fit_model(model,data,estimator='DWLS')
}

# Availability gate only; calibration must not proceed with a missing policy arm.
dwls_preflight <- function() {
  cells <- dwls_cells()
  cells <- cells[cells$family=='nested' & cells$nesting=='thresholds' &
    cells$cross==0 & cells$factors==2 & cells$n==400 & cells$role=='null',]
  do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
    cell <- cells[i,]; seed <- 817130001L+10000L*cell$cell_id+1L
    data <- dwls_draw_data(cell,seed)
    h1 <- dwls_fit(cell,data,'loadings')
    h0 <- dwls_fit(cell,data,c('loadings','thresholds'))
    test <- magmaanlab::policy_nested(h1,h0)$lr
    data.frame(cell_id=cell$cell_id,categories=cell$categories,n=cell$n,
      groups=cell$groups,parameterization=cell$parameterization,seed=seed,
      converged_h1=h1$converged,converged_h0=h0$converged,
      available=test$available,reason=test$reason,detail=test$detail,
      df=test$df,spectrum_size=length(test$eigenvalues))
  }))
}
