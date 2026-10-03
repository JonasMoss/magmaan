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
  power_n <- base('nested',2:3,c(2,5),'symmetric',c(400,1000),2,'theta',nesting='metric',role='power')
  x <- rbind(g,mg,nested,coverage,subset,power_g,power_n)
  x$cell_id <- seq_len(nrow(x)); x$production_reps <- ifelse(x$role=='power',1000L,2000L)
  x
}

dwls_mode_cells <- function(mode) {
  cells <- dwls_cells()
  if(mode %in% c('explore','confirm')) cells <- cells[cells$family=='global',]
  cells
}

dwls_global_arms <- function() c('policy_sb','policy_peba4','scaled_shifted',
  'mean_variance','scaled_f','all','pall','eba2','eba4','eba6','peba2','peba6','pols')

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
    h1 <- dwls_fit(cell,data,'thresholds')
    h0 <- dwls_fit(cell,data,c('loadings','thresholds'))
    test <- magmaanlab::policy_nested(h1,h0)$lr
    data.frame(cell_id=cell$cell_id,categories=cell$categories,n=cell$n,
      groups=cell$groups,parameterization=cell$parameterization,seed=seed,
      converged_h1=h1$converged,converged_h0=h0$converged,
      available=test$available,reason=test$reason,detail=test$detail,
      df=test$df,spectrum_size=length(test$eigenvalues))
  }))
}

# All distribution and covariance calculations use lab primitives.
dwls_calibrate <- function(statistic, df, eigenvalues) {
  core <- magmaanlab::magmaan_core
  sb <- core$robust_satorra_bentler(statistic,df,eigenvalues)
  peba <- core$robust_fmg_test(statistic,df,eigenvalues,'peba',4)
  list(p_sb=pchisq(sb$chi2_scaled,df,lower.tail=FALSE),p_peba4=peba$p_value)
}

dwls_targets <- function(fit, covariance=NULL) {
  pt <- fit$partable
  row <- function(lhs,op,rhs) {
    i <- which(pt$lhs==lhs & pt$op==op & pt$rhs==rhs & pt$group==1)
    if(length(i)!=1) stop('Target row missing or ambiguous: ',lhs,op,rhs)
    pt$free[i]
  }
  value <- function(i) { if(i<1) stop('Target is fixed'); fit$theta[i] }
  l <- row('f1','=~','x2'); t <- row('x2','|','t1')
  targets <- list(loading=list(value=value(l),gradient=replace(numeric(fit$npar),l,1)),
                  threshold=list(value=value(t),gradient=replace(numeric(fit$npar),t,1)))
  if(grepl('f2 ~ f1',fit$syntax,fixed=TRUE)) {
    i <- row('f2','~','f1')
    targets$path <- list(value=value(i),gradient=replace(numeric(fit$npar),i,1))
  } else {
    i <- row('f1','~~','f2'); a <- row('f1','~~','f1'); b <- row('f2','~~','f2')
    v <- value(i)/sqrt(value(a)*value(b)); g <- numeric(fit$npar)
    g[i] <- 1/sqrt(value(a)*value(b)); g[a] <- -v/(2*value(a)); g[b] <- -v/(2*value(b))
    targets$correlation <- list(value=v,gradient=g)
  }
  do.call(rbind,lapply(names(targets),function(n) {
    z <- targets[[n]]
    data.frame(target=n,estimate=z$value,se=if(is.null(covariance)) NA_real_ else
      sqrt(as.numeric(crossprod(z$gradient,covariance%*%z$gradient))))
  }))
}

dwls_population_key <- function(cell) paste(cell$model,cell$categories,cell$groups,
  cell$parameterization,cell$cross,sep='_')

dwls_population <- function(cells,out) {
  x <- cells[cells$family=='coverage',]
  if(!nrow(x)) return(NULL)
  x <- x[!duplicated(vapply(seq_len(nrow(x)),function(i) dwls_population_key(x[i,]),'')),]
  start <- proc.time()
  result <- vector('list',nrow(x))
  for(i in seq_len(nrow(x))) {
    c <- x[i,]; seed <- 817120001L+i*10000L
    cat('Population target ',i,'/',nrow(x),'\n',sep=''); flush.console()
    f <- dwls_fit(c,dwls_draw_data(c,seed,n=100000L*c$groups))
    if(!isTRUE(f$converged)) stop('Population fit did not converge: ',dwls_population_key(c))
    z <- dwls_targets(f); z$key <- dwls_population_key(c); z$seed <- seed
    z$n_per_group <- 100000L; result[[i]] <- z
    write_csv(do.call(rbind,result[seq_len(i)]),file.path(out,'population_targets.csv'))
  }
  elapsed <- proc.time()-start
  write_csv(data.frame(elapsed_seconds=unname(elapsed['elapsed']),cpu_seconds=unname(sum(elapsed[c('user.self','sys.self')]))),file.path(out,'population_timing.csv'))
  do.call(rbind,result)
}

dwls_replicate <- function(cell,replicate,seed_base,population) {
  seed <- seed_base+10000L*cell$cell_id+replicate
  start <- proc.time()
  spectrum <- numeric(); rows <- list(); gap <- NA_real_; error <- ''; h1_converged <- h0_converged <- NA
  add <- function(arm,p=NA_real_,statistic=NA_real_,df=NA_integer_,spectrum_size=NA_integer_,
                  target='',covered=NA,estimate=NA_real_,se=NA_real_,reason='available') {
    rows[[length(rows)+1L]] <<- data.frame(arm=arm,p=p,statistic=statistic,df=df,
      spectrum_size=spectrum_size,target=target,covered=covered,estimate=estimate,se=se,reason=reason)
  }
  tryCatch({
    d <- dwls_draw_data(cell,seed)
    equal1 <- if(cell$family=='nested' && cell$nesting=='thresholds') 'thresholds' else NULL
    h1 <- dwls_fit(cell,d,equal1); h1_converged <- h1$converged
    if(!isTRUE(h1$converged)) stop('H1 did not converge')
    core <- magmaanlab::magmaan_core
    if(cell$family=='nested') {
      equal0 <- if(cell$nesting=='metric') 'loadings' else c('thresholds','loadings')
      h0 <- dwls_fit(cell,d,equal0); h0_converged <- h0$converged
      if(!isTRUE(h0$converged)) stop('H0 did not converge')
      policy <- magmaanlab::policy_nested(h1,h0)$lr
      if(!isTRUE(policy$available)) stop('Policy unavailable: ',policy$reason,': ',policy$detail)
      explicit <- core$ordinal_profile_lrt(h1,h0,h1$ordinal_stats)
      e <- explicit$eigvals; e <- e[e>1e-8*max(e)]
      e <- sort(c(rep(0,max(0,explicit$df_diff-length(e))),e))
      cal <- dwls_calibrate(max(0,explicit$T_diff),explicit$df_diff,e)
      cal$p_peba4 <- core$robust_fmg_test(max(0,explicit$T_diff),length(e),e,'peba',4)$p_value
      gap <- max(abs(c(policy$statistic-explicit$T_diff,policy$p_sb-cal$p_sb,
                        policy$p_peba4-cal$p_peba4)))
      for(a in c('sb','peba4')) add(paste0('policy_',a),policy[[paste0('p_',a)]],
        policy$statistic/policy$sb_scale,policy$df,length(policy$eigenvalues))
      fixed <- magmaanlab::robust_nested_lrt(h1,h0,data=h1$ordinal_stats,
        gamma='empirical',method='restriction_map',A.method='exact',weight='DWLS')
      cal <- dwls_calibrate(fixed$T_diff,fixed$df_diff,fixed$eigenvalues)
      for(a in c('sb','peba4')) add(paste0('fixed_',a),cal[[paste0('p_',a)]],
        fixed$T_diff/mean(fixed$eigenvalues),fixed$df_diff,length(fixed$eigenvalues))
    } else {
      policy <- magmaanlab::policy_inference(h1)
      if(cell$family=='coverage') {
        if(!isTRUE(policy$covariance_available)) stop('Policy covariance unavailable: ',policy$covariance_reason)
        ij <- core$robust_ordinal_ij(h1,h1$ordinal_stats)$vcov
        gap <- max(abs(policy$covariance-ij))
        covariances <- list(policy_ij=policy$covariance,
          expected=core$robust_ordinal(h1,h1$ordinal_stats,bread='expected')$vcov,
          observed=core$robust_ordinal(h1,h1$ordinal_stats,bread='observed')$vcov)
        truth <- population[population$key==dwls_population_key(cell),]
        for(a in names(covariances)) {
          z <- dwls_targets(h1,covariances[[a]])
          for(i in seq_len(nrow(z))) {
            v <- truth$estimate[match(z$target[i],truth$target)]
            if(!is.finite(v)) stop('Population target missing')
            add(a,target=z$target[i],covered=abs(z$estimate[i]-v)<=qnorm(.975)*z$se[i],
              estimate=z$estimate[i],se=z$se[i])
          }
        }
      } else {
        policy <- policy$score
        if(!isTRUE(policy$available)) stop('Policy global unavailable: ',policy$reason,': ',policy$detail)
        fixed <- core$robust_ordinal(h1,h1$ordinal_stats,bread='expected')
        spectrum <- policy$eigenvalues
        cal <- dwls_calibrate(fixed$chisq_standard,policy$df,spectrum)
        gap <- max(abs(c(policy$statistic-fixed$chisq_standard,policy$p_sb-cal$p_sb,
                          policy$p_peba4-cal$p_peba4)))
        for(a in c('sb','peba4')) add(paste0('policy_',a),policy[[paste0('p_',a)]],
          policy$statistic/policy$sb_scale,policy$df,length(policy$eigenvalues))
        ss <- fixed$scaled_shifted; mv <- fixed$mean_var_adjusted
        add('scaled_shifted',pchisq(ss$chi2_adj,ss$df,lower.tail=FALSE),ss$chi2_adj,ss$df)
        add('mean_variance',pchisq(mv$chi2_adj,mv$df_adj,lower.tail=FALSE),mv$chi2_adj,mv$df_adj)
        methods <- c(scaled_f='scaled_f',all='all',pall='penalized_all',
          eba2='eba',eba4='eba',eba6='eba',peba2='peba',peba6='peba',pols='pols')
        for(a in names(methods)) {
          # pOLS uses the bound primitive's default gamma = 4 explicitly.
          param <- if(grepl('eba',a)) as.numeric(sub('.*eba','',a)) else 4
          test <- core$robust_fmg_test(fixed$chisq_standard,policy$df,spectrum,
            methods[[a]],param,truncate_negative=TRUE)
          if(!is.finite(test$p_value)) stop('Nonfinite FMG arm: ',a)
          add(a,test$p_value,fixed$chisq_standard,policy$df,length(spectrum))
        }
      }
    }
    if(!is.finite(gap) || gap>1e-7) stop('Policy gap exceeds 1e-7')
  },error=function(e) { error <<- conditionMessage(e) })
  if(!length(rows)) add('failure',reason=error)
  z <- do.call(rbind,rows); z$cell_id <- cell$cell_id; z$replicate <- replicate; z$seed <- seed
  z$h1_converged <- h1_converged; z$h0_converged <- h0_converged
  z$policy_gap <- gap; z$error <- error; z$elapsed_seconds <- unname((proc.time()-start)['elapsed'])
  z$eigenvalues <- rep(list(spectrum),nrow(z))
  z$cpu_seconds <- unname(sum((proc.time()-start)[c('user.self','sys.self')]))
  z
}
