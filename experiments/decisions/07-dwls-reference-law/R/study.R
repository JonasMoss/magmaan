reference_cells <- function() {
  x <- expand.grid(model=c('ptsd8','mtmm9','coping12','bifactor15','long18','long30','worland11','mdd9'),
    categories=c(2L,5L,7L),shape=c('symmetric','skewed'),n=c(250L,500L,1000L,2000L),stringsAsFactors=FALSE)
  x$cell_id <- seq_len(nrow(x)); x
}
reference_replicate <- function(cell,replicate,base,here) {
  seed <- base+10000*cell$cell_id+replicate; set.seed(seed); start <- proc.time()
  pop <- jsonlite::fromJSON(file.path(here,'populations',paste0(cell$model,'.json')))
  cuts <- if(cell$shape=='symmetric') qnorm(seq_len(cell$categories-1)/cell$categories) else
    qnorm(cumsum(switch(as.character(cell$categories),'2'=c(.85,.15),'5'=c(.45,.25,.15,.10,.05),'7'=c(.35,.25,.15,.10,.07,.05,.03)))[seq_len(cell$categories-1)])
  data <- do.call(rbind,lapply(seq_len(pop$groups),function(g) {
    y <- matrix(rnorm(cell$n/pop$groups*length(pop$observed)),cell$n/pop$groups)%*%chol(pop$correlation)
    d <- as.data.frame(lapply(seq_along(pop$observed),function(j) ordered(findInterval(y[,j],cuts)+1,levels=seq_len(cell$categories))))
    names(d) <- pop$observed; if(pop$groups>1) d$group <- factor(g,levels=seq_len(pop$groups)); d
  }))
  fit <- function(syntax,equal=NULL) {
    spec <- magmaanlab::model_spec(syntax,ordered=pop$observed,parameterization='theta',
      group=if(pop$groups>1) 'group' else NULL,group_labels=if(pop$groups>1) c('1','2') else NULL,group_equal=equal)
    magmaanlab::fit_model(spec,data,estimator='DWLS')
  }
  tests <- if(pop$groups>1) c('global','metric','thresholds') else c('global','nested')
  output <- list(); h1 <- tryCatch(fit(pop$h1),error=function(e)e)
  for(test in tests) {
    conv1 <- if(inherits(h1,'error')) NA else h1$converged; conv0 <- NA
    z <- tryCatch({
      if(inherits(h1,'error')) stop(conditionMessage(h1))
      if(!isTRUE(conv1)) stop('h1_not_converged')
      if(test=='global') magmaanlab::policy_inference(h1)$score else {
        h0 <- if(test=='nested') fit(pop$h0) else fit(pop$h1,if(test=='metric') 'loadings' else 'thresholds')
        conv0 <- h0$converged; if(!isTRUE(conv0)) stop('h0_not_converged')
        magmaanlab::policy_nested(h1,h0)$lr
      }
    },error=function(e) list(available=FALSE,reason='error',detail=conditionMessage(e)))
    output[[test]] <- list(cell_id=cell$cell_id,replicate=replicate,seed=seed,test=test,
      h1_converged=conv1,h0_converged=conv0,available=isTRUE(z$available),
      reason=if(isTRUE(z$available)) 'available' else z$reason %||% 'unavailable',detail=z$detail %||% '',
      statistic=z$statistic %||% NA_real_,df=z$df %||% NA_real_,eigenvalues=z$eigenvalues %||% numeric())
  }
  elapsed <- proc.time()-start
  for(test in tests) { output[[test]]$elapsed_seconds <- unname(elapsed['elapsed']); output[[test]]$cpu_seconds <- unname(sum(elapsed[c('user.self','sys.self')])) }
  output
}
reference_pvalues <- function(z) {
  methods <- c(All='all',SB='sb',PEBA4='peba',scaled_shifted='ss',mean_variance='mv',scaled_F='scaled_f',
    EBA2='eba',EBA4='eba',EBA6='eba',pEBA2='peba',pEBA4='peba',pEBA6='peba',pOLS='pols',pAll='penalized_all')
  vapply(names(methods),function(a) {
    if(!z$available) return(NA_real_)
    parameter <- if(grepl('EBA',a)) as.numeric(sub('.*EBA','',a)) else 4
    tryCatch(magmaanlab::magmaan_core$robust_fmg_test(z$statistic,z$df,z$eigenvalues,
      methods[[a]],parameter,truncate_negative=TRUE)$p_value,error=function(e) NA_real_)
  },0)
}
reference_summarize <- function(raw,cells,out) {
  rows <- lapply(raw,function(z) {
    p <- reference_pvalues(z)
    data.frame(cell_id=z$cell_id,replicate=z$replicate,seed=z$seed,test=z$test,reference=names(p),p=unname(p),
      reason=ifelse(z$available & !is.finite(p),'reference_numeric_failure',z$reason),detail=z$detail,
      statistic=z$statistic,df=z$df,spectrum_size=length(z$eigenvalues),h1_converged=z$h1_converged,h0_converged=z$h0_converged)
  })
  rows <- do.call(rbind,rows); saveRDS(rows,file.path(out,'pvalues.rds'))
  groups <- split(rows,interaction(rows$cell_id,rows$test,rows$reference,drop=TRUE))
  summary <- do.call(rbind,lapply(groups,function(x) {
    valid <- is.finite(x$p); n <- sum(valid); k <- sum(x$p[valid]<.05); rate <- if(n) k/n else NA_real_
    den <- 1+qnorm(.975)^2/max(1,n); center <- (rate+qnorm(.975)^2/(2*max(1,n)))/den
    half <- qnorm(.975)*sqrt(rate*(1-rate)/max(1,n)+qnorm(.975)^2/(4*max(1,n)^2))/den
    data.frame(cell_id=x$cell_id[1],test=x$test[1],reference=x$reference[1],attempted=nrow(x),available=n,failed=nrow(x)-n,
      rejected=k,rate,wilson_low=center-half,wilson_high=center+half,attempted_rate=k/nrow(x))
  }))
  # Joint availability is required for ranking; marginal rates remain in summary.
  joint <- aggregate(is.finite(rows$p),rows[c('cell_id','replicate','test')],all); names(joint)[4] <- 'joint'
  rows <- merge(rows,joint); paired <- rows[rows$joint,]
  summary$joint_available <- 0L; summary$joint_rate <- NA_real_
  for(i in seq_len(nrow(summary))) {
    x <- paired[paired$cell_id==summary$cell_id[i] & paired$test==summary$test[i] & paired$reference==summary$reference[i],]
    summary$joint_available[i] <- nrow(x); if(nrow(x)) summary$joint_rate[i] <- mean(x$p<.05)
  }
  summary <- merge(summary,cells); summary$registered_flag <- summary$n>=500 & is.finite(summary$joint_rate) & (summary$joint_rate<.03|summary$joint_rate>.07)
  write_csv(summary,file.path(out,'summary.csv'))
  write_csv(rows[rows$reason!='available',],file.path(out,'failures.csv'))
  ranking <- do.call(rbind,lapply(c(unique(summary$model),'overall'),function(model) {
    x <- if(model=='overall') summary else summary[summary$model==model,]
    x$sample_band <- ifelse(x$n>=500,'N>=500','N<500')
    do.call(rbind,lapply(split(x,interaction(x$test,x$reference,x$sample_band,drop=TRUE)),function(a)
      data.frame(model,test=a$test[1],reference=a$reference[1],sample_band=a$sample_band[1],cells=nrow(a),
        unavailable_cells=sum(!is.finite(a$joint_rate)),outside_band=sum(a$registered_flag),rmse=sqrt(mean((a$joint_rate-.05)^2,na.rm=TRUE)))))
  }))
  write_csv(ranking,file.path(out,'ranking.csv'))
  timing <- do.call(rbind,lapply(split(raw,vapply(raw,function(z) as.character(z$cell_id),'')),function(x) {
    x <- x[!duplicated(vapply(x,function(z) z$replicate,0))]
    data.frame(cell_id=x[[1]]$cell_id,attempted=length(x),mean_seconds=mean(vapply(x,function(z) z$elapsed_seconds,0)),mean_cpu_seconds=mean(vapply(x,function(z) z$cpu_seconds,0)))
  }))
  write_csv(merge(timing,cells),file.path(out,'timing.csv'))
  write_csv(data.frame(full_cells=nrow(reference_cells()),observed_cells=nrow(cells),unpriced_cells=nrow(reference_cells())-nrow(cells),
    measured_cells_cpu_hours=sum(timing$mean_cpu_seconds)*1000/3600,full_grid_cpu_hours=if(nrow(cells)==192) sum(timing$mean_cpu_seconds)*1000/3600 else NA_real_),file.path(out,'cost.csv'))
}
