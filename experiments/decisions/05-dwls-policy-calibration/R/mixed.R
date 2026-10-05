mixed_seed_base <- function(mode) {
  c(smoke=1217000001L,pilot=1417000001L,production=1617000001L)[[mode]]
}
mixed_ordered <- function() paste0('x',c(4:6,10:12))
mixed_cells <- function() {
  base <- function(family,n,groups=1,parameterization='delta',cross=0,
                   nesting='none',role='null',design=1:4) {
    x <- expand.grid(design=design,n=n,groups=groups,parameterization=parameterization,
      cross=cross,nesting=nesting,role=role,stringsAsFactors=FALSE)
    x$family <- family; x
  }
  ns <- c(300,500,1000,2000)
  x <- rbind(base('global',ns),base('global',ns,2,c('delta','theta')),
    base('nested',ns,2,c('delta','theta'),c(0,.3),'metric'),
    base('nested',ns,2,'theta',c(0,.3),'moment',design=3:4),
    base('coverage',c(300,1000),cross=c(0,.2,.4)),
    base('global',ns,cross=.3,role='power',design=3),
    base('nested',ns,2,'theta',nesting='metric',role='power',design=3))
  x$categories <- ifelse(x$design<=2,2,5)
  x$skew <- ifelse(x$design %% 2==1,'symmetric','skewed')
  x$factors <- 2L; x$model <- 'cfa'; x$cell_id <- seq_len(nrow(x))
  x$production_reps <- ifelse(x$role=='power',1000L,2000L)
  x
}
mixed_draw <- function(cell,seed,n=cell$n) {
  # Reuse the Gaussian latent-response generator, changing only thresholding.
  set.seed(seed); p <- 12L; phi <- matrix(c(1,.3,.3,1),2)
  loading <- rep(c(.65,.7,.75,.8,.7,.75),2)
  cuts <- if(cell$categories==2) qnorm(if(cell$skew=='skewed') .85 else .5) else
    qnorm(if(cell$skew=='skewed') c(.45,.70,.85,.95) else c(.2,.4,.6,.8))
  do.call(rbind,lapply(seq_len(cell$groups),function(g) {
    lambda <- matrix(0,p,2); lambda[cbind(1:p,rep(1:2,each=6))] <- loading
    lambda[7,1] <- cell$cross
    if(cell$family=='nested' && cell$role=='power' && g==2) lambda[2,1] <- lambda[2,1]+.15
    common <- lambda%*%phi%*%t(lambda)
    sigma <- common+diag(1-diag(common))
    z <- matrix(rnorm(as.integer(n/cell$groups)*p),ncol=p)%*%chol(sigma)
    d <- as.data.frame(z); names(d) <- paste0('x',1:p)
    for(v in mixed_ordered()) d[[v]] <- as.integer(cut(d[[v]],c(-Inf,cuts,Inf)))
    if(cell$groups==2) d$group <- c('a','b')[g]
    d
  }))
}
mixed_fit <- function(cell,data,equal=NULL) {
  spec <- magmaanlab::model_spec(dwls_syntax(cell),ordered=mixed_ordered(),
    parameterization=cell$parameterization,group=if(cell$groups==2) 'group' else NULL,
    group_labels=if(cell$groups==2) c('a','b') else NULL,group_equal=equal)
  magmaanlab::fit_model(spec,data,estimator='DWLS')
}
mixed_lavaan <- function(cell,data,equal=NULL) {
  lavaan::cfa(dwls_syntax(cell),data=data,ordered=mixed_ordered(),
    estimator='WLSMV',parameterization=cell$parameterization,
    group=if(cell$groups==2) 'group' else NULL,group.equal=if(is.null(equal)) character() else equal)
}
mixed_targets <- function(pt,theta,covariance=NULL) {
  row <- function(lhs,op,rhs='') {
    i <- which(pt$lhs==lhs & pt$op==op & pt$rhs==rhs & pt$group==1)
    if(length(i)!=1 || pt$free[i]<1) stop('Missing free target ',lhs,op,rhs)
    pt$free[i]
  }
  ids <- c(continuous_loading=row('f1','=~','x2'),ordinal_loading=row('f1','=~','x5'),
    threshold=row('x5','|','t1'),intercept=row('x2','~1'))
  gradients <- lapply(ids,function(i) replace(numeric(length(theta)),i,1))
  values <- theta[ids]
  i <- row('f1','~~','f2'); a <- row('f1','~~','f1'); b <- row('f2','~~','f2')
  v <- theta[i]/sqrt(theta[a]*theta[b]); g <- numeric(length(theta))
  g[i] <- 1/sqrt(theta[a]*theta[b]); g[a] <- -v/(2*theta[a]); g[b] <- -v/(2*theta[b])
  gradients$correlation <- g; values <- c(values,correlation=v)
  data.frame(target=names(gradients),estimate=as.numeric(values),
    se=vapply(gradients,function(g) if(is.null(covariance)) NA_real_ else
      sqrt(as.numeric(crossprod(g,covariance%*%g))),numeric(1)))
}
mixed_population_key <- function(cell) paste(cell$design,cell$cross,sep='_')
mixed_population <- function(cells,out) {
  coverage <- mixed_cells(); coverage <- coverage[coverage$family=='coverage',]
  keys <- unique(vapply(seq_len(nrow(coverage)),function(i) mixed_population_key(coverage[i,]),''))
  need <- unique(vapply(which(cells$family=='coverage'),function(i) mixed_population_key(cells[i,]),''))
  if(!length(need)) return(NULL)
  result <- list(); start <- proc.time()
  for(key in need) {
    cell <- coverage[match(key,vapply(seq_len(nrow(coverage)),function(i) mixed_population_key(coverage[i,]),'')),]
    seed <- 1017000001L+10000L*match(key,keys)
    cat('Population ',key,'\n'); flush.console()
    fit <- mixed_fit(cell,mixed_draw(cell,seed,n=100000L))
    if(!isTRUE(fit$converged)) stop('Population fit did not converge: ',key)
    z <- mixed_targets(fit$partable,fit$theta); z$key <- key; z$seed <- seed; z$n_per_group <- 100000L
    result[[key]] <- z
    write_csv(do.call(rbind,result),file.path(out,'population_targets.csv'))
  }
  write_csv(data.frame(elapsed_seconds=unname((proc.time()-start)['elapsed']),
    cpu_seconds=unname(sum((proc.time()-start)[c('user.self','sys.self')]))),file.path(out,'population_timing.csv'))
  do.call(rbind,result)
}
mixed_arms <- function(cell) if(cell$family=='coverage') c('policy_ij','lavaan_robust') else
  if(cell$family=='global') c('policy_all','policy_sb','policy_peba4','lavaan_wlsmv') else
    c('policy_sb','policy_peba4','all','lavaan_lrt')
mixed_replicate <- function(cell,replicate,seed_base,population) {
  seed <- seed_base+10000L*cell$cell_id+replicate; start <- proc.time()
  rows <- list(); h1_converged <- h0_converged <- NA; gap <- NA_real_
  add <- function(arm,p=NA_real_,target='',covered=NA,estimate=NA_real_,se=NA_real_,
                  statistic=NA_real_,df=NA_real_,spectrum_size=NA_integer_,reason='available') {
    rows[[length(rows)+1L]] <<- data.frame(arm,p,target,covered,estimate,se,statistic,df,spectrum_size,reason)
  }
  d <- mixed_draw(cell,seed)
  equal0 <- if(cell$family=='nested') if(cell$nesting=='metric') 'loadings' else 'thresholds' else NULL
  policy_error <- ''
  tryCatch({
    h1 <- mixed_fit(cell,d); h1_converged <- h1$converged
    if(!isTRUE(h1$converged)) stop('H1 did not converge')
    if(cell$family=='nested') {
      h0 <- mixed_fit(cell,d,equal0); h0_converged <- h0$converged
      if(!isTRUE(h0$converged)) stop('H0 did not converge')
      test <- magmaanlab::policy_nested(h1,h0)$lr
      if(!isTRUE(test$available)) stop('Policy unavailable: ',test$reason,': ',test$detail)
      cal <- dwls_calibrate(test$statistic,test$df,test$eigenvalues)
      gap <- max(abs(c(test$p_sb-cal$p_sb,test$p_peba4-cal$p_peba4)))
      for(a in c('sb','peba4')) add(paste0('policy_',a),test[[paste0('p_',a)]],
        statistic=test$statistic,df=test$df,spectrum_size=length(test$eigenvalues))
      # Merger amendment 2026-10-05: All on the same nested spectrum, a
      # comparator for the TASK-81 reference decision (not rule-gating).
      all <- magmaanlab::magmaan_core$robust_fmg_test(test$statistic,test$df,test$eigenvalues,'all',0,truncate_negative=TRUE)
      add('all',all$p_value,statistic=test$statistic,df=test$df,spectrum_size=length(test$eigenvalues))
    } else {
      policy <- magmaanlab::policy_inference(h1)
      if(cell$family=='coverage') {
        if(!isTRUE(policy$covariance_available)) stop('Policy covariance unavailable: ',policy$covariance_detail)
        ij <- magmaanlab::magmaan_core$robust_mixed_ordinal_ij(h1,h1$mixed_ordinal_stats)$vcov
        gap <- max(abs(policy$covariance-ij))
        z <- mixed_targets(h1$partable,h1$theta,policy$covariance)
        truth <- population[population$key==mixed_population_key(cell),]
        for(i in seq_len(nrow(z))) add('policy_ij',target=z$target[i],
          covered=abs(z$estimate[i]-truth$estimate[match(z$target[i],truth$target)])<=qnorm(.975)*z$se[i],
          estimate=z$estimate[i],se=z$se[i])
      } else {
        test <- policy$score
        if(!isTRUE(test$available)) stop('Policy unavailable: ',test$reason,': ',test$detail)
        all <- magmaanlab::magmaan_core$robust_fmg_test(test$statistic,test$df,test$eigenvalues,'all',0,truncate_negative=TRUE)
        gap <- abs(test$p_all-all$p_value)
        cal <- dwls_calibrate(test$statistic,test$df,test$eigenvalues)
        add('policy_all',test$p_all,statistic=test$statistic,df=test$df,spectrum_size=length(test$eigenvalues))
        for(a in c('sb','peba4')) add(paste0('policy_',a),cal[[paste0('p_',a)]],
          statistic=test$statistic,df=test$df,spectrum_size=length(test$eigenvalues))
      }
    }
  },error=function(e) policy_error <<- conditionMessage(e))
  comparator_error <- ''
  tryCatch({
    l1 <- suppressWarnings(mixed_lavaan(cell,d))
    if(!isTRUE(lavaan::lavInspect(l1,'converged'))) stop('lavaan H1 did not converge')
    if(cell$family=='nested') {
      l0 <- suppressWarnings(mixed_lavaan(cell,d,equal0))
      if(!isTRUE(lavaan::lavInspect(l0,'converged'))) stop('lavaan H0 did not converge')
      tab <- suppressWarnings(lavaan::lavTestLRT(l1,l0))
      add('lavaan_lrt',tail(tab[['Pr(>Chisq)']],1),statistic=tail(tab[['Chisq diff']],1),df=tail(tab[['Df diff']],1))
    } else if(cell$family=='global') {
      fm <- lavaan::fitMeasures(l1,c('pvalue.scaled','chisq.scaled','df.scaled'))
      add('lavaan_wlsmv',fm[1],statistic=fm[2],df=fm[3])
    } else {
      pt <- lavaan::parTable(l1)
      z <- mixed_targets(pt,lavaan::coef(l1),lavaan::vcov(l1))
      truth <- population[population$key==mixed_population_key(cell),]
      for(i in seq_len(nrow(z))) add('lavaan_robust',target=z$target[i],
        covered=abs(z$estimate[i]-truth$estimate[match(z$target[i],truth$target)])<=qnorm(.975)*z$se[i],
        estimate=z$estimate[i],se=z$se[i])
    }
  },error=function(e) comparator_error <<- conditionMessage(e))
  targets <- if(cell$family=='coverage') c('continuous_loading','ordinal_loading','threshold','intercept','correlation') else ''
  for(arm in mixed_arms(cell)) for(target in targets) {
    if(!any(vapply(rows,function(x) x$arm==arm && x$target==target,logical(1))))
      add(arm,target=target,reason=if(startsWith(arm,'policy') || arm=='all') policy_error else comparator_error)
  }
  z <- do.call(rbind,rows); z$cell_id <- cell$cell_id; z$replicate <- replicate; z$seed <- seed
  z$h1_converged <- h1_converged; z$h0_converged <- h0_converged; z$policy_gap <- gap
  z$error <- ifelse(z$reason=='available','',z$reason)
  z$elapsed_seconds <- unname((proc.time()-start)['elapsed'])
  z$cpu_seconds <- unname(sum((proc.time()-start)[c('user.self','sys.self')]))
  z$comparator_judge <- 'lavaan convergence only; mixed evaluate_at adapter unavailable'
  z
}
mixed_summarize <- function(raw,cells,out) {
  attempts <- raw[!duplicated(raw[c('cell_id','replicate')]),]
  timing <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
    x <- attempts[attempts$cell_id==cells$cell_id[i],]
    data.frame(cell_id=cells$cell_id[i],attempted=nrow(x),mean_seconds=mean(x$elapsed_seconds),
      mean_cpu_seconds=mean(x$cpu_seconds),production_reps=cells$production_reps[i],
      production_cpu_hours=mean(x$cpu_seconds)*cells$production_reps[i]/3600)
  }))
  summary <- list()
  for(i in seq_len(nrow(cells))) {
    cell <- cells[i,]; total <- timing$attempted[i]
    targets <- if(cell$family=='coverage') c('continuous_loading','ordinal_loading','threshold','intercept','correlation') else ''
    for(arm in mixed_arms(cell)) for(target in targets) {
      x <- raw[raw$cell_id==cell$cell_id & raw$arm==arm & raw$target==target,]
      valid <- !nzchar(x$error) & if(target=='') is.finite(x$p) else !is.na(x$covered)
      k <- if(target=='') sum(x$p[valid]<.05) else sum(x$covered[valid]); n <- sum(valid)
      rate <- if(total) k/total else NA_real_; ci <- dwls_wilson(k,total)
      adjusted <- NA_real_; null_count <- 0L
      if(cell$role=='power') {
        matched <- cells$role=='null' & cells$cross==0
        for(col in c('family','design','n','groups','parameterization','nesting')) matched <- matched & cells[[col]]==cell[[col]]
        nx <- raw[raw$cell_id %in% cells$cell_id[matched] & raw$arm==arm & !nzchar(raw$error) & is.finite(raw$p),]
        null_count <- nrow(nx)
        if(n && null_count) adjusted <- sum(x$p[valid]<quantile(nx$p,.05,type=8))/total
      }
      flag <- cell$role=='null' && startsWith(arm,'policy') && is.finite(rate) &&
        if(target=='') cell$n>=500 && (rate<.03 || rate>.07) else cell$n>=300 && rate<.93
      summary[[length(summary)+1L]] <- data.frame(cell_id=cell$cell_id,arm,target,
        attempted=total,successful=n,failed=total-n,successes=k,rate,
        successful_rate=if(n) k/n else NA_real_,wilson_low=ci[1],wilson_high=ci[2],
        size_adjusted_power=adjusted,matched_null_successful=null_count,finite_sample_followup=flag,
        unavailable=n==0,df=if(nrow(x)) x$df[1] else NA,spectrum_size=if(nrow(x)) x$spectrum_size[1] else NA)
    }
  }
  write_csv(do.call(rbind,summary),file.path(out,'summary.csv'))
  write_csv(timing,file.path(out,'timing.csv'))
  write_csv(raw[nzchar(raw$error),setdiff(names(raw),c('estimate','se','covered','p'))],file.path(out,'failures.csv'))
  write_csv(data.frame(priced_cells=sum(timing$attempted>0),total_cells=nrow(cells),
    production_cpu_hours_priced=sum(timing$production_cpu_hours,na.rm=TRUE),
    complete_grid_price=all(timing$attempted>0)),file.path(out,'cost.csv'))
}
mixed_run <- function(args,mode,workers,here,opt) {
  if(!mode %in% c('smoke','pilot','production')) stop('Mixed lane supports smoke, pilot, production')
  run_id <- opt('--run-id',paste0('mixed-',mode))
  if(!grepl('^[a-zA-Z0-9_-]+$',run_id)) stop('Invalid run ID')
  out <- opt('--out-dir',file.path(here,'results','mixed',run_id))
  if(dir.exists(out)) stop('Run exists; choose a fresh run ID')
  cells <- mixed_cells(); family <- opt('--family','all')
  if(!family %in% c('all','global','nested')) stop('Unknown family')
  if(family!='all') cells <- cells[cells$family==family,]
  only <- as.integer(strsplit(opt('--cell',''),',',fixed=TRUE)[[1]])
  if(length(only)) {
    if(any(!only %in% cells$cell_id)) stop('Unknown cell ID')
    cells <- cells[cells$cell_id %in% only,]
  }
  reps <- as.integer(opt('--reps',if(mode=='smoke') '1' else if(mode=='pilot') '2' else NA_character_))
  if(!is.na(reps) && reps<1) stop('reps must be positive')
  dir.create(out,recursive=TRUE)
  binary <- list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE)
  files <- c(file.path(here,'run_experiment.R'),sort(list.files(file.path(here,'R'),full.names=TRUE)),
    file.path(here,'criteria','mixed_policy.md'),binary)
  write_metadata(file.path(out,'metadata.csv'),list(lane='mixed',mode=mode,workers=workers,
    seed_base=mixed_seed_base(mode),family=family,selected_cells=paste(cells$cell_id,collapse=','),
    source_hashes=paste(tools::md5sum(files),collapse=','),
    native_md5=paste(tools::md5sum(binary),collapse=','),
    git_head=git_scalar(c('rev-parse','HEAD')),git_dirty=git_dirty(),reps_override=reps,
    population_n_per_group=100000L,population_seed_base=1017000001L,
    comparator_judge='lavaan convergence only; mixed evaluate_at adapter unavailable'),packages=c('magmaanlab','lavaan'))
  write_csv(cells,file.path(out,'cells.csv'))
  population <- mixed_population(cells,out)
  jobs <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i)
    data.frame(cell_id=cells$cell_id[i],replicate=seq_len(if(is.na(reps)) cells$production_reps[i] else reps))))
  rows <- list(); start <- proc.time()[['elapsed']]
  for(first in seq(1L,nrow(jobs),by=4L)) {
    indices <- first:min(first+3L,nrow(jobs))
    batch <- parallel::mclapply(indices,function(j) {
      set_single_threaded_math()
      mixed_replicate(cells[match(jobs$cell_id[j],cells$cell_id),],jobs$replicate[j],mixed_seed_base(mode),population)
    },mc.cores=workers,mc.preschedule=TRUE)
    if(any(vapply(batch,inherits,logical(1),'try-error'))) stop('Worker process failed')
    rows <- c(rows,batch); raw <- do.call(rbind,rows)
    saveRDS(raw,file.path(out,'raw.rds'))
    elapsed <- proc.time()[['elapsed']]-start
    write_csv(data.frame(completed=max(indices),total=nrow(jobs),elapsed_seconds=elapsed),file.path(out,'progress.csv'))
    cat('Mixed completed ',max(indices),'/',nrow(jobs),' in ',round(elapsed,1),'s\n',sep=''); flush.console()
  }
  mixed_summarize(raw,cells,out)
  if(any(is.finite(raw$policy_gap) & raw$policy_gap>1e-7)) stop('Policy equivalence gate failed')
  # Completion marker is written only after all draws and gates; raw.rds alone
  # is a checkpoint, never evidence that a preempted cell is complete.
  writeLines('complete',file.path(out,'COMPLETE'))
  cat('Mixed results: ',out,'\n',sep='')
}
