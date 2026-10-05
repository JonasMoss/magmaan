exact_seed_base <- function(mode) {
  c(smoke=317000001L,pilot=517000001L,production=1817000001L)[[mode]]
}
exact_cells <- function() dwls_cells()
exact_validate_seeds <- function() {
  fresh <- c(117000001,317000001,517000001,1817000001)
  old <- c(817120001,817130001,817140001,817150001,817160001,
           1017000001,1217000001,1417000001,1617000001)
  stopifnot(all(abs(outer(fresh,old,'-'))>=100000000),
            all(as.matrix(dist(fresh))[upper.tri(diag(4))]>=100000000),
            max(fresh)+10000*146+2000 < .Machine$integer.max)
}
exact_source_files <- function(here) {
  binary <- list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE)
  c(file.path(here,'run_experiment.R'),sort(list.files(file.path(here,'R'),full.names=TRUE)),
    file.path(here,'criteria','exact_first_stage.md'),binary)
}
exact_population <- function(cells,out) {
  full <- exact_cells(); full <- full[full$family=='coverage',]
  keys <- unique(vapply(seq_len(nrow(full)),function(i) dwls_population_key(full[i,]),''))
  need <- unique(vapply(which(cells$family=='coverage'),function(i) dwls_population_key(cells[i,]),''))
  if(!length(need)) return(NULL)
  result <- list(); start <- proc.time()
  for(key in need) {
    cell <- full[match(key,vapply(seq_len(nrow(full)),function(i) dwls_population_key(full[i,]),'')),]
    seed <- 117000001L+10000L*match(key,keys)
    cat('Population ',key,'\n'); flush.console()
    fit <- dwls_fit(cell,dwls_draw_data(cell,seed,n=100000L*cell$groups))
    if(!isTRUE(fit$converged)) stop('Population fit did not converge: ',key)
    z <- dwls_targets(fit); z$key <- key; z$seed <- seed; z$n_per_group <- 100000L
    result[[key]] <- z
    write_csv(do.call(rbind,result),file.path(out,'population_targets.csv'))
  }
  write_csv(data.frame(elapsed_seconds=unname((proc.time()-start)['elapsed']),
    cpu_seconds=unname(sum((proc.time()-start)[c('user.self','sys.self')]))),file.path(out,'population_timing.csv'))
  do.call(rbind,result)
}
exact_arms <- function(cell) if(cell$family=='coverage') c('policy_ij','opg_ij') else
  if(cell$family=='global') c('policy_all','opg_all') else
    c('policy_sb','policy_peba4','exact_all','opg_sb','opg_peba4','opg_all')
exact_nested_spectrum <- function(fit,parts,vcov) {
  # H is per-observation observed curvature in active K coordinates; N*vcov
  # is H^-1 B H^-1. Restricting it by A yields exactly r spectrum terms.
  K <- parts$K; A <- parts$A
  H <- crossprod(K,parts$hessian_total%*%K)/fit$ntotal
  L <- solve(crossprod(K),t(K))
  V <- fit$ntotal*L%*%vcov%*%t(L)
  C <- A%*%solve(H,t(A)); S <- A%*%V%*%t(A)
  Ri <- solve(chol(C))
  sort(eigen(t(Ri)%*%S%*%Ri,symmetric=TRUE,only.values=TRUE)$values)
}
exact_replicate <- function(cell,replicate,seed_base,population) {
  seed <- seed_base+10000L*cell$cell_id+replicate; start <- proc.time()
  rows <- list(); h1_converged <- h0_converged <- NA; gap <- NA_real_
  spectra <- list(exact=numeric(),opg=numeric())
  add <- function(arm,p=NA_real_,target='',covered=NA,estimate=NA_real_,se=NA_real_,
                  statistic=NA_real_,df=NA_real_,spectrum_size=NA_integer_,reason='available') {
    if(reason=='available' && ((target=='' && !is.finite(p)) ||
      (target!='' && (!is.finite(se) || is.na(covered))))) reason <- 'numeric_failure'
    rows[[length(rows)+1L]] <<- data.frame(arm,p,target,covered,estimate,se,statistic,df,spectrum_size,reason)
  }
  h1 <- h0 <- NULL; fit_error <- ''
  tryCatch({
    d <- dwls_draw_data(cell,seed)
    equal1 <- if(cell$family=='nested' && cell$nesting=='thresholds') 'thresholds' else NULL
    h1 <- dwls_fit(cell,d,equal1); h1_converged <- h1$converged
    if(!isTRUE(h1$converged)) stop('H1 did not converge')
    if(cell$family=='nested') {
      equal0 <- if(cell$nesting=='metric') 'loadings' else c('thresholds','loadings')
      h0 <- dwls_fit(cell,d,equal0); h0_converged <- h0$converged
      if(!isTRUE(h0$converged)) stop('H0 did not converge')
    }
  },error=function(e) fit_error <<- conditionMessage(e))
  errors <- c(exact=fit_error,opg=fit_error)
  core <- magmaanlab::magmaan_core
  if(!nzchar(fit_error)) for(stage in c('exact','opg')) {
    tryCatch({
      if(cell$family=='coverage') {
        ij <- core$robust_ordinal_ij(h1,h1$ordinal_stats,first_stage=stage)$vcov
        if(stage=='exact') {
          policy <- magmaanlab::policy_inference(h1)
          if(!isTRUE(policy$covariance_available)) stop('Policy covariance unavailable: ',policy$covariance_reason)
          gap <- max(abs(policy$covariance-ij)); covariance <- policy$covariance
        } else covariance <- ij
        z <- dwls_targets(h1,covariance)
        truth <- population[population$key==dwls_population_key(cell),]
        for(i in seq_len(nrow(z))) {
          value <- truth$estimate[match(z$target[i],truth$target)]
          if(!is.finite(value)) stop('Population target missing')
          add(if(stage=='exact') 'policy_ij' else 'opg_ij',target=z$target[i],
            covered=abs(z$estimate[i]-value)<=qnorm(.975)*z$se[i],estimate=z$estimate[i],se=z$se[i])
        }
      } else if(cell$family=='global') {
        if(stage=='exact') {
          test <- magmaanlab::policy_inference(h1)$score
          if(!isTRUE(test$available)) stop('Policy unavailable: ',test$reason,': ',test$detail)
          spectra[[stage]] <- test$eigenvalues
          all <- core$robust_fmg_test(test$statistic,test$df,test$eigenvalues,'all',0,truncate_negative=TRUE)
          gap <- abs(test$p_all-all$p_value)
          add('policy_all',test$p_all,statistic=test$statistic,df=test$df,spectrum_size=length(test$eigenvalues))
        } else {
          test <- core$robust_ordinal(h1,h1$ordinal_stats,bread='expected')
          spectra[[stage]] <- test$eigvals
          all <- core$robust_fmg_test(test$chisq_standard,test$df,test$eigvals,'all',0,truncate_negative=TRUE)
          add('opg_all',all$p_value,statistic=test$chisq_standard,df=test$df,spectrum_size=length(test$eigvals))
        }
      } else {
        test <- magmaanlab::policy_nested(h1,h0)$lr
        if(!isTRUE(test$available)) stop('Policy unavailable: ',test$reason,': ',test$detail)
        parts <- magmaanlab:::ordinal_nested_diagnostic_impl(h1,h0)
        ij <- core$robust_ordinal_ij(h1,h1$ordinal_stats,first_stage=stage)$vcov
        spectrum <- exact_nested_spectrum(h1,parts,ij); spectra[[stage]] <- spectrum
        cal <- dwls_calibrate(test$statistic,test$df,spectrum)
        if(stage=='exact') {
          gap <- max(abs(c(test$p_sb-cal$p_sb,test$p_peba4-cal$p_peba4)),
            sqrt(sum((sort(test$eigenvalues)-spectrum)^2)/sum(spectrum^2)))
        }
        for(a in c('sb','peba4')) add(paste0(if(stage=='exact') 'policy_' else 'opg_',a),
          if(stage=='exact') test[[paste0('p_',a)]] else cal[[paste0('p_',a)]],
          statistic=test$statistic,df=test$df,spectrum_size=length(spectrum))
        all <- core$robust_fmg_test(test$statistic,test$df,spectrum,'all',0,truncate_negative=TRUE)
        add(paste0(stage,'_all'),all$p_value,statistic=test$statistic,df=test$df,spectrum_size=length(spectrum))
      }
    },error=function(e) errors[stage] <<- conditionMessage(e))
  }
  targets <- if(cell$family=='coverage') c('loading','threshold',if(cell$model=='sem') 'path' else 'correlation') else ''
  for(arm in exact_arms(cell)) for(target in targets) {
    stage <- if(startsWith(arm,'opg')) 'opg' else 'exact'
    if(!any(vapply(rows,function(x) x$arm==arm && x$target==target,logical(1))))
      add(arm,target=target,reason=if(nzchar(errors[stage])) errors[stage] else 'arm unavailable')
  }
  z <- do.call(rbind,rows); z$cell_id <- cell$cell_id; z$replicate <- replicate; z$seed <- seed
  z$h1_converged <- h1_converged; z$h0_converged <- h0_converged; z$policy_gap <- gap
  z$error <- ifelse(z$reason=='available','',z$reason)
  z$elapsed_seconds <- unname((proc.time()-start)['elapsed'])
  z$cpu_seconds <- unname(sum((proc.time()-start)[c('user.self','sys.self')]))
  z$exact_eigenvalues <- c(list(spectra$exact),rep(list(NULL),nrow(z)-1L))
  z$opg_eigenvalues <- c(list(spectra$opg),rep(list(NULL),nrow(z)-1L))
  z
}
exact_run <- function(args,mode,workers,here,opt) {
  if(!mode %in% c('smoke','pilot','production')) stop('Exact first-stage lane supports smoke, pilot, production')
  run_id <- opt('--run-id',paste0('exact-',mode))
  if(!grepl('^[a-zA-Z0-9_-]+$',run_id)) stop('Invalid run ID')
  out <- opt('--out-dir',file.path(here,'results','exact-first-stage',run_id))
  if(dir.exists(out)) stop('Run exists; choose a fresh run ID')
  cells <- exact_cells(); family <- opt('--family','all')
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
  files <- exact_source_files(here)
  if(any(!file.exists(files))) stop('Missing source fingerprint input')
  write_metadata(file.path(out,'metadata.csv'),list(lane='exact-first-stage',mode=mode,workers=workers,
    seed_base=exact_seed_base(mode),family=family,selected_cells=paste(cells$cell_id,collapse=','),
    source_hashes=paste(tools::md5sum(files),collapse=','),
    native_md5=paste(tools::md5sum(binary),collapse=','),
    git_head=git_scalar(c('rev-parse','HEAD')),git_dirty=git_dirty(),reps_override=reps,
    population_n_per_group=100000L,population_seed_base=117000001L,
    comparator_judge='same library fits and convergence for both arms'),packages=c('magmaanlab','lavaan'))
  write_csv(cells,file.path(out,'cells.csv'))
  exact_validate_seeds()
  population <- exact_population(cells,out)
  jobs <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i)
    data.frame(cell_id=cells$cell_id[i],replicate=seq_len(if(is.na(reps)) cells$production_reps[i] else reps))))
  rows <- list(); start <- proc.time()[['elapsed']]
  for(first in seq(1L,nrow(jobs),by=4L)) {
    indices <- first:min(first+3L,nrow(jobs))
    batch <- parallel::mclapply(indices,function(j) {
      set_single_threaded_math()
      exact_replicate(cells[match(jobs$cell_id[j],cells$cell_id),],jobs$replicate[j],exact_seed_base(mode),population)
    },mc.cores=workers,mc.preschedule=TRUE)
    if(any(vapply(batch,inherits,logical(1),'try-error'))) stop('Worker process failed')
    rows <- c(rows,batch); raw <- do.call(rbind,rows)
    saveRDS(raw,file.path(out,'raw.rds'))
    elapsed <- proc.time()[['elapsed']]-start
    write_csv(data.frame(completed=max(indices),total=nrow(jobs),elapsed_seconds=elapsed),file.path(out,'progress.csv'))
    cat('Exact first-stage completed ',max(indices),'/',nrow(jobs),' in ',round(elapsed,1),'s\n',sep=''); flush.console()
  }
  exact_summarize(raw,cells,out)
  if(any(is.finite(raw$policy_gap) & raw$policy_gap>1e-7)) stop('Policy equivalence gate failed')
  # Completion marker is written only after all draws and gates; raw.rds alone
  # is a checkpoint, never evidence that a preempted cell is complete.
  writeLines('complete',file.path(out,'COMPLETE'))
  cat('Exact first-stage results: ',out,'\n',sep='')
}

exact_summarize <- function(raw,cells,out) {
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
    targets <- if(cell$family=='coverage') c('loading','threshold',if(cell$model=='sem') 'path' else 'correlation') else ''
    for(arm in exact_arms(cell)) for(target in targets) {
      x <- raw[raw$cell_id==cell$cell_id & raw$arm==arm & raw$target==target,]
      valid <- !nzchar(x$error) & if(target=='') is.finite(x$p) else !is.na(x$covered)
      k <- if(target=='') sum(x$p[valid]<.05) else sum(x$covered[valid]); n <- sum(valid)
      rate <- if(total) k/total else NA_real_; ci <- dwls_wilson(k,total)
      adjusted <- NA_real_; null_count <- 0L
      if(cell$role=='power') {
        matched <- cells$role=='null' & cells$cross==0
        for(col in c('family','factors','categories','skew','n','groups','parameterization','nesting','model')) matched <- matched & cells[[col]]==cell[[col]]
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
  exact_paired(raw,cells,out)
  write_csv(raw[nzchar(raw$error),setdiff(names(raw),c('estimate','se','covered','p','exact_eigenvalues','opg_eigenvalues'))],file.path(out,'failures.csv'))
  write_csv(data.frame(priced_cells=sum(timing$attempted>0),total_cells=nrow(exact_cells()),unpriced_cell_ids=paste(setdiff(exact_cells()$cell_id,cells$cell_id[timing$attempted>0]),collapse=','),
    production_cpu_hours_priced=sum(timing$production_cpu_hours,na.rm=TRUE),
    complete_grid_price=nrow(cells)==nrow(exact_cells()) && all(timing$attempted>0)),file.path(out,'cost.csv'))
}

exact_paired <- function(raw,cells,out) {
  pairs <- list(); index <- 0L
  for(i in seq_len(nrow(cells))) {
    cell <- cells[i,]
    arms <- if(cell$family=='coverage') c(policy_ij='opg_ij') else
      if(cell$family=='global') c(policy_all='opg_all') else
        c(policy_sb='opg_sb',policy_peba4='opg_peba4',exact_all='opg_all')
    targets <- if(cell$family=='coverage') c('loading','threshold',if(cell$model=='sem') 'path' else 'correlation') else ''
    for(a in names(arms)) for(target in targets) {
      index <- index+1L
      x <- raw[raw$cell_id==cell$cell_id & raw$arm==a & raw$target==target,]
      y <- raw[raw$cell_id==cell$cell_id & raw$arm==arms[[a]] & raw$target==target,]
      z <- merge(x,y,by=c('cell_id','replicate','seed'),suffixes=c('_exact','_opg'))
      valid <- !nzchar(z$error_exact) & !nzchar(z$error_opg)
      if(target=='') {
        valid <- valid & is.finite(z$p_exact) & is.finite(z$p_opg)
        xe <- is.finite(z$p_exact) & !nzchar(z$error_exact) & z$p_exact<.05
        yo <- is.finite(z$p_opg) & !nzchar(z$error_opg) & z$p_opg<.05
      } else {
        valid <- valid & !is.na(z$covered_exact) & !is.na(z$covered_opg)
        xe <- !is.na(z$covered_exact) & !nzchar(z$error_exact) & z$covered_exact
        yo <- !is.na(z$covered_opg) & !nzchar(z$error_opg) & z$covered_opg
      }
      difference <- as.numeric(xe[valid])-as.numeric(yo[valid]); n <- length(difference)
      pair_index <- match(a,names(arms))*10L+match(target,targets)
      set.seed(617000001L+10000L*cell$cell_id+pair_index)
      ci <- if(n) quantile(replicate(2000L,mean(sample(difference,n,replace=TRUE))),c(.025,.975),names=FALSE) else c(NA_real_,NA_real_)
      pairs[[index]] <- data.frame(cell_id=cell$cell_id,exact_arm=a,opg_arm=arms[[a]],target,
        attempted=nrow(z),paired=n,excluded=nrow(z)-n,
        exact_minus_opg=if(n) mean(difference) else NA_real_,bootstrap_low=ci[1],bootstrap_high=ci[2],
        attempted_difference=if(nrow(z)) mean(as.numeric(xe)-as.numeric(yo)) else NA_real_)
    }
  }
  write_csv(do.call(rbind,pairs),file.path(out,'paired.csv'))
}
