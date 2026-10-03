dwls_wilson <- function(k,n) {
  if(!n) return(c(NA_real_,NA_real_))
  z <- qnorm(.975); p <- k/n; center <- (p+z*z/(2*n))/(1+z*z/n)
  half <- z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
  c(center-half,center+half)
}

dwls_summarize <- function(raw,cells,out) {
  attempts <- raw[!duplicated(raw[c('cell_id','replicate')]),]
  timing <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
    x <- attempts[attempts$cell_id==cells$cell_id[i],]
    data.frame(cell_id=cells$cell_id[i],attempted=nrow(x),failed=sum(nzchar(x$error)),
      mean_seconds=mean(x$elapsed_seconds),mean_cpu_seconds=mean(if('cpu_seconds' %in% names(x)) x$cpu_seconds else x$elapsed_seconds),production_reps=cells$production_reps[i],
      production_cpu_hours=mean(if('cpu_seconds' %in% names(x)) x$cpu_seconds else x$elapsed_seconds)*cells$production_reps[i]/3600,
      max_policy_gap=if(all(is.na(x$policy_gap))) NA_real_ else max(x$policy_gap,na.rm=TRUE))
  }))
  write_csv(timing,file.path(out,'timing.csv'))
  write_csv(attempts[nzchar(attempts$error),c('cell_id','replicate','seed','h1_converged','h0_converged','error')],file.path(out,'failures.csv'))
  summaries <- list()
  for(i in seq_len(nrow(cells))) {
    cell <- cells[i,]
    arms <- if(cell$family=='coverage') c('policy_ij','expected','observed') else
      if(cell$family=='nested') c(dwls_global_arms(),'profile_sb','profile_peba4','fixed_sb','fixed_peba4') else
        dwls_global_arms()
    targets <- if(cell$family=='coverage') c('loading','threshold',if(cell$model=='sem') 'path' else 'correlation') else ''
    for(a in arms) for(t in targets) {
      x <- raw[raw$cell_id==cell$cell_id & raw$arm==a & raw$target==t,]
      valid <- !nzchar(x$error) & if(t=='') is.finite(x$p) else !is.na(x$covered)
      n <- sum(valid); k <- if(t=='') sum(x$p[valid]<.05) else sum(x$covered[valid])
      ci <- dwls_wilson(k,n); rate <- if(n) k/n else NA_real_
      adjusted <- NA_real_; null_count <- NA_integer_
      if(cell$role=='power') {
        match_cols <- c('family','factors','categories','skew','n','groups','parameterization','nesting','model')
        matched <- cells$role=='null' & cells$cross==0
        for(col in match_cols) matched <- matched & cells[[col]]==cell[[col]]
        ids <- cells$cell_id[matched]; if(length(ids)>1) stop('Ambiguous matched null for cell ',i)
        # A selected power-only subset has no null draws for size adjustment.
        nx <- raw[raw$cell_id %in% ids & raw$arm==a & !nzchar(raw$error) & is.finite(raw$p),]
        null_count <- nrow(nx)
        # Chi-square-equivalent statistics are monotone in each arm's p-value.
        if(n && null_count) {
          null_stat <- qchisq(nx$p,df=1,lower.tail=FALSE)
          threshold <- unname(quantile(null_stat,.95,type=8))
          adjusted <- mean(qchisq(x$p[valid],df=1,lower.tail=FALSE)>threshold)
        }
      }
      flag <- cell$role=='null' && is.finite(rate) &&
        if(t=='') cell$n>=500 && (rate<.03 || rate>.07) else
          a=='policy_ij' && cell$n>=300 && rate<.93
      summaries[[length(summaries)+1L]] <- data.frame(cell_id=cell$cell_id,arm=a,target=t,
        attempted=timing$attempted[i],successful=n,failed=timing$attempted[i]-n,
        successes=k,rate=rate,wilson_low=ci[1],wilson_high=ci[2],
        size_adjusted_power=adjusted,matched_null_successful=null_count,
        finite_sample_followup=flag,
        df=if(nrow(x)) x$df[1] else NA,spectrum_size=if(nrow(x)) x$spectrum_size[1] else NA)
    }
  }
  write_csv(do.call(rbind,summaries),file.path(out,'summary.csv'))
  write_csv(data.frame(cells=nrow(cells),replicates=nrow(attempts),
    failures=sum(nzchar(attempts$error)),production_cpu_hours=sum(timing$production_cpu_hours),
    ideal_four_worker_hours=sum(timing$production_cpu_hours)/4),file.path(out,'cost.csv'))
}
