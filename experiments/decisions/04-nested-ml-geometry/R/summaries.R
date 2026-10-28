wilson <- function(k,n) {
  if (!n) return(c(NA_real_,NA_real_))
  z <- qnorm(.975); center <- (k/n+z^2/(2*n))/(1+z^2/n)
  half <- z*sqrt(k/n*(1-k/n)/n+z^2/(4*n^2))/(1+z^2/n)
  c(center-half,center+half)
}
geometry_summaries <- function(raw,cells,mode) {
  rates <- list(); paired <- list()
  # Runs before the 2026-10-28 amendment have exact LR rows only.
  if (is.null(raw$restriction)) raw$restriction <- ifelse(raw$test=='score','none','exact')
  for (id in cells$cell_id) for (test in c('score','lr')) for (method in c('sb','peba4')) {
    p_col <- paste0('p_',method)
    arms <- unique(raw[raw$test==test,c('geometry','restriction')])
    for (arm in seq_len(nrow(arms))) {
      geometry <- arms$geometry[arm]; restriction <- arms$restriction[arm]
      x <- raw[raw$cell_id==id & raw$test==test & raw$geometry==geometry & raw$restriction==restriction,]
      ok <- !nzchar(x$error) & is.finite(x[[p_col]])
      p <- x[[p_col]][ok]; interval <- wilson(sum(p<.05),length(p))
      cell <- cells[cells$cell_id==id,]
      critical <- NA_real_; adjusted <- NA_real_
      if (cell$role=='power') {
        null_id <- cells$cell_id[cells$n==cell$n & cells$distribution==cell$distribution &
          cells$larger==cell$larger & cells$role=='null']
        y <- raw[raw$cell_id==null_id & raw$test==test & raw$geometry==geometry &
          raw$restriction==restriction & !nzchar(raw$error),p_col]
        y <- y[is.finite(y)]
        if (length(y)) critical <- unname(quantile(y,.05,type=1))
        if (length(p) && is.finite(critical)) adjusted <- mean(p<critical)
      }
      rates[[length(rates)+1L]] <- cbind(cell,geometry=geometry,restriction=restriction,test=test,method=method,
        draws=nrow(x),available=length(p),failures=sum(!ok),rejection=if(length(p)) mean(p<.05) else NA_real_,
        lower=interval[1],upper=interval[2],null_p_critical=critical,size_adjusted_power=adjusted)
    }
    if (cells$role[cells$cell_id==id]=='null') {
      # The registered pair: the expected-geometry comparator (exact rows for
      # LR, as lavaan) against the policy arm (observed; carried rows for LR).
      comparator <- if (test=='lr') 'exact' else 'none'
      policy_rows <- if (test=='lr' && any(raw$restriction=='invariant')) 'invariant' else comparator
      x <- raw[raw$cell_id==id & raw$test==test & raw$geometry=='expected' & raw$restriction==comparator,]
      y <- raw[raw$cell_id==id & raw$test==test & raw$geometry=='observed' & raw$restriction==policy_rows,]
      y <- y[match(x$rep,y$rep),]
      ok <- !nzchar(x$error) & !nzchar(y$error) & is.finite(x[[p_col]]) & is.finite(y[[p_col]])
      a <- x[[p_col]][ok]<.05; b <- y[[p_col]][ok]<.05
      delta <- lo <- hi <- NA_real_
      if (length(a)>1) {
        delta <- abs(mean(b)-.05)-abs(mean(a)-.05)
        set.seed(826100051L+id*100+match(test,c('score','lr'))*10+match(method,c('sb','peba4')))
        boot <- replicate(10000,{at <- sample.int(length(a),replace=TRUE)
          abs(mean(b[at])-.05)-abs(mean(a[at])-.05)})
        ci <- quantile(boot,c(.025,.975)); lo <- ci[1]; hi <- ci[2]
      }
      paired[[length(paired)+1L]] <- data.frame(cell_id=id,test=test,method=method,
        policy_restriction=policy_rows,pairs=length(a),
        size_error_difference=delta,lower=lo,upper=hi,
        qualifying_loss=is.finite(delta) && delta>.01 && is.finite(lo) && lo>0)
    }
  }
  gaps <- do.call(rbind,lapply(cells$cell_id,function(id) {
    do.call(rbind,lapply(c('score','lr'),function(test) {
      comparator <- if (test=='lr') 'exact' else 'none'
      x <- raw[raw$cell_id==id & raw$test==test & raw$geometry=='expected' & raw$restriction==comparator,]
      y <- raw[raw$cell_id==id & raw$test==test & raw$geometry=='observed' & raw$restriction==comparator,]
      y <- y[match(x$rep,y$rep),]
      ok <- !nzchar(x$error) & !nzchar(y$error) & is.finite(x$p_peba4) & is.finite(y$p_peba4)
      relative <- abs(y$statistic[ok]-x$statistic[ok])/abs(x$statistic[ok])
      cbind(cells[cells$cell_id==id,],test=test,pairs=sum(ok),
        median_absolute_peba4_gap=if(any(ok)) median(abs(y$p_peba4[ok]-x$p_peba4[ok])) else NA_real_,
        median_relative_score_gap=if(test=='score' && any(is.finite(relative)))
          median(relative[is.finite(relative)]) else NA_real_)
    }))
  }))
  # Carried against exact LR rows within each geometry (2026-10-28 amendment).
  restriction_gaps <- if (any(raw$restriction=='invariant')) do.call(rbind,lapply(cells$cell_id,function(id) {
    do.call(rbind,lapply(c('expected','observed'),function(geometry) {
      x <- raw[raw$cell_id==id & raw$test=='lr' & raw$geometry==geometry & raw$restriction=='exact',]
      y <- raw[raw$cell_id==id & raw$test=='lr' & raw$geometry==geometry & raw$restriction=='invariant',]
      y <- y[match(x$rep,y$rep),]
      ok <- !nzchar(x$error) & !nzchar(y$error) & is.finite(x$p_peba4) & is.finite(y$p_peba4)
      cbind(cells[cells$cell_id==id,],geometry=geometry,pairs=sum(ok),
        median_absolute_peba4_gap=if(any(ok)) median(abs(y$p_peba4[ok]-x$p_peba4[ok])) else NA_real_,
        max_absolute_peba4_gap=if(any(ok)) max(abs(y$p_peba4[ok]-x$p_peba4[ok])) else NA_real_,
        changed_peba4_decisions=sum((y$p_peba4[ok]<.05)!=(x$p_peba4[ok]<.05)))
    }))
  })) else NULL
  draws <- raw[!duplicated(paste(raw$cell_id,raw$rep)),]
  timing <- do.call(rbind,lapply(cells$cell_id,function(id) {
    t <- draws$elapsed_s[draws$cell_id==id]
    data.frame(cell_id=id,draws=length(t),mean_seconds=mean(t),median_seconds=median(t),
      total_seconds=sum(t),production_reps=cells$production_reps[cells$cell_id==id],
      production_cpu_hours=mean(t)*cells$production_reps[cells$cell_id==id]/3600)
  }))
  paired <- do.call(rbind,paired)
  gated <- paired$cell_id %in% cells$cell_id[cells$larger!='correct' & cells$role=='null']
  out <- list(rates=do.call(rbind,rates),paired=paired,timing=timing,geometry_gaps=gaps,
    failures=raw[nzchar(raw$error),c('cell_id','rep','seed','geometry','restriction','test','converged_h0','converged_h1','error')],
    checks=data.frame(draws=nrow(draws),failed_draws=sum(!draws$converged_h0 | !draws$converged_h1),
      failed_arms=sum(nzchar(raw$error)),max_policy_gap=max(c(0,raw$policy_gap),na.rm=TRUE)),
    decisions=data.frame(status=if(mode!='production') 'open_development_only' else
      if(any(paired$qualifying_loss[gated])) 'reconsider_observed' else 'retain_observed'))
  if (!is.null(restriction_gaps)) out$restriction_gaps <- restriction_gaps
  out
}
