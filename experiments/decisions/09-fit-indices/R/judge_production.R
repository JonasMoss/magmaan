# Apply the approved criteria to frozen cell summaries, without fitting.
judge_production <- function(summary, cells) {
  s <- merge(summary,cells[c('id','model','estimator','N','generator','delta')],by.x='cell',by.y='id')
  stopifnot(all(s$estimator=='ML'),length(unique(s$cell))==768L)
  point <- s[s$kind=='point',]
  base <- point[point$arm=='misspec_robust',c('cell','index','rmse')]
  names(base)[3] <- 'default_rmse'
  point <- merge(point,base,by=c('cell','index'))
  point$comparison <- ifelse(!is.finite(point$rmse)|!is.finite(point$default_rmse),'unavailable',
    ifelse(point$rmse <= .9*point$default_rmse,'win',
      ifelse(point$default_rmse <= .9*point$rmse,'loss','tie')))
  # The default is not a candidate against itself.
  point$comparison[point$arm=='misspec_robust'] <- 'tie'
  family <- do.call(rbind,lapply(split(point,interaction(point$model,point$index,point$arm,drop=TRUE)),function(x) {
    ok <- is.finite(x$rmse)
    data.frame(model=x$model[1],index=x$index[1],arm=x$arm[1],cells=nrow(x),
      availability=sum(x$available)/sum(x$attempted),
      bias=if(any(ok)) weighted.mean(x$bias[ok],x$available[ok]) else NA_real_,
      rmse=if(any(ok)) sqrt(weighted.mean(x$rmse[ok]^2,x$available[ok])) else NA_real_,
      wins=sum(x$comparison=='win'),losses=sum(x$comparison=='loss'),
      ties=sum(x$comparison=='tie'),unavailable=sum(x$comparison=='unavailable'))
  }))
  candidates <- do.call(rbind,lapply(split(family[family$arm!='misspec_robust',],interaction(
    family$index[family$arm!='misspec_robust'],family$arm[family$arm!='misspec_robust'],drop=TRUE)),function(x)
      data.frame(index=x$index[1],arm=x$arm[1],wins=sum(x$wins),losses=sum(x$losses),
        losing_families=sum(x$losses>x$wins),eligible=all(x$losses<=x$wins) && sum(x$wins)-sum(x$losses)>=.1*768)))
  interval <- s[s$kind=='interval',]
  interval$flag <- ifelse(is.finite(interval$coverage), interval$coverage<.875 | interval$coverage>.925, NA)
  interval$below_floor <- ifelse(is.finite(interval$coverage),interval$coverage<.8,NA)
  intervals <- do.call(rbind,lapply(split(interval,interaction(interval$index,interval$arm,drop=TRUE)),function(x)
    data.frame(index=x$index[1],arm=x$arm[1],cells=nrow(x),min_availability=min(x$availability),
      availability_failures=sum(x$availability<.95),flagged=sum(x$flag,na.rm=TRUE),
      undefined_coverage=sum(!is.finite(x$coverage)),below_floor=sum(x$below_floor,na.rm=TRUE),
      min_coverage=if(any(is.finite(x$coverage))) min(x$coverage,na.rm=TRUE) else NA_real_,
      median_cell_width=if(any(is.finite(x$width))) median(x$width,na.rm=TRUE) else NA_real_,
      eligible=all(x$availability>=.95) && all(is.finite(x$coverage)) &&
        sum(x$flag,na.rm=TRUE)<=.05*nrow(x) && all(x$coverage>=.8))))
  decisions <- do.call(rbind,lapply(c('rmsea','cfi','tli','srmr'),function(index) {
    p <- candidates[candidates$index==index & candidates$eligible,]
    i <- intervals[intervals$index==index & intervals$eligible,]
    data.frame(index=index,point=if(nrow(p)) p$arm[which.max(p$wins)] else 'misspec_robust',
      interval=if(nrow(i)) i$arm[which.min(i$median_cell_width)] else 'none')
  }))
  list(point=point,family=family,candidates=candidates,interval=interval,intervals=intervals,decisions=decisions,
    losses=point[point$arm!='misspec_robust' & point$comparison=='loss',])
}
