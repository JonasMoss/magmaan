#!/usr/bin/env Rscript
# Retained-draw diagnostic only. No fitting, simulation or policy selection.
args <- commandArgs(TRUE)
if ('--help' %in% args || !length(args) %in% c(3L,4L)) {
  cat('Usage: Rscript scripts/diagnose_wald.R RAW_RUN CORRECTED_EVIDENCE OUTPUT [MAX_CELLS]\n')
  quit(status=if ('--help' %in% args) 0L else 1L)
}
raw_run <- normalizePath(args[1]); evidence <- normalizePath(args[2]); out <- args[3]
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
root <- normalizePath(file.path(dirname(script),'../../../..'))
cells <- read.csv(file.path(evidence,'cells.csv')); targets <- read.csv(file.path(evidence,'targets.csv'))
stopifnot(!anyDuplicated(cells$id), all(cells$id %in% targets$cell))
if (length(args)==4L) cells <- head(cells,as.integer(args[4]))
mean_safe <- function(x) if (length(x)) mean(x) else NA_real_
var_safe <- function(x) if (length(x)>1L) var(x) else NA_real_
z <- qnorm(.95); summaries <- vector('list',nrow(cells)); start <- proc.time()[3]
for (i in seq_len(nrow(cells))) {
  cell <- cells[i,]; target <- targets[match(cell$id,targets$cell),]
  basis <- jsonlite::fromJSON(file.path(root,'experiments/_support/basis/sem_score',paste0(cell$model,'.json')),simplifyVector=FALSE)
  groups <- length(basis$groups)
  paths <- file.path(raw_run,'raw',cell$id,sprintf('rep_%05d.rds',1:1000))
  stopifnot(file.exists(file.path(raw_run,'raw',cell$id,'COMPLETE')),all(file.exists(paths)))
  rows <- do.call(rbind,lapply(paths,readRDS))
  stopifnot(nrow(rows)==24000L,all(rows$cell==cell$id))
  summaries[[i]] <- do.call(rbind,lapply(c('rmsea','cfi'),function(index) {
    w <- rows[rows$arm=='wald_f' & rows$index==index,]
    p <- rows[rows$arm=='misspec_robust' & rows$index==index,]
    stopifnot(!anyDuplicated(w$replicate),!anyDuplicated(p$replicate))
    p <- p[match(w$replicate,p$replicate),]
    valid <- w$converged & p$converged & is.na(w$reason) & is.na(p$reason) &
      is.finite(w$lower) & is.finite(w$upper) & is.finite(p$estimate)
    w <- w[valid,]; p <- p[valid,]
    clipped <- if(index=='rmsea') w$lower<=0 else w$lower<=0 | w$upper>=1
    transform <- if(index=='rmsea') function(x) target$df*x^2/groups else identity
    points <- transform(p$estimate)
    v <- ((transform(w$upper[!clipped])-transform(w$lower[!clipped]))/(2*z))^2
    empirical <- var_safe(points); conditional <- var_safe(points[!clipped])
    coverage <- mean_safe(w$lower<=target[[index]] & w$upper>=target[[index]])
    data.frame(cell=cell$id,model=cell$model,delta=abs(cell$delta),signed_delta=cell$delta,
      N=cell$N,generator=cell$generator,index=index,df=target$df,groups=groups,target=target[[index]],
      attempted=1000,available=nrow(w),unclipped=sum(!clipped),clipped_share=mean_safe(clipped),
      point_zero_or_one_share=mean_safe(if(index=='rmsea') p$estimate==0 else p$estimate>=1 | p$estimate<=0),
      mean_variance=mean_safe(v),empirical_variance=empirical,ratio=mean_safe(v)/empirical,
      empirical_variance_unclipped=conditional,ratio_unclipped=mean_safe(v)/conditional,
      coverage=coverage,coverage_se=sqrt(coverage*(1-coverage)/nrow(w)),
      below_target=mean_safe(w$upper<target[[index]]),above_target=mean_safe(w$lower>target[[index]]))
  }))
  if(i%%16L==0L || i==nrow(cells)) {
    cat(i,'/',nrow(cells),'cells;',round(proc.time()[3]-start,1),'seconds\n'); flush.console()
  }
}
s <- do.call(rbind,summaries)
# Confirm the matched-convergence denominator preserves corrected production evidence.
for (index in c('rmsea','cfi')) {
  old <- read.csv(file.path(evidence,paste0('summary_',index,'.csv')))
  old <- old[old$arm=='wald_f' & old$cell %in% cells$id,]
  now <- s[s$index==index,]; now <- now[match(old$cell,now$cell),]
  stopifnot(identical(old$available,now$available),
    isTRUE(all.equal(old$coverage,now$coverage,tolerance=1e-12)))
}
# Pool cell diagnostics descriptively; never combine points with different targets.
aggregate_cells <- function(keys) {
  do.call(rbind,lapply(split(s,interaction(s[keys],drop=TRUE)),function(x) {
    r <- x[1,keys,drop=FALSE]
    cbind(r,data.frame(cells=nrow(x),finite_ratio_cells=sum(is.finite(x$ratio)),
      median_ratio=median(x$ratio[is.finite(x$ratio)]),
      mean_ratio=mean_safe(x$ratio[is.finite(x$ratio)]),
      variance_weighted_ratio=sum(x$mean_variance,na.rm=TRUE)/sum(x$empirical_variance[is.finite(x$mean_variance)],na.rm=TRUE),
      mean_clipped_share=mean(x$clipped_share),mean_coverage=mean(x$coverage),
      over_925=sum(x$coverage>.925),under_875=sum(x$coverage<.875),
      mean_point_boundary_share=mean(x$point_zero_or_one_share)))
  }))
}
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write <- function(x,name) write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE,na='')
write(s,'cells'); write(aggregate_cells(c('model','delta','N','generator','index')),'by_model')
write(aggregate_cells(c('delta','N','generator','index')),'pooled')
write(aggregate_cells(c('delta','N','index')),'by_delta_n')
write(data.frame(key=c('raw_input','corrected_evidence','cells','draws_per_cell','elapsed_seconds','R_version','target_md5','cells_md5','manifest_md5','interpretation'),
  value=c(raw_run,evidence,nrow(cells),1000,proc.time()[3]-start,R.version.string,
  unname(tools::md5sum(file.path(evidence,'targets.csv'))),unname(tools::md5sum(file.path(evidence,'cells.csv'))),
  unname(tools::md5sum(file.path(raw_run,'raw','manifest.rds'))),
  'Variance mean excludes clipped intervals; empirical variance uses all matched converged available points. RMSEA inversion recovers max(F_tilde,0), not negative F_tilde. Conditional ratio also supplied. CFI excludes either clipped endpoint. Pooled ratios are descriptive, not decisions.')),'metadata')
cat('Wrote diagnostic CSVs to',out,'\n')
