#!/usr/bin/env Rscript
# Independent population oracle; no simulation or package installation.
file <- sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1])
leaf <- normalizePath(file.path(dirname(file), '..'))
root <- normalizePath(file.path(leaf, '../../..'))
source(file.path(root, 'experiments/_support/R/helpers.R'))
source(support_path('R', 'basis.R'))
for (name in c('config','populations','targets')) source(file.path(leaf,'R',paste0(name,'.R')))
set_single_threaded_math()
cells <- study_config('smoke')[c(1,2,4),]
cells$model <- c('cfa_ptsd8','growth6','sem_worland11')
cells$delta <- .4; cells$estimator <- 'ML'; cells$missing_rate <- 0
answer <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) {
  cell <- as.list(cells[i,]); p <- build_population(cell)
  calibration <- list(groups=lapply(seq_along(p$stats$S),function(g)
    list(sigma=p$stats$S[[g]],mu=p$stats$mean[[g]])))
  target <- compute_targets(p,calibration,cell,1L)
  stopifnot(target$status=='available',length(p$stats$S)==1L)
  oracle <- lavaan::sem(p$record$h1,sample.cov=p$stats$S[[1]],
    sample.mean=if(p$record$meanstructure) p$stats$mean[[1]] else NULL,
    sample.nobs=1e7,likelihood='normal',sample.cov.rescale=FALSE,
    meanstructure=p$record$meanstructure,fixed.x=FALSE)
  stopifnot(lavaan::lavInspect(oracle,'converged'))
  measures <- lavaan::fitMeasures(oracle,c('rmsea','cfi','tli'))
  corrected <- target$own[names(measures)]
  relative <- abs(corrected-measures)/abs(measures)
  stopifnot(all(relative<1e-3))
  data.frame(model=cell$model,index=names(measures),target=unname(corrected),
    lavaan=unname(measures),relative_difference=unname(relative))
}))
print(answer,row.names=FALSE)
write.csv(answer,file.path(leaf,'results/fit_indices/2026-10-09-ml-b/target_check.csv'),row.names=FALSE)
