#!/usr/bin/env Rscript
# Rejudge retained draws after correcting F = 2 fmin; never refit a draw.
args <- commandArgs(TRUE)
if(length(args)!=3L || '--help' %in% args) {
  cat('Usage: Rscript scripts/correct_targets.R ORIGINAL_INPUT FROZEN_EVIDENCE CORRECTED_INPUT\n')
  quit(status=if('--help' %in% args) 0L else 1L)
}
input <- normalizePath(args[1]); frozen <- normalizePath(args[2]); output <- args[3]
dir.create(output,recursive=TRUE,showWarnings=FALSE)
read_csv <- function(dir,name) read.csv(file.path(dir,paste0(name,'.csv')),stringsAsFactors=FALSE)
raw <- file.path(input,'raw')
cells <- read_csv(input,'cells')
stopifnot(nrow(cells)==768L,file.exists(file.path(raw,'manifest.rds')),
  sum(dir.exists(file.path(raw,list.files(raw))))==768L,
  all(file.exists(file.path(raw,cells$id,'COMPLETE'))))
original <- read_csv(frozen,'targets'); targets <- original
for(name in c('F0','F0_b','complete_F0','complete_F0_b')) targets[[name]] <- 2*original[[name]]
for(name in c('rmsea','complete_rmsea')) targets[[name]] <- sqrt(2)*original[[name]]
unchanged <- setdiff(names(targets),c('F0','F0_b','complete_F0','complete_F0_b','rmsea','complete_rmsea'))
stopifnot(identical(targets[unchanged],original[unchanged]))
old <- do.call(rbind,lapply(c('rmsea','cfi','tli','srmr'),function(index) read_csv(frozen,paste0('summary_',index))))
key <- function(x) paste(x$cell,x$arm,x$index,sep=':')
summary <- vector('list',nrow(cells))
for(i in seq_len(nrow(cells))) {
  id <- cells$id[i]; paths <- file.path(raw,id,sprintf('rep_%05d.rds',1:1000))
  stopifnot(all(file.exists(paths)))
  rows <- do.call(rbind,lapply(paths,readRDS))
  stopifnot(nrow(rows)==24000L,all(rows$cell==id))
  summary[[i]] <- do.call(rbind,lapply(split(rows,interaction(rows$arm,rows$index,drop=TRUE)),function(x) {
    z <- targets[match(id,targets$cell),]; target <- z[[x$index[1]]]
    available <- is.na(x$reason) & is.finite(x$estimate)
    if(x$kind[1]=='interval') available <- available & is.finite(x$lower) & is.finite(x$upper)
    error <- x$estimate[available]-target; coverage <- x$lower<=target & x$upper>=target
    data.frame(cell=id,arm=x$arm[1],index=x$index[1],kind=x$kind[1],target=target,target_status=z$status,
      attempted=nrow(x),available=sum(available),availability=mean(available),convergence=mean(x$converged),
      bias=if(length(error)) mean(error) else NA_real_,rmse=if(length(error)) sqrt(mean(error^2)) else NA_real_,
      coverage=if(x$kind[1]=='interval' && any(available)) mean(coverage[available]) else NA_real_,
      coverage_attempted=if(x$kind[1]=='interval') mean(available & coverage,na.rm=TRUE) else NA_real_,
      width=if(x$kind[1]=='interval' && any(available)) mean(x$upper[available]-x$lower[available]) else NA_real_)
  }))
  if(i%%16L==0L) { cat('Read',i,'/ 768 cells\n'); flush.console() }
}
s <- do.call(rbind,summary); s <- s[match(key(old),key(s)),]; rownames(s)<-NULL
for(index in c('cfi','tli','srmr')) {
  keep <- old$index==index
  stopifnot(isTRUE(all.equal(s[keep,],old[keep,],tolerance=1e-12,check.attributes=FALSE)))
  # Preserve original CSV precision exactly for the unaffected indices.
  s[keep,] <- old[keep,]
}
for(name in setdiff(list.files(input,pattern='\\.csv$'),c('targets.csv','summary.csv'))) {
  dest <- file.path(output,name)
  if(!file.exists(dest)) stopifnot(file.symlink(file.path(input,name),dest))
}
write.csv(targets,file.path(output,'targets.csv'),row.names=FALSE,na='')
write.csv(s,file.path(output,'summary.csv'),row.names=FALSE,na='')
cat('Verified unchanged CFI, TLI, SRMR summaries; wrote corrected input\n')
file <- sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
source(file.path(dirname(file),'../R/judge_production.R'))
before <- judge_production(old,cells); after <- judge_production(s,cells)
for(name in c('decisions','family','candidates','intervals','losses','interval')) {
  a <- before[[name]]; b <- after[[name]]
  a <- a[a$index!='rmsea',]; b <- b[b$index!='rmsea',]
  stopifnot(identical(a,b))
}
write.csv(data.frame(key=c('correction_date','correction','raw_manifest_md5','unaffected_indices'),
  value=c('2026-10-10','F = 2 fmin; raw draws re-summarized; criteria unchanged',
    unname(tools::md5sum(file.path(raw,'manifest.rds'))),'CFI, TLI, SRMR summaries and judgments asserted unchanged')),
  file.path(output,'target_correction_metadata.csv'),row.names=FALSE)
print(after$decisions); print(after$candidates[after$candidates$index=='rmsea',]);
print(after$intervals[after$intervals$index=='rmsea',])
invariant <- c('cell','arm','index','kind','target_status','attempted','available','availability','convergence','width')
stopifnot(isTRUE(all.equal(s[invariant],old[invariant],tolerance=1e-12,check.attributes=FALSE)))
cat('Verified unchanged widths, availability, convergence and attempted denominators\n')
