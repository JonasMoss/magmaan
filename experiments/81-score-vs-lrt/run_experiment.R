#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if('--help' %in% args) {
  cat('Usage: Rscript run_experiment.R [--pilot|--smoke] [--reps N] [--workers N]\n',
      ' [--seed-base N] [--budget-minutes N] [--output NAME]\n',
      'Default: 10 reps/cell, 1 worker, pilot output; no main run is automatic.\n',
      '18 cells: p=10,20; N=100,200,500; normal, t10, vm2.\n',
      '--smoke: 1 rep/cell. --output: new subdirectory under results/.\n',
      'Budget is for projection only. Output folders must not already exist.\n',sep='')
  quit(status=0)
}
opts <- list(reps=10L,workers=1L,seed_base=20260916L,budget_minutes=60,output='pilot')
i <- 1L
while(i<=length(args)) {
  a <- args[i]
  if(a=='--pilot') opts$reps <- 10L else if(a=='--smoke') opts$reps <- 1L else {
    key <- c('--reps'='reps','--workers'='workers','--seed-base'='seed_base',
             '--budget-minutes'='budget_minutes','--output'='output')[a]
    if(is.na(key)||i==length(args)) stop('unknown/incomplete option: ',a)
    i <- i+1L;opts[[key]] <- if(key=='output') args[i] else as.numeric(args[i])
  }
  i <- i+1L
}
stopifnot(opts$reps>=1,opts$reps==floor(opts$reps),opts$workers>=1,
  opts$workers==floor(opts$workers),opts$budget_minutes>0,
  opts$seed_base>=0,opts$seed_base<2e9,opts$seed_base==floor(opts$seed_base),opts$reps<100000,grepl('^[a-zA-Z0-9_-]+$',opts$output))
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
base <- dirname(script)
source(file.path(base,'..','_support','R','helpers.R'))
source(file.path(base,'R','design.R'))
set_single_threaded_math()
suppressPackageStartupMessages(library(magmaan))
outdir <- file.path(base,'results',opts$output)
if(dir.exists(outdir)) stop('Output already exists; choose another --output')
dir.create(outdir,recursive=TRUE)
write_out <- function(x,name) write.csv(x,file.path(outdir,name),row.names=FALSE,na='')
package_files <- list.files(find.package('magmaan'),pattern='\\.(so|rdb|rdx)$|^DESCRIPTION$',recursive=TRUE,full.names=TRUE)
package_hashes <- tools::md5sum(package_files)
started <- Sys.time(); start <- clock_seconds()
power <- function() {
  paths <- Sys.glob('/sys/class/power_supply/*/online')
  vals <- vapply(paths,function(p)paste(readLines(p,warn=FALSE),collapse=''),character(1))
  if(!length(vals)) 'unknown' else if(any(vals=='1')) 'AC' else 'battery'
}
power_start <- power()
version_hash <- function(path) if(file.exists(path)) unname(tools::md5sum(path)) else NA_character_
root <- repo_root(base)
head <- system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE)
status <- system2('git',c('-C',shQuote(root),'status','--short'),stdout=TRUE)
writeLines(status,file.path(outdir,'worktree-status.txt'))
writeLines(capture.output(sessionInfo()),file.path(outdir,'session.txt'))
grid <- expand.grid(p=c(10L,20L),n=c(100L,200L,500L),
 distribution=c('normal','t10','vm2'),stringsAsFactors=FALSE)
grid$cell_id <- seq_len(nrow(grid));write_out(grid,'design.csv')
contexts <- list()
for(p in c(10,20)) for(d in unique(grid$distribution)) contexts[[paste(p,d)]] <- prepare(p,d)
setup_seconds <- clock_seconds()-start
cat(sprintf('Prepared 18 cells in %.2fs; %s; %d worker(s), %d reps/cell\n',
 setup_seconds,power_start,opts$workers,opts$reps));flush.console()
# Warm both dimensions; these calls are excluded from the measured loop.
wt <- clock_seconds()
for(p in c(10,20)) {
  cell <- grid[grid$p==p & grid$n==500 & grid$distribution=='normal',]
  invisible(one_rep(cell,0L,contexts[[paste(p,'normal')]],opts$seed_base))
}
warmup_seconds <- clock_seconds()-wt
rows <- list();cells <- list();loop_start <- clock_seconds()
for(j in seq_len(nrow(grid))) {
  cell <- grid[j,];ctx <- contexts[[paste(cell$p,cell$distribution)]];t0 <- clock_seconds()
  chunks <- split(seq_len(opts$reps),ceiling(seq_len(opts$reps)/100))
  batch <- list()
  for(ids in chunks) {
    if(opts$workers==1) piece <- lapply(ids,function(r)one_rep(cell,r,ctx,opts$seed_base)) else
      piece <- parallel::mclapply(ids,function(r)one_rep(cell,r,ctx,opts$seed_base),
        mc.cores=opts$workers,mc.preschedule=TRUE,mc.set.seed=FALSE)
    if(any(vapply(piece,inherits,logical(1),'try-error'))) stop('worker failed')
    batch <- c(batch,piece)
    if(opts$reps>100) {
      cat(sprintf('  cell %d/18: %d/%d reps, %.1fs elapsed in cell\n',
        j,max(ids),opts$reps,clock_seconds()-t0));flush.console()
    }
  }
  elapsed <- clock_seconds()-t0
  result <- do.call(rbind,batch);rows[[j]] <- result
  append_csv(result,file.path(outdir,'replicates.csv'))
  cells[[j]] <- data.frame(cell_id=cell$cell_id,p=cell$p,n=cell$n,distribution=cell$distribution,
    reps=opts$reps,seconds=elapsed,seconds_per_rep=elapsed/opts$reps,power=power())
  progress <- data.frame(completed_cells=j,total_cells=nrow(grid),
    elapsed_seconds=clock_seconds()-loop_start,eta_seconds=(clock_seconds()-loop_start)/j*(nrow(grid)-j))
  write_out(progress,'progress.csv')
  cat(sprintf('%2d/18 p=%d N=%d %-6s %.2fs; %d/%d valid method results; ETA %.0fs\n',
    j,cell$p,cell$n,cell$distribution,elapsed,sum(!nzchar(result$error)),nrow(result),progress$eta_seconds))
  flush.console()
}
loop_seconds <- clock_seconds()-loop_start
raw <- do.call(rbind,rows);ct <- do.call(rbind,cells);write_out(ct,'cell_timing.csv')
summary <- do.call(rbind,lapply(split(raw,interaction(raw$cell_id,raw$method,drop=TRUE)),function(x){
  ok <- !nzchar(x$error)&is.finite(x$p_value);v <- sum(ok)
  rate <- if(v) mean(x$p_value[ok]<.05) else NA_real_
  data.frame(x[1,c('cell_id','p','n','distribution','method')],attempted=nrow(x),valid=v,
    failed=sum(!ok),rejection_05=rate,mcse=if(v) sqrt(rate*(1-rate)/v) else NA_real_,
    median_postfit_ms=if(all(is.na(x$postfit_seconds))) NA_real_ else median(x$postfit_seconds,na.rm=TRUE)*1000,
    median_batch_ms=median(x$postfit_batch_seconds,na.rm=TRUE)*1000,
    median_pipeline_ms=median(x$pipeline_seconds,na.rm=TRUE)*1000)
}))
write_out(summary,'score_lrt_summary.csv')
# Paired differences use only common valid replicates, with their denominator.
paired <- list()
for(ci in grid$cell_id) for(method in setdiff(unique(raw$method),'sb_ml')) {
  x <- raw[raw$cell_id==ci & raw$method==method,c('rep','p_value','error')]
  y <- raw[raw$cell_id==ci & raw$method=='sb_ml',c('rep','p_value','error')]
  z <- merge(x,y,by='rep',suffixes=c('_method','_lr_sb'))
  ok <- !nzchar(z$error_method)&!nzchar(z$error_lr_sb)&
    is.finite(z$p_value_method)&is.finite(z$p_value_lr_sb)
  delta <- as.numeric(z$p_value_method[ok]<.05)-as.numeric(z$p_value_lr_sb[ok]<.05)
  paired[[length(paired)+1L]] <- data.frame(cell_id=ci,method=method,common_valid=sum(ok),
    paired_rejection_difference=if(length(delta))mean(delta) else NA_real_,
    paired_mcse=if(length(delta)>1)sd(delta)/sqrt(length(delta)) else NA_real_)
}
write_out(do.call(rbind,paired),'paired_vs_lr_sb.csv')
# Predict serial/same-worker wall time, never multiply by an assumed speedup.
per_round <- loop_seconds/opts$reps
fixed <- setup_seconds+warmup_seconds
projection <- data.frame(budget_minutes=opts$budget_minutes,workers=opts$workers,
  fixed_seconds=fixed,seconds_per_grid_round=per_round,
  projected_2000_seconds=fixed+2000*per_round,
  reps_per_cell_with_20pct_reserve=max(0,floor((.8*opts$budget_minutes*60-fixed)/per_round)))
write_out(projection,'budget_projection.csv')
write_out(data.frame(phase=c('setup','warmup','simulation_loop','total_runner'),
 elapsed_seconds=c(setup_seconds,warmup_seconds,loop_seconds,clock_seconds()-start),
 workers=opts$workers,blas_threads=1,run_id=opts$output),'score_lrt_timing.csv')
stopifnot(identical(package_hashes,tools::md5sum(package_files)))
write_out(data.frame(path=names(package_hashes),md5=unname(package_hashes)),'package_fingerprints.csv')
dll <- getLoadedDLLs()[['magmaan']][['path']]
meta <- c(started_utc=format(started,tz='UTC',usetz=TRUE),finished_utc=format(Sys.time(),tz='UTC',usetz=TRUE),
 power_start=power_start,power_end=power(),command=paste(commandArgs(),collapse=' '),
 seed_base=opts$seed_base,reps=opts$reps,workers=opts$workers,blas_threads=1,git_head=head,
 package_path=find.package('magmaan'),package_version=as.character(packageVersion('magmaan')),
 dll_md5=version_hash(dll),r_database_md5=version_hash(file.path(find.package('magmaan'),'R','magmaan.rdb')),
 cpu=paste(unique(sub('.*: ','',grep('model name',readLines('/proc/cpuinfo'),value=TRUE))),collapse='; '),
 active_researcher_seconds=NA,question_to_report_seconds=NA)
write_out(data.frame(key=names(meta),value=unname(meta)),'metadata.csv')
print(projection,row.names=FALSE)
cat('Results:',outdir,'\n')
