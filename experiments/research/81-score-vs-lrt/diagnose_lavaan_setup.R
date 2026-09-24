#!/usr/bin/env Rscript
# Bounded check of model parsing in the reference, holding each sample fixed.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
base<-dirname(script);source(file.path(base,'R','design.R'))
if('--help' %in% commandArgs(TRUE)) {cat('Usage: Rscript diagnose_lavaan_setup.R [--smoke]\n');quit(status=0)}
calls<-if('--smoke' %in% commandArgs(TRUE))2L else 20L
models<-setNames(lapply(c(10,20),function(p)lavaan::lavaanify(population(p)$syntax,
  model_type='cfa',auto=TRUE,auto_fix_first=TRUE,auto_fix_single=TRUE,
  auto_var=TRUE,auto_cov_lv_x=TRUE,auto_cov_y=TRUE)),c('10','20'))
grid<-expand.grid(p=c(10L,20L),n=c(100L,200L,500L),distribution=c('normal','t10','vm2'),stringsAsFactors=FALSE)
rows<-list()
for(j in seq_len(nrow(grid))) {
 c<-grid[j,];ctx<-prepare(c$p,c$distribution);d<-draw_sample(ctx,c$n,c$distribution,20260917+j*100000+1)
 funs<-list(syntax=function()lavaan::cfa(ctx$syntax,d,se='none',test='Satorra.Bentler',baseline=FALSE),
   partable=function()lavaan::cfa(models[[as.character(c$p)]],d,se='none',test='Satorra.Bentler',baseline=FALSE))
 a<-suppressWarnings(funs$syntax());b<-suppressWarnings(funs$partable())
 diff<-max(abs(lavaan::coef(a)-lavaan::coef(b)))
 stopifnot(diff<1e-7,abs(a@test$standard$stat-b@test$standard$stat)<1e-7)
 for(k in names(funs)) {
  gc(FALSE);t<-Sys.time();for(i in seq_len(calls))invisible(suppressWarnings(funs[[k]]()))
  rows[[length(rows)+1L]]<-data.frame(cell_id=j,c,phase=k,ms=1000*as.numeric(difftime(Sys.time(),t,units='secs'))/calls,parameter_diff=diff)
 }
}
outdir<-file.path(base,'results','runtime-audit');dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
x<-do.call(rbind,rows);write.csv(x,file.path(outdir,'lavaan_setup.csv'),row.names=FALSE)
print(aggregate(ms~phase,x,mean),row.names=FALSE)
