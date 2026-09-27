#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(FALSE), value=TRUE)[1]))
here <- dirname(dirname(script)); args <- commandArgs(TRUE)
if('--help' %in% args) {
  cat('Usage: Rscript scripts/summarize_psd_scale.R\nReads absolute-controls and relative-final runs; writes comparison summaries and paired losses.\n')
  quit(save='no')
}
root <- file.path(here,'results','psd-scale');out <- file.path(root,'comparison')
x <- do.call(rbind,lapply(c('absolute-controls','relative-final'),function(run)
  cbind(scale=if(run=='relative-final') 'relative' else 'absolute',read.csv(file.path(root,run,'fits.csv')))))
keys <- c('pop','model','n','rep','transform','arm')
stopifnot(!anyDuplicated(x[c('scale',keys)]),all(x$returned | !x$success))
# The best of these four attempts is only a comparison target, not a global reference.
problem <- c('pop','model','n','rep','transform')
x$best <- ave(ifelse(x$success,x$fmin,Inf),interaction(x[problem],drop=TRUE),FUN=min)
x$hit <- x$success & is.finite(x$best) & x$fmin <= x$best+1e-6*(1+abs(x$best))
x$fit_count <- 1L
summary <- aggregate(x[c('fit_count','success','hit','seconds')],x[c('scale','transform','arm')],sum)
family <- aggregate(x[c('fit_count','success','hit','seconds')],x[c('scale','family','transform','arm')],sum)
pair <- merge(x[x$scale=='absolute',c(keys,'family','success','hit','fmin','seconds','message')],
              x[x$scale=='relative',c(keys,'success','hit','fmin','seconds','message')],by=keys,suffixes=c('_absolute','_relative'))
stopifnot(nrow(pair)*2==nrow(x))
pair$gain <- !pair$success_absolute & pair$success_relative
pair$loss <- pair$success_absolute & !pair$success_relative
pair$objective_gain <- !pair$hit_absolute & pair$hit_relative
pair$objective_loss <- pair$hit_absolute & !pair$hit_relative
changes <- aggregate(pair[c('gain','loss','objective_gain','objective_loss')],pair[c('arm','transform')],sum)
r <- x[x$scale=='relative',]
route <- merge(r[r$arm=='direct',c(problem,'family','success','hit','fmin')],
               r[r$arm=='two_stage',c(problem,'success','hit','fmin')],by=problem,suffixes=c('_direct','_two_stage'))
route$gain <- !route$success_direct & route$success_two_stage
route$loss <- route$success_direct & !route$success_two_stage
route$objective_gain <- !route$hit_direct & route$hit_two_stage
route$objective_loss <- route$hit_direct & !route$hit_two_stage
route_summary <- aggregate(route[c('gain','loss','objective_gain','objective_loss')],route['transform'],sum)
dir.create(out,showWarnings=FALSE)
for(name in c('summary','family','pair','changes','route','route_summary'))
  write.csv(get(name),file.path(out,paste0(name,'.csv')),row.names=FALSE)
print(summary,row.names=FALSE);print(changes,row.names=FALSE);print(route_summary,row.names=FALSE)
cat('Wrote ',out,'\n',sep='')
