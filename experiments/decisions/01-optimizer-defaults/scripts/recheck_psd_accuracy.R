#!/usr/bin/env Rscript
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script));args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/recheck_psd_accuracy.R [--smoke] [--run-id NAME] [--workers 4]\nRe-audit saved normalized pilot points; no fitting. Requires the normalized PSD accuracy build.\n');quit(save='no')}
source(file.path(here,'..','..','_support','R','helpers.R'));set_single_threaded_math()
for(f in c('families.R','fits.R','psd_normalization.R'))source(file.path(here,'R',f))
opt<-function(k,d)if(k %in% args)args[match(k,args)+1L]else d
smoke<-'--smoke' %in% args;run<-opt('--run-id',if(smoke)'smoke'else'points')
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run))
workers<-as.integer(opt('--workers','4'));stopifnot(length(workers)==1L,!is.na(workers),workers>=1L)
input<-file.path(here,'results','psd-normalization','pilot')
out<-file.path(here,'results','psd-accuracy-normalization',run)
if(file.exists(out))stop('choose a fresh run ID')
x<-read.csv(file.path(input,'fits.csv'));x<-x[x$arm=='normalized' & x$returned,]
keys<-c('pop','model','n','rep','chart','transform','route','arm')
key<-function(z)do.call(paste,c(z[keys],sep='|'))
files<-list.files(file.path(input,'raw'),pattern='^parameters_.*csv$',full.names=TRUE)
parameters<-do.call(rbind,lapply(files,read.csv));parameters<-parameters[parameters$arm=='normalized',]
points<-split(parameters,key(parameters))
if(smoke)x<-rbind(head(x,6),x[x$success & !x$original_accuracy_passed,])
pops<-normalization_populations()
check_point<-function(i){
 row<-x[i,];p<-pops[[row$pop]]
 fm<-p$models[[which(vapply(p$models,`[[`,'','key')==row$model)]]
 fm$std_lv<-row$chart=='std_lv';model<-build_model(fm)
 moments<-draw_moments(p,row$n,row$seed)
 sample<-sample_in_units(moments,transform_factors(row$transform,p$p),fm$meanstructure)
 setup<-normalize_unconstrained_sample(model,sample,fm$std_lv)
 z<-points[[key(row)]];theta<-z$value[order(z$parameter)]
 stopifnot(length(theta)==max(model$partable$free))
 audit<-function(s,t){
  ev<-magmaanlab::magmaan_core$evaluate_at(model,s,t,estimator='ML')
  a<-magmaanlab::frontier_newton_accuracy(ev,psd=TRUE)
  stopifnot(isTRUE(a$unit_normalized))
  a
 }
 a<-audit(sample,theta);b<-audit(setup$sample,theta/setup$theta_units)
 cbind(row[c(keys,'seed','fit_success','original_accuracy','original_accuracy_passed')],
  status=a$status,passed=a$passed,normalized_status=b$status,normalized_passed=b$passed,
  condition=a$condition,normalized_condition=b$condition,distance=a$distance,
  normalized_distance=b$distance,null_directions=a$null_directions,normalized_null_directions=b$null_directions,
  constrained_directions=a$constrained_directions,normalized_constrained_directions=b$constrained_directions)
}
dir.create(out,recursive=TRUE)
write_metadata(file.path(out,'metadata.csv'),values=list(points=nrow(x),smoke=smoke,
 git_head=magmaan_cache_ref()$git_head,git_dirty=magmaan_cache_ref()$git_dirty,
 library=find.package('magmaanlab'),built=utils::packageDescription('magmaanlab')$Built,
 input='psd-normalization/pilot saved theta; samples regenerated with original seeds',
 procedure='no refitting; original and normalized representations of each saved point',
 budget=.01,max_condition=1e12),packages='magmaanlab')
rows<-list();batches<-split(seq_len(nrow(x)),ceiling(seq_len(nrow(x))/200));t0<-proc.time()[['elapsed']]
for(b in seq_along(batches)){
 z<-parallel::mclapply(batches[[b]],check_point,mc.cores=workers,mc.preschedule=TRUE)
 stopifnot(all(vapply(z,is.data.frame,logical(1))))
 rows[[b]]<-do.call(rbind,z)
 done<-max(batches[[b]]);elapsed<-proc.time()[['elapsed']]-t0
 cat(sprintf('%d/%d points; %.1fs; estimated %.1fs remaining\n',done,nrow(x),elapsed,elapsed*(nrow(x)/done-1)))
}
y<-do.call(rbind,rows)
y$verdict_disagreement<-y$status!=y$normalized_status | y$passed!=y$normalized_passed
y$geometry_disagreement<-y$null_directions!=y$normalized_null_directions | y$constrained_directions!=y$normalized_constrained_directions
y$gain<-!y$original_accuracy_passed & y$passed
y$loss<-y$original_accuracy_passed & !y$passed
y$points<-1L
summary<-aggregate(y[c('points','gain','loss','verdict_disagreement','geometry_disagreement')],y[c('chart','route')],sum)
write.csv(y,file.path(out,'point_checks.csv'),row.names=FALSE)
write.csv(summary,file.path(out,'summary.csv'),row.names=FALSE)
write.csv(y[y$gain | y$loss | y$verdict_disagreement | y$geometry_disagreement,],file.path(out,'changes.csv'),row.names=FALSE)
print(summary,row.names=FALSE)
cat('Wrote ',out,'\n',sep='')
