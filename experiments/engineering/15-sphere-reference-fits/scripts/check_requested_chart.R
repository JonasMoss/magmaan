#!/usr/bin/env Rscript
# Required-failure witness for promotion, not a change to the library threshold.
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
source(file.path(here,'R','designs.R'))
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript scripts/check_requested_chart.R\nChecks saved sphere endpoints without fitting or changing markers.\nThe 184,000-loading endpoint must be rejected at the explicit study pole tolerance 1e-4.\nAlso records translation at 1e-6; neither tolerance is promoted here.\n');quit(save='no')}
p<-file.path(here,'results','sphere-translations-inspected')
x<-read.csv(file.path(p,'source_endpoints.csv'));pts<-read.csv(file.path(p,'sphere_parameters.csv'))
base<-magmaanlab::model_spec(model_syntax)
rows<-lapply(seq_len(nrow(x)),function(i){
 point<-pts[pts$case_id==i,];pt<-base$partable;key<-function(z)paste(z$lhs,z$op,z$rhs)
 idx<-match(key(pt),key(point));stopifnot(!anyNA(idx))
 pt$est<-point$est[idx];pt$free<-seq_len(nrow(pt));pt$ustart<-pt$est
 accepted<-function(tol)!inherits(tryCatch(magmaanlab::frontier_reidentify(pt,base,pole_tol=tol),error=function(e)e),'error')
 a<-accepted(1e-6);b<-accepted(1e-4)
 required<-x$batch[i]=='fresh' && x$design[i]=='weak_marker' && x$n[i]==100 && x$rep[i]==4
 if(required)stopifnot(a,!b,x$chart_level[i]<1e-4,x$local_checked[i])
 data.frame(batch=x$batch[i],design=x$design[i],n=x$n[i],rep=x$rep[i],
  chart_level=x$chart_level[i],local_checked=x$local_checked[i],
  translates_at_1e6=a,translates_at_1e4=b,
  study_requested_chart_status=if(b)'passes_chart_proximity_check' else 'reject_requested_chart_near_pole',
  named_required_failure=required)
})
rows<-do.call(rbind,rows);stopifnot(sum(rows$named_required_failure)==1)
out<-file.path(here,'results','requested-chart-policy');dir.create(out,recursive=TRUE,showWarnings=FALSE)
write.csv(rows,file.path(out,'endpoint_checks.csv'),row.names=FALSE)
cat('Required near-pole endpoint rejected at study tolerance 1e-4; unchanged 1e-6 still admits it.\nNo optimizer run or automatic marker replacement.\n')
