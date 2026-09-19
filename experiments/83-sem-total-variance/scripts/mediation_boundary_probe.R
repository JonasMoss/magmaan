#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Usage: Rscript scripts/mediation_boundary_probe.R [--smoke] [--reps N] [--budget-sec N] [--seed-base N] [--results-dir PATH]\n',
      'Diagnose stressed mediation draw 4 saved by native_structure_probe.R.\n',
      'Default: 6 random perturbations per endpoint, 42 fits, 30s soft budget.\n',
      'Smoke: 1 perturbation per endpoint, 22 fits. No oracle supplied as a start.\n',sep='')
  quit()
}
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script)); repo <- normalizePath(file.path(here,'../..'))
o <- list(reps=6L,budget_sec=30,seed_base=20260920L,results_dir=file.path(here,'results/mediation-boundary'))
i <- 1L
while(i<=length(args)) {
 a<-args[i]
 if(a=='--smoke') o$reps<-1L else {
  if(i==length(args)) stop('Missing value: ',a)
  i<-i+1L;k<-gsub('-','_',sub('^--','',a))
  if(!k %in% names(o)) stop('Unknown option: ',a)
  o[[k]]<-if(k=='results_dir')args[i] else as.numeric(args[i])
 };i<-i+1L
}
stopifnot(is.finite(o$reps),o$reps>=1,o$reps==as.integer(o$reps),
 is.finite(o$budget_sec),o$budget_sec>0,is.finite(o$seed_base),o$seed_base>=0,o$seed_base<2e9)
source(file.path(repo,'experiments/_support/R/helpers.R'));set_single_threaded_math()
input<-file.path(here,'results/native-structures/discrepancies.rds')
if(!file.exists(input))stop('Run scripts/native_structure_probe.R first')
ds<-readRDS(input)
selected<-Filter(function(d)d$case=='mediation' && d$setting=='stress' && d$replicate==4,ds)
stopifnot(length(selected)==1);d<-selected[[1]];pt<-d$spec$partable
stopifnot(all(pt$op %in% c('=~','~','~~')),all(pt$lhs[pt$op=='~~']==pt$rhs[pt$op=='~~']))
idx<-function(lhs,op,rhs) { k<-pt$free[pt$lhs==lhs & pt$op==op & pt$rhs==rhs];stopifnot(length(k)==1,k>0);k }
# This diagnostic is deliberately restricted to the three-factor mediation model.
mat<-function(x) {
 L<-matrix(0,9,3);B<-matrix(0,3,3);P<-diag(3)*0;T<-diag(9)*0
 for(k in seq_len(nrow(pt))) {
  r<-pt[k,];v<-if(r$free>0)x[r$free] else r$ustart
  i<-as.integer(sub('^[fx]','',r$lhs));j<-as.integer(sub('^[fx]','',r$rhs))
  if(r$op=='=~')L[j,i]<-v else if(r$op=='~')B[i,j]<-v else if(startsWith(r$lhs,'f'))P[i,j]<-v else T[i,j]<-v
 }
 A<-solve(diag(3)-B);C<-A%*%P%*%t(A)
 list(L=L,B=B,P=P,T=T,A=A,C=C,Sigma=L%*%C%*%t(L)+T)
}
objective<-function(x) {
 V<-mat(x)$Sigma;S<-d$stats$S[[1]]
 as.numeric(determinant(V,logarithm=TRUE)$modulus+sum(diag(solve(V,S)))-
            determinant(S,logarithm=TRUE)$modulus-nrow(S))
}
dir.create(o$results_dir,recursive=TRUE,showWarnings=FALSE)
unlink(file.path(o$results_dir,c('parameters.csv','geometry.csv','ridge.csv','fits.csv','metadata.csv')))
t0<-Sys.time();elapsed<-function()as.numeric(difftime(Sys.time(),t0,units='secs'))
q2<-idx('f2','~~','f2');a<-idx('f2','~','f1');b<-idx('f3','~','f2');cprime<-idx('f3','~','f1')
parameters<-pt[pt$free>0,c('lhs','op','rhs','free')]
for(nm in names(d$fits)) parameters[[nm]]<-d$fits[[nm]]$theta[parameters$free]
write.csv(parameters,file.path(o$results_dir,'parameters.csv'),row.names=FALSE)
geometry<-do.call(rbind,lapply(names(d$fits),function(nm) {
 x<-d$fits[[nm]]$theta;m<-mat(x)
 stopifnot(abs(objective(x)-2*d$fits[[nm]]$fmin)<1e-10)
 data.frame(endpoint=nm,objective=objective(x),mediator_disturbance=m$P[2,2],
   outcome_disturbance=m$P[3,3],direct=x[cprime],mediated_path=x[b],total=x[cprime]+x[a]*x[b],
   latent_min_eigen=min(eigen(m$C,symmetric=TRUE,only.values=TRUE)$values))
}))
write.csv(geometry,file.path(o$results_dir,'geometry.csv'),row.names=FALSE)
# At q2=0, changing b and compensating c leaves the entire observed covariance fixed.
x<-d$fits$diagonal$theta;stopifnot(x[q2]<1e-8);x[q2]<-0
m<-mat(x);W<-solve(m$Sigma);G<-W-W%*%d$stats$S[[1]]%*%W
ridge<-do.call(rbind,lapply(c(-3,-1,0,.5,1,2,3),function(slope) {
 z<-x;z[b]<-slope;z[cprime]<-x[cprime]-x[a]*(slope-x[b])
 v<-m$L[,2]+slope*m$L[,3];score<-drop(t(v)%*%G%*%v)
 h<-1e-6*m$C[2,2];zz<-z;zz[q2]<-h
 fd<-(objective(zz)-objective(z))/h
 native<-magmaan::magmaan_core$evaluate_at(d$spec,d$stats,z,estimator='ML')
 opened<-magmaan::magmaan_core$evaluate_at(d$spec,d$stats,zz,estimator='ML')
 stopifnot(max(abs(mat(z)$Sigma-m$Sigma))<1e-12,abs(fd-score)<1e-4,
  abs(2*native$fmin-objective(z))<1e-10,abs(2*opened$fmin-objective(zz))<1e-10)
 data.frame(slope=slope,objective=objective(z),covariance_gap=max(abs(mat(z)$Sigma-m$Sigma)),
   variance_derivative=score,finite_difference=fd,opened_objective=objective(zz))
}))
write.csv(ridge,file.path(o$results_dir,'ridge.csv'),row.names=FALSE)
rows<-list();exhausted<-FALSE;planned<-4*(2+o$reps)+10
for(endpoint in c('none','diagonal')) {
 x<-d$fits[[endpoint]]$theta;m<-mat(x)
 variances<-pt[pt$op=='~~' & pt$free>0,]
 reopen<-function(z) {
  for(k in seq_len(nrow(variances))) {
   r<-variances[k,];j<-as.integer(sub('^[fx]','',r$lhs))
   scale<-if(startsWith(r$lhs,'f'))m$C[j,j] else d$stats$S[[1]][j,j]
   z[r$free]<-max(z[r$free],.01*scale)
  };z
 }
 starts<-list(unchanged=x,reopen=reopen(x))
 for(r in seq_len(o$reps)) {
  set.seed(o$seed_base+r);z<-reopen(x)
  for(k in which(pt$free>0)) {
   p<-pt[k,];j<-p$free
   if(p$op=='~') {
    lhs<-as.integer(sub('f','',p$lhs));rhs<-as.integer(sub('f','',p$rhs))
    z[j]<-z[j]+rnorm(1)*sqrt(m$C[lhs,lhs]/m$C[rhs,rhs])
   } else z[j]<-z[j]*exp(rnorm(1,sd=.2))
  }
  starts[[paste0('jitter_',r)]]<-z
 }
 if(endpoint=='diagonal') for(shift in c(-2,-1,1,2)) {
  z<-x;delta<-shift*sqrt(m$C[3,3]/m$C[2,2]);z[b]<-z[b]+delta;z[cprime]<-z[cprime]-x[a]*delta
  starts[[paste0('ridge_',shift)]]<-reopen(z)
 }
 if(endpoint=='diagonal') {
  W<-solve(m$Sigma);G<-W-W%*%d$stats$S[[1]]%*%W
  candidates<-c(-2,-1,1,2)*sqrt(m$C[3,3]/m$C[2,2])+x[b]
  scores<-vapply(candidates,function(slope) {v<-m$L[,2]+slope*m$L[,3];drop(t(v)%*%G%*%v)},0.0)
  if(min(scores)< -1e-8) {
   z<-x;z[b]<-candidates[which.min(scores)];z[cprime]<-x[cprime]-x[a]*(z[b]-x[b])
   starts$score_guided<-reopen(z)
  }
 }
 for(policy in names(starts)) for(method in c('none','diagonal')) {
  if(elapsed()>o$budget_sec){exhausted<-TRUE;break}
  z<-starts[[policy]];mm<-mat(z)
  stopifnot(min(diag(mm$P))>=0,min(diag(mm$T))>=0)
  # Explicit starts in the same SEM; the PSD solver retains all constraints.
  spec<-d$spec;k<-spec$partable$free;free<-k>0;spec$partable$ustart[free]<-z[k[free]]
  tick<-elapsed()
  f<-tryCatch(magmaan::frontier_fit_ml_psd(spec,d$stats,preconditioning=method,
   control=list(max_iter=1000L,gtol=1e-8,ftol=1e-12)),error=function(e)e)
  seconds<-elapsed()-tick;err<-inherits(f,'error');g<-if(err)NULL else f$diagnostics$geometric_stationarity
  rows[[length(rows)+1L]]<-data.frame(endpoint=endpoint,policy=policy,preconditioning=method,
   start_objective=objective(z),objective=if(err)NA_real_ else 2*f$fmin,
   seconds=seconds,evaluations=if(err)NA_integer_ else f$f_evals,
   accepted=!err && isTRUE(f$converged) && isTRUE(f$diagnostics$admissibility$admissible) && isTRUE(g$cone_stationary),
   nullity=if(err)NA_integer_ else g$covariance_nullity,
   mediator_disturbance=if(err)NA_real_ else f$theta[q2],
   mediated_path=if(err)NA_real_ else f$theta[b],
   error=if(err)conditionMessage(f) else '')
  write.csv(do.call(rbind,rows),file.path(o$results_dir,'fits.csv'),row.names=FALSE)
 }
 cat(sprintf('%s: %d/%d fits, %.2fs elapsed\n',endpoint,length(rows),planned,elapsed()));flush.console()
 if(exhausted)break
}
write_metadata(file.path(o$results_dir,'metadata.csv'),values=c(o,list(complete=!exhausted,
 elapsed_seconds=elapsed(),planned_fits=planned,completed_fits=length(rows),
 git_head=git_scalar(c('rev-parse','HEAD'),root=repo),input_md5=unname(tools::md5sum(input)),
 source_md5=unname(tools::md5sum(script)))),packages='magmaan')
cat('Results: ',o$results_dir,'\n',sep='');if(exhausted)quit(status=2L)
