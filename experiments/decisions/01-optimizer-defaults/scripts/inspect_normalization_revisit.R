#!/usr/bin/env Rscript
# Descriptive covariance checks at saved points; no optimization or reclassification.
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script)); args <- commandArgs(TRUE)
if('--help' %in% args) {cat('Usage: Rscript scripts/inspect_normalization_revisit.R [RUN_ID=paired]\n');quit(save='no')}
run <- if(length(args))args[1]else'paired';stopifnot(grepl('^[a-zA-Z0-9_-]+$',run))
for(f in c('families.R','fits.R'))source(file.path(here,'R',f))
out<-file.path(here,'results','normalization-revisit',run)
p<-read.csv(file.path(out,'pairs.csv'));p<-p[p$objective_gain|p$objective_loss,]
theta<-do.call(rbind,lapply(list.files(file.path(out,'raw'),pattern='^parameters_',full.names=TRUE),read.csv))
keys<-c('pop','n','rep','seed','family','model','chart','transform','route')
pops<-all_populations();rows<-list()
for(i in seq_len(nrow(p)))for(arm in c('original_units','normalized')) {
 r<-p[i,keys];fm<-Filter(function(m)m$key==r$model,pops[[r$pop]]$models)[[1]]
 pt<-build_model(fm)$partable
 z<-theta[theta$arm==arm,];for(k in keys)z<-z[z[[k]]==r[[k]],]
 z<-z[order(z$parameter),];stopifnot(nrow(z)>0)
 e<-pt$ustart;free<-pt$free>0;e[free]<-z$value[pt$free[free]]
 lv<-unique(pt$lhs[pt$op=='=~']);ov<-unique(pt$rhs[pt$op=='=~'])
 for(block in c('latent','residual')) {
  names<-if(block=='latent')lv else ov
  C<-matrix(0,length(names),length(names),dimnames=list(names,names))
  for(j in which(pt$op=='~~' & pt$lhs %in% names & pt$rhs %in% names))C[pt$lhs[j],pt$rhs[j]]<-C[pt$rhs[j],pt$lhs[j]]<-e[j]
  if(!length(C)||any(!is.finite(C)))next
  scale<-sqrt(abs(diag(C)));scale[scale==0]<-1
  mineig<-min(eigen(C/outer(scale,scale),symmetric=TRUE,only.values=TRUE)$values)
  rows[[length(rows)+1L]]<-cbind(r,arm,block,min_diagonal_scaled_eigenvalue=mineig)
 }
}
z<-do.call(rbind,rows);write.csv(z,file.path(out,'objective_covariance_checks.csv'),row.names=FALSE)
print(aggregate(z$min_diagonal_scaled_eigenvalue < -1e-6,z[c('route','arm','block')],sum))
