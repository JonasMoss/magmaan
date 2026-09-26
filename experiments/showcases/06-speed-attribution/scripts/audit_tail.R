#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args) {
  cat('Rscript scripts/audit_tail.R [pilot-run-id] [audit-run-id]\n',
      'Defaults: pilot tail-audit. Requires the pilot inputs and Python mpmath.\n',
      'Writes tolerance sweep and an independent Erlang-chain reference; no timing claims.\n',sep='');quit()
}
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here<-dirname(dirname(script))
run<-if(length(args)) args[1] else 'pilot'
id<-if(length(args)>1L) args[2] else 'tail-audit'
stopifnot(all(grepl('^[A-Za-z0-9_-]+$',c(run,id))))
out<-file.path(here,'results',id)
if(dir.exists(out)) stop('Audit output already exists: ',out)
suppressPackageStartupMessages(library(magmaan))
source(file.path(here,'R/workloads.R'))
check_lavaan_adapter()
cases<-readRDS(file.path(here,'results',run,'inputs.rds'))
f<-lav_raw(cases$hs,'peba4_wald')
eig<-sort(f@test$peba4_ml$UGamma.eigenvalues,decreasing=TRUE)
statistic<-f@test$standard$stat
stopifnot(length(eig)==24L,identical(f@Options$information,c('expected','expected')))
# Reconstruct the four penalized blocks from their definition.
k<-length(eig)/4
weights<-(rep(colMeans(matrix(eig,nrow=k)),each=k)+mean(eig))/2
pm<-calibrate_quadratic(quadratic_reference(statistic,length(eig),eig),'peba4')$p_value
rows<-lapply(c(1e-6,1e-8,1e-10,1e-12,1e-13),function(eps) {
  z<-lavaan:::lav_test_fmg_imhof(statistic,weights,epsabs=eps,epsrel=eps)
  data.frame(tolerance=eps,lavaan_p=z$Qq,magmaan_p=pm,
    integral_error_estimate=z$abserr,relative_gap=z$Qq/pm-1)
})
x<-do.call(rbind,rows)
stopifnot(abs(x$lavaan_p[1]-f@test$peba4_ml$pvalue)<1e-14)
dir.create(out,recursive=TRUE)
write.csv(x,file.path(out,'tolerance-sweep.csv'),row.names=FALSE)
writeLines(c('weight',sprintf('%.17g',weights)),file.path(out,'weights.csv'))
writeLines(sprintf('%.17g',statistic),file.path(out,'statistic.txt'))
status<-system2('python3',c(shQuote(file.path(here,'scripts/erlang_reference.py')),shQuote(out)))
if(status!=0L) stop('Independent reference failed; install mpmath or inspect its diagnostic.')
ref<-read.csv(file.path(out,'independent.csv'))$p_value
stopifnot(abs(pm/ref-1)<1e-6,abs(tail(x$lavaan_p,1)/ref-1)<1e-6)
writeLines(c(capture.output(sessionInfo()),'Independent engine: Python mpmath, 50/70 decimal digits.',
  'Same statistic and spectrum throughout; no fits or information matrices changed.'),
  file.path(out,'session.txt'))
print(x,digits=10,row.names=FALSE)
cat('Independent reference:',sprintf('%.17g',ref),'\nResults:',out,'\n')
