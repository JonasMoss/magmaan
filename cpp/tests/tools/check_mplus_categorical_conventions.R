#!/usr/bin/env Rscript
# Independent reference for the P-IV2 convention decision. First run
# Rscript cpp/tests/tools/regen_mplus_probes.R --categorical.
# Reads only that generator's ignored synthetic data; no frontend syntax.
suppressPackageStartupMessages(library(lavaan))
Sys.setenv(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
options(digits=12)
script <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
root <- normalizePath(file.path(dirname(script), '../../..'))
pin <- trimws(readLines(file.path(root, 'cpp/tests/fixtures/lavaan_version.txt'))[1])
stopifnot(gsub('-', '.', pin, fixed=TRUE) == as.character(packageVersion('lavaan')))
base <- path.expand('~/.cache/magmaan-logs/mplus-probes-categorical/P-IV2')
for (par in c('delta','theta')) {
 d <- read.table(file.path(base,paste0(par,'_ordinal_scalar'),'probe.dat'))
 names(d) <- c(paste0('u',1:6),'g')
 syntax <- c('f1 =~ c(1,1)*u1 + c(l2,l2)*u2 + c(l3,l3)*u3',
 'f2 =~ c(1,1)*u4 + c(l5,l5)*u5 + c(l6,l6)*u6',
 'f1 ~~ c(NA,NA)*f1; f2 ~~ c(NA,NA)*f2; f1 ~~ c(NA,NA)*f2',
 'f1 ~ c(0,NA)*1; f2 ~ c(0,NA)*1')
 for (j in 1:6) {
 syntax <- c(syntax, sprintf('u%d | c(t%d1,t%d1)*t1 + c(t%d2,t%d2)*t2',j,j,j,j,j),sprintf('u%d ~ c(0,0)*1',j))
 syntax <- c(syntax, if (par=='delta') sprintf('u%d ~*~ c(1,NA)*u%d',j,j) else sprintf('u%d ~~ c(1,NA)*u%d',j,j))
 }
 for (convention in c('lavaan','Mplus')) {
 fit <- lavaan(paste(syntax,collapse='\n'),data=d,group='g', ordered=paste0('u',1:6),parameterization=par,estimator='WLSMV',mimic=convention,auto.var=FALSE,auto.fix.first=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,meanstructure=TRUE)
 stopifnot(lavInspect(fit,'converged'), all(unlist(lavInspect(fit,'theta')) >= -1e-8))
 cat(convention,par, lavInspect(fit,'converged'), '\n')
 values <- fitMeasures(fit,c('npar','df','chisq','chisq.scaled','df.scaled','chisq.scaling.factor'))
 for (name in names(values)) cat(name, sprintf('%.12f', values[[name]]), '\n')
 }
}
