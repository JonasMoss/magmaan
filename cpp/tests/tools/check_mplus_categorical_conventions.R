#!/usr/bin/env Rscript
# Independent P-IV2 references and optional native-route gate. First run
# Rscript cpp/tests/tools/regen_mplus_probes.R --categorical.
# Reads only that generator's ignored synthetic data; no frontend syntax.
suppressPackageStartupMessages(library(lavaan))
Sys.setenv(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
options(digits=12)
check_native <- '--native' %in% commandArgs(trailingOnly=TRUE)
if (check_native) suppressPackageStartupMessages(library(magmaanlab))
native_failures <- character()
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
 syntax <- paste(syntax,collapse='\n')
 for (convention in c('lavaan','Mplus')) {
 fit <- lavaan(syntax,data=d,group='g', ordered=paste0('u',1:6),parameterization=par,estimator='WLSMV',mimic=convention,auto.var=FALSE,auto.fix.first=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,meanstructure=TRUE)
 stopifnot(lavInspect(fit,'converged'), all(unlist(lavInspect(fit,'theta')) >= -1e-8))
 cat(convention,par, lavInspect(fit,'converged'), '\n')
 values <- fitMeasures(fit,c('npar','df','chisq','chisq.scaled','df.scaled','chisq.scaling.factor'))
 for (name in names(values)) cat(name, sprintf('%.12f', values[[name]]), '\n')
 if (check_native && convention == 'lavaan') {
   spec <- model_spec(syntax, ordered=paste0('u',1:6),
     parameterization=par, group='g', group_labels=c('1','2'),
     auto_var=FALSE, auto_fix_first=FALSE, auto_cov_lv_x=FALSE,
     auto_cov_y=FALSE, meanstructure=TRUE)
   for (route in c('native','lavaan-0.7.2')) {
     actual <- tryCatch(fit_model(spec,d,estimator='DWLS',
       options=if (route=='native') NULL else list(preset=route)),error=identity)
     if (inherits(actual,'error')) {
       detail <- paste(par,route,conditionMessage(actual))
       cat('native-route error:',detail,'\n')
       native_failures <- c(native_failures,detail)
       next
     }
     mp <- actual$partable; lp <- parTable(fit)
     key <- function(p) paste(p$lhs,p$op,p$rhs,p$group,sep='\r')
     idx <- match(key(mp),key(lp))
     use <- !is.na(idx) & mp$op %in% c('=~','|','~1','~*~')
     # Retained-estimate reporting uses the existing lavaan compatibility bundle.
     reporting <- convention_inference(actual, 'WLSMV')
     common_free <- !is.na(idx) & mp$free > 0L &
       mp$op %in% c('=~','|','~1')
     if (!reporting$covariance_available || !reporting$test$available) {
       cat('convention-unavailable',par,route,reporting$covariance_reason,
         reporting$covariance_detail,reporting$test$reason,reporting$test$detail,'\n')
     } else {
       actual_se <- sqrt(diag(reporting$covariance))[mp$free[common_free]]
       oracle_se <- lp$se[idx[common_free]]
       se_error <- abs(actual_se-oracle_se)
       test_error <- abs(reporting$test$statistic-fitMeasures(fit,'chisq.scaled'))
       cat('retained-reporting',par,route,'df',reporting$test$df,
         'max_common_se_difference',sprintf('%.12f',max(se_error)),
         'test_difference',sprintf('%.12f',test_error),'\n')
       if (any(se_error > 1e-5*(1+pmax(abs(actual_se),abs(oracle_se)))) ||
           test_error > 1e-5*(1+abs(fitMeasures(fit,'chisq.scaled'))) ||
           reporting$test$df != fitMeasures(fit,'df'))
         native_failures <- c(native_failures,paste(par,route,
           'available WLSMV convention disagrees with default lavaan'))
     }
     errors <- abs(mp$est[use]-lp$est[idx[use]])
     difference <- max(errors)
     within_tolerance <- all(errors <= 1e-5 *
       (1 + pmax(abs(mp$est[use]),abs(lp$est[idx[use]]))))
     scales <- mp$group==2 & mp$lhs %in% paste0('u',1:6) & mp$lhs==mp$rhs &
       mp$op=='~~'
     cat('native-route',par,route,'converged',actual$converged,
       'max_common_estimate_difference',sprintf('%.12f',difference),
       'free_group2_scale_rows',sum(mp$free[scales]>0),'\n')
     if (!isTRUE(actual$converged) || !within_tolerance ||
         sum(mp$free[scales]>0)!=6L)
       native_failures <- c(native_failures,paste(par,route,
         'does not preserve the explicit SCALAR model'))
   }
 }
 }
}

if (length(native_failures)) stop(paste(native_failures,collapse='\n'),call.=FALSE)
