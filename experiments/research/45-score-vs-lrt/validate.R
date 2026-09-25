#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
base <- dirname(script)
source(file.path(base,'R','design.R'))
suppressPackageStartupMessages(library(magmaanlab))
rows <- list()
check <- function(name,actual,expected,tol) {
  delta <- max(abs(actual-expected));ok <- is.finite(delta)&&delta<=tol
  rows[[length(rows)+1L]] <<- data.frame(check=name,max_abs_diff=delta,tolerance=tol,passed=ok)
}
for(p in c(10,20)) for(distribution in c('normal','vm2')) {
  ctx <- prepare(p,distribution);d <- draw_sample(ctx,500,distribution,20260916+p)
  f <- fit_sample(ctx,d);stopifnot(f$converged)
  lr <- fmg_tests(f,tests=lr_tests,data=d)
  sc <- global_score_flip_test(f,d,n_flips=1L,sensitivity='expected',metric='expected')
  key <- paste(p,distribution,sep='_')
  projected <- project_scores(score_components(prepare_inference(f,d)))
  reference <- score_spectrum(projected)
  check(paste0(key,'_primitive_statistic'),projected$statistic,sc$statistic_effective,1e-7)
  check(paste0(key,'_primitive_spectrum'),reference$eigenvalues,sc$eigenvalues,1e-7)
  check(paste0(key,'_primitive_SB'),calibrate_quadratic(reference,'sb')$p_value,sc$p_mean_scaled,1e-9)

  check(paste0(key,'_df'),sc$df,if(p==10)34 else 169,0)
  check(paste0(key,'_score_RLS'),sc$statistic_effective,
    lr$base_statistic[lr$label=='std_rls'],1e-5)
  # Independent Gaussian covariance-score quadratic at the fitted null.
  imp <- magmaanlab:::model_implied(f)
  sigma <- imp$sigma[[1L]]
  sm <- cov(d)*(nrow(d)-1)/nrow(d)
  resid <- solve(sigma,sm)-diag(p)
  check(paste0(key,'_score_trace'),sc$statistic_effective,nrow(d)/2*sum(diag(resid%*%resid)),1e-5)
  sb <- magmaanlab:::infer_fmg_test(sc$statistic_effective,sc$df,sc$eigenvalues,method='sb')
  check(paste0(key,'_score_SB'),sb$p_value,sc$p_mean_scaled,1e-10)
  # Under the regular normal-theory limit the spectrum is all ones;
  # finite-sample empirical spectra need not equal this limit.
  for(method in c('sb','peba')) {
    nt <- magmaanlab:::infer_fmg_test(sc$statistic_effective,sc$df,rep(1,sc$df),
      method=method,param=4)
    check(paste0(key,'_normal_spectrum_',method),nt$p_value,
      pchisq(sc$statistic_effective,sc$df,lower.tail=FALSE),1e-7)
  }
  lv <- lavaan::cfa(ctx$syntax,data=d,estimator='ML',test='Satorra.Bentler')
  check(paste0(key,'_lavaan_LR'),lr$base_statistic[1],unname(lavaan::fitMeasures(lv,'chisq')),1e-4)
  check(paste0(key,'_lavaan_SB'),lr$p_value[2],unname(lavaan::fitMeasures(lv,'pvalue.scaled')),1e-5)
  refs <- as.numeric(semTests::pvalues(lv,tests=lr_tests))
  check(paste0(key,'_semTests'),lr$p_value,refs,1e-4)
}
dir.create(file.path(base,'results','validation'),recursive=TRUE,showWarnings=FALSE)
result <- do.call(rbind,rows)
write.csv(result,file.path(base,'results','validation','checks.csv'),row.names=FALSE)
print(result,row.names=FALSE)
if(!all(result$passed)) stop('validation failed')
