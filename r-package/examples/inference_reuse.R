suppressPackageStartupMessages(library(magmaan))
set.seed(903)
x <- as.data.frame(matrix(rt(360*6,8),360,6) + rnorm(360))
names(x) <- paste0('x',1:6)
syntax <- 'f =~ x1 + a*x2 + b*x3 + x4 + x5 + x6'
control <- list(max_iter=3000L,ftol=1e-12,gtol=1e-8)
close <- function(x,y,tol=1e-7) stopifnot(max(abs(x-y))<tol)
reject <- function(expr, pattern) {
  e <- tryCatch({force(expr);NULL},error=identity)
  stopifnot(inherits(e,'error'),grepl(pattern,conditionMessage(e)))
}
f1 <- magmaan(syntax,x,meanstructure=TRUE,control=control)
f0 <- magmaan(paste(syntax,'a == b',sep='\n'),x,meanstructure=TRUE,control=control)
d <- prepare_inference_data(f1,x)
g1 <- prepare_inference(f1,d); g0 <- prepare_inference(f0,d)
h <- prepare_hypothesis(g0,g1)
for (g in list(g0,g1)) {
  q <- inference_quadratic(g,'score')
  old <- project_scores(score_components(g))
  close(q$statistic,old$statistic)
  close(score_spectrum(q)$eigenvalues,score_spectrum(old)$eigenvalues)
  lr <- inference_quadratic(g,'lr')
  fresh <- g$original_fit
  gof <- fmg_tests(fresh,x,tests=c('sb_ml','peba4_ml'))
  close(calibrate_quadratic(lr,c('sb','peba4'))$p_value,gof$p_value)
  close(fmg_tests(g,tests=c('sb_ml','peba4_ml','peba4_ug_ml'))$p_value,
        fmg_tests(fresh,x,tests=c('sb_ml','peba4_ml','peba4_ug_ml'))$p_value)
  close(inference_information(g),magmaan:::infer_information_expected(fresh))
  close(inference_covariance(g),parameter_covariance(g,inference_information(g)))
  close(inference_covariance(g,TRUE),magmaan:::infer_robust_se_raw(fresh,as.matrix(x))$vcov)
  V <- inference_covariance(g,TRUE)
  R <- diag(length(g$theta))[1,,drop=FALSE]
  close(wald_test(g,R,V)$chi2,drop(crossprod(R%*%g$theta,solve(R%*%V%*%t(R),R%*%g$theta))))
}
qs <- inference_quadratic(h,'score'); ql <- inference_quadratic(h,'lr')
oldscore <- project_scores(score_components(g0,H1=f1))
close(qs$statistic,oldscore$statistic)
close(score_spectrum(qs)$eigenvalues,score_spectrum(oldscore)$eigenvalues)
oldlr <- robust_nested_lrt(f1,f0,x)
close(ql$statistic,oldlr$T_diff)
close(score_spectrum(ql)$eigenvalues,oldlr$eigenvalues)
close(robust_nested_lrt(g1,g0)$eigenvalues,oldlr$eigenvalues)
# All repeated consumers use their retained geometry and reductions.
before <- list(inference_reuse(g0),inference_reuse(g1))
for (i in 1:3) {
  calibrate_quadratic(inference_quadratic(h,'score'),c('sb','peba4'))
  calibrate_quadratic(inference_quadratic(h,'lr'),c('sb','peba4'))
  fmg_tests(g1,tests=c('sb_ml','peba4_ug_ml'))
  robust_nested_lrt(g1,g0)
  inference_covariance(g0,TRUE)
}
stopifnot(identical(before,list(inference_reuse(g0),inference_reuse(g1))),
          before[[1]]$contribution_builds==1,before[[1]]$geometry_builds==1,
          before[[2]]$geometry_builds==1,before[[1]]$u_builds==1,before[[2]]$u_builds==1)
# Explicit tiled storage gives the same result without storing all moment rows.
dt <- prepare_inference_data(f1,x,storage='tiled')
gt <- prepare_inference(f1,dt)
close(score_spectrum(inference_quadratic(gt,'score'))$eigenvalues,
      score_spectrum(inference_quadratic(g1,'score'))$eigenvalues)
close(score_spectrum(inference_quadratic(gt,'lr'))$eigenvalues,
      score_spectrum(inference_quadratic(g1,'lr'))$eigenvalues)
stopifnot(inference_reuse(gt)$contribution_builds==0,inference_reuse(gt)$projection_passes==1)
reject(prepare_hypothesis(g0,gt),'share one prepared')
reject(inference_quadratic(unserialize(serialize(g1,NULL))),'prepare it again')
reject(wald_test(g0,diag(length(g0$theta))[1,,drop=FALSE],inference_covariance(g1)), 'another fit snapshot')
# Edited extracted fits must not silently reuse stale geometry.
changed <- g1$fit; changed$theta[1] <- changed$theta[1]*1.01
cached <- magmaan:::infer_fmg_ugamma_spectra(changed,as.matrix(x))
attr(changed,'magmaan_ntml') <- NULL
close(cached$biased,magmaan:::infer_fmg_ugamma_spectra(changed,as.matrix(x))$biased)
y <- x; y$x1[1] <- y$x1[1]+1
reject(magmaan:::infer_fmg_ugamma_spectra(g1$fit,as.matrix(y)),'observations differ')
# A restricted mean needs the fitted-mean correction, not centered GOF rows.
fm <- magmaan(paste(syntax,'x1 ~ 0*1',sep='\n'),transform(x,x1=x1+1),meanstructure=TRUE,control=control)
gm <- prepare_inference(fm,transform(x,x1=x1+1))
qm <- inference_quadratic(gm,'score'); om <- project_scores(score_components(gm))
close(qm$statistic,om$statistic)
close(score_spectrum(qm)$eigenvalues,score_spectrum(om)$eigenvalues)
gmt <- prepare_inference(fm,prepare_inference_data(fm,transform(x,x1=x1+1),storage='tiled'))
close(score_spectrum(inference_quadratic(gmt,'score'))$eigenvalues,score_spectrum(qm)$eigenvalues)
close(calibrate_quadratic(inference_quadratic(gmt,'score'),'sb')$p_value,
      calibrate_quadratic(qm,'sb')$p_value)
cat('Shared NTML score/LR/GOF/Wald, tiled paths, ownership and construction counts passed.\n')
