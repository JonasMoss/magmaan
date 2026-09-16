# Bounded local timings, no simulation rerun. Run after just r-dev.
suppressPackageStartupMessages(library(magmaan))
set.seed(392)
time_ms <- function(f,n=15L) {
  f()
  1000*system.time(for (i in seq_len(n)) f())[['elapsed']]/n
}
results <- list()
for (p in c(10L,20L)) {
  x <- as.data.frame(matrix(rt(400*p,8),400,p)+rnorm(400))
  names(x) <- paste0('x',seq_len(p))
  syntax <- paste('f =~',paste(c('x1','a*x2','b*x3',paste0('x',4:p)),collapse=' + '))
  opts <- list(max_iter=3000L,ftol=1e-12,gtol=1e-8)
  f1 <- magmaan(syntax,x,meanstructure=TRUE,control=opts)
  f0 <- magmaan(paste(syntax,'a == b',sep='\n'),x,meanstructure=TRUE,control=opts)
  old_global <- function() {
    old_context <- prepare_inference(f0,x)
    calibrate_quadratic(project_scores(score_components(old_context)),c('sb','peba4'))
    fmg_tests(f0,x,tests=c('sb_ml','peba4_ml'))
  }
  prepare <- function() {
    d <- prepare_inference_data(f1,x)
    list(g0=prepare_inference(f0,d),g1=prepare_inference(f1,d))
  }
  fresh_global <- function() {
    d <- prepare_inference_data(f0,x)
    g <- prepare_inference(f0,d)
    calibrate_quadratic(inference_quadratic(g,'score'),c('sb','peba4'))
    calibrate_quadratic(inference_quadratic(g,'lr'),c('sb','peba4'))
  }
  fresh_nested <- function() {
    g <- prepare(); h <- prepare_hypothesis(g$g0,g$g1)
    calibrate_quadratic(inference_quadratic(h,'score'),c('sb','peba4'))
    calibrate_quadratic(inference_quadratic(h,'lr'),c('sb','peba4'))
  }
  old_nested <- function() {
    old_context <- prepare_inference(f0,x)
    calibrate_quadratic(project_scores(score_components(old_context,H1=f1)),c('sb','peba4'))
    robust_nested_lrt(f1,f0,x)
  }
  g <- prepare(); h <- prepare_hypothesis(g$g0,g$g1)
  repeat_all <- function() {
    for (object in list(g$g0,h)) for (test in c('score','lr'))
      calibrate_quadratic(inference_quadratic(object,test),c('sb','peba4'))
    inference_covariance(g$g0,TRUE)
  }
  repeat_all(); before <- list(inference_reuse(g$g0),inference_reuse(g$g1))
  values <- c(prepare=time_ms(prepare),legacy_global=time_ms(old_global),
              shared_global_including_prepare=time_ms(fresh_global),
              legacy_nested=time_ms(old_nested),shared_nested_including_prepare=time_ms(fresh_nested),
              repeat_four_calibrations_and_covariance=time_ms(repeat_all))
  stopifnot(identical(before,list(inference_reuse(g$g0),inference_reuse(g$g1))))
  results[[length(results)+1L]] <- data.frame(p=p,n=400,phase=names(values),milliseconds=unname(values))
}
print(do.call(rbind,results),row.names=FALSE)
