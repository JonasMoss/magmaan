#!/usr/bin/env Rscript
# Bounded diagnostic; no changes to the simulation or its statistical procedures.
args <- commandArgs(TRUE)
if('--help' %in% args) {
  cat('Usage: Rscript diagnose_runtime.R [--smoke]\n18 fixed design draws; 20 warmed calls per phase (2 with --smoke).\n')
  quit(status=0)
}
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
base <- dirname(script); source(file.path(base,'R','design.R'))
suppressPackageStartupMessages(library(magmaanlab))
# Diagnostic composition for single-group complete-data covariance models only.
# One U-factor and one reduced empirical meat; retain the uncentered score correction.
shared_primary <- function(f, d) {
 u <- magmaan_core$robust_build_u_factor_fit(f, bread = "expected")
 Z <- magmaan_core$robust_casewise_contributions(f$partable,as.matrix(d))
 Y <- Z %*% u$B
 M <- crossprod(Y)/nrow(d)
 E <- u$blocks[[1]]$S-u$blocks[[1]]$Sigma_hat
 v <- sqrt(nrow(d))*drop(crossprod(u$B,E[lower.tri(E,diag=TRUE)]))
 lr_ev <- magmaan_core$robust_ugamma_eigenvalues(M)
 sc_ev <- magmaan_core$robust_ugamma_eigenvalues(M+tcrossprod(v)/nrow(d))
 # Match the existing calibrators' treatment of roundoff at zero eigenvalues.
 stopifnot(min(lr_ev)>-1e-8,min(sc_ev)>-1e-8)
 lr_ev <- pmax(lr_ev,0); sc_ev <- pmax(sc_ev,0)
 t_lr <- magmaanlab:::infer_chi2_stat(magmaanlab:::fit_sample_stats(f),f$fmin)
 list(lr=calibrate_quadratic(quadratic_reference(t_lr,u$df,lr_ev),c('sb','peba4')),
      score=calibrate_quadratic(quadratic_reference(sum(v*v),u$df,sc_ev),c('sb','peba4')),
      lr_ev=lr_ev,score_ev=sc_ev,statistic=sum(v*v))
}

outdir <- file.path(base,'results','runtime-audit');dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
calls <- if('--smoke' %in% args) 2L else 20L
measure_calls <- function(fun) {
  invisible(fun());gc(FALSE)
  t <- Sys.time(); for(i in seq_len(calls)) invisible(fun())
  1000*as.numeric(difftime(Sys.time(),t,units='secs'))/calls
}
grid <- expand.grid(p=c(10L,20L),n=c(100L,200L,500L),distribution=c('normal','t10','vm2'),stringsAsFactors=FALSE)
# Exactly two structural models, prepared outside all cell/replicate timers.
models <- setNames(lapply(c(10,20),function(p)prepare_model(population(p)$syntax)),c('10','20'))
rows <- checks <- list()
for(j in seq_len(nrow(grid))) {
 cell <- grid[j,];ctx <- prepare(cell$p,cell$distribution);model <- models[[as.character(cell$p)]]
 d <- draw_sample(ctx,cell$n,cell$distribution,20260917+j*100000+1)
 prepared_data <- prepare_data(model,d)
 control <- list(max_iter=4000L,ftol=1e-12,gtol=1e-8)
 f <- fit_sample(ctx,d)
 fp <- estimate(model,prepared_data,control=control)
 stopifnot(f$converged,fp$converged,max(abs(f$theta-fp$theta))<1e-5)
 context <- prepare_inference(f,d);snapshot <- magmaanlab:::.inference_fit(context)
 components <- score_components(context);projected <- project_scores(components)
 spectrum <- score_spectrum(projected)
 centered <- score_spectrum(project_scores(components,center=TRUE))
 X <- as.matrix(d)
 ev <- magmaanlab:::infer_fmg_ugamma_spectra(snapshot,X,FALSE)
 lr <- fmg_tests(context,tests=lr_tests)
 primary <- function(context) {
   a <- fmg_tests(context,tests=c('sb_ml','peba4_ml'))
   b <- calibrate_quadratic(score_spectrum(project_scores(score_components(context))),c('sb','peba4'))
   list(lr=a,score=b)
 }
 # Diagnostic hypothesis only: centered/uncentered meats from one retained projection.
 # Independently compare the resulting LR spectrum and all four p-values.
 centered_meat <- projected$meat-tcrossprod(projected$score)/cell$n
 reused <- score_spectrum(score_quadratic(projected$score,projected$metric,centered_meat))
 shared <- shared_primary(snapshot,d)
 expected <- primary(context)
 checks[[j]] <- data.frame(cell_id=j,cell,
   parameter_diff=max(abs(f$theta-fp$theta)),
   shared_score_stat_diff=abs(shared$statistic-projected$statistic),
   shared_score_spectrum_diff=max(abs(shared$score_ev-spectrum$eigenvalues)),
   shared_four_p_diff=max(abs(c(shared$lr$p_value,shared$score$p_value)-
                            c(expected$lr$p_value,expected$score$p_value))),
   centered_rank_one_diff=max(abs(sort(reused$eigenvalues)-sort(centered$eigenvalues))),
   lr_vs_centered_spectrum=max(abs(sort(ev$biased)-sort(centered$eigenvalues))),
   lr_vs_uncentered_spectrum=max(abs(sort(ev$biased)-sort(spectrum$eigenvalues))))
 phases <- list(
   model_preparation=function()prepare_model(ctx$syntax),
   data_preparation=function()prepare_data(model,d),
   fit_raw_syntax=function()fit_sample(ctx,d),
   fit_prepared_model_fresh_data=function()estimate(model,prepare_data(model,d),control=control),
   fit_prepared_data=function()estimate(model,prepared_data,control=control),
   inference_context=function()prepare_inference(f,d),
   lr_eight_tests=function()fmg_tests(context,tests=lr_tests),
   lr_primary_two=function()fmg_tests(context,tests=c('sb_ml','peba4_ml')),
   lr_sb_only=function()fmg_tests(context,tests='sb_ml'),
   lr_spectra_biased=function()magmaanlab:::infer_fmg_ugamma_spectra(snapshot,X,FALSE),
   lr_spectra_both=function()magmaanlab:::infer_fmg_ugamma_spectra(snapshot,X,TRUE),
   score_components=function()score_components(context),
   score_projection=function()project_scores(components),
   score_spectrum=function()score_spectrum(projected),
   score_calibration_three=function()calibrate_quadratic(spectrum,c('sb','peba2','peba4')),
   score_calibration_primary=function()calibrate_quadratic(spectrum,c('sb','peba4')),
   four_tests_existing_context=function()primary(context),
   four_tests_shared_geometry=function()shared_primary(snapshot,d),
   shared_prepared_pipeline=function(){fit<-estimate(model,prepare_data(model,d),control=control);shared_primary(fit,d)},
   generation=function()draw_sample(ctx,cell$n,cell$distribution,20260917+j*100000+1),
   full_raw_pipeline=function(){
     fit<-fit_sample(ctx,d); ic<-prepare_inference(fit,d)
     lr<-fmg_tests(ic,tests=lr_tests)
     q<-project_scores(score_components(ic)); ref<-score_spectrum(q)
     cal<-calibrate_quadratic(ref,c('sb','peba2','peba4'))
     sw<-tryCatch(calibrate_quadratic(score_sandwich(q),'std'),error=identity)
     list(lr=lr,score=cal,sandwich=sw)
   },
   runner_replication=function()one_rep(transform(cell,cell_id=j),1L,ctx,20260917),
   primary_raw_pipeline=function(){fit<-fit_sample(ctx,d);primary(prepare_inference(fit,d))},
   primary_prepared_pipeline=function(){fit<-estimate(model,prepare_data(model,d),control=control);primary(prepare_inference(fit))},
   lavaan_fit_sb=function()suppressWarnings(lavaan::cfa(ctx$syntax,d,estimator='ML',se='none',test='Satorra.Bentler',baseline=FALSE))
 )
 for(phase in names(phases)) {
   z <- tryCatch(measure_calls(phases[[phase]]),error=identity)
   rows[[length(rows)+1L]] <- data.frame(cell_id=j,cell,phase=phase,calls=calls,
     ms=if(inherits(z,'error'))NA_real_ else z,error=if(inherits(z,'error'))conditionMessage(z) else '')
 }
 message('Runtime audit ',j,'/18')
}
result<-do.call(rbind,rows)
write.csv(result,file.path(outdir,'phases.csv'),row.names=FALSE)
check_table <- do.call(rbind,checks)
write.csv(check_table,file.path(outdir,'checks.csv'),row.names=FALSE)
stopifnot(max(check_table$shared_four_p_diff)<1e-7,
          max(check_table$shared_score_stat_diff)<1e-7,
          max(check_table$shared_score_spectrum_diff)<1e-7)
writeLines(c(capture.output(sessionInfo()),paste('Calls per phase:',calls),
 'Seed base: 20260917; replicate 1 in each cell; one worker, one BLAS thread.'),file.path(outdir,'session.txt'))
print(aggregate(ms~phase,result,mean),row.names=FALSE)
