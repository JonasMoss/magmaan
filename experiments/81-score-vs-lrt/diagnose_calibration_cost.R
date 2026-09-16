#!/usr/bin/env Rscript
# Compare fitting, LR inference and calibration under identical prepared-input boundaries.
args<-commandArgs(TRUE)
if('--help'%in%args){cat('Usage: Rscript diagnose_calibration_cost.R [--smoke] [--legacy-and-wrappers]\nSix cases, three warmed batches per phase; --smoke uses one short batch.\n');quit(status=0)}
script<-normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
base<-dirname(script);source(file.path(base,'R','design.R'))
suppressPackageStartupMessages(library(magmaan))
outdir<-file.path(base,'results',if('--legacy-and-wrappers'%in%args)'calibration-wrapper-audit' else 'calibration-audit');dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
files<-unlist(lapply(c('magmaan','lavaan',if('--legacy-and-wrappers'%in%args)'semTests'),function(pkg)list.files(find.package(pkg),pattern='\\.(so|rdb|rdx)$|^DESCRIPTION$',recursive=TRUE,full.names=TRUE)))
hashes<-tools::md5sum(files)
started<-Sys.time()
cases<-data.frame(case=c('old_6','old_9','old_12','score_10','score_20','score_20_small_skew'),
 p=c(6,9,12,10,20,20),n=c(200,500,1000,500,500,100),
 distribution=c(rep('normal',5),'vm2'),meanstructure=c(rep(TRUE,3),rep(FALSE,3)))
measure<-function(fun,calls){t<-Sys.time();for(i in seq_len(calls))invisible(fun());1000*as.numeric(difftime(Sys.time(),t,units='secs'))/calls}
rows<-checks<-list();core<-magmaan_core
for(j in seq_len(nrow(cases))){
 c<-cases[j,]
 if(j<=3){
  set.seed(20260916+j);eta<-rnorm(c$n)
  d<-as.data.frame(.7*matrix(rep(eta,c$p),c$n,c$p)+sqrt(.51)*matrix(rnorm(c$n*c$p),c$n,c$p))
  names(d)<-paste0('x',seq_len(c$p));syntax<-paste('f =~',paste(names(d),collapse=' + '))
 }else{
  ctx<-prepare(c$p,c$distribution)
  cell_id<-if(j==4)5L else if(j==5)6L else 14L
  d<-draw_sample(ctx,c$n,c$distribution,20260917+cell_id*100000+1);syntax<-ctx$syntax
 }
 model<-prepare_model(syntax,meanstructure=c$meanstructure);pd<-prepare_data(model,d)
 ctrl<-list(max_iter=4000L,ftol=1e-12,gtol=1e-8)
 magfit<-function()estimate(model,pd,control=ctrl)
 # The earlier benchmark used default controls; retain this additional comparator.
 magfit_default<-function()estimate(model,pd)
 unfitted<-lapply(c(none='none',standard='standard',sb='Satorra.Bentler',peba='peba4_ml'),function(test){
  x<-lavaan::cfa(syntax,d,meanstructure=c$meanstructure,se='none',test=test,baseline=FALSE,h1=FALSE,do.fit=FALSE)
  x@Options$do.fit<-TRUE;x
 })
 lavfit<-function(kind){x<-unfitted[[kind]];suppressWarnings(lavaan::lavaan(
  slotOptions=x@Options,slotParTable=x@ParTable,slotSampleStats=x@SampleStats,
  slotData=x@Data,slotModel=x@Model,slotCache=x@Cache))}
 f<-magfit();lf<-lavfit('peba');stopifnot(f$converged,lavaan::lavInspect(lf,'converged'))
 ic<-prepare_inference(f);snapshot<-magmaan:::.inference_fit(ic)
 X<-as.matrix(d);fm<-fmg_tests(ic,tests=c('sb_ml','peba4_ml'))
 pscore<-project_scores(score_components(ic));sscore<-score_spectrum(pscore)
 ev<-magmaan:::infer_fmg_ugamma_spectra(snapshot,X,FALSE)$biased
 df<-fm$df[1];T<-fm$base_statistic[1];eig<-sort(pmax(ev,0),decreasing=TRUE)
 ug<-lavaan:::lav_test_fmg_ugamma(lavobject=lf)
 u<-core$robust_build_u_factor_fit(snapshot)
 # Materialize Gamma only outside timing to isolate the reduced eigensolve.
 Z<-core$robust_casewise_contributions(f$partable,X)
 if(c$meanstructure)Z<-cbind(scale(X,center=TRUE,scale=FALSE),Z)
 M<-crossprod(Z%*%u$B)/c$n
 stopifnot(max(abs(sort(core$robust_ugamma_eigenvalues(M))-sort(ev)))<1e-7)
 magcal<-function(method)magmaan:::infer_fmg_test(T,df,eig,method=method,param=4)$p_value
 lavcal<-function(method)if(method=='sb')lavaan:::lav_test_fmg_sb(T,eig) else lavaan:::lav_test_fmg_peba(T,eig,j=4L)
 ls<-lavfit('sb')
 checks[[j]]<-data.frame(c,df=df,
  fit_peba_p_diff=abs(fm$p_value[2]-lf@test$peba4_ml$pvalue),
  fit_sb_p_diff=abs(fm$p_value[1]-ls@test$satorra.bentler$pvalue),
  same_spectrum_peba_diff=abs(magcal('peba')-lavcal('peba')),
  same_spectrum_sb_diff=abs(magcal('sb')-lavcal('sb')),
  default_control_parameter_diff=max(abs(f$theta-magfit_default()$theta)))
 phases<-list(
  mag_fit=magfit,mag_fit_default=magfit_default,
  lav_fit=function()lavfit('none'),
  mag_fit_sb=function(){ff<-magfit();fmg_tests(ff,tests='sb_ml')},
  lav_fit_sb=function()lavfit('sb'),
  mag_fit_peba=function(){ff<-magfit();fmg_tests(ff,tests='peba4_ml')},
  lav_fit_peba=function()lavfit('peba'),
  mag_fit_peba_score=function(){ff<-magfit();cx<-prepare_inference(ff);a<-fmg_tests(cx,tests='peba4_ml');b<-calibrate_quadratic(score_spectrum(project_scores(score_components(cx))),'peba4');list(a,b)},
  mag_postfit_sb=function()fmg_tests(ic,tests='sb_ml'),
  mag_postfit_peba=function()fmg_tests(ic,tests='peba4_ml'),
  lav_postfit_peba=function()lavaan:::lav_test_fmg(lavobject=lf,test='peba4_ml'),
  mag_spectra=function()magmaan:::infer_fmg_ugamma_spectra(snapshot,X,FALSE),
  mag_eigen_only=function()core$robust_ugamma_eigenvalues(M),
  lav_ugamma=function()lavaan:::lav_test_fmg_ugamma(lavobject=lf),
  lav_eigen_only=function()lavaan:::lav_test_fmg_ugamma_eigenvalues(ug,df),
  mag_cal_sb=function()magcal('sb'),mag_cal_peba=function()magcal('peba'),
  lav_cal_sb=function()lavcal('sb'),lav_cal_peba=function()lavcal('peba'),
  mag_score_components=function()score_components(ic),
  mag_score_project=function()project_scores(score_components(ic)),
  mag_score_spectrum=function()score_spectrum(pscore),
  mag_score_cal_peba=function()calibrate_quadratic(sscore,'peba4'),
  # Tiny native call, same wrapper family: an empirical lower bound on call overhead.
  mag_native_chisq_call=function()magmaan:::infer_chi2_pvalue(T,as.integer(df))
 )
 if('--legacy-and-wrappers'%in%args){
  ss<-magmaan:::fit_sample_stats(snapshot);implied<-magmaan:::model_implied(snapshot)
  res<-magmaan:::infer_fmg_test(T,df,eig,method='peba',param=4)
  row<-list(input='peba4_ml',label='peba4_ml',p_value=res$p_value,df=res$df,
    base='ml',base_statistic=res$chi2_source,method=res$method,param=res$param,
    ug=FALSE,chi2_equiv=res$chi2_equiv,n_truncated=res$n_truncated,
    eigenvalues=eig,lambdas_raw=res$lambdas_raw,lambdas=res$lambdas,
    lambdas_reference=res$lambdas_reference)
  oldref<-lavfit('standard')
  legacy<-tryCatch(semTests::pvalues(oldref,tests='peba4_ml'),error=identity)
  checks[[j]]$legacy_peba_p_diff<-if(inherits(legacy,'error'))NA_real_ else abs(as.numeric(legacy)-fm$p_value[2])
  checks[[j]]$legacy_error<-if(inherits(legacy,'error'))conditionMessage(legacy) else ''
  phases<-list(
   mag_fit_peba=phases$mag_fit_peba,lav_fit_peba=phases$lav_fit_peba,
   lav_fit_peba_semtests_legacy=function(){z<-lavfit('standard');semTests::pvalues(z,tests='peba4_ml')},
   semtests_peba_postfit_legacy=function()semTests::pvalues(oldref,tests='peba4_ml'),
   mag_postfit_peba=phases$mag_postfit_peba,
   mag_postfit_sample_extract=function()magmaan:::fit_sample_stats(snapshot),
   mag_postfit_df=function()magmaan:::infer_df_stat(snapshot$partable,ss),
   mag_postfit_implied=function()magmaan:::model_implied(snapshot),
   mag_postfit_unused_rls=function()magmaan:::infer_rls_chi2_fit(snapshot,implied),
   mag_postfit_test_parse=function()magmaan:::.fmg_parse_test('peba4_ml'),
   mag_postfit_result_frame=function()magmaan:::.fmg_rows_to_df(list(row)),
   mag_postfit_raw_extract=function()magmaan:::.fmg_raw_from_fit_or_data(snapshot,NULL),
   mag_postfit_raw_check=function()magmaan:::.fmg_validate_complete_raw(snapshot,X),
   mag_spectra=phases$mag_spectra,mag_cal_peba=phases$mag_cal_peba
  )
 }
 for(phase in names(phases)){
  fun<-phases[[phase]];warm<-tryCatch({invisible(fun());NULL},error=identity)
  if(inherits(warm,'error')){
   rows[[length(rows)+1L]]<-data.frame(c,df=df,phase=phase,batch=0L,calls=0L,ms=NA_real_,error=conditionMessage(warm))
   next
  }
  calls<-if('--smoke'%in%args)2L else max(5L,min(200L,ceiling(40/max(measure(fun,2),.001))))
  for(batch in seq_len(if('--smoke'%in%args)1L else 3L)){
   gc(FALSE);rows[[length(rows)+1L]]<-data.frame(c,df=df,phase=phase,batch=batch,calls=calls,ms=measure(fun,calls),error='')
  }
 }
 message('Calibration audit ',j,'/6: ',c$case)
}
stopifnot(identical(hashes,tools::md5sum(files)))
x<-do.call(rbind,rows);ch<-do.call(rbind,checks)
write.csv(x,file.path(outdir,'samples.csv'),row.names=FALSE)
summary<-aggregate(ms~case+p+n+df+phase,x,median)
write.csv(summary,file.path(outdir,'summary.csv'),row.names=FALSE)
write.csv(ch,file.path(outdir,'checks.csv'),row.names=FALSE)
write.csv(data.frame(path=names(hashes),md5=unname(hashes)),file.path(outdir,'fingerprints.csv'),row.names=FALSE)
meta<-c(started=as.character(started),finished=as.character(Sys.time()),command=paste(commandArgs(),collapse=' '),worker=1,blas_threads=1)
write.csv(data.frame(key=names(meta),value=unname(meta)),file.path(outdir,'metadata.csv'),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(outdir,'session.txt'))
print(ch,row.names=FALSE)
if('--legacy-and-wrappers'%in%args)stopifnot(max(ch$legacy_peba_p_diff,na.rm=TRUE)<1e-4)
stopifnot(max(ch$fit_peba_p_diff)<1e-4,max(ch$fit_sb_p_diff)<1e-4,
 max(ch$same_spectrum_peba_diff)<1e-6,max(ch$same_spectrum_sb_diff)<1e-10)
