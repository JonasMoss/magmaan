#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(script); root <- dirname(dirname(dirname(here)))
value <- function(flag,default) { i<-match(flag,args); if(is.na(i)) default else {
  if(i==length(args)) stop('Missing value for ',flag); args[i+1L] } }
if('--help' %in% args) {
  cat('Speed attribution pilot: complete-data ML on HS CFA and PoliticalDemocracy.\n',
      'Usage: Rscript run_experiment.R [--smoke] [--sessions 3] [--batches 7]\n',
      '       [--target-ms 100] [--out pilot] [--cases hs,democracy]\n',
      'Output is a new directory under this experiment\'s results/.\n',
      'Set R_LIBS to select an optimized magmaan installation. Requires lavaan 0.7-2.\n',
      'The parent launches serial fresh R sessions with one BLAS/OpenMP thread.\n',
      'Smoke: one session, two batches, 20 ms target; not publication evidence.\n',
      'Failed parity rows retain diagnostic times but receive no speed ratio.\n',sep='');quit()
}
smoke <- '--smoke' %in% args
sessions <- as.integer(value('--sessions',if(smoke) 1 else 3))
batches <- as.integer(value('--batches',if(smoke) 2 else 7))
target <- as.numeric(value('--target-ms',if(smoke) 20 else 100))
run_id <- value('--out',if(smoke) 'smoke' else 'pilot')
stopifnot(grepl('^[A-Za-z0-9_-]+$',run_id),is.finite(sessions),sessions>0,
          is.finite(batches),batches>0,is.finite(target),target>0)
out <- file.path(here,'results',run_id)
source(file.path(root,'benchmarks/r/timing.R'))
source(file.path(here,'R/workloads.R'))
write_csv <- function(x,name) write.csv(x,file.path(out,name),row.names=FALSE,na='')
case_ids <- strsplit(value('--cases','hs,democracy'),',',fixed=TRUE)[[1]]
stopifnot(all(case_ids %in% c('hs','democracy')), !anyDuplicated(case_ids))
if(!'--worker' %in% args) {
  if(dir.exists(out)) stop('Output already exists; choose a new --out: ',out)
  dir.create(out,recursive=TRUE)
  Sys.setenv(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             BLIS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',TZ='UTC')
  message('Approximate timing budget: ',round(length(case_ids)*14*2*batches*target*sessions/1000),
          ' seconds plus calibration, validation, and diagnostics.')
  for(s in seq_len(sessions)) {
    message('Fresh session ',s,'/',sessions)
    status <- system2(file.path(R.home('bin'),'Rscript'),
      c(shQuote(script),vapply(args,shQuote,''),'--worker','--session',s))
    if(status!=0L) stop('Worker failed; partial output retained: ',out)
  }
  source(file.path(here,'R/summarize.R'))
  summarize_run(out)
  message('Wrote ',out,'; render with quarto render ',file.path(here,'report.qmd'),
          ' -P run:',run_id)
  quit()
}
session <- as.integer(value('--session',1))
suppressPackageStartupMessages(library(magmaan))
check_lavaan_adapter()
stopifnot(all(c('prepare_inference','inference_covariance','inference_quadratic',
  'score_spectrum','calibrate_quadratic') %in% getNamespaceExports('magmaan')))
read_syntax <- function(id) paste(readLines(file.path(root,'benchmarks/cases',id,'model.lav')),collapse='\n')
cases <- list(hs=list(syntax=read_syntax('hs_3factor_cfa'),data=lavaan::HolzingerSwineford1939),
              democracy=list(syntax=read_syntax('bollen_democracy_sem'),data=lavaan::PoliticalDemocracy))
cases <- cases[case_ids]
package_paths <- unlist(lapply(c('magmaan','lavaan'),function(pkg)
  list.files(find.package(pkg),pattern='\\.(so|rdb|rdx)$|^DESCRIPTION$',recursive=TRUE,full.names=TRUE)))
fingerprints <- tools::md5sum(package_paths)
code_paths <- c(script,list.files(file.path(here,'R'),full.names=TRUE),file.path(root,'benchmarks/r/timing.R'))
metadata <- c(started_utc=format(Sys.time(),tz='UTC',usetz=TRUE),command=paste(commandArgs(),collapse=' '),session=session,
  source_revision=system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE),
  R=R.version.string,magmaan=as.character(packageVersion('magmaan')),
  lavaan=as.character(packageVersion('lavaan')),magmaan_path=find.package('magmaan'),
  lavaan_path=find.package('lavaan'),batches=batches,target_ms=target,
  cases=paste(case_ids,collapse=','),seed='none: fixed package datasets',
  build_provenance='installed package fingerprints; source identity requires matching build record',
  cpu=paste(unique(readLines('/proc/cpuinfo')[grepl('model name',readLines('/proc/cpuinfo'))]),collapse=';'),
  extSoftVersion(),Sys.getenv(c('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')),
  setNames(as.character(fingerprints),paste0('md5:',names(fingerprints))),
  setNames(as.character(tools::md5sum(code_paths)),paste0('code_md5:',code_paths)))
write_csv(data.frame(key=names(metadata),value=unname(metadata)),paste0('metadata-',session,'.csv'))
writeLines(capture.output(sessionInfo()),file.path(out,paste0('session-',session,'.txt')))
if(session==1L) {
  saveRDS(cases,file.path(out,'inputs.rds'))
  saveRDS(setNames(lapply(workloads,lav_options),workloads),file.path(out,'options.rds'))
  write_csv(expand.grid(case=case_ids,session=seq_len(sessions)), 'planned.csv')
}
samples <- checks <- diagnostics <- stages <- errors <- values <- list()
append_samples <- function(x,id,boundary,w) {
  x$case<-id; x$boundary<-boundary; x$workload<-w
  samples[[length(samples)+1L]] <<- x
}
add_check <- function(x) {x$session<-session;checks[[length(checks)+1L]] <<- x}
flush <- function() {
  for(n in c('samples','checks','diagnostics','stages','errors','values')) {
    rows<-get(n)
    if(length(rows)) write_csv(do.call(rbind,rows),paste0(n,'-',session,'.csv'))
  }
}
for(id in names(cases)) {
 tryCatch({
  case<-cases[[id]]; message(id,': validating fit and inference routes')
  pm<-prepare_model(case$syntax,meanstructure=FALSE,fixed_x=FALSE,auto_cov_y=TRUE)
  pd<-prepare_data(pm,case$data)
  u<-do.call(lavaan::sem,c(list(model=case$syntax,data=case$data,do.fit=FALSE),lav_options('fit')))
  m0<-mag_raw(case); l0<-lav_raw(case,'fit')
  ov<-unlist(m0$ov_names); X<-as.matrix(case$data[,ov,drop=FALSE])
  stopifnot(!anyNA(X))
  for(engine in c('magmaan','lavaan')) {
    f<-if(engine=='magmaan') m0 else l0
    diagnostics[[length(diagnostics)+1L]]<-data.frame(session=session,case=id,engine=engine,
      n=nrow(X),p=ncol(X),npar=length(if(engine=='magmaan') f$theta else f@optim$x),
      converged=if(engine=='magmaan') isTRUE(f$converged) else isTRUE(f@optim$converged),
      admissible=if(engine=='magmaan') isTRUE(f$diagnostics$admissibility$admissible) else
        isTRUE(lavaan::lavInspect(f,'post.check')),
      iterations=if(engine=='magmaan') f$iterations else f@optim$iterations,
      f_evals=if(engine=='magmaan') f$f_evals else NA_real_,
      g_evals=if(engine=='magmaan') f$g_evals else NA_real_,
      gradient_max=if(engine=='magmaan') f$grad_norm else max(abs(lavaan::lavInspect(f,'gradient'))))
  }
  accepted_fit<-isTRUE(m0$converged) && isTRUE(m0$diagnostics$admissibility$admissible) &&
    isTRUE(l0@optim$converged) && isTRUE(lavaan::lavInspect(l0,'post.check')) &&
    max(abs(lavaan::lavInspect(l0,'gradient')))<1e-5
  # Keep even rejected cells in the denominator and diagnostic output.
  add_check(data.frame(case=id,boundary='all',workload='all',comparison='acceptance',
    metric='fit_acceptance',max_abs=as.numeric(!accepted_fit),atol=0,rtol=0,passed=accepted_fit))
  for(w in workloads) {
    # Bind loop variables before callbacks are timed.
    ref<-lav_raw(case,w); options<-ref@Options
    raw_arms<-list(magmaan=function() extract_mag(mag_raw(case),w,case$data),
                  lavaan=function() extract_lav(lav_raw(case,w),w))
    prep_arms<-list(magmaan=function() extract_mag(estimate(pm,pd),w,pd),
      lavaan=function() {f<-lav_prepared_fit(u);if(w=='fit') extract_lav(f,w) else lav_post(f,w,options)})
    boundaries<-list(raw=raw_arms,prepared=prep_arms)
    if(w!='fit') boundaries$postfit<-list(magmaan=function() extract_mag(m0,w,case$data),
                                        lavaan=function() lav_post(l0,w,options))
    reference<-lapply(raw_arms,function(fun)fun())
    # Coarse internal lavaan timing is diagnostic only: proc.time quantization
    # and instrumentation prevent using these single-fit slots for attribution.
    for(stage in setdiff(names(ref@timing),'start_time'))
      stages[[length(stages)+1L]]<-data.frame(session=session,case=id,workload=w,
        stage=stage,ms=as.numeric(ref@timing[[stage]])*1000,
        kind='lavaan_builtin_diagnostic')
    for(boundary in names(boundaries)) {
      arms<-boundaries[[boundary]]; result<-lapply(arms,function(fun)fun())
      add_check(compare_outputs(result$magmaan,result$lavaan,id,boundary,w))
      if(boundary!='raw') for(engine in names(arms))
        add_check(compare_outputs(result[[engine]],reference[[engine]],id,boundary,w,paste0(engine,'_adapter')))
      for(engine in names(arms)) {
        z<-result[[engine]]
        for(metric in names(z$tests)) values[[length(values)+1L]]<-data.frame(
          session=session,case=id,boundary=boundary,workload=w,engine=engine,
          metric=metric,value=unname(z$tests[metric]))
      }
      message(id,' / ',w,' / ',boundary)
      append_samples(time_paired(arms,batches,target,session=session),id,boundary,w)
      flush()
    }
  }
  # Distinct setup probes: independent timers, never stacked or subtracted.
  probes<-list(magmaan_model=function()prepare_model(case$syntax,meanstructure=FALSE,
                       fixed_x=FALSE,auto_cov_y=TRUE),
               magmaan_data=function()prepare_data(pm,case$data),
               lavaan_unfitted=function()do.call(lavaan::sem,c(list(model=case$syntax,
                     data=case$data,do.fit=FALSE),lav_options('fit'))))
  for(stage in names(probes)) append_samples(time_paired(setNames(probes[stage],sub('_.*','',stage)),
    batches,target,session=session),id,'setup',stage)
  # Fixed-theta lavaan probes include parameter-to-matrix placement, but no
  # optimizer. These are unpaired diagnostics pending a native replay bridge.
  x<-l0@optim$x
  objective<-function() {g<-li('lav_model_x2glist')(l0@Model,x)
    li('lav_model_objective')(l0@Model,g,l0@SampleStats,l0@Data,l0@Cache)}
  gradient<-function() {g<-li('lav_model_x2glist')(l0@Model,x)
    li('lav_model_grad')(l0@Model,g,l0@SampleStats,l0@Data,l0@Cache)}
  stopifnot(abs(objective()-l0@optim$fx)<1e-8,
    max(abs(gradient()-lavaan::lavInspect(l0,'gradient')))<1e-7)
  for(stage in c('objective','gradient')) append_samples(time_paired(
    list(lavaan=get(stage)),batches,target,session=session),id,'fixed_theta',stage)
  # Reuse the *same* spectrum/statistic to localize pEBA disagreement, without
  # claiming which implementation is correct or changing either method.
  lf<-lav_raw(case,'peba4_wald'); eig<-lf@test$peba4_ml$UGamma.eigenvalues
  base<-lf@test$standard$stat; df<-lf@test$standard$df
  pp<-calibrate_quadratic(quadratic_reference(base,df,eig),'peba4')$p_value
  values[[length(values)+1L]]<-data.frame(session=session,case=id,boundary='same_spectrum',
    workload='peba4_wald',engine=c('magmaan','lavaan'),metric='peba4',
    value=c(pp,lf@test$peba4_ml$pvalue))
  flush()
 },error=function(e) {
   errors[[length(errors)+1L]]<<-data.frame(session=session,case=id,error=conditionMessage(e))
   message('Retained failed case ',id,': ',conditionMessage(e));flush()
 })
}
stopifnot(identical(fingerprints,tools::md5sum(package_paths)))
writeLines(format(Sys.time(),tz='UTC',usetz=TRUE),file.path(out,paste0('finished-',session,'.txt')))
if(session==1L) write_csv(data.frame(file=c('inputs.rds','options.rds'),
  md5=unname(tools::md5sum(file.path(out,c('inputs.rds','options.rds'))))), 'input-hashes.csv')
flush()
