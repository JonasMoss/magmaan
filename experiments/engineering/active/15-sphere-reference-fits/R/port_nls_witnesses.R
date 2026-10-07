run_port_nls_witnesses <- function(args,here) {
if(any(args %in% c('--help','-h'))) {
 cat('Usage: run_experiment.R --ordinary --port-nls-witnesses [--smoke] [--fresh] [--run-id NAME] [--probe EXECUTABLE]\n',
 'Replays retained witnesses; --smoke selects one per start/coordinate pair.\n',
 'Use --fresh for 12 fixed-seed confirmation fits; --before-run NAME compares saved estimates.\n',
 'Writes original/affine objectives and exact inputs for scripts/profile_port_nls.cpp.\n',
 'The C++ probe prints profile/original objectives and cold backend evaluation-history checks.\n')
 return(invisible(NULL))
}
source(file.path(here,'../../../_support/R/helpers.R'))
source(file.path(here,'R/designs.R'))
source(file.path(here,'R/uls_reliability_study.R'))
reg <- magmaanlab::model_spec(model_syntax)
cov <- magmaanlab::model_spec('X =~ x1 + x2 + x3\nY =~ y1 + y2 + y3\nX ~~ Y')
w <- read.csv(file.path(here,'results/uls-reliability-final-v2/inconsistency_witnesses.csv'))
w <- w[c('case_id','role','design','rep','seed','start_id','route','coordinates','optimizer')]
tasks <- read.csv(file.path(here,'results/uls-reliability-final-v2/tasks.csv'))
frozen <- read.csv(file.path(here,'results/uls-unit-final-60/covariances.csv'))
known <- c('--ordinary','--port-nls-witnesses','--smoke','--fresh','--run-id','--before-run','--probe')
if(any(startsWith(args,'--') & !args %in% known)) stop('unknown replay option')
value <- function(k, default) {
 j <- match(k,args)
 if(is.na(j)) return(default)
 if(j==length(args) || startsWith(args[j+1L],'--')) stop('missing value for ',k)
 args[j+1L]
}
run <- value('--run-id','port-nls-witnesses')
out <- file.path(here,'results',run)
if(!grepl('^[A-Za-z0-9_-]+$',run) || dir.exists(out)) stop('choose a fresh run-id')
dir.create(out,recursive=TRUE)
if('--fresh' %in% args) {
 grid <- expand.grid(design=c('ernst','weak_marker'),start_id=c('fabin3','layered','signed_-1_-1'),coordinates=c('mixed','standardized'),stringsAsFactors=FALSE)
 w <- w[rep(1L,nrow(grid)),]; w$case_id <- 1001L+seq_len(nrow(grid)); w$role <- 'fresh'; w$design <- grid$design; w$start_id <- grid$start_id; w$coordinates <- grid$coordinates
 w$seed <- 817260700L+seq_len(nrow(grid)); w$rep <- seq_len(nrow(grid))
}
if('--smoke' %in% args) w <- w[!duplicated(paste(w$start_id,w$coordinates)),]
write_metadata(file.path(out,'metadata.csv'),values=list(command=paste(args,collapse=' '),
  witnesses=nrow(w),input_md5=unname(tools::md5sum(file.path(here,'results/uls-reliability-final-v2/inconsistency_witnesses.csv'))),
  git_head=magmaan_cache_ref()$git_head,
  source_md5=paste(tools::md5sum(c(file.path(here,'R/port_nls_witnesses.R'),file.path(here,'scripts/profile_port_nls.cpp'))),collapse=';'),
  installed_library_md5=unname(tools::md5sum(system.file('libs',paste0('magmaanlab',.Platform$dynlib.ext),package='magmaanlab'))),
  probe_md5=if(nzchar(value('--probe','')))unname(tools::md5sum(value('--probe',''))) else '',
  threads=1,fresh_seed_base=if('--fresh' %in% args)817260700L else NA_integer_,
  scope='retained replay or bounded fresh endpoint confirmation; no acceptance/default change'),packages='magmaanlab')
rows <- list()
for(i in seq_len(nrow(w))) {
 z <- w[i,]; task <- if('--fresh' %in% args)z else tasks[tasks$case_id==z$case_id,]
 if(task$role=='retained') {
  f <- frozen[frozen$case_id==task$rep,]; s <- matrix(0,6,6); s[cbind(f$row,f$col)] <- f$value; dimnames(s)<-list(ov_names,ov_names)
  sample <- list(S=list(s),nobs=100L)
 } else {
  d <- draw_data(design_sigma(designs_all()[[task$design]]),100L,task$seed)
  d[] <- sweep(as.matrix(d),2,c(.01,100,2,.3,10,.1),'*')
  sample <- magmaanlab::df_to_data(d,reg,scaling='n'); s <- sample$S[[1]]
 }
 if(z$start_id %in% c('fabin3','layered')) {
  method <- z$start_id
  theta <- magmaanlab::magmaan_core$estimate_start_values(reg$partable,sample,start=method,transport=if(method=='layered')'native' else 'auto')
  pt <- reg$partable; pt$est <- pt$ustart; pt$est[pt$free>0] <- theta[pt$free[pt$free>0]]; start <- uls_study_point(pt)
 } else {sg <- as.numeric(strsplit(z$start_id,'_')[[1]][2:3]); start <- uls_signed_moment_point(sample,sg[1],sg[2])}
 unit <- if(z$coordinates=='mixed')rep(1,6) else sqrt(diag(s))
 tr <- uls_study_transport(start,cov,unit); target <- cov; free <- target$partable$free
 target$partable$ustart[free>0] <- tr$theta[free[free>0]]
 weights <- diag(outer(unit,unit)[lower.tri(s,diag=TRUE)]^2)
 t0 <- proc.time()[['elapsed']]
 fit <- magmaanlab::magmaan_core$fit_wls_snlls(target,list(S=list(s/outer(unit,unit)),nobs=100L),W=weights,optimizer='port-nls',control=list(start=as.numeric(tr$theta),coordinate_scaling='sample_units'))
 p <- uls_study_point(fit$partable); L <- p$loading*unit; sigma <- L%*%p$phi%*%t(L)+diag(p$residual*unit^2)
 original <- sum((sigma-s)[lower.tri(s,diag=TRUE)]^2)/2
 # Independent linear design at the returned loading coordinates; residual
 # variances plus three latent covariance entries are the profiled block.
 mask <- lower.tri(s,diag=TRUE)
 h <- sapply(1:9,function(k){
  if(k<=6) {m<-matrix(0,6,6);m[k,k]<-1;return(m[mask])}
  a<-L[,1];b<-L[,2]
  m<-switch(as.character(k),'7'=outer(a,a),'8'=outer(b,b),'9'=outer(a,b)+outer(b,a));m[mask]
 })
 alpha <- c(p$residual*unit^2,p$phi[1,1],p$phi[2,2],p$phi[1,2])
 expanded_affine <- sum((h%*%alpha-s[mask])^2)/2
 restored <- uls_study_transport(list(loading=L,phi=p$phi,residual=p$residual*unit^2),reg)
 checked <- magmaanlab::magmaan_core$evaluate_at(reg$partable,sample,restored$theta,estimator='ULS',bounds=list(lower=rep(-Inf,length(restored$theta)),upper=rep(Inf,length(restored$theta))))
 rows[[i]] <- cbind(z,common_original=checked$fmin,after_audit=checked$verdict$status,after_criterion=checked$verdict$criterion,after_newton_status=checked$diagnostics$newton_accuracy$status,native_after_audit=fit$verdict$status,backend=fit$fmin,expanded_original=original,expanded_affine=expanded_affine,max_loading=max(abs(L)),max_alpha=max(abs(alpha)),optimizer_status=fit$optimizer_status,seconds=proc.time()[['elapsed']]-t0)
 write.csv(do.call(rbind,rows),file.path(out,'replay.csv'),row.names=FALSE)
 saveRDS(fit,file.path(out,sprintf('fit_%02d.rds',i)))
 writeLines(sprintf("%.17g",c(as.vector(s/outer(unit,unit)),diag(weights),tr$theta,fit$partable$est[fit$partable$free>0][order(fit$partable$free[fit$partable$free>0])])),file.path(out,sprintf('input_%02d.txt',i)))
 cat(i,fit$fmin,original,expanded_affine,'seconds',rows[[i]]$seconds,'\n')
}

probe <- value('--probe','')
if(nzchar(probe)) {
 rows <- do.call(rbind,rows)
 keys <- c('profile_recomputed','profile_expanded_original','returned_original_core','expansion_gap',
   'backend_cold','historical_f_gap','historical_x_gap','last_endpoint_f','cold_endpoint_gap','stored_objective','returned_x_objective','best_point_substituted','raw_stop_code')
 for(i in seq_len(nrow(rows))) {
   err <- file.path(out,'probe_stderr.txt')
   result <- system2(probe,shQuote(file.path(out,sprintf('input_%02d.txt',i))),stdout=TRUE,stderr=err)
   if(!is.null(attr(result,'status'))) stop('core profile probe failed: ',i)
   values <- as.numeric(strsplit(paste(c(result,readLines(err)),collapse=','),',',fixed=TRUE)[[1]])
   if(length(values)!=length(keys) || any(!is.finite(values))) stop('invalid core probe output: ',i)
   for(j in seq_along(keys)) rows[i,keys[j]] <- values[j]
 }
 rows$relative_original_gap <- abs(rows$backend-rows$expanded_original)/(1+abs(rows$expanded_original))
 rows$consistent <- rows$relative_original_gap <= 1e-12 & rows$expansion_gap == 0 & rows$cold_endpoint_gap == 0
 rows$no_historical_loss <- rows$backend <= rows$stored_objective + 1e-14*pmax(abs(rows$stored_objective),abs(rows$backend))
 rows$disposition <- ifelse(rows$consistent & rows$no_historical_loss,if('--fresh' %in% args)'consistent_endpoint' else 'fixed_endpoint_inconsistency','unexplained')
 if(!'--fresh' %in% args) {
   before <- read.csv(file.path(here,'results/port-nls-endpoint-replay-final/comparisons.csv'))
   key <- function(d)paste(d$case_id,d$start_id,d$coordinates)
   j <- match(key(rows),key(before))
   rows$before_reported <- before$backend[j]
   rows$before_original <- before$expanded_original[j]
   rows$before_audit <- before$common_audit[j]
 }
 before_run <- value('--before-run','')
 if(nzchar(before_run)) {
   if(!grepl('^[A-Za-z0-9_-]+$',before_run)) stop('invalid before-run')
   before <- read.csv(file.path(here,'results',before_run,'replay.csv'))
   key <- function(d)paste(d$case_id,d$start_id,d$coordinates,d$seed)
   if(!identical(key(before),key(rows))) stop('before-run design differs')
   rows$before_reported <- before$backend
   rows$before_original <- before$expanded_original
   rows$before_audit <- before$after_audit
   rows$estimate_bit_identical <- vapply(seq_len(nrow(rows)),function(i) {
     old <- readRDS(file.path(here,'results',before_run,sprintf('fit_%02d.rds',i)))
     new <- readRDS(file.path(out,sprintf('fit_%02d.rds',i)))
     identical(old$partable$est,new$partable$est)
   },FALSE)
   rows$objective_loss <- rows$backend > rows$before_original + 1e-12*(1+abs(rows$before_original))
   rows$audit_loss <- rows$before_audit=='passed' & rows$after_audit!='passed'
 }
 write.csv(rows,file.path(out,'comparisons.csv'),row.names=FALSE)
 # The same profile problems expose scalar drmngb/drmng endpoint mismatches.
 baseline <- read.csv(file.path(here,'results/port-scalar-before/points.csv'))
 scalar <- list()
 baseline_run <- if('--fresh' %in% args)'port-nls-fresh-before' else 'port-nls-endpoint-replay-final'
 for(i in seq_len(nrow(rows)))for(routine in c('scalar','unbounded')) {
   input <- sprintf('input_%02d.txt',i)
   old <- baseline[baseline$run==baseline_run & baseline$case_id==rows$case_id[i] & baseline$start_id==rows$start_id[i] & baseline$coordinates==rows$coordinates[i] & baseline$routine==routine,]
   if(nrow(old)!=1L)stop('missing scalar baseline')
   result <- system2(probe,c(shQuote(file.path(out,input)),routine),stdout=TRUE)
   if(!is.null(attr(result,'status')))stop('scalar probe failed')
   v <- as.numeric(strsplit(result,',',fixed=TRUE)[[1]])
   if(length(v)!=11L || any(!is.finite(v)))stop('invalid scalar probe')
   old_x <- as.numeric(old[paste0('x',1:4)])
   scalar[[length(scalar)+1L]] <- data.frame(case_id=rows$case_id[i],start_id=rows$start_id[i],coordinates=rows$coordinates[i],routine=routine,
     before_reported=old$reported,before_recomputed=old$recomputed,before_stop=old$stop,after_stop=v[11],stop_unchanged=v[11]==old$stop,after_reported=v[1],stored_objective=v[6],returned_x_objective=v[7],substituted=v[8]!=0,
     before_consistent=old$reported==old$recomputed,estimate_bit_identical=identical(old_x,unname(v[2:5])),
     recomputed=v[9],original=v[10],objective_consistent=max(abs(v[1]-v[9]),abs(v[1]-v[10]))<=1e-12*(1+abs(v[1])),
     no_loss=v[1]<=old$reported+1e-14*max(abs(old$reported),abs(v[1])))
 }
 write.csv(do.call(rbind,scalar),file.path(out,'scalar_checks.csv'),row.names=FALSE)

}
cat("Wrote",out,"\n")
}
