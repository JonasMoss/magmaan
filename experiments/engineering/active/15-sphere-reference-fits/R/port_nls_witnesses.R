run_port_nls_witnesses <- function(args,here) {
if(any(args %in% c('--help','-h'))) {
 cat('Usage: run_experiment.R --ordinary --port-nls-witnesses [--smoke] [--run-id NAME] [--probe EXECUTABLE]\n',
 'Replays only retained witnesses; --smoke selects one per start/coordinate pair.\n',
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
tasks <- read.csv(file.path(here,'results/uls-reliability-final-v2/tasks.csv'))
frozen <- read.csv(file.path(here,'results/uls-unit-final-60/covariances.csv'))
known <- c('--ordinary','--port-nls-witnesses','--smoke','--run-id','--probe')
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
if('--smoke' %in% args) w <- w[!duplicated(paste(w$start_id,w$coordinates)),]
write_metadata(file.path(out,'metadata.csv'),values=list(command=paste(args,collapse=' '),
  witnesses=nrow(w),input_md5=unname(tools::md5sum(file.path(here,'results/uls-reliability-final-v2/inconsistency_witnesses.csv'))),
  git_head=magmaan_cache_ref()$git_head,
  source_md5=paste(tools::md5sum(c(file.path(here,'R/port_nls_witnesses.R'),file.path(here,'scripts/profile_port_nls.cpp'))),collapse=';'),
  probe_md5=if(nzchar(value('--probe','')))unname(tools::md5sum(value('--probe',''))) else '',
  threads=1,scope='retained endpoints only; no acceptance/default change'),packages='magmaanlab')
rows <- list()
for(i in seq_len(nrow(w))) {
 z <- w[i,]; task <- tasks[tasks$case_id==z$case_id,]
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
 rows[[i]] <- cbind(z,backend=fit$fmin,expanded_original=original,expanded_affine=expanded_affine,max_loading=max(abs(L)),max_alpha=max(abs(alpha)),optimizer_status=fit$optimizer_status,seconds=proc.time()[['elapsed']]-t0)
 write.csv(do.call(rbind,rows),file.path(out,'replay.csv'),row.names=FALSE)
 saveRDS(fit,file.path(out,sprintf('fit_%02d.rds',i)))
 writeLines(sprintf("%.17g",c(as.vector(s/outer(unit,unit)),diag(weights),tr$theta,fit$partable$est[fit$partable$free>0][order(fit$partable$free[fit$partable$free>0])])),file.path(out,sprintf('input_%02d.txt',i)))
 cat(i,fit$fmin,original,expanded_affine,'seconds',rows[[i]]$seconds,'\n')
}

probe <- value('--probe','')
if(nzchar(probe)) {
 rows <- do.call(rbind,rows)
 keys <- c('profile_recomputed','profile_expanded_original','returned_original_core','expansion_gap',
   'backend_cold','historical_f_gap','historical_x_gap','last_endpoint_f','cold_endpoint_gap')
 for(i in seq_len(nrow(rows))) {
   err <- file.path(out,'probe_stderr.txt')
   result <- system2(probe,shQuote(file.path(out,sprintf('input_%02d.txt',i))),stdout=TRUE,stderr=err)
   if(!is.null(attr(result,'status'))) stop('core profile probe failed: ',i)
   values <- as.numeric(strsplit(paste(c(result,readLines(err)),collapse=','),',',fixed=TRUE)[[1]])
   if(length(values)!=length(keys) || any(!is.finite(values))) stop('invalid core probe output: ',i)
   for(j in seq_along(keys)) rows[i,keys[j]] <- values[j]
 }
 attributed <- rows$historical_f_gap <= 1e-10*(1+abs(rows$backend_cold)) &
   rows$historical_x_gap > 0 & rows$cold_endpoint_gap == 0 & rows$expansion_gap == 0 &
   abs(rows$profile_recomputed-rows$expanded_original) <= 1e-10*(1+abs(rows$expanded_original)) &
   abs(rows$backend_cold-rows$backend) <= 1e-10*(1+abs(rows$backend))
 rows$attribution <- ifelse(attributed,'backend_stored_objective_from_different_evaluated_point','unexplained')
 rows$disposition <- 'unfixed_stopping_endpoint_inconsistency'
 write.csv(rows,file.path(out,'comparisons.csv'),row.names=FALSE)
}
cat("Wrote",out,"\n")
}
