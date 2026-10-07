# Healthy one-factor regression controls for endpoint retention. Run before and
# after the core repair using separate installed packages and fresh output IDs.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)<1L || any(args %in% c('--help','-h'))) {
 cat('Usage: Rscript scripts/port_regression_controls.R RUN_ID [BEFORE_RUN_ID]\n')
 quit(status=if(length(args))0L else 1L)
}
script <- sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
here <- dirname(dirname(normalizePath(script)))
if(any(!grepl('^[A-Za-z0-9_-]+$',args)))stop('invalid run ID')
out <- file.path(here,'results',args[1]); if(dir.exists(out))stop('choose fresh run ID')
dir.create(out,recursive=TRUE)
model <- magmaanlab::model_spec('f =~ x1 + x2 + x3 + x4')
lambda <- c(1,.85,.7,.6)
s <- 2*outer(lambda,lambda)+diag(c(.6,.5,.8,.7))
s[1,4] <- s[4,1] <- s[1,4]+.05
sample <- list(S=list(s),nobs=300L)
rows <- list()
for(kind in c('uls','gls','wls'))for(backend in c('port','port-nls')) {
 id <- paste(kind,backend,sep='_')
 extra <- if(kind=='wls')list(W=list(diag(10)))else list()
 fit <- do.call(magmaanlab::magmaan_core[[paste0('fit_',kind,'_snlls')]],c(list(model,sample,optimizer=backend),extra))
 saveRDS(fit,file.path(out,paste0(id,'.rds')))
 telemetry <- fit$audit$port_endpoint
 row <- data.frame(id=id,objective=fit$fmin,audit=fit$verdict$status,
   substituted=if(is.null(telemetry))NA else telemetry$best_point_substituted,
   raw_stop=fit$audit$raw_backend_status,
   stored_objective=if(is.null(telemetry))NA_real_ else telemetry$stored_objective,
   returned_x_objective=if(is.null(telemetry))NA_real_ else telemetry$returned_x_objective)
 if(length(args)>1L) {
   old <- readRDS(file.path(here,'results',args[2],paste0(id,'.rds')))
   row$before_objective <- old$fmin
   row$estimate_bit_identical <- identical(old$partable$est,fit$partable$est)
 }
 rows[[length(rows)+1L]] <- row
}
write.csv(do.call(rbind,rows),file.path(out,'comparisons.csv'),row.names=FALSE)
source(file.path(here,'../../../_support/R/helpers.R'))
write_metadata(file.path(out,'metadata.csv'),values=list(command=paste(args,collapse=' '),
 git_head=magmaan_cache_ref()$git_head,source_md5=unname(tools::md5sum(script)),
 installed_library_md5=unname(tools::md5sum(system.file('libs',paste0('magmaanlab',.Platform$dynlib.ext),package='magmaanlab'))),
 scope='six healthy one-factor PORT/SNLLS controls; exact before/after estimates'),packages='magmaanlab')
print(do.call(rbind,rows))
