#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script)); args <- commandArgs(TRUE)
if('--help' %in% args) {
 cat('Usage: Rscript scripts/summarize_normalization_revisit.R [RUN_ID=paired]\n'); quit(save='no')
}
run <- if(length(args)) args[1] else 'paired'
stopifnot(grepl('^[a-zA-Z0-9_-]+$',run))
out <- file.path(here,'results','normalization-revisit',run)
x <- read.csv(file.path(out,'fits.csv'),stringsAsFactors=FALSE)
write <- function(z,name) write.csv(z,file.path(out,paste0(name,'.csv')),row.names=FALSE)
keys <- c('pop','n','rep','seed','family','model','chart','transform','route')
stopifnot(!anyDuplicated(x[c(keys,'arm')]),!anyNA(x$success))
a <- x[x$arm=='original_units',]; b <- x[x$arm=='normalized',]
p <- merge(a,b,by=keys,suffixes=c('_original','_normalized'))
stopifnot(nrow(p)==nrow(a),nrow(a)==nrow(b))
p$gain <- !p$success_original & p$success_normalized
p$loss <- p$success_original & !p$success_normalized
p$both_success <- p$success_original & p$success_normalized
p$objective_delta <- p$fmin_normalized-p$fmin_original
p$objective_gain <- p$both_success & is.finite(p$objective_delta) & p$objective_delta < -1e-6*(1+abs(p$fmin_original))
p$objective_loss <- p$both_success & is.finite(p$objective_delta) & p$objective_delta > 1e-6*(1+abs(p$fmin_original))
summarize <- function(z,groups,cols) {
 z$attempts <- 1L
 aggregate(z[c('attempts',cols)],z[groups],function(v)sum(v,na.rm=TRUE))
}
write(summarize(x,c('family','route','arm'),c('returned','certified','success','screened_success','extent_flag','seconds')),'family')
write(summarize(x,c('family','route','arm','transform'),c('success','screened_success','extent_flag')),'family_transform')
write(summarize(p,c('family','route'),c('gain','loss','both_success','objective_gain','objective_loss')),'paired_summary')
write(p[p$gain|p$loss|p$objective_gain|p$objective_loss,],'changes')
# Compare each rescaled fit against its own native-unit counterpart.
covs <- do.call(rbind,lapply(list.files(file.path(out,'raw'),pattern='^covariances_',full.names=TRUE),read.csv))
covkeys <- c(keys,'arm')
covmax <- function(z) {
 native <- z[z$transform=='native',]; scaled <- z[z$transform!='native',]
 k <- c(setdiff(covkeys,'transform'),'element')
 q <- merge(scaled,native[setdiff(names(native),'transform')],by=k,suffixes=c('_scaled','_native'))
 q$max_covariance_difference <- abs(q$value_scaled-q$value_native)
 aggregate(q['max_covariance_difference'],q[covkeys],max)
}
cm <- covmax(covs)
k <- setdiff(covkeys,'transform')
u <- merge(x[x$transform!='native',],x[x$transform=='native',setdiff(names(x),'transform')],by=k,suffixes=c('_scaled','_native'))
u <- merge(u,cm,by=covkeys,all.x=TRUE)
u$verdict_changed <- u$success_scaled != u$success_native
u$both_success <- u$success_scaled & u$success_native
u$objective_mismatch <- u$both_success & is.finite(u$fmin_scaled) & is.finite(u$fmin_native) & abs(u$fmin_scaled-u$fmin_native)>1e-6*(1+abs(u$fmin_native))
u$covariance_mismatch <- u$both_success & !is.na(u$max_covariance_difference) & u$max_covariance_difference>1e-5
write(summarize(u,c('family','route','arm'),c('verdict_changed','both_success','objective_mismatch','covariance_mismatch')),'invariance')
write(u[u$verdict_changed|u$objective_mismatch|u$covariance_mismatch,],'unit_exceptions')
# Full comparison rows are local; selected exceptions above are frozen evidence.
write(p,'pairs'); write(u,'unit_pairs')
cat('Fits:',nrow(x),' Cases:',nrow(p)/3,'\n')
print(summarize(x,c('route','arm'),c('success','screened_success','extent_flag')))
print(summarize(p,'route',c('gain','loss','both_success','objective_gain','objective_loss')))
print(summarize(u,c('route','arm'),c('verdict_changed','objective_mismatch','covariance_mismatch')))
