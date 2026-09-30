#!/usr/bin/env Rscript
# Preserve factor-level identification evidence alongside the PORT non-successes.
args<-commandArgs(TRUE);root<-normalizePath(args[1]);out<-normalizePath(args[2])
script<-normalizePath(sub("--file=","",grep("--file=",commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),"../../../../_support/R/helpers.R"));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),"inputs.R"));source(file.path(dirname(script),"arms.R"))
manifest<-read.csv(file.path(root,"manifest.csv"),stringsAsFactors=FALSE)
d<-read.csv(file.path(out,"comparison.csv"),stringsAsFactors=FALSE)
z<-subset(d,arm=="port_default" & !accepted_match)
z$failure_class<-ifelse(!z$returned,ifelse(grepl("nonlinear equality",z$message),"unsupported constraints","solver error"),
                       ifelse(!z$converged,"audit rejected","accepted, worse objective"))
rows<-list();factors<-list();constraints<-list()
for(i in seq_len(nrow(z))) {
  j<-z[i,];c<-read_case(root,manifest$case_dir[match(j$case,manifest$case_id)],j$estimator)
  p<-c$model$partable
  if(!j$case %in% vapply(factors,function(x)x$case[1],character(1))) {
    for(group in unique(p$group[p$op=="=~"])) for(factor in unique(p$lhs[p$op=="=~" & p$group==group])) {
      l<-p[p$op=="=~" & p$lhs==factor & p$group==group,]
      fixed<-l$free==0 & is.finite(l$ustart)
      v<-p[p$op=="~~" & p$lhs==factor & p$rhs==factor & p$group==group,]
      factors[[length(factors)+1]]<-data.frame(case=j$case,group=group,factor=factor,
        fixed_loadings=if(any(fixed)) paste(paste0(l$rhs[fixed],"=",l$ustart[fixed]),collapse="; ") else "",
        free_loadings=paste(l$rhs[!fixed],collapse="; "),
        variance_fixed=if(nrow(v) && v$free[1]==0) v$ustart[1] else NA_real_)
    }
    eq<-p[p$op %in% c("==","<",">"),intersect(c("lhs","op","rhs","group"),names(p)),drop=FALSE]
    if(nrow(eq)) constraints[[length(constraints)+1]]<-data.frame(case=j$case,eq)
  }
  x<-magmaan_core$estimate_start_values(p,c$sample,start=if(j$estimator=="ML") "scaled-fabin" else "fabin3",
                                        transport=if(j$estimator=="ML") "auto" else "native")
  f<-tryCatch(one_fit(c,j$estimator,optimizer_arms()$port_default,x),error=function(e)e)
  j$newton_status<-NA_character_;j$geometry_stationary<-NA;j$rerun_matches<-FALSE
  if(!inherits(f,"error")) {
    j$newton_status<-f$diagnostics$newton_accuracy$status
    j$geometry_stationary<-isTRUE(f$diagnostics$geometric_stationarity$ambient_stationary)
    j$rerun_matches<-isTRUE(f$converged)==j$converged && abs(f$fmin-j$f)<=1e-8*(1+abs(j$f))
  } else j$rerun_matches<-!j$returned && identical(conditionMessage(f),j$message)
  rows[[i]]<-j
  cat(sprintf("[%d/%d] %s %s: %s\n",i,nrow(z),j$case,j$estimator,j$newton_status))
}
stopifnot(all(vapply(rows,function(x)x$rerun_matches,logical(1))))
write.csv(do.call(rbind,rows),file.path(out,"port_failures.csv"),row.names=FALSE)
write.csv(do.call(rbind,factors),file.path(out,"port_factor_identification.csv"),row.names=FALSE)
write.csv(do.call(rbind,constraints),file.path(out,"port_identification_constraints.csv"),row.names=FALSE)
