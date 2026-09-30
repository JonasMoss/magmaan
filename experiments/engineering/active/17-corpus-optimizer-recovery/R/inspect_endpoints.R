#!/usr/bin/env Rscript
args<-commandArgs(TRUE);root<-normalizePath(args[1]);out<-normalizePath(args[2])
script<-normalizePath(sub("--file=","",grep("--file=",commandArgs(),value=TRUE)[1]))
source(file.path(dirname(script),"../../../../_support/R/helpers.R"));set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(dirname(script),"inputs.R"));source(file.path(dirname(script),"arms.R"))
manifest<-read.csv(file.path(root,"manifest.csv"),stringsAsFactors=FALSE)
ids<-c("newsom_2015_ex5_4","newsom_2015_ex5_4c","little_2013_ch3_fig_3_11_longitudinal_cfa_phantom")
rows<-list();thetas<-list()
for(id in ids) for(est in c("ML","GLS")) {
  c<-read_case(root,manifest$case_dir[match(id,manifest$case_id)],est)
  x<-magmaan_core$estimate_start_values(c$model$partable,c$sample,
      start=if(est=="ML") "scaled-fabin" else "fabin3",transport=if(est=="ML") "auto" else "native")
  ref<-reference_fit(c);rt<-map_theta(c$model,ref)
  for(arm in c("lbfgs_default","port_default","slsqp_default","lbfgs_reference_start")) {
    f<-one_fit(c,est,optimizer_arms()[[if(arm=="lbfgs_reference_start") "lbfgs_default" else arm]],
               if(arm=="lbfgs_reference_start") rt else x)
    a<-f$diagnostics$newton_accuracy;g<-f$diagnostics$geometric_stationarity
    rows[[length(rows)+1]]<-data.frame(case=id,estimator=est,arm=arm,f=f$fmin,
      accepted=f$converged,backend_status=f$optimizer_status,
      terminal_gradient=f$audit$grad_inf_norm,geometry_l2=g$ambient_residual_l2,
      newton_checked=a$checked,newton_status=a$status,newton_distance=a$distance,
      newton_passed=a$passed,newton_condition=a$condition,
      lin_eq_residual=f$diagnostics$lin_eq_residual_inf,
      nl_eq_residual=f$diagnostics$nl_eq_residual_inf)
    thetas[[length(thetas)+1]]<-data.frame(case=id,estimator=est,arm=arm,index=seq_along(f$theta),theta=f$theta)
  }
}
write.csv(do.call(rbind,rows),file.path(out,"endpoints.csv"),row.names=FALSE)
write.csv(do.call(rbind,thetas),file.path(out,"endpoint_parameters.csv"),row.names=FALSE)
