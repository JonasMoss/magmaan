#!/usr/bin/env Rscript
script <- normalizePath(sub("^--file=","",grep("^--file=",commandArgs(),value=TRUE)[1]))
here <- dirname(script)
if ("--help" %in% commandArgs(TRUE)) {
  cat("Corrected-corpus optimizer recovery (ordinary ML and fixed-weight GLS).\n",
      "Usage: Rscript run_experiment.R [--smoke] [--cases id,id] [options]\n",
      "  --corpus PATH    Optional textbook-corpus mount (default: external/textbook-corpus)\n",
      "  --results PATH   Output directory (default: results/current)\n",
      "  --arms name,name Restrict policies (useful for isolated timeout follow-ups)\n",
      "  --workers N      Concurrent case/estimator processes (default: 3)\n",
      "  --timeout SEC    Wall-clock cap per case/estimator, all arms (default: 180)\n",
      "  --engine-source REV  Commit used to build installed magmaanlab (optional provenance)\n",
      "Requires the local corpus plus installed magmaanlab, lavaan and jsonlite.\n",
      "Arms: L-BFGS default/budget/tight/memory50; PORT default/tight;\n",
      "SLSQP default/tight; explicit L-BFGS-to-SLSQP fallback.\n",
      "Every ordinary arm uses the same current constructed start. A separate\n",
      "reference-start arm diagnoses starting-point dependence. No PSD substitution.\n",
      "ML retains its current coordinate scaling; controls are in R/arms.R.\n",
      "The smoke profile runs the two historical Newsom GLS cases and Little's\n",
      "phantom model. Use this timing trial before scheduling the full corpus.\n",
      "Jobs checkpoint completed arms; timeouts do not mean numerical failure.\n",
      "After completion: Rscript R/complete_timeouts.R CORPUS RESULTS;\n",
      "Rscript R/validate_inputs.R CORPUS RESULTS;\n",
      "Rscript R/summarize.R RESULTS; quarto render report.qmd.\n", sep="")
  quit(save="no")
}
source(file.path(here,"../../_support/R/helpers.R"))
set_single_threaded_math()
suppressPackageStartupMessages({library(magmaanlab);library(lavaan);library(jsonlite)})
source(file.path(here,"R/inputs.R"));source(file.path(here,"R/arms.R"))
argv<-commandArgs(TRUE)
value<-function(key,default=NULL) {i<-match(key,argv);if(is.na(i)) default else argv[i+1L]}
root<-normalizePath(value("--corpus",corpus_root()))
out<-normalizePath(value("--results",file.path(here,"results/current")),mustWork=FALSE)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
manifest<-read.csv(file.path(root,"manifest.csv"),stringsAsFactors=FALSE)
write_row<-function(row,path) {
  row<-as.data.frame(row,stringsAsFactors=FALSE)
  write.table(row,path,sep=",",row.names=FALSE,col.names=!file.exists(path),append=file.exists(path),qmethod="double")
}
worker<-function(id,estimator) {
  path<-file.path(out,paste0(id,"__",estimator,".csv"))
  base<-list(case=id,estimator=estimator,arm="preparation",returned=FALSE,converged=FALSE,
    f=NA_real_,status="error",stationary=FALSE,gradient=NA_real_,raw_gradient=NA_real_,
    geometry_checked=FALSE,geometry_l2=NA_real_,verdict="",verdict_reason="",
    audit_rhs=NA_real_,f_consistent=FALSE,admissible=FALSE,f_evals=NA_integer_,g_evals=NA_integer_,
    iterations=NA_integer_,seconds=NA_real_,reference_f=NA_real_,reference_converged=FALSE,
    reference_full_gradient=NA_real_,start_f=NA_real_,start_method="",start_transport="",message="")
  save_error<-function(e) {r<-base;r$message<-conditionMessage(e);write_row(r,path)}
  tryCatch({
    case<-read_case(root,manifest$case_dir[match(id,manifest$case_id)],estimator)
    method<-if(estimator=="ML") "scaled-fabin" else "fabin3"
    x0<-magmaan_core$estimate_start_values(case$model$partable,case$sample,start=method,
                                          transport=if(estimator=="ML") "auto" else "native")
    base$start_method<-attr(x0,"start_method") %||% method
    base$start_transport<-attr(x0,"start_transport") %||% ""
    initial<-tryCatch(magmaan_core$evaluate_at(case$model,case$sample,x0,estimator),error=function(e)NULL)
    if(!is.null(initial)) base$start_f<-initial$fmin
    # Evaluate the reference under OUR original objective, avoiding statistic scaling differences.
    ref<-tryCatch(reference_fit(case),error=function(e)NULL)
    if(!is.null(ref)) {
      rt<-map_theta(case$model,ref)
      at<-tryCatch(magmaan_core$evaluate_at(case$model,case$sample,rt,estimator),error=function(e)NULL)
      base$reference_converged<-isTRUE(lavaan::lavInspect(ref,"converged"))
      if(!is.null(at)) {base$reference_f<-at$fmin;base$reference_full_gradient<-at$audit$grad_inf_norm}
    }
    selected_arms<-names(optimizer_arms())
    if(length(value("--arms"))) selected_arms<-intersect(selected_arms,strsplit(value("--arms"),",",fixed=TRUE)[[1]])
    for(name in selected_arms) {
      base$arm<-name;t0<-proc.time()[["elapsed"]]
      row<-tryCatch({fit<-one_fit(case,estimator,optimizer_arms()[[name]],x0)
        r<-modifyList(base,fit_fields(fit));r$message<-"";r
      },error=function(e){r<-base;r$message<-conditionMessage(e);r})
      row$seconds<-proc.time()[["elapsed"]]-t0
      write_row(row,path)
    }
    # Separate verified-solution restart: not counted as a default-start arm.
    want_reference<-is.null(value("--arms")) || "lbfgs_reference_start" %in% strsplit(value("--arms"),",",fixed=TRUE)[[1]]
    if(want_reference && !is.null(ref) && base$reference_converged) {
      base$arm<-"lbfgs_reference_start";t0<-proc.time()[["elapsed"]]
      row<-tryCatch(modifyList(base,fit_fields(one_fit(case,estimator,optimizer_arms()$lbfgs_default,rt))),
                    error=function(e){r<-base;r$message<-conditionMessage(e);r})
      row$seconds<-proc.time()[["elapsed"]]-t0;write_row(row,path)
    }
  },error=save_error)
}
if("--worker" %in% argv) {worker(value("--worker"),value("--estimator"));quit(save="no")}
ids<-manifest$case_id
if("--smoke" %in% argv) ids<-c("newsom_2015_ex5_4","newsom_2015_ex5_4c","little_2013_ch3_fig_3_11_longitudinal_cfa_phantom")
if(length(value("--cases"))) ids<-strsplit(value("--cases"),",",fixed=TRUE)[[1]]
jobs<-expand.grid(case=ids,estimator=c("ML","GLS"),stringsAsFactors=FALSE)
# Manifest/case fingerprints identify the exact corpus used. Check them again at completion.
paths<-c(file.path(root,"manifest.csv"),unlist(lapply(manifest$case_dir[match(ids,manifest$case_id)],function(d)
  list.files(file.path(root,d),pattern="\\.(json|lav|csv)$",recursive=TRUE,full.names=TRUE))))
hashes<-tools::md5sum(paths)
write.csv(data.frame(path=names(hashes),md5=unname(hashes)),file.path(out,"input_hashes.csv"),row.names=FALSE)
write_metadata(file.path(out,"metadata.csv"),values=list(
  git_head=system2("git",c("rev-parse","HEAD"),stdout=TRUE),
  engine_source_commit=value("--engine-source","not recorded"),
  corpus_head=system2("git",c("-C",shQuote(root),"rev-parse","HEAD"),stdout=TRUE),
  workers=value("--workers","3"), timeout_seconds=value("--timeout","180"),
  start_policy="ML auto FABIN3; GLS native FABIN3; same supplied vector for all ordinary arms"),packages=c("magmaanlab","lavaan"))
library_paths<-list.files(find.package("magmaanlab"),recursive=TRUE,full.names=TRUE)
library_hashes<-tools::md5sum(library_paths)
write.csv(data.frame(path=names(library_hashes),md5=unname(library_hashes)),
          file.path(out,"library_hashes.csv"),row.names=FALSE)
statuses<-parallel::mclapply(seq_len(nrow(jobs)),function(i) {
  j<-jobs[i,];path<-file.path(out,paste0(j$case,"__",j$estimator,".csv"))
  if(file.exists(path)) unlink(path)
  log<-file.path(out,paste0(j$case,"__",j$estimator,".log"))
  args<-c(value("--timeout","180"),file.path(R.home("bin"),"Rscript"),shQuote(script),
    "--worker",j$case,"--estimator",j$estimator,"--corpus",shQuote(root),"--results",shQuote(out))
  if(length(value("--arms"))) args<-c(args,"--arms",value("--arms"))
  code<-system2("timeout",args,stdout=log,stderr=log)
  cat(sprintf("[%d/%d] %s %s exit=%d\n",i,nrow(jobs),j$case,j$estimator,code))
  data.frame(case=j$case,estimator=j$estimator,exit=code)
},mc.cores=as.integer(value("--workers","3")),mc.preschedule=FALSE)
write.csv(do.call(rbind,statuses),file.path(out,"jobs.csv"),row.names=FALSE)
if(!identical(unname(hashes),unname(tools::md5sum(paths)))) stop("Corpus inputs changed during run; results are not a pinned comparison")
if(!identical(unname(library_hashes),unname(tools::md5sum(library_paths)))) stop("Installed engine changed during run")
cat("Corpus hashes unchanged. Run R/summarize.R with the result directory.\n")
