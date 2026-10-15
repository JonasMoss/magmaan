#!/usr/bin/env Rscript
# Non-confirmatory smoke: --run-id smoke-2026-10-15 --seed-base 2026101502 --reps 2 --workers 2
args <- commandArgs(TRUE)
if ("--help" %in% args) {
  cat("PSD factor paired runner\n--run-id ID --seed-base INTEGER --reps INTEGER --workers INTEGER --families all|F1,F2,F3,F4,controls\nOne atomic CSV per population/model/N/estimator cell; completed cells are skipped.\nProduction: seed 2026101501, reps 100; requires author's separate go.\n")
  quit()
}
arg <- function(key,default=NULL) {i<-match(paste0("--",key),args); if(is.na(i)) default else {if(i==length(args)) stop("missing ",key); args[i+1L]}}
run_id <- arg("run-id"); seed_base <- as.integer(arg("seed-base")); reps <- as.integer(arg("reps","2")); workers <- as.integer(arg("workers","2"))
if (is.null(run_id) || !grepl("^[A-Za-z0-9_-]+$",run_id) || anyNA(c(seed_base,reps,workers)) || reps<1 || workers<1) stop("invalid arguments; see --help")
Sys.setenv(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
script <- sub("^--file=","",commandArgs()[grepl("^--file=",commandArgs())][1])
root <- normalizePath(file.path(dirname(script),".."))
source(file.path(root,"R/families.R")); source(file.path(root,"R/fits.R"))
source(file.path(root,"R/psd_factor_families.R"))
out <- file.path(root,"results/psd-factor",run_id); dir.create(file.path(out,"raw"),recursive=TRUE,showWarnings=FALSE)
selected <- strsplit(arg("families","all"),",",fixed=TRUE)[[1]]
if (!all(selected %in% c("all","F1","F2","F3","F4","controls"))) stop("unknown family")
# A source fingerprint prevents silently mixing resumed runs across implementations.
sources <- c(file.path(root,c("R/families.R","R/fits.R","R/psd_factor_families.R","scripts/psd_factor.R","criteria/psd-factor.md")))
package_files <- unlist(lapply(c("magmaanlab","magmaan"), function(pkg) list.files(find.package(pkg), pattern="[.](so|rdb)$",recursive=TRUE,full.names=TRUE)))
fingerprint <- paste(unname(tools::md5sum(c(sources,package_files))),collapse=";")
metadata <- data.frame(key=c("run_id","seed_base","reps","families","source_md5","command","git_commit","magmaanlab_version","magmaan_version","library","R_version","ordinal_thresholds","mcar","purpose"),
 value=c(run_id,seed_base,reps,paste(selected,collapse=","),fingerprint,paste(commandArgs(),collapse=" "),
 system2("git",c("-C",shQuote(root),"rev-parse","HEAD"),stdout=TRUE),as.character(packageVersion("magmaanlab")),as.character(packageVersion("magmaan")),
 find.package("magmaanlab"),R.version.string,"qnorm(c(.2,.4,.6,.8)) times population marginal SD","15% independent MCAR",if(seed_base==2026101502) "smoke; decides nothing" else "production; separate author authorization required"))
meta_path <- file.path(out,"metadata.csv")
pops <- psd_factor_populations(); cells <- list()
for(p in pops) for(fm in p$models) for(n in p$ns) {
 families <- p$family
 if(p$family=="F2") families <- c("F2","F4")
 for(family in families) {
  group <- if(p$role=="control") "controls" else if(startsWith(family,"F1")) "F1" else family
  if(!("all" %in% selected || group %in% selected)) next
  estimators <- if(family %in% c("F2","F3")) c("ML","FIML","ULS","DWLS") else "ML"
  for(estimator in estimators) cells[[length(cells)+1L]] <- list(p=p,fm=fm,n=n,family=family,estimator=estimator)
 }
}
metadata <- rbind(metadata,data.frame(key=c("expected_cells","expected_rows"),value=c(length(cells),length(cells)*reps*2L)))
if(file.exists(meta_path)) {
 old <- read.csv(meta_path); keys <- c("seed_base","reps","families","source_md5","magmaanlab_version","magmaan_version","library")
 if(!identical(old$value[match(keys,old$key)],metadata$value[match(keys,metadata$key)])) stop("resume metadata mismatch")
} else write.csv(metadata,meta_path,row.names=FALSE)
fit_call <- function(model,data,estimator,arm,start=NULL) {
 fun <- switch(estimator,ML=magmaanlab::frontier_fit_ml_psd,FIML=magmaanlab::frontier_fit_fiml_psd,
  ULS=magmaanlab::frontier_fit_uls_psd,DWLS=magmaanlab::frontier_fit_ordinal_psd)
 a <- list(model=model,data=data)
 if(arm=="square") a$psd_factor <- "square"
 if(!is.null(start)) a$control <- list(start=start)
 do.call(fun,a)
}
run_cell <- function(cell) {
 p<-cell$p; fm<-cell$fm; n<-cell$n; est<-cell$estimator; family<-cell$family
 path <- file.path(out,"raw",paste(family,p$key,fm$key,n,est,sep="__")); path<-paste0(path,".csv")
 if(file.exists(path)) return(paste("skip",basename(path)))
 rows <- list(); cell_started <- proc.time()[[3]]
 for(rep in seq_len(reps)) {
  seed <- draw_seed(seed_base,p$key,n,rep); moments <- draw_moments(p,n,seed); d<-moments$X
  # F2 and F4 deliberately share draws. Missingness uses a separate stable stream.
  if(est=="FIML") {set.seed(draw_seed(seed_base,paste0(p$key,"_mcar"),n,rep)); d[ matrix(runif(n*p$p)<.15,n,p$p)]<-NA_real_}
  model <- build_model(fm); input<-d; prep_error<-""
  if(est=="DWLS") {
   for(j in seq_len(p$p)) d[[j]] <- ordered(cut(d[[j]],c(-Inf,qnorm(c(.2,.4,.6,.8))*sqrt(p$Sigma[j,j]),Inf),labels=FALSE),levels=1:5)
   input <- tryCatch(magmaanlab::magmaan_core$data_ordinal_stats_from_df(d,model,ordered=names(d)),error=function(e){prep_error<<-conditionMessage(e);NULL})
  }
  start<-NULL; ordinary_fmin<-NA_real_; ordinary_error<-""; ordinary_seconds<-0
  if(est=="ML") {
   t<-proc.time()[[3]]
   ordinary_fmin<-tryCatch(magmaan::as_lab_fit(magmaan::magmaan(model,d,covariance="psd",inference=FALSE))$fmin,
    error=function(e){ordinary_error<<-conditionMessage(e);NA_real_})
   ordinary_seconds<-proc.time()[[3]]-t
  }
  if(family=="F4") {
   # Obtain the untouched library start, then change only the first Psi column.
   base <- fit_call(model,input,est,"triangular")
   start <- base$start$theta; pt<-base$partable
   ix<-which(pt$lhs=="f1" & pt$op=="~~" & pt$rhs %in% paste0("f",1:3))
   stopifnot(length(ix)==3L,all(pt$free[ix]>0))
   start[pt$free[ix]]<-ifelse(pt$rhs[ix]=="f1",1e-6,0)
  }
  for(arm in c("triangular","square")) {
   error<-prep_error; t<-proc.time()[[3]]
   fit<-if(nzchar(error)) NULL else tryCatch(fit_call(model,input,est,arm,start),error=function(e){error<<-conditionMessage(e);NULL})
   seconds<-proc.time()[[3]]-t
   verified<-NA
   if(family=="F4" && !is.null(fit)) {
    verified<-identical(as.numeric(fit$start$theta),as.numeric(start))
    if(!verified) stop("F4 start transport failed: Needs decision")
   }
   rows[[length(rows)+1L]]<-data.frame(family=family,role=p$role,population=p$key,model=fm$key,estimator=est,N=n,
    replication=rep,seed=seed,arm=arm,converged=!is.null(fit)&&isTRUE(fit$converged),
    fmin=if(is.null(fit)) NA_real_ else fit$fmin,objective_evaluations=if(is.null(fit)) NA_integer_ else fit$f_evals,
    seconds=seconds,error=error,ordinary_fmin=ordinary_fmin,ordinary_error=ordinary_error,ordinary_seconds=ordinary_seconds,
    start_verified=verified,start_variance=if(is.null(start)) NA_real_ else start[pt$free[ix[pt$rhs[ix]=="f1"]]],
    start_covariance_12=if(is.null(start)) NA_real_ else start[pt$free[ix[pt$rhs[ix]=="f2"]]],
    start_covariance_13=if(is.null(start)) NA_real_ else start[pt$free[ix[pt$rhs[ix]=="f3"]]])
  }
 }
 records<-do.call(rbind,rows); records$cell_elapsed_seconds <- proc.time()[[3]]-cell_started
 tmp<-paste0(path,".tmp");write.csv(records,tmp,row.names=FALSE);stopifnot(file.rename(tmp,path))
 paste("done",basename(path))
}
cat(length(cells),"cells;",reps,"replications;",workers,"workers\n")
# Bounded batches keep parent-visible progress and make interruptions resumable.
for(first in seq.int(1L,length(cells),by=workers)) {
 result<-parallel::mclapply(cells[first:min(length(cells),first+workers-1L)],run_cell,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
 if(any(vapply(result,inherits,FALSE,"try-error"))) stop("cell failed; inspect errors above")
 cat(paste(unlist(result),collapse="\n"),"\n");flush.console()
}
cat("Raw cells:",file.path(out,"raw"),"\n")
