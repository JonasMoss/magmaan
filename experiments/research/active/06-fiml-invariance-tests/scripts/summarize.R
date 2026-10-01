#!/usr/bin/env Rscript
# Curate original runs without pooling different designs or fitting models.
script <- sub("^--file=", "", grep("^--file=",commandArgs(),value=TRUE)[1L])
root <- dirname(dirname(normalizePath(script)))
if (identical(commandArgs(TRUE),"--help")) {
  cat("Usage: Rscript scripts/summarize.R\nSummarizes retained calibration, flip-stress, pilot and oracle runs; writes source hashes.\n")
  quit(status=0L)
}
out <- file.path(root,"results","overview")
sources <- character()
read_result <- function(name) {
  p <- file.path(root,"results",name)
  if (!file.exists(p)) stop("Missing historical result: ",p,
    "; recover that original run before rebuilding the overview.")
  sources <<- c(sources,p)
  read.csv(p,stringsAsFactors=FALSE)
}
s <- read_result("calibration/summary_rejection.csv")
meta <- read_result("calibration/metadata.csv")
# Complete/MCAR permit the stated covariance null; nonnormal MAR is stress.
s <- s[s$truth=="h0" & s$outcome=="nested" & s$rung %in% c("weak","strict") &
       (s$mech != "MAR" | s$dist == "norm") & s$method %in% c("naive","SB","pEBA4"),]
calibration <- do.call(rbind,lapply(split(s,interaction(s$estimator,s$rung,s$method)),function(z)
  data.frame(estimator=z$estimator[1],step=z$rung[1],method=z$method[1],cells=nrow(z),
    min_rate=min(z$reject),max_rate=max(z$reject),
    mean_cell_error=mean(abs(z$reject-.05)),min_valid=min(z$n_reps),max_valid=max(z$n_reps))))
flips <- do.call(rbind,lapply(c("screen","confirm"),function(profile) {
  z <- read_result(paste0("score-flips/",profile,"/method_summary.csv"))
  z <- z[z$method %in% c("flip_effective","flip_standardized","score_sb","score_peba4","nested_peba4_ml"),]
  z$region <- ifelse(z$distribution != "normal" & grepl("mar",z$missingness),
                     "nonnormal MAR stress","complete/MCAR or normal MAR")
  do.call(rbind,lapply(split(z,interaction(z$region,z$method)),function(x)
    data.frame(profile=profile,region=x$region[1],method=x$method[1],cells=nrow(x),
      mean_cell_rate=mean(x$rejection_rate),min_rate=min(x$rejection_rate),max_rate=max(x$rejection_rate),
      min_valid=min(x$n),max_valid=max(x$n))))
}))
parity <- read_result("oracle/lavaan_parity.csv")
parity <- do.call(rbind,lapply(split(parity,parity$metric),function(z)
  data.frame(metric=z$metric[1],comparisons=nrow(z),max_abs_difference=max(z$abs_diff,na.rm=TRUE))))
pilot <- read_result("pilot/matched_power.csv")
pilot_meta <- read_result("pilot/metadata.csv")
dir.create(out,recursive=TRUE,showWarnings=FALSE)
artifacts <- list(calibration=calibration,calibration_metadata=meta,flip_stress=flips,
  oracle_checks=parity,pilot_power=pilot,pilot_metadata=pilot_meta)
for (name in names(artifacts)) write.csv(artifacts[[name]],file.path(out,paste0(name,".csv")),row.names=FALSE)
write.csv(data.frame(source=sub(paste0(root,"/"),"",sources,fixed=TRUE),
  md5=unname(tools::md5sum(sources))),file.path(out,"provenance.csv"),row.names=FALSE)
cat("Wrote retained overview to",out,"\n")
