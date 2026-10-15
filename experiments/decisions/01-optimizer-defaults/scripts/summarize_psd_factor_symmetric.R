#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ("--help" %in% args || length(args)!=2L || args[1]!="--run-id") {
 cat("Usage: Rscript scripts/summarize_psd_factor_symmetric.R --run-id ID\n"); quit(status=if("--help" %in% args) 0 else 1)
}
script <- sub("^--file=","",commandArgs()[grepl("^--file=",commandArgs())][1])
root <- normalizePath(file.path(dirname(script),".."))
out <- file.path(root,"results/psd-factor-symmetric",args[2])
source(file.path(root,"R/psd_factor_pairing.R"))
paths <- list.files(file.path(out,"raw"),pattern="[.]csv$",full.names=TRUE)
if(!length(paths)) stop("no raw cells")
cells <- lapply(paths,read.csv,colClasses=c(error="character",ordinary_error="character"))
x <- do.call(rbind,cells)
meta <- read.csv(file.path(out,"metadata.csv"),colClasses="character")
stopifnot(length(paths)==as.integer(meta$value[meta$key=="expected_cells"]),nrow(x)==as.integer(meta$value[meta$key=="expected_rows"]))
x <- psd_factor_best(x)
rescue <- x[x$arm=="triangular",]
sym <- x[x$arm=="symmetric",]; sym <- sym[match(rescue$key,sym$key),]
use <- !rescue$converged
seconds <- rescue$seconds
rescue[use,] <- sym[use,]
rescue$seconds <- seconds + ifelse(use,sym$seconds,0)
rescue$arm <- "rescue"
x <- rbind(x,rescue)
by <- c("family","role","estimator")
write <- function(d,name) write.csv(d,file.path(out,paste0(name,".csv")),row.names=FALSE,na="NA")
summary <- do.call(rbind,lapply(split(x,key(x,c(by,"arm"))),function(d)cbind(d[1,c(by,"arm")],data.frame(
 fits=nrow(d),successes=sum(d$converged),best_known=sum(d$best),errors=sum(nzchar(d$error)),total_seconds=sum(d$seconds),median_seconds=median(d$seconds)))))
pairs <- lapply(c("symmetric","rescue","square"),function(arm) {
 p <- psd_factor_pair(x,arm)
 p$arm <- arm
 p$candidate_seconds <- p[[paste0("seconds_",arm)]]
 p$candidate_fmin <- p[[paste0("fmin_",arm)]]
 p$candidate_converged <- p[[paste0("converged_",arm)]]
 p$candidate_evaluations <- p[[paste0("objective_evaluations_",arm)]]
 p[,c("key",by,"arm","success_win","success_loss","best_win","best_loss","win","loss","candidate_seconds","seconds_triangular","candidate_fmin","fmin_triangular","candidate_converged","converged_triangular","candidate_evaluations","objective_evaluations_triangular")]
})
p <- do.call(rbind,pairs)
paired <- do.call(rbind,lapply(split(p,key(p,c(by,"arm"))),function(d)cbind(d[1,c(by,"arm")],data.frame(draws=nrow(d),success_wins=sum(d$success_win),success_losses=sum(d$success_loss),best_wins=sum(d$best_win),best_losses=sum(d$best_loss),wins=sum(d$win),losses=sum(d$loss),candidate_seconds=sum(d$candidate_seconds),baseline_seconds=sum(d$seconds_triangular)))))
f5 <- x[x$transform=="mixed",]
native <- x[x$transform=="native",]
ck <- c("population","model","estimator","N","replication","seed","arm")
idx <- match(key(f5,ck),key(native,ck))
if(anyNA(idx)) stop("F5 consistency requires corresponding native cells; select all or F1,F2,F3,F5")
f5$different <- f5$converged != native$converged[idx]
consistency <- do.call(rbind,lapply(split(f5,key(f5,c("family","arm"))),function(d)data.frame(family=d$family[1],arm=d$arm[1],draws=nrow(d),success_differs=sum(d$different))))
rules <- list()
add <- function(rule,family,estimator,clause,value,limit,pass) {
 rules[[length(rules)+1L]] <<- data.frame(rule=rule,family=family,estimator=estimator,clause=clause,value=value,limit=limit,pass=pass)
}
for(i in seq_len(nrow(paired))) {
 d <- paired[i,]; if(d$role!="test" || d$arm=="square") next
 rule <- if(d$arm=="symmetric") 1L else 2L
 add(rule,d$family,d$estimator,"success_losses_minus_wins",d$success_losses-d$success_wins,0,d$success_losses<=d$success_wins)
 add(rule,d$family,d$estimator,"best_losses_minus_wins",d$best_losses-d$best_wins,0,d$best_losses<=d$best_wins)
 ratio <- d$candidate_seconds/d$baseline_seconds
 add(rule,d$family,d$estimator,"seconds_ratio",ratio,1.25,is.finite(ratio)&&ratio<=1.25)
 if(startsWith(d$family,"F5_")) {
  z <- consistency[consistency$family==d$family,]
  value <- z$success_differs[z$arm==d$arm]; limit <- z$success_differs[z$arm=="triangular"]
  add(rule,d$family,d$estimator,"F5_consistency",value,limit,value<=limit)
 }
}
rules <- do.call(rbind,rules)
# Rule 2 gates only on cost; its success/best/consistency clauses are reported diagnostics.
r1 <- all(rules$pass[rules$rule==1])
r2 <- all(rules$pass[rules$rule==2 & rules$clause=="seconds_ratio"])
rules <- rbind(rules,data.frame(rule=c(1,2),family="overall",estimator="all",clause="verdict",value=c(r1,!r1&&r2),limit=1,pass=c(r1,!r1&&r2)))
m <- x[x$arm=="triangular" & x$estimator=="ML",]
m$delta_T <- 2*m$N*(m$fmin-m$ordinary_fmin)
agreement <- do.call(rbind,lapply(split(m,key(m,by)),function(d)cbind(d[1,by],data.frame(draws=nrow(d),comparable=sum(is.finite(d$delta_T)),different_T_gt_1e_4=sum(abs(d$delta_T)>1e-4,na.rm=TRUE),ordinary_errors=sum(nzchar(d$ordinary_error)),ordinary_seconds=sum(d$ordinary_seconds)))))
write(summary,"summary"); write(paired,"paired"); write(consistency,"consistency"); write(rules,"rules")
write(p[p$loss & p$arm %in% c("symmetric","rescue"),],"losses"); write(agreement,"agreement")
write(x[nzchar(x$error)|nzchar(x$ordinary_error),],"errors")
reps <- as.integer(meta$value[meta$key=="reps"])
cell_seconds <- sum(vapply(cells,function(d)d$cell_elapsed_seconds[1],0.0))
cost <- data.frame(cells=length(cells),rows=sum(vapply(cells,nrow,0L)),errors=sum(nzchar(x$error[x$arm!="rescue"])),cell_seconds=cell_seconds,mean_cell_seconds=cell_seconds/length(cells),smoke_wall_seconds=as.numeric(difftime(max(file.info(paths)$mtime),as.POSIXct(meta$value[meta$key=="started_utc"],tz="UTC"),units="secs")),production_cpu_hours_100_reps=cell_seconds/reps*100/3600,production_wall_hours_6_workers=cell_seconds/reps*100/3600/6)
write(cost,"cost")
extra <- data.frame(key=c("summary_command","summary_source_md5","summary_note"),value=c(paste(commandArgs(),collapse=" "),paste(unname(tools::md5sum(c(script,file.path(root,"R/psd_factor_pairing.R")))),collapse=";"),"Smoke verdicts are diagnostic only; production needs separate authorization."))
write(rbind(meta[!meta$key %in% extra$key,],extra),"metadata")
print(cost,row.names=FALSE); cat("Written:",out,"\n")
