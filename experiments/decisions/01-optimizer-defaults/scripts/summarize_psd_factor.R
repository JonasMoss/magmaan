#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if("--help" %in% args || length(args)!=2L || args[1]!="--run-id") {
 cat("Usage: Rscript scripts/summarize_psd_factor.R --run-id ID\nWrites per-family/estimator/arm rates, paired counts, every loss, agreement and failures.\n");quit(status=if("--help" %in% args)0 else 1)
}
script<-sub("^--file=","",commandArgs()[grepl("^--file=",commandArgs())][1])
root<-normalizePath(file.path(dirname(script),".."));out<-file.path(root,"results/psd-factor",args[2])
paths<-list.files(file.path(out,"raw"),pattern="[.]csv$",full.names=TRUE)
if(!length(paths))stop("no raw cells")
cells<-lapply(paths,read.csv,stringsAsFactors=FALSE,colClasses=c(error="character",ordinary_error="character"))
x<-do.call(rbind,cells)
meta<-read.csv(file.path(out,"metadata.csv"),colClasses="character")
stopifnot(length(paths)==as.integer(meta$value[meta$key=="expected_cells"]),
 nrow(x)==as.integer(meta$value[meta$key=="expected_rows"]))
key<-function(d,cols)do.call(paste,c(d[cols],sep="|"))
x$key<-key(x,c("family","population","model","estimator","N","replication","seed"))
stopifnot(!anyDuplicated(paste(x$key,x$arm)))
x$best<-FALSE
for(ids in split(seq_len(nrow(x)),x$key)) {
 d<-x[ids,];stopifnot(setequal(d$arm,c("triangular","square")))
 success<-d$converged & is.finite(d$fmin)
 if(any(success)) {best<-min(d$fmin[success]);x$best[ids]<-success & d$fmin<=best+1e-6*(1+abs(best))}
}
by<-c("family","role","estimator")
summary<-do.call(rbind,lapply(split(x,key(x,c(by,"arm"))),function(d)cbind(d[1,c(by,"arm")],
 data.frame(fits=nrow(d),successes=sum(d$converged),best_known=sum(d$best),errors=sum(nzchar(d$error)),
 total_seconds=sum(d$seconds),median_seconds=median(d$seconds)))))
a<-x[x$arm=="square",]; b<-x[x$arm=="triangular",]
p<-merge(a,b,by="key",suffixes=c("_square","_triangular"))
for(k in by)p[[k]]<-p[[paste0(k,"_square")]]
p$success_win<-p$converged_square & !p$converged_triangular
p$success_loss<-!p$converged_square & p$converged_triangular
p$best_win<-p$best_square & !p$best_triangular
p$best_loss<-!p$best_square & p$best_triangular
p$win<-p$success_win | (p$converged_square & p$converged_triangular & p$best_win)
p$loss<-p$success_loss | (p$converged_square & p$converged_triangular & p$best_loss)
paired<-do.call(rbind,lapply(split(p,key(p,by)),function(d)cbind(d[1,by],data.frame(draws=nrow(d),
 success_wins=sum(d$success_win),success_losses=sum(d$success_loss),best_wins=sum(d$best_win),best_losses=sum(d$best_loss),
 wins=sum(d$win),losses=sum(d$loss),triangular_seconds=sum(d$seconds_triangular),square_seconds=sum(d$seconds_square)))))
# Complete-data ML uses fmin=F/2, so T=2*N*fmin (ML covariance uses divisor N).
m<-x[x$arm=="triangular" & x$estimator=="ML",]
m$delta_T<-2*m$N*(m$fmin-m$ordinary_fmin)
agreement<-do.call(rbind,lapply(split(m,key(m,by)),function(d)cbind(d[1,by],data.frame(draws=nrow(d),
 comparable=sum(is.finite(d$delta_T)),different_T_gt_1e_4=sum(abs(d$delta_T)>1e-4,na.rm=TRUE),
 ordinary_errors=sum(nzchar(d$ordinary_error)),ordinary_seconds=sum(d$ordinary_seconds)))))
write<-function(d,name)write.csv(d,file.path(out,paste0(name,".csv")),row.names=FALSE,na="NA")
write(summary,"summary");write(paired,"paired");write(p[p$loss,],"losses")
write(agreement,"agreement");write(m[is.finite(m$delta_T)&abs(m$delta_T)>1e-4,],"agreement_differences")
write(x[nzchar(x$error)|nzchar(x$ordinary_error),],"errors")
write(x[x$family=="F4",c("population","N","replication","seed","arm","start_verified","start_variance","start_covariance_12","start_covariance_13")],"start_verification")
cat("Per-family summaries (no automatic decision):\n");print(summary,row.names=FALSE);print(paired,row.names=FALSE)
meta<-read.csv(file.path(out,"metadata.csv"),colClasses="character")
reps<-as.integer(meta$value[meta$key=="reps"])
cell_seconds<-sum(vapply(cells,function(d)d$cell_elapsed_seconds[1],0.0))
cost<-data.frame(fits=nrow(x),cells=length(cells),mean_fit_seconds=mean(x$seconds),
 median_fit_seconds=median(x$seconds),arm_fit_seconds=sum(x$seconds),
 ordinary_fit_seconds=sum(m$ordinary_seconds),cell_seconds=cell_seconds,
 production_cpu_hours_100_reps=cell_seconds/reps*100/3600)
write(cost,"cost")
cat("Measured cost (cell time includes preparation and F4 start retrieval):\n");print(cost,row.names=FALSE)
extra<-data.frame(key=c("summary_source_md5","summary_command","summary_note","control_factor_correlations"),
 value=c(unname(tools::md5sum(script)),paste(commandArgs(),collapse=" "),
 if (meta$value[meta$key=="seed_base"]=="2026101502") "Smoke only; no default decision. Earlier development attempts retained under ignored raw/attempt-*." else "Production summary; apply registered rules separately.","weak_marker_three_factor: independent factors"))
meta<-rbind(meta[!meta$key %in% extra$key,],extra)
write.csv(meta,file.path(out,"metadata.csv"),row.names=FALSE)
cat("Written:",out,"\n")
