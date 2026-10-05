#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
get <- function(key) { i <- match(key,args); if(is.na(i) || i==length(args)) stop("Missing ",key); args[i+1L] }
file <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
study <- dirname(dirname(normalizePath(file))); root <- normalizePath(file.path(study,"../../.."))
source(file.path(root,"experiments/_support/R/helpers.R")); source(file.path(study,"R/study.R"))
run <- get("--run-dir"); mode <- get("--mode")
seed <- c(smoke=818130001L,pilot=818140001L,production=830150001L,confirm=840150001L)[[mode]]
cells <- threshold_cells()
if("--cell" %in% args) cells <- cells[cells$cell_id %in% as.integer(strsplit(get("--cell"),",",fixed=TRUE)[[1]]),]
files <- file.path(run,"cells",sprintf("cell_%03d",cells$cell_id),"raw.rds")
if(!all(file.exists(files))) stop("Incomplete cell outputs")
meta <- lapply(dirname(files),function(d) read.csv(file.path(d,"metadata.csv")))
if(!all(vapply(meta,function(m) identical(m[m$key %in% c("source_md5","criteria_md5"),],meta[[1]][meta[[1]]$key %in% c("source_md5","criteria_md5"),]),FALSE))) stop("Provenance mismatch")
raw <- do.call(rbind,lapply(files,readRDS)); raw <- raw[order(raw$cell_id,raw$replicate,raw$arm),]; rownames(raw) <- NULL
if(any(raw$seed!=seed+10000L*raw$cell_id+raw$replicate)) stop("Seed mismatch")
for(i in seq_len(nrow(cells))) {
  reps <- if(mode %in% c("production","confirm")) cells$production_reps[i] else if(mode=="pilot") 20L else 2L
  x <- raw[raw$cell_id==cells$cell_id[i],]
  if(nrow(x)!=4L*reps || anyDuplicated(x[c("replicate","arm")]) || !setequal(x$replicate,seq_len(reps))) stop("Incomplete or duplicate draws")
}
out <- file.path(run,"final"); if(dir.exists(out)) stop("Final output exists")
dir.create(out); saveRDS(raw,file.path(out,"raw.rds")); write_csv(cells,file.path(out,"cells.csv"))
write_metadata(file.path(out,"metadata.csv"),list(mode=mode,seed_base=seed,executor="modal",source_metadata= paste(dirname(files),collapse=";")),packages=c("magmaanlab","lavaan"))
threshold_summarize(raw,cells,out); cat("Combined into",out,"\n")
