#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if("--help" %in% args) {
  cat("Threshold invariance calibration (criteria draft; production requires merger registration).\n",
      "Rscript run_experiment.R --smoke|--pilot|--production [--cell 1,2] [--workers 1|2] --out-dir PATH\n",
      "Smoke: 2 draws/cell; pilot: 20; production: 2000 null/1000 power.\n",
      "Seeds: 818130001/818140001/830150001 + 10000*cell + replicate. Fresh output required.\n")
  quit(status=0)
}
file <- sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])
study <- dirname(normalizePath(file)); root <- normalizePath(file.path(study,"../../.."))
source(file.path(root,"experiments/_support/R/helpers.R"))
source(file.path(study,"R/study.R"))
opt <- function(key,default=NULL) { i <- match(key,args); if(is.na(i)) default else {
  if(i==length(args)) stop("Missing value for ",key); args[i+1L] } }
modes <- c("smoke","pilot","production"); mode <- modes[paste0("--",modes) %in% args]
if(length(mode)!=1L) stop("Choose exactly one mode; see --help")
workers <- as.integer(opt("--workers","1")); if(!workers %in% 1:2) stop("Use 1 or 2 workers")
cells <- threshold_cells()
if(!is.null(opt("--cell"))) {
  ids <- as.integer(strsplit(opt("--cell"),",",fixed=TRUE)[[1]])
  if(anyNA(ids) || any(!ids %in% cells$cell_id)) stop("Unknown cell")
  cells <- cells[cells$cell_id %in% ids,]
}
out <- opt("--out-dir",file.path(study,"results",mode))
if(dir.exists(out)) stop("Output exists: use a fresh directory")
dir.create(out,recursive=TRUE)
seed_base <- c(smoke=818130001L,pilot=818140001L,production=830150001L)[[mode]]
write_csv(cells,file.path(out,"cells.csv"))
write_metadata(file.path(out,"metadata.csv"),list(mode=mode,seed_base=seed_base,
  workers=workers,git_head=system2("git",c("-C",root,"rev-parse","HEAD"),stdout=TRUE),
  package_dll_md5=unname(tools::md5sum(system.file("libs",paste0("magmaanlab",.Platform$dynlib.ext),package="magmaanlab"))),
  criteria_status="draft; pilot is non-gating",
  source_md5=paste(tools::md5sum(c(file,file.path(study,"R/study.R"))),collapse=";"),
  criteria_md5=unname(tools::md5sum(file.path(study,"criteria/threshold_invariance.md")))),packages=c("magmaanlab","lavaan"))
raw <- list()
for(i in seq_len(nrow(cells))) {
  cell <- cells[i,]; reps <- if(mode=="production") cell$production_reps else if(mode=="pilot") 20L else 2L
  cat("Cell",cell$cell_id,"(",i,"/",nrow(cells),"):",reps,"draws\n")
  rows <- parallel::mclapply(seq_len(reps),function(r) threshold_replicate(cell,r,seed_base),
    mc.cores=workers,mc.preschedule=TRUE,mc.set.seed=FALSE)
  if(any(vapply(rows,inherits,FALSE,"try-error"))) stop("Worker failed")
  raw[[i]] <- do.call(rbind,rows)
  saveRDS(do.call(rbind,raw),file.path(out,"checkpoint.rds"))
}
raw <- do.call(rbind,raw); saveRDS(raw,file.path(out,"raw.rds"))
threshold_summarize(raw,cells,out); cat("Wrote",normalizePath(out),"\n")
