#!/usr/bin/env Rscript
# Frozen production evidence: reduce bulky inputs without changing denominators.
args <- commandArgs(TRUE)
if (length(args) != 2L || '--help' %in% args) {
  cat('Usage: Rscript scripts/compact_production.R INPUT_DIRECTORY OUTPUT_DIRECTORY\n')
  quit(status = if ('--help' %in% args) 0L else 1L)
}
input <- normalizePath(args[1]); output <- args[2]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
read_input <- function(name) read.csv(file.path(input, paste0(name,'.csv')), stringsAsFactors=FALSE)
write_evidence <- function(x, name) {
  path <- file.path(output,paste0(name,'.csv'))
  write.csv(x,path,row.names=FALSE,na='')
  if (file.info(path)$size >= 1000000) stop('Evidence exceeds 1 MB: ',path)
}
for (name in c('metadata','cells','targets','cost','arms','generator_checks','setup_timing','target_diagnostics'))
  write_evidence(read_input(name),name)
if (file.exists(file.path(input,'target_correction_metadata.csv')))
  write_evidence(read_input('target_correction_metadata'),'target_correction_metadata')
s <- read_input('summary')
stopifnot(nrow(s)==768L*24L, all(s$attempted==1000), all(s$target_status=='available'))
for (index in unique(s$index)) write_evidence(s[s$index==index,],paste0('summary_',index))
# Stream failures so compaction never requires the full 443 MB table in memory.
con <- file(file.path(input,'failures.csv'),'r'); header <- readLines(con,n=1)
counts <- list(); defects <- list(); chunk <- 0L
repeat {
  lines <- readLines(con,n=50000L)
  if (!length(lines)) break
  x <- read.csv(text=paste(c(header,lines),collapse='\n'),stringsAsFactors=FALSE)
  defect <- grepl('replacement has 10 rows, data has 24',x$reason,fixed=TRUE)
  if (any(defect)) defects[[length(defects)+1L]] <- unique(x[defect,c('cell','replicate')])
  x$reason_class <- ifelse(defect,'arm_row_assignment_defect',
    ifelse(grepl('^not_converged:',x$reason),'not_converged',x$reason))
  chunk <- chunk+1L
  counts[[chunk]] <- aggregate(rep(1L,nrow(x)),x[c('cell','arm','index','reason_class')],sum)
}
close(con)
x <- do.call(rbind,counts)
x <- aggregate(x$x,x[c('cell','arm','index','reason_class')],sum); names(x)[5] <- 'count'
write_evidence(x,'failure_counts')
d <- unique(do.call(rbind,defects)); d <- d[order(d$cell,d$replicate),]
write_evidence(d,'harness_affected_draws')
write_evidence(aggregate(rep(1L,nrow(d)),list(cell=d$cell),sum),'harness_affected_cells')
t <- read_input('timing')
timing <- do.call(rbind,lapply(split(t$seconds,t$cell),function(v)
  data.frame(attempted=length(v),mean=mean(v),q00=min(v),q25=unname(quantile(v,.25)),
    q50=median(v),q75=unname(quantile(v,.75)),q95=unname(quantile(v,.95)),q100=max(v))))
timing$cell <- rownames(timing); rownames(timing) <- NULL
write_evidence(timing,'timing_summary')
provenance <- data.frame(file=list.files(input,pattern='\\.csv$'),md5=unname(tools::md5sum(list.files(input,pattern='\\.csv$',full.names=TRUE))))
write_evidence(provenance,'input_checksums')
write_evidence(data.frame(key=c('purpose','library_commit','harness_defect_draws','harness_defect_cells','compaction'),
  value=c('Frozen ML production; approved criteria unchanged; original large inputs remain ignored','dd6325ad',nrow(d),length(unique(d$cell)),
  'summary split by index; failure reason classes counted; timing quantiles per cell; no refitting')),'compaction_metadata')
cat('Compacted',nrow(s),'summary rows;',nrow(d),'defect draws in',length(unique(d$cell)),'cells\n')

# Derived decision tables are reproducible from the bounded summaries alone.
file <- sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1])
leaf <- normalizePath(file.path(dirname(file),'..'))
source(file.path(leaf,'R/judge_production.R'))
j <- judge_production(s,read_input('cells'))
for (name in c('decisions','family','candidates','intervals')) write_evidence(j[[name]],name)
for (index in c('rmsea','cfi','tli','srmr')) write_evidence(
  j$losses[j$losses$index==index,c('cell','model','index','arm','N','generator','delta','target','rmse','default_rmse','availability')],
  paste0('point_losses_',index))
for (index in c('rmsea','cfi','tli')) write_evidence(
  j$interval[j$interval$index==index,c('cell','model','index','arm','N','generator','delta','target','attempted','available','availability','coverage','width','flag','below_floor')],
  paste0('interval_cells_',index))
print(j$decisions)

# A correction must preserve unaffected frozen judgments and CSV precision.
if (file.exists(file.path(input,'target_correction_metadata.csv'))) {
  for (name in c('decisions','family','candidates','intervals',
      paste0('point_losses_',c('cfi','tli','srmr')),
      paste0('interval_cells_',c('cfi','tli')))) {
    original <- read_input(name)
    path <- file.path(output,paste0(name,'.csv'))
    corrected <- read.csv(path,stringsAsFactors=FALSE)
    a <- original[original$index!='rmsea',]
    keep <- corrected$index!='rmsea'
    b <- corrected[keep,]
    keys <- intersect(c('cell','model','index','arm'),names(a))
    key <- function(x) do.call(paste,c(x[keys],sep=':'))
    a <- a[match(key(b),key(a)),]
    stopifnot(isTRUE(all.equal(a,b,tolerance=1e-12,check.attributes=FALSE)))
    if (all(keep)) stopifnot(file.copy(file.path(input,paste0(name,'.csv')),path,overwrite=TRUE)) else {
      corrected[keep,] <- a
      write_evidence(corrected,name)
    }
  }
  cat('Verified and preserved unchanged CFI, TLI, SRMR judgment rows\n')
}
