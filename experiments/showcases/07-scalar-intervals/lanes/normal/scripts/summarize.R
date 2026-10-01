#!/usr/bin/env Rscript
# Small result-only summaries; no estimation or simulation occurs here.
script <- sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1])
root <- dirname(dirname(normalizePath(script)))
if(identical(commandArgs(trailingOnly=TRUE),'--help')) {
  cat('Rscript scripts/summarize.R\nReads the four prespecified run directories and writes results/comparison/.\n')
  quit(status=0L)
}
runs <- c('main','bartlett-pilot','bartlett-large','bartlett-independent')
paths <- file.path(root,'results',runs,'intervals.csv')
if(!all(file.exists(paths))) stop('Complete main, bartlett-pilot, bartlett-large and bartlett-independent first.')
x <- setNames(lapply(paths,read.csv,stringsAsFactors=FALSE),runs)
expected <- c(main=4800L,'bartlett-pilot'=250L,'bartlett-large'=50L,'bartlett-independent'=50L)
if(any(vapply(x,nrow,integer(1))!=expected)) stop('The prespecified run sizes are not complete.')
out <- file.path(root,'results','comparison'); dir.create(out,showWarnings=FALSE)
# The Bartlett arms must reuse the exact ordinary datasets and constructions.
for(run in runs[-1]) {
  ordinary <- x[[run]][x[[run]]$method!='lr_bartlett',]
  matched <- merge(ordinary,x$main,by=c('n','replicate','target','method'),suffixes=c('_pilot','_main'))
  stopifnot(nrow(matched)==nrow(ordinary),all(matched$seed_pilot==matched$seed_main),
            all(matched$valid_pilot==matched$valid_main))
  good <- matched$valid_pilot & matched$valid_main
  stopifnot(max(abs(matched$lower_pilot[good]-matched$lower_main[good]))<1e-10,
            max(abs(matched$upper_pilot[good]-matched$upper_main[good]))<1e-10)
}
base <- x[['bartlett-pilot']]
a <- base[base$method=='lr_bartlett',]
ordinary <- base[base$method=='lr',c('replicate','width')]
names(ordinary)[2] <- 'ordinary_width'
a <- merge(a,ordinary,by='replicate')
rows <- list()
for(name in c('bartlett-large','bartlett-independent')) {
  b <- x[[name]]; b <- b[b$method=='lr_bartlett',]
  paired <- merge(a,b,by='replicate',suffixes=c('_399','_check'))
  valid <- paired$valid_399 & paired$valid_check
  ratio <- rep(NA_real_,nrow(paired))
  ratio[valid] <- pmax(abs(paired$lower_check[valid]-paired$lower_399[valid]),
                       abs(paired$upper_check[valid]-paired$upper_399[valid]))/paired$ordinary_width[valid]
  rows[[name]] <- data.frame(replicate=paired$replicate,check=name,valid=valid,
    endpoint_shift_fraction=ratio,unstable=ifelse(valid,ratio>.05,NA),
    width_399=paired$width_399,width_check=paired$width_check,
    covered_399=paired$covered_399,covered_check=paired$covered_check,
    error_399=paired$error_399,error_check=paired$error_check)
}
stability <- do.call(rbind,rows)
write.csv(stability,file.path(out,'stability.csv'),row.names=FALSE)
bartlett_pairs <- do.call(rbind,lapply(runs[-1],function(run) {
  z <- x[[run]]
  pair <- merge(z[z$method=='lr',],z[z$method=='lr_bartlett',],by='replicate')
  valid <- pair$valid.x & pair$valid.y
  d <- as.numeric(pair$covered.y[valid])-as.numeric(pair$covered.x[valid])
  data.frame(run=run,attempted=nrow(pair),common_valid=sum(valid),
    ordinary_successful_covering=sum(pair$valid.x & pair$covered.x,na.rm=TRUE)/nrow(pair),
    bartlett_successful_covering=sum(pair$valid.y & pair$covered.y,na.rm=TRUE)/nrow(pair),
    paired_coverage_difference=if(length(d)) mean(d) else NA_real_,
    paired_difference_mcse=if(length(d)>1) sd(d)/sqrt(length(d)) else NA_real_,
    paired_mean_width_difference=if(any(valid)) mean(pair$width.y[valid]-pair$width.x[valid]) else NA_real_)
}))
write.csv(bartlett_pairs,file.path(out,'bartlett_vs_lr.csv'),row.names=FALSE)
# Bootstrap-noise check at fixed B=1999, distinct from increasing B.
b <- x[['bartlett-large']]; b <- b[b$method=='lr_bartlett',]
c <- x[['bartlett-independent']]; c <- c[c$method=='lr_bartlett',]
p <- merge(b,c,by='replicate',suffixes=c('_large','_independent'))
p <- merge(p,ordinary,by='replicate'); good <- p$valid_large & p$valid_independent
ratio <- rep(NA_real_,nrow(p)); ratio[good] <- pmax(
  abs(p$lower_large[good]-p$lower_independent[good]),
  abs(p$upper_large[good]-p$upper_independent[good]))/p$ordinary_width[good]
noise <- data.frame(replicate=p$replicate,valid=good,endpoint_shift_fraction=ratio,
                    unstable=ifelse(good,ratio>.05,NA))
write.csv(noise,file.path(out,'bootstrap_noise.csv'),row.names=FALSE)
# Boundary diagnostics distinguish unrestricted improper solutions from visits
# to inadmissible primitive covariances during an otherwise valid inversion.
m <- x$main
boundary <- do.call(rbind,lapply(split(m,interaction(m$n,m$target,m$method,drop=TRUE)),function(z) {
  good <- z$valid & z$unrestricted_admissible
  data.frame(n=z$n[1],target=z$target[1],method=z$method[1],attempted=nrow(z),
    unrestricted_inadmissible=sum(!z$unrestricted_admissible),
    any_candidate_inadmissible=sum(z$candidate_inadmissible>0),
    admissible_unrestricted_valid=sum(good),
    admissible_unrestricted_coverage=if(any(good)) mean(z$covered[good]) else NA_real_)
}))
write.csv(boundary,file.path(out,'admissibility.csv'),row.names=FALSE)
# Endpoint-specific correction uncertainty, not a pool of adaptive root visits.
ends <- list()
for(name in runs[-1]) {
  ints <- x[[name]]; ints <- ints[ints$method=='lr_bartlett' & ints$valid,]
  cand <- read.csv(file.path(root,'results',name,'candidates.csv'))
  for(i in seq_len(nrow(ints))) for(side in c('lower','upper')) {
    z <- cand[cand$method=='lr_bartlett' & cand$replicate==ints$replicate[i] &
      abs(cand$candidate-ints[[side]][i])<1e-10,,drop=FALSE]
    if(!nrow(z)) stop('Missing endpoint diagnostic')
    z <- z[nrow(z),]
    ends[[length(ends)+1L]] <- data.frame(run=name,replicate=ints$replicate[i],side=side,
      correction=z$correction,mcse=z$correction_mcse,statistic_gap=abs(z$statistic-qchisq(.95,1)))
  }
}
endpoint_rows <- if(length(ends)) do.call(rbind,ends) else data.frame(
  run=character(),replicate=integer(),side=character(),correction=numeric(),mcse=numeric(),statistic_gap=numeric())
write.csv(endpoint_rows,file.path(out,'endpoint_corrections.csv'),row.names=FALSE)
print(do.call(rbind,lapply(split(stability,stability$check),function(z)
  data.frame(check=z$check[1],attempted=nrow(z),valid=sum(z$valid),unstable=sum(z$unstable,na.rm=TRUE)))))
cat('Wrote',out,'\n')

# Deliberate small evidence snapshot: raw/per-candidate data remain ignored.
frozen <- file.path(root,'results','frozen')
dir.create(frozen,showWarnings=FALSE)
copy <- function(from,to) {
  if(!file.copy(from,file.path(frozen,to),overwrite=TRUE)) stop('Evidence copy failed: ',from)
}
for(run in runs) {
  copy(file.path(root,'results',run,'summary.csv'),paste0(run,'_summary.csv'))
  copy(file.path(root,'results',run,'metadata.csv'),paste0(run,'_metadata.csv'))
  copy(file.path(root,'results',run,'failures.csv'),paste0(run,'_failures.csv'))
}
copy(file.path(root,'results','main','paired.csv'),'main_paired.csv')
copy(file.path(root,'results','main','checks','validation.csv'),'validation.csv')
for(name in c('stability','bootstrap_noise','admissibility','endpoint_corrections','bartlett_vs_lr'))
  copy(file.path(out,paste0(name,'.csv')),paste0(name,'.csv'))
cat('Updated compact evidence snapshot:',frozen,'\n')
