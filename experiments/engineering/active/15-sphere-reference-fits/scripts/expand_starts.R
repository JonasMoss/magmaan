#!/usr/bin/env Rscript
script <- normalizePath(sub('^--file=', '', grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here,'..', '..', '..', '_support','R','helpers.R')); set_single_threaded_math()
for(f in c('designs.R','fit.R','start_design.R','expanded_designs.R')) source(file.path(here,'R',f))
args <- commandArgs(TRUE)
if('--help' %in% args) {
 cat('Usage: Rscript scripts/expand_starts.R [--smoke] [--run-id NAME]\nThree new model families, N=30/100, 20 draws per cell (smoke: one).\nCanonical, positive spectral control, every negative subset; ML only, both backends.\nWrites paired coverage, references, raw fits and starts; never overwrites a completed run.\n');quit(save='no')
}
smoke <- '--smoke' %in% args
run <- if('--run-id' %in% args) args[match('--run-id',args)+1L] else if(smoke) 'expanded-smoke' else 'expanded-families'
stopifnot(length(run)==1, !is.na(run), grepl('^[a-zA-Z0-9_-]+$',run))
out <- file.path(here,'results',run); if(file.exists(file.path(out,'fits.csv'))) stop('choose a fresh --run-id')
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_out <- function(x,name) write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
designs <- expanded_designs(); reps <- if(smoke) 1L else 20L
seed_base <- if(smoke) 1110000000L else 1100000000L
rows <- starts <- parameters <- list(); t0 <- proc.time()[['elapsed']]; task_id <- 0L
for(j in seq_along(designs)) for(n in c(30L,100L)) for(rep in seq_len(reps)) {
 d <- designs[[j]]; seed <- seed_base+j*100000L+n*100L+rep
 set.seed(seed); data <- as.data.frame(matrix(rnorm(n*ncol(d$sigma)),n) %*% chol(d$sigma)); names(data) <- colnames(d$sigma)
 sample <- sample_moments(data); spec <- magmaanlab::model_spec(d$syntax)
 recipes <- expanded_recipes(spec,sample)
 for(name in names(recipes)) {
  theta <- recipes[[name]]
  if(!is.null(theta)) {
   ev <- magmaanlab::magmaan_core$evaluate_at(spec$partable,sample,theta,estimator='ML')
   stopifnot(is.finite(ev$fmin),ev$diagnostics$admissibility$implied_sigma_pd)
   starts[[length(starts)+1L]] <- data.frame(design=names(designs)[j],n=n,rep=rep,seed=seed,start_id=name,parameter=seq_along(theta),value=theta)
  }
  for(backend in c('nlopt-lbfgs','port')) {
   result <- run_fit(spec,data,sample,'ML','sphere',backend,name,theta,
     assessment=list(marker_model=expanded_marker(d),extent=expanded_extent))
   id <- length(rows)+1L
   rows[[id]] <- cbind(data.frame(fit_id=id,design=names(designs)[j],n=n,rep=rep,seed=seed,
     transform='native',domain='ML',route='sphere',backend=backend,start_id=name), result$record)
   if(!is.null(result$partable)) parameters[[length(parameters)+1L]] <- cbind(fit_id=id,result$partable[c('lhs','op','rhs','est')])
  }
 }
 task_id <- task_id+1L
 cat(sprintf('[%d/%d] %s N=%d rep=%d; %.1fs\n',task_id,length(designs)*2*reps,names(designs)[j],n,rep,proc.time()[['elapsed']]-t0))
}
fits <- do.call(rbind,rows); refs <- reference_rows(fits); cmp <- compare_rows(fits,refs)
coverage <- list()
for(group in split(cmp,interaction(cmp$design,cmp$n,cmp$rep,drop=TRUE))) {
 single <- grep('^negative_.$',group$start_id,value=TRUE)
 portfolios <- list(canonical='canonical',single_negative=c('canonical',single),
   all_signed=setdiff(unique(group$start_id),'spectral_positive'),positive_control=unique(group$start_id))
 for(backend in c('both','port','nlopt-lbfgs')) for(p in names(portfolios)) {
  z <- group[group$start_id %in% portfolios[[p]] & (backend=='both' | group$backend==backend),]
  coverage[[length(coverage)+1L]] <- data.frame(design=z$design[1],n=z$n[1],rep=z$rep[1],seed=z$seed[1],backend=backend,portfolio=p,
   attempts=nrow(z),has_candidate=any(z$screened),matches_best_observed=any(z$comparison=='matches_sphere_reference'),seconds=sum(z$seconds))
 }
}
coverage <- do.call(rbind,coverage)
summary <- aggregate(coverage[c('attempts','has_candidate','matches_best_observed','seconds')],coverage[c('design','backend','portfolio')],sum)
write_out(fits,'fits');write_out(do.call(rbind,starts),'starts');write_out(do.call(rbind,parameters),'parameters')
write_out(refs,'references');write_out(cmp,'comparisons');write_out(coverage,'coverage');write_out(summary,'summary')
write_out(aggregate(list(fits=cmp$fit_id),cmp[c('design','start_id','backend','label')],length),'labels')
ref <- magmaan_cache_ref()
write_metadata(file.path(out,'metadata.csv'),values=list(tasks=task_id,reps=reps,seed_base=seed_base,
 seed_rule='seed_base + family_index*100000 + N*100 + rep',strength=.5,
 families=names(designs),starts='canonical; positive spectral; all nonempty negative subsets (at most three factors)',
 screen='unchanged thresholds; generalized strongest-marker and standardized extent',git_head=ref$git_head,git_dirty=ref$git_dirty,
 magmaanlab_built=utils::packageDescription('magmaanlab')$Built),packages='magmaanlab')
cat('Wrote ',out,'\n',sep='')
