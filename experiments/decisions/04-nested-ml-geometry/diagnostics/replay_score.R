#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if ('--help' %in% args) {
  cat('Replay production null cells at N=100: --reps 200 --run-id ID\nTwo workers; <=280 seconds; --reps 2 is the smoke path. Requires installed magmaan/magmaanlab/lavaan.\n')
  quit(save='no')
}
opt <- function(k,d) { i <- match(k,args); if(is.na(i)) d else args[i+1] }
reps <- as.integer(opt('--reps','200')); stopifnot(reps>=2,reps<=200)
script <- normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1]))
here <- dirname(dirname(script))
source(file.path(here,'R','compute.R'))
source(file.path(here,'..','..','_support','R','helpers.R'))
set_single_threaded_math()
out <- file.path(here,'results','score-diagnostic',opt('--run-id','replay'))
if(dir.exists(out)) stop('Choose a fresh run-id')
dir.create(file.path(out,'raw'),recursive=TRUE)
started <- proc.time()[['elapsed']]
cells <- read.csv(file.path(here,'results','nested-geometry','production-2026-10-03','cells.csv'))
cells <- subset(cells,n==100 & role=='null' & larger %in% c('correct','mild'))
meta <- read.csv(file.path(here,'results','nested-geometry','production-2026-10-03','metadata.csv'))
base <- as.integer(meta$value[meta$key=='seed_base'])
syntax <- 'f1 =~ x1+x2+x3+x4\nf2 =~ x5+x6+x7+x8'
make_data <- function(cell,seed,pop=FALSE) {
  set.seed(seed)
  do.call(rbind,lapply(c('a','b'),function(g) {
    x <- lavaan::simulateData(population_syntax(cell,g),sample.nobs=if(pop) 1000 else cell$n,
      empirical=pop,skewness=if(!pop && cell$distribution=='skewed') rep(2,8) else NULL,
      kurtosis=if(!pop && cell$distribution=='skewed') rep(7,8) else NULL)
    x$group <- g; x
  }))
}
components <- function(d) {
  fit <- function(eq=NULL) magmaan::as_lab_fit(magmaan::magmaan(
    magmaan::magmaan_model(syntax,prototype=d,group='group',group.equal=eq),d,estimator='ML'))
  f0 <- fit('loadings'); f1 <- fit()
  if(!isTRUE(f0$converged)||!isTRUE(f1$converged)) stop('library convergence verdict failed')
  shared <- magmaanlab::prepare_inference_data(f1)
  i0 <- magmaanlab::prepare_inference(f0,shared); i1 <- magmaanlab::prepare_inference(f1,shared)
  list(c=magmaanlab::score_components(i0,H1=f1),hyp=magmaanlab::prepare_hypothesis(i0,i1))
}
# Parameter coordinates are identical across these fits. Per-case population
# observed sensitivity comes from exact population moments, not a random pilot.
pop <- lapply(seq_len(nrow(cells)),function(i) {
  z <- components(make_data(cells[i,],1L,TRUE))$c
  list(H=z$sensitivity/2000,M=z$metric/2000,K=z$nuisance,D=z$directions)
})
project <- function(c,H) {
  K <- c$nuisance; D <- c$directions
  G <- D-K%*%solve(crossprod(K,H%*%K),crossprod(K,H%*%D))
  list(G=G,u=as.numeric(crossprod(G,c$score))/sqrt(nrow(c$rows)),
    V=crossprod(c$rows%*%G)/nrow(c$rows),
    M=crossprod(G,c$metric%*%G)/nrow(c$rows))
}
one <- function(i,r) tryCatch({
  seed <- base+10000L*cells$cell_id[i]+r
  z <- components(make_data(cells[i,],seed)); c <- z$c
  stopifnot(max(abs(c$nuisance-pop[[i]]$K))<1e-12,max(abs(c$directions-pop[[i]]$D))<1e-12)
  a <- list(observed=project(c,c$sensitivity),population=project(c,pop[[i]]$H),expected=project(c,c$metric))
  a$population_geometry <- a$population
  a$population_geometry$M <- crossprod(a$population$G,pop[[i]]$M%*%a$population$G)
  for(name in names(a)) {
    x <- a[[name]]; L <- chol(x$M); W <- solve(t(L)); meat <- W%*%x$V%*%t(W)
    x$stat <- as.numeric(crossprod(x$u,solve(x$M,x$u)))
    x$spectrum <- eigen(meat,symmetric=TRUE,only.values=TRUE)$values
    if(name %in% c('observed','expected')) {
      q <- magmaanlab::inference_quadratic(z$hyp,'score',name)
      if(abs(q$statistic-x$stat)>1e-7) stop('policy statistic mismatch')
    }
    x$Hgap <- norm(c$sensitivity/nrow(c$rows)-pop[[i]]$H,'F')/norm(pop[[i]]$H,'F')
    a[[name]] <- x
  }
  list(cell=i,rep=r,seed=seed,arms=a,error='')
},error=function(e) list(cell=i,rep=r,error=conditionMessage(e)))
all <- list()
for(batch in split(seq_len(reps),ceiling(seq_len(reps)/10))) {
  if(proc.time()[['elapsed']]-started>260) break
  jobs <- expand.grid(i=seq_len(nrow(cells)),r=batch)
  v <- parallel::mclapply(seq_len(nrow(jobs)),function(j) one(jobs$i[j],jobs$r[j]),mc.cores=2,mc.preschedule=TRUE)
  all <- c(all,v)
  saveRDS(v,file.path(out,'raw',paste0('batch-',batch[1],'.rds')))
  cat(length(all),'draws,',round(proc.time()[['elapsed']]-started,1),'seconds\n');flush.console()
}
fail <- Filter(function(x) nzchar(x$error),all)
write_csv(if(length(fail)) do.call(rbind,lapply(fail,function(x) data.frame(cell=x$cell,rep=x$rep,error=x$error))) else data.frame(cell=integer(),rep=integer(),error=character()),file.path(out,'failures.csv'))
summary <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) do.call(rbind,lapply(c('observed','population','population_geometry','expected'),function(arm) {
  v <- lapply(Filter(function(x) x$cell==i && !nzchar(x$error),all),function(x)x$arms[[arm]])
  U <- do.call(rbind,lapply(v,`[[`,'u')); V <- Reduce(`+`,lapply(v,`[[`,'V'))/length(v)
  C <- cov(U); eig <- function(x) eigen(x,symmetric=TRUE,only.values=TRUE)$values
  s <- vapply(v,`[[`,numeric(1),'stat'); spectra <- do.call(rbind,lapply(v,`[[`,'spectrum'))
  # Mixture across fitted reference laws: include variation of conditional means.
  refmean <- rowSums(spectra); refvar <- 2*rowSums(spectra^2)
  data.frame(cell_id=cells$cell_id[i],larger=cells$larger[i],distribution=cells$distribution[i],arm=arm,draws=length(v),
    score_mean_norm=sqrt(sum(colMeans(U)^2)),mc_trace=sum(diag(C)),meat_trace=sum(diag(V)),trace_ratio=sum(diag(V))/sum(diag(C)),
    statistic_mean=mean(s),statistic_variance=var(s),reference_mean=mean(refmean),reference_variance=mean(refvar)+var(refmean),
    sensitivity_relative_gap=mean(vapply(v,`[[`,numeric(1),'Hgap')),
    t(setNames(eig(C),paste0('mc_eigen_',1:6))),t(setNames(eig(V),paste0('meat_eigen_',1:6))))
}))))
write_csv(summary,file.path(out,'summary.csv'))
write_metadata(file.path(out,'metadata.csv'),list(seed_base=base,requested_reps=reps,workers=2,elapsed_s=proc.time()[['elapsed']]-started,
  git_head=git_scalar(c('rev-parse','HEAD')),git_dirty=git_dirty(),script_md5=unname(tools::md5sum(script)),
  native_md5=unname(tools::md5sum(list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE))),
  population_n_per_group=1000,population_moments='exact empirical population covariance; ML Hessian depends only on first two moments'),packages=c('magmaanlab','magmaan','lavaan'))
cat('Results:',out,'\n')
if(length(fail)) stop('Failures saved; inspect before interpreting')
