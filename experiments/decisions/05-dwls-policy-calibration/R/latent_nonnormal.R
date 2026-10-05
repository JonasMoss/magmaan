latent_seed_base <- function(mode) c(smoke=717000001L,pilot=1917000001L,production=2017000001L)[[mode]]
latent_validate_seeds <- function() {
  old <- c(117000001,317000001,517000001,1817000001,
    817120001,817130001,817140001,817150001,817160001,
    1017000001,1217000001,1417000001,1617000001)
  fresh <- c(15000001,217000001,717000001,1917000001,2017000001)
  stopifnot(all(abs(outer(fresh,old,'-'))>=100000000),
    min(dist(fresh))>=100000000, max(fresh)+10000*200+2000<.Machine$integer.max)
}
latent_cells <- function() {
  base <- expand.grid(factors=1:2,categories=5L,skew=c('symmetric','skewed'),
    generator=c('gaussian','chisq8','chisq2'),n=c(300L,1000L,4000L),stringsAsFactors=FALSE)
  binary <- expand.grid(factors=1:2,categories=2L,skew='symmetric',
    generator=c('gaussian','chisq8','chisq2'),n=c(300L,1000L,4000L),stringsAsFactors=FALSE)
  b <- rbind(binary,base)
  b$model <- 'cfa'; b$cross <- 0; b$role <- 'null'; b$nesting <- 'none'
  b$groups <- 1L; b$parameterization <- 'delta'
  coverage <- b; coverage$family <- 'coverage'
  global <- b; global$family <- 'global'
  nested <- b; nested$family <- 'nested'; nested$groups <- 2L; nested$nesting <- 'metric'
  theta <- nested[nested$n==1000 & nested$skew=='symmetric',]
  theta$parameterization <- 'theta'; theta$nesting <- 'thresholds'
  x <- rbind(coverage,global,nested,theta)
  x$cell_id <- seq_len(nrow(x)); x$production_reps <- 2000L
  x
}
latent_key <- function(cell) paste(cell$factors,cell$categories,cell$skew,cell$generator,sep='_')
latent_draw <- function(cell,seed,n=cell$n) {
  set.seed(seed)
  k <- cell$factors; p <- 6*k
  lambda <- rep(c(.65,.7,.75,.8,.7,.75),k)
  phi <- matrix(.3,k,k); diag(phi) <- 1
  innovation <- function(n) if(cell$generator=='gaussian') rnorm(n) else {
    df <- if(cell$generator=='chisq8') 8 else 2
    (rchisq(n,df)-df)/sqrt(2*df)
  }
  cuts <- qnorm(if(cell$categories==2) .5 else if(cell$skew=='skewed') c(.45,.70,.85,.95) else c(.2,.4,.6,.8))
  do.call(rbind,lapply(seq_len(cell$groups),function(g) {
    ng <- as.integer(n/cell$groups)
    eta <- matrix(innovation(ng*k),ncol=k)%*%chol(phi)
    z <- eta[,rep(seq_len(k),each=6),drop=FALSE]*rep(lambda,each=ng) +
      matrix(innovation(ng*p),ncol=p)*rep(sqrt(1-lambda^2),each=ng)
    d <- as.data.frame(lapply(seq_len(p),function(j) as.integer(findInterval(z[,j],cuts)+1L)))
    names(d) <- paste0('x',seq_len(p))
    if(cell$groups==2) d$group <- c('a','b')[g]
    d
  }))
}
latent_sources <- function(here) {
  binary <- list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE)
  c(file.path(here,'run_experiment.R'),sort(list.files(file.path(here,'R'),full.names=TRUE)),
    file.path(here,'criteria','latent_nonnormal.md'),binary)
}
# Bind the existing same-draw comparator machinery to this lane's populations.
# Source into an isolated environment so lane A retains its original design.
latent_engine <- function(here) {
  e <- new.env(parent=environment(latent_engine))
  sys.source(file.path(here,'R','compute.R'),e)
  sys.source(file.path(here,'R','exact_first_stage.R'),e)
  e$dwls_draw_data <- latent_draw; e$dwls_population_key <- latent_key
  e$exact_cells <- latent_cells
  e
}
latent_population_stats <- function(cell,seed,n,chunk=10000L) {
  cell$groups <- 1L; p <- 6*cell$factors; c <- cell$categories
  pairs <- which(lower.tri(matrix(0,p,p)),arr.ind=TRUE)
  tables <- lapply(seq_len(nrow(pairs)),function(i) matrix(0L,c,c))
  # Chunks have independent deterministic substreams. These exact counts, not
  # averages of chunk polychorics, define the large-draw Stage-1 estimate.
  for(first in seq(1L,n,by=chunk)) {
    d <- as.matrix(latent_draw(cell,seed+as.integer((first-1L)/chunk),n=min(chunk,n-first+1L)))
    for(i in seq_len(nrow(pairs))) tables[[i]] <- tables[[i]]+
      matrix(tabulate(d[,pairs[i,1]]+c*(d[,pairs[i,2]]-1L),nbins=c*c),c,c)
  }
  stats <- magmaanlab::magmaan_core$data_ordinal_stats_from_raw(
    as.matrix(latent_draw(cell,seed,n=2000L)),full_wls_weight=FALSE)
  q <- p*(c-1)+nrow(pairs); thresholds <- numeric(p*(c-1)); W <- numeric(q)
  R <- diag(p)
  for(i in seq_len(nrow(pairs))) {
    counts <- tables[[i]]
    idx <- rep(seq_len(c*c),as.vector(counts))
    d <- cbind((idx-1L)%%c+1L,(idx-1L)%/%c+1L)
    pair <- magmaanlab::magmaan_core$data_ordinal_stats_from_raw(d,full_wls_weight=FALSE)
    a <- pairs[i,1]; b <- pairs[i,2]
    ia <- ((a-1L)*(c-1L)+1L):(a*(c-1L)); ib <- ((b-1L)*(c-1L)+1L):(b*(c-1L))
    thresholds[ia] <- pair$thresholds[[1]][seq_len(c-1L)]
    thresholds[ib] <- pair$thresholds[[1]][c:(2L*(c-1L))]
    wp <- diag(pair$W_dwls[[1]])
    W[ia] <- wp[seq_len(c-1L)]; W[ib] <- wp[c:(2L*(c-1L))]
    W[p*(c-1L)+i] <- tail(wp,1)
    R[a,b] <- R[b,a] <- pair$R[[1]][2,1]
    rm(pair,d,idx); gc(FALSE)
  }
  stats$ov_names <- list(paste0('x',seq_len(p))); stats$ordered <- stats$ov_names[[1]]
  dimnames(R) <- list(stats$ordered,stats$ordered)
  stats$R <- list(R); stats$thresholds <- list(thresholds)
  stats$moments <- list(c(thresholds,R[lower.tri(R)]))
  stats$W_dwls <- list(diag(W)); stats$NACOV <- list(diag(1/W))
  stats$W_wls <- list(matrix(numeric(),0,0)); stats$nobs <- as.integer(n)
  # Population fits consume only moments and weights. Casewise rows would be
  # meaningless after pair reconstruction; explicitly omit them from this input.
  stats$moment_influence <- list(matrix(numeric(),0,q))
  stats$int_data <- list(matrix(integer(),0,p))
  stats$moment_bread <- list(matrix(numeric(),0,0))
  stats
}
latent_population <- function(cells,out,mode) {
  full <- latent_cells(); full <- full[full$family=='coverage',]
  keys <- unique(vapply(seq_len(nrow(full)),function(i) latent_key(full[i,]),''))
  need <- unique(vapply(seq_len(nrow(cells)),function(i) latent_key(cells[i,]),''))
  n <- if(mode=='smoke') 20000L else 1000000L
  result <- list(); stage1 <- list(); start <- proc.time()
  for(key in need) for(draw in 1:2) {
    cell <- full[match(key,vapply(seq_len(nrow(full)),function(i) latent_key(full[i,]),'')),]
    seed <- c(217000001L,15000001L)[draw]+10000L*match(key,keys)
    cat('Population ',key,' draw ',draw,' N=',n,'\n',sep=''); flush.console()
    stats <- latent_population_stats(cell,seed,n)
    fit <- dwls_fit(cell,stats)
    if(!isTRUE(fit$converged)) stop('Population fit did not converge: ',key)
    z <- dwls_targets(fit); z$key <- key; z$seed <- seed; z$n_per_group <- n; z$draw <- draw
    z$population_F <- fit$fmin
    df <- length(stats$moments[[1]])-fit$npar
    z$population_df <- df; z$population_rmsea <- sqrt(max(0,fit$fmin)/df)
    result[[length(result)+1L]] <- z
    stage1[[length(stage1)+1L]] <- data.frame(key,draw,moment=seq_along(stats$moments[[1]]),
      value=stats$moments[[1]],opg_weight=diag(stats$W_dwls[[1]]))
    write_csv(do.call(rbind,result),file.path(out,'population_targets.csv'))
    write_csv(do.call(rbind,stage1),file.path(out,'population_stage1.csv'))
  }
  pop <- do.call(rbind,result)
  a <- pop[pop$draw==1,]; b <- pop[pop$draw==2,]
  delta <- merge(a,b,by=c('key','target'),suffixes=c('_first','_second'))
  delta$target_difference <- delta$estimate_second-delta$estimate_first
  write_csv(delta,file.path(out,'population_uncertainty.csv'))
  write_csv(data.frame(elapsed_seconds=unname((proc.time()-start)['elapsed']),
    cpu_seconds=unname(sum((proc.time()-start)[c('user.self','sys.self')]))),file.path(out,'population_timing.csv'))
  a
}
latent_summarize <- function(raw,cells,out,here) {
  e <- latent_engine(here); e$exact_summarize(raw,cells,out)
  path <- file.path(out,'summary.csv'); s <- read.csv(path)
  # Global tests are descriptive and Gaussian cells are development controls.
  i <- match(s$cell_id,cells$cell_id)
  policy <- startsWith(s$arm,'policy') | s$arm=='exact_all'
  s$finite_sample_followup <- policy & cells$generator[i]!='gaussian' &
    ((cells$family[i]=='coverage' & cells$n[i]>=300 & s$rate<.93) |
     (cells$family[i]=='nested' & cells$n[i]>=500 & (s$rate<.03 | s$rate>.07)))
  s$interpretation <- ifelse(cells$family[i]=='global','descriptive_global',
    ifelse(cells$generator[i]=='gaussian','development_control','pseudo_target_calibration'))
  write_csv(s,path)
}
latent_run <- function(args,mode,workers,here,opt) {
  if(!mode %in% c('smoke','pilot','production')) stop('Latent lane supports smoke, pilot, production')
  latent_validate_seeds()
  cells <- latent_cells(); family <- opt('--family','all')
  if(!family %in% c('all','global','nested','coverage')) stop('Unknown family')
  if(family!='all') cells <- cells[cells$family==family,]
  ids <- as.integer(strsplit(opt('--cell',''),',',fixed=TRUE)[[1]])
  if(length(ids)) {
    if(any(!ids %in% cells$cell_id)) stop('Unknown cell ID')
    cells <- cells[cells$cell_id %in% ids,]
  }
  reps <- as.integer(opt('--reps',if(mode=='smoke') '1' else if(mode=='pilot') '2' else NA_character_))
  if(!is.na(reps) && reps<1) stop('Invalid replicates')
  id <- opt('--run-id',paste0('latent-',mode))
  if(!grepl('^[a-zA-Z0-9_-]+$',id)) stop('Invalid run ID')
  out <- opt('--out-dir',file.path(here,'results','latent-nonnormal',id))
  if(dir.exists(out)) stop('Run exists; choose a fresh ID')
  dir.create(out,recursive=TRUE)
  binary <- list.files(file.path(find.package('magmaanlab'),'libs'),'\\.so$',full.names=TRUE)
  write_metadata(file.path(out,'metadata.csv'),list(lane='latent-nonnormal',mode=mode,workers=workers,
    seed_base=latent_seed_base(mode),selected_cells=paste(cells$cell_id,collapse=','),family=family,
    source_hashes=paste(tools::md5sum(latent_sources(here)),collapse=','),
    native_md5=paste(tools::md5sum(binary),collapse=','),reps_override=reps,
    git_head=git_scalar(c('rev-parse','HEAD')),git_dirty=git_dirty(),
    population_n_per_group=if(mode=='smoke') 20000L else 1000000L,
    population_seed_base=217000001L,population_second_seed_base=15000001L,
    population_chunk_size=10000L,n_convention='per group',
    population_method='stream pair counts; library pair Stage-1; OPG DWLS weights',
    comparator_judge='same library fits and convergence for both arms'),packages=c('magmaanlab','lavaan'))
  write_csv(cells,file.path(out,'cells.csv'))
  population <- latent_population(cells,out,mode)
  e <- latent_engine(here)
  jobs <- do.call(rbind,lapply(seq_len(nrow(cells)),function(i) data.frame(cell_id=cells$cell_id[i],
    replicate=seq_len(if(is.na(reps)) cells$production_reps[i] else reps))))
  rows <- list(); start <- proc.time()[['elapsed']]
  for(first in seq(1L,nrow(jobs),by=4L)) {
    indices <- first:min(first+3L,nrow(jobs))
    batch <- parallel::mclapply(indices,function(j) {
      set_single_threaded_math()
      cell <- cells[match(jobs$cell_id[j],cells$cell_id),]
      # Comparator code uses total N; the design explicitly stores N per group.
      cell$n <- cell$n*cell$groups
      e$exact_replicate(cell,jobs$replicate[j],latent_seed_base(mode),population)
    },mc.cores=workers,mc.preschedule=TRUE)
    if(any(vapply(batch,inherits,logical(1),'try-error'))) stop('Worker failed')
    rows <- c(rows,batch); raw <- do.call(rbind,rows)
    saveRDS(raw,file.path(out,'raw.rds'))
    elapsed <- proc.time()[['elapsed']]-start
    write_csv(data.frame(completed=max(indices),total=nrow(jobs),elapsed_seconds=elapsed),file.path(out,'progress.csv'))
    cat('Latent completed ',max(indices),'/',nrow(jobs),' in ',round(elapsed,1),'s\n',sep=''); flush.console()
  }
  latent_summarize(raw,cells,out,here)
  if(any(is.finite(raw$policy_gap) & raw$policy_gap>1e-7)) stop('Policy equivalence gate failed')
  if(any(grepl('Policy unavailable|Policy covariance unavailable|Population target missing',raw$error)))
    stop('Required policy component unavailable; retained failures')
  writeLines('complete',file.path(out,'COMPLETE'))
  cat('Latent results: ',out,'\n',sep='')
}
