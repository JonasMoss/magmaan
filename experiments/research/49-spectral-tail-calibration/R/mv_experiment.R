# Invoked by run_experiment.R --mv; uses base and args from that entry point.
if('--help'%in%args) {
  cat('Usage: Rscript run_experiment.R --mv [--metrics] [--smoke] [--reps N] [--seed-base N] [--output NAME]\n',
      'Default: 500 reps, 24 cells: one/two-factor CFA; p=8,12; n=100,200,500; normal/skewed.\n',
      'Compares SB, MV, MV tau2-only U4, MV both moments, ALL, pEBA4, independent tau2 diagnostic.\n',
      '--metrics: also compare observed-H0 score weight, with expected sensitivity held fixed.\n',
      '--smoke: 2 reps/cell. Output directory must be new.\n')
} else {
compare_metrics <- '--metrics'%in%args
args <- args[args!='--metrics']
opt <- list(reps=500L,seed_base=20260926L,output='mv_pilot500')
i <- 1L
while(i<=length(args)) {
  if(args[i]=='--smoke') opt$reps <- 2L else {
    key <- c('--reps'='reps','--seed-base'='seed_base','--output'='output')[args[i]]
    if(is.na(key)||i==length(args)) stop('Unknown or incomplete option')
    i <- i+1L;opt[[key]] <- if(key=='output') args[i] else as.integer(args[i])
  }; i <- i+1L
}
stopifnot(opt$reps>0,opt$reps<100000,opt$seed_base>0,opt$seed_base<1e9,
          grepl('^[a-zA-Z0-9_-]+$',opt$output))
source(file.path(base,'R','spectral.R'));source(file.path(base,'R','mv.R'))
suppressPackageStartupMessages(library(magmaanlab))
outdir <- file.path(base,'results',opt$output)
if(dir.exists(outdir)) stop('Output exists; choose a new --output')
dir.create(outdir,recursive=TRUE)
write_out <- function(x,name)write.csv(x,file.path(outdir,name),row.names=FALSE,na='')
root <- normalizePath(file.path(base,'../../..'))
writeLines(capture.output(sessionInfo()),file.path(outdir,'session.txt'))
writeLines(system2('git',c('-C',shQuote(root),'status','--short'),stdout=TRUE),file.path(outdir,'worktree-status.txt'))
src <- file.path(base,c('run_experiment.R','R/mv_experiment.R','R/mv.R','R/spectral.R'))
write_out(data.frame(path=basename(src),md5=unname(tools::md5sum(src))),'source_hashes.csv')
write_out(data.frame(key=c('args','seed','magmaanlab','git','package_so_md5'),value=c(
 paste(c(if(compare_metrics)'--metrics',args),collapse=' '),opt$seed_base,as.character(packageVersion('magmaanlab')),
 system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE),
 paste(tools::md5sum(list.files(find.package('magmaanlab'),pattern='\\.so$',recursive=TRUE,full.names=TRUE)),collapse=';'))),'metadata.csv')
grid <- expand.grid(model=c('one_factor','two_factor'),p=c(8L,12L),n=c(100L,200L,500L),
                    distribution=c('normal','skewed'),stringsAsFactors=FALSE)
grid$cell <- seq_len(nrow(grid));write_out(grid,'design.csv')
draw <- function(cell) {
  n <- cell$n;p <- cell$p;Z <- matrix(rnorm(n*(p+2)),n,p+2)
  if(cell$distribution=='skewed') Z <- (Z^2-1)/sqrt(2)
  if(cell$model=='one_factor') {
    loading <- seq(.45,.8,length.out=p); signal <- outer(Z[,1],loading)
  } else {
    loading <- rep(seq(.45,.8,length.out=p/2),2)
    signal <- cbind(outer(Z[,1],loading[1:(p/2)]),
                    outer(.4*Z[,1]+sqrt(.84)*Z[,2],loading[1:(p/2)]))
  }
  X <- signal+sweep(Z[,-c(1,2)],2,sqrt(1-loading^2),'*')
  colnames(X) <- paste0('x',seq_len(p));as.data.frame(X)
}
methods <- c('sb','mv','mv_tau2','mv_both','all','peba4','mv_tau2_independent')
one_mv <- function(cell,rep_id) {
  seed <- opt$seed_base+cell$cell*100000L+rep_id;set.seed(seed)
  ans <- expand.grid(method=methods,statistic_kind=c('score','lrt',if(compare_metrics)'score_observed'),stringsAsFactors=FALSE)
  cells <- cell[rep(1L,nrow(ans)),,drop=FALSE];rownames(cells) <- NULL
  ans <- cbind(cells,rep=rep_id,seed=seed,ans)
  ans$p_value <- ans$tau1 <- ans$tau2 <- ans$tau2_plugin <- ans$effective_df <- NA_real_
  ans$df <- ans$statistic <- ans$spectral_error <- ans$moment_seconds <- NA_real_
  ans$moment_infeasible <- NA;ans$error <- ''
  start <- proc.time()[['elapsed']]
  finish <- function() {ans$seconds <- proc.time()[['elapsed']]-start;ans}
  ing <- tryCatch({
    X <- draw(cell);p <- cell$p
    syntax <- if(cell$model=='one_factor') paste('f =~',paste(names(X),collapse='+')) else
      paste(paste('f1 =~',paste(names(X)[1:(p/2)],collapse='+')),
            paste('f2 =~',paste(names(X)[(p/2+1):p],collapse='+')),sep='\n')
    f <- fit_model(syntax,X,estimator='ML')
    if(!isTRUE(f$converged)||!isTRUE(f$diagnostics$admissibility$admissible)) stop('Fit not converged/admissible')
    ctx <- prepare_inference(f,X);c <- score_components(ctx,sensitivity='expected',metric='expected');s <- project_scores(c)
    W <- s$projection%*%solve(chol(s$metric))*sqrt(cell$n);Y <- c$rows%*%W
    sr <- score_spectrum(s);lr <- fmg_tests(ctx,tests='all_ml')
    t0 <- proc.time()[['elapsed']];m <- mv_cross_moments(Y);mt <- proc.time()[['elapsed']]-t0
    Xi <- draw(cell);sigma <- magmaanlab:::model_implied(f)$sigma[[1]];mu <- colMeans(X)
    if(max(abs(fixed_covariance_scores(X,mu,sigma)-c$rows))>1e-7) stop('Score transport identity failed')
    mi <- mv_cross_moments(fixed_covariance_scores(Xi,mu,sigma)%*%W)
    expected <- list(Y=Y,m=m,mi=mi,mt=mt,q=s$statistic,eigen=sr$eigenvalues)
    lrt <- list(Y=scale(Y,scale=FALSE),m=m,mi=mi,mt=mt,
                q=lr$base_statistic[1],eigen=lr$eigenvalues[[1]])
    observed <- if(compare_metrics) tryCatch({
      co <- score_components(ctx,sensitivity='expected',metric='observed')
      so <- project_scores(co)
      Wo <- so$projection%*%solve(chol(so$metric))*sqrt(cell$n)
      Yo <- co$rows%*%Wo
      if(max(abs(co$rows-c$rows))>1e-7) stop('Metric changed score rows')
      if(max(abs(tcrossprod(so$projection)-tcrossprod(s$projection)))>1e-7)
        stop('Metric changed nuisance complement')
      # Independent check of the statistic as well as its calibration spectrum.
      qo <- sum(colSums(Yo)^2)/cell$n
      if(abs(qo-so$statistic)>1e-7*max(1,abs(qo))) stop('Observed quadratic identity failed')
      t0 <- proc.time()[['elapsed']];mo <- mv_cross_moments(Yo)
      mto <- proc.time()[['elapsed']]-t0
      list(Y=Yo,m=mo,mi=mv_cross_moments(fixed_covariance_scores(Xi,mu,sigma)%*%Wo),
           mt=mto,q=so$statistic,eigen=score_spectrum(so)$eigenvalues)
    },error=function(e)e) else NULL
    list(score=expected,lrt=lrt,score_observed=observed)
  },error=function(e)e)
  if(inherits(ing,'error')) {ans$error <- conditionMessage(ing);return(finish())}
  for(kind in unique(ans$statistic_kind)) {
    ix <- which(ans$statistic_kind==kind);z <- ing[[kind]]
    if(inherits(z,'error')) {ans$error[ix] <- conditionMessage(z);next}
    Y <- z$Y
    raw <- eigen(crossprod(Y)/cell$n,symmetric=TRUE,only.values=TRUE)$values
    q <- z$q;d <- length(raw);t1 <- sum(raw);t2 <- sum(raw^2)
    ix <- which(ans$statistic_kind==kind)
    err <- max(abs(sort(raw)-sort(z$eigen)))
    ans$df[ix] <- d;ans$statistic[ix] <- q;ans$spectral_error[ix] <- err;ans$moment_seconds[ix] <- z$mt
    if(err>1e-7*max(1,max(raw))) {ans$error[ix] <- 'Spectrum identity failed';next}
    for(j in ix) {
      method <- ans$method[j]
      a <- if(method=='mv_both') z$m['tau1'] else t1
      b <- if(method%in%c('mv_tau2','mv_both')) z$m['tau2'] else
        if(method=='mv_tau2_independent') z$mi['tau2'] else t2
      ans$tau1[j] <- a;ans$tau2[j] <- b;ans$tau2_plugin[j] <- t2
      ans$effective_df[j] <- a^2/b;ans$moment_infeasible[j] <- b<a^2/d
      v <- tryCatch({
        value <- switch(method,sb=fmg_tail(q,raw,'sb'),all=fmg_tail(q,raw),peba4=fmg_tail(q,raw,'peba'),
                        mv_moment_tail(q,a,b))
        if(!is.finite(value)||value<0||value>1) stop('Invalid tail probability')
        if(method=='mv' && abs(value-fmg_tail(q,raw,'mean_var_adjusted'))>1e-10) stop('Core MV parity failed')
        value
      },error=function(e)e)
      if(inherits(v,'error')) ans$error[j] <- conditionMessage(v) else ans$p_value[j] <- v
    }
  };finish()
}
started <- Sys.time();all <- list()
for(i in seq_len(nrow(grid))) {
  rows <- vector('list',opt$reps)
  for(r in seq_len(opt$reps)) {
    rows[[r]] <- one_mv(grid[i,],r)
    if(r%%100==0) {cat(sprintf('MV cell %d/24: %d/%d\n',i,r,opt$reps));flush.console()}
  }
  all[[i]] <- do.call(rbind,rows)
  write.table(all[[i]],file.path(outdir,'replicates.csv'),sep=',',row.names=FALSE,col.names=i==1,append=i>1,na='')
  elapsed <- as.numeric(difftime(Sys.time(),started,units='secs'))
  cat(sprintf('MV cell %d/24 complete; %.1fs elapsed; ETA %.1fs\n',i,elapsed,elapsed/i*(24-i)));flush.console()
}
raw <- do.call(rbind,all)
summary <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$statistic_kind,raw$method,drop=TRUE)),function(x) {
  ok <- !nzchar(x$error)&is.finite(x$p_value);n <- sum(ok);hits <- sum(x$p_value[ok]<.05)
  ci <- if(n)binom.test(hits,n)$conf.int else c(NA,NA)
  data.frame(x[1,c('cell','model','p','n','distribution','statistic_kind','method','df')],
             d_over_n=x$df[1]/x$n[1],attempted=nrow(x),valid=n,failed=sum(!ok),
             rejection=if(n)hits/n else NA_real_,lower=ci[1],upper=ci[2],
             mean_tau2_ratio=mean(x$tau2[ok]/x$tau2_plugin[ok]),
             infeasible_fraction=mean(x$moment_infeasible[ok]))
}))
write_out(summary,'summary.csv')
paired <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$statistic_kind,drop=TRUE)),function(x) {
  do.call(rbind,lapply(c('mv','peba4'),function(reference) {
    ref <- x[x$method==reference,]
    do.call(rbind,lapply(setdiff(methods,reference),function(method) {
      z <- x[x$method==method,];idx <- match(z$rep,ref$rep)
      ok <- !nzchar(z$error)&!nzchar(ref$error[idx])&is.finite(z$p_value)&is.finite(ref$p_value[idx])
      delta <- as.numeric(z$p_value[ok]<.05)-as.numeric(ref$p_value[idx][ok]<.05)
      data.frame(z[1,c('cell','model','p','n','distribution','statistic_kind','method')],reference=reference,
                 paired=sum(ok),difference=mean(delta),mcse=if(length(delta)>1)sd(delta)/sqrt(length(delta)) else NA_real_)
    }))
  }))
}))
write_out(paired,'paired.csv')
if(compare_metrics) {
  metric_pairs <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$method,drop=TRUE)),function(x) {
    e <- x[x$statistic_kind=='score',];o <- x[x$statistic_kind=='score_observed',]
    o <- o[match(e$rep,o$rep),]
    ok <- !nzchar(e$error)&!nzchar(o$error)&is.finite(e$p_value)&is.finite(o$p_value)
    delta <- as.numeric(o$p_value[ok]<.05)-as.numeric(e$p_value[ok]<.05)
    data.frame(e[1,c('cell','model','p','n','distribution','method')],paired=sum(ok),
      expected_failed=sum(nzchar(e$error)),observed_failed=sum(nzchar(o$error)),
      expected=mean(e$p_value[ok]<.05),observed=mean(o$p_value[ok]<.05),
      difference=mean(delta),mcse=if(length(delta)>1)sd(delta)/sqrt(length(delta)) else NA_real_)
  }))
  write_out(metric_pairs,'metric_pairs.csv')
}
cat('Wrote MV study results to ',outdir,'\n',sep='')
}
