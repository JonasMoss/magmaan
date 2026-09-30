#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
if('--oracle'%in%args) {
  args <- args[args!='--oracle']
  base <- dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])))
  source(file.path(base,'R','oracle_experiment.R'))
  quit(status=0)
}
if('--mv'%in%args) {
  args <- args[args!='--mv']
  base <- dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])))
  source(file.path(base,'R','mv_experiment.R'))
  quit(status=0)
}
if('--help'%in%args) {
  cat('Usage: Rscript run_experiment.R [--smoke] [--reps N] [--seed-base N] [--output NAME]\n',
      'Use --oracle to replay saved MV results with population calibration and constrained MV.\n',
      'Use --mv for the focused debiased-MV study (--mv --help for its options).\n',
      'Default: 200 reps, 12 cells: p=8,12; n=100,400; normal, skewed, heterogeneous.\n',
      'Null one-factor CFA; output is a new results/ subdirectory. --smoke: 2 reps.\n')
  quit()
}
opt <- list(reps=200L,seed_base=20260925L,output='pilot')
i <- 1L
while(i<=length(args)) {
  if(args[i]=='--smoke') opt$reps <- 2L else {
    key <- c('--reps'='reps','--seed-base'='seed_base','--output'='output')[args[i]]
    if(is.na(key)||i==length(args)) stop('Unknown or incomplete option')
    i <- i+1L; opt[[key]] <- if(key=='output') args[i] else as.integer(args[i])
  }; i <- i+1L
}
stopifnot(opt$reps>0,opt$reps<100000,opt$seed_base>0,opt$seed_base<1e9,
          grepl('^[a-zA-Z0-9_-]+$',opt$output))
base <- dirname(normalizePath(sub('^--file=','',grep('^--file=',commandArgs(FALSE),value=TRUE)[1])))
source(file.path(base,'R','spectral.R'))
suppressPackageStartupMessages(library(magmaanlab))
outdir <- file.path(base,'results',opt$output)
if(dir.exists(outdir)) stop('Output exists; choose a new --output')
dir.create(outdir,recursive=TRUE)
write_out <- function(x,name)write.csv(x,file.path(outdir,name),row.names=FALSE,na='')
writeLines(capture.output(sessionInfo()),file.path(outdir,'session.txt'))
root <- normalizePath(file.path(base,'../../../..'))
writeLines(system2('git',c('-C',shQuote(root),'status','--short'),stdout=TRUE),file.path(outdir,'worktree-status.txt'))
metadata <- data.frame(key=c('args','seed','magmaanlab','git','source_md5','package_so_md5'),
 value=c(paste(args,collapse=' '),opt$seed_base,as.character(packageVersion('magmaanlab')),
 system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE),
 paste(tools::md5sum(c(file.path(base,'R/spectral.R'),file.path(base,'run_experiment.R'))),collapse=';'),
 paste(tools::md5sum(list.files(find.package('magmaanlab'),pattern='\\.so$',recursive=TRUE,full.names=TRUE)),collapse=';')))
write_out(metadata,'metadata.csv')
grid <- expand.grid(p=c(8L,12L),n=c(100L,400L),distribution=c('normal','skewed','heterogeneous'),stringsAsFactors=FALSE)
grid$cell <- seq_len(nrow(grid)); write_out(grid,'design.csv')
draw <- function(n,p,distribution) {
  Z <- matrix(rnorm(n*(p+1)),n,p+1)
  if(distribution=='skewed') Z <- (Z^2-1)/sqrt(2)
  if(distribution=='heterogeneous') {
    j <- seq(1,p+1,by=2); Z[,j] <- (Z[,j]^2-1)/sqrt(2)
    Z[,2] <- (exp(.7*Z[,2])-exp(.7^2/2))/sqrt((exp(.7^2)-1)*exp(.7^2))
  }
  loading <- seq(.45,.8,length.out=p)
  X <- outer(Z[,1],loading)+sweep(Z[,-1],2,sqrt(1-loading^2),'*')
  colnames(X) <- paste0('x',seq_len(p)); as.data.frame(X)
}
methods <- c('peba4','all','sb','mv','ans','raw_saddle','mom4','mom6','cgf4','cgf6','mom4_independent','cgf4_independent')
one <- function(cell,rep) {
  seed <- opt$seed_base+cell$cell*100000L+rep; set.seed(seed)
  ans <- expand.grid(method=methods,statistic_kind=c('score','lrt'),stringsAsFactors=FALSE)
  cell_rep <- cell[rep(1L,nrow(ans)),,drop=FALSE]; rownames(cell_rep) <- NULL
  ans <- cbind(cell_rep,rep=rep,seed=seed,ans)
  ans$p_value <- NA_real_; ans$reject <- NA; ans$error <- ''; ans$below_mean_bound <- FALSE
  ans$df <- NA_integer_; ans$statistic <- ans$moment_residual <- ans$spectral_error <- NA_real_
  start <- proc.time()[['elapsed']]
  finish <- function() {ans$seconds <- proc.time()[['elapsed']]-start;ans}
  ingredients <- tryCatch({
    X <- draw(cell$n,cell$p,cell$distribution)
    f <- fit_model(paste('f =~',paste(names(X),collapse='+')),X,estimator='ML')
    if(!isTRUE(f$converged)||!isTRUE(f$diagnostics$admissibility$admissible)) stop('Fit not converged/admissible')
    ctx <- prepare_inference(f,X); c <- score_components(ctx); s <- project_scores(c)
    W <- s$projection%*%solve(chol(s$metric))*sqrt(cell$n)
    Y <- c$rows%*%W
    lr <- fmg_tests(ctx,tests='all_ml'); sr <- score_spectrum(s)
    # Extra, independent calibration observations: diagnostic, not equal-cost method.
    Xi <- draw(cell$n,cell$p,cell$distribution)
    sigma <- magmaanlab:::model_implied(f)$sigma[[1]]
    mu <- colMeans(X)
    check <- fixed_covariance_scores(X,mu,sigma)
    if(max(abs(check-c$rows))>1e-7) stop('Independent-score transport identity failed')
    Yi <- fixed_covariance_scores(Xi,mu,sigma)%*%W
    list(Y=Y,Yi=Yi,q=c(score=s$statistic,lrt=lr$base_statistic[1]),
         eigen=list(score=sr$eigenvalues,lrt=lr$eigenvalues[[1]]))
  },error=function(e)e)
  if(inherits(ingredients,'error')) {ans$error <- conditionMessage(ingredients);return(finish())}
  z <- ingredients; moments <- cycle_moments(z$Y); mi <- cycle_moments(z$Yi)
  for(kind in c('score','lrt')) {
    Y <- if(kind=='lrt') scale(z$Y,scale=FALSE) else z$Y
    raw <- eigen(crossprod(Y)/cell$n,symmetric=TRUE,only.values=TRUE)$values
    q <- z$q[[kind]]; d <- length(raw)
    err <- max(abs(sort(raw)-sort(z$eigen[[kind]])))
    ix <- which(ans$statistic_kind==kind); ans$df[ix] <- d;ans$statistic[ix] <- q;ans$spectral_error[ix] <- err
    if(err>1e-7*max(1,max(raw))) {ans$error[ix] <- 'Spectrum identity failed';next}
    for(j in ix) {
      method <- ans$method[j]
      result <- tryCatch({
        residual <- NA_real_; bound <- FALSE
        p <- switch(method,
          peba4=fmg_tail(q,raw,'peba'), all=fmg_tail(q,raw), sb=fmg_tail(q,raw,'sb'),
          mv=fmg_tail(q,raw,'mean_var_adjusted'), ans=fmg_tail(q,nonlinear_weights(raw,cell$n)),
          raw_saddle=spectral_saddle(q,raw), {
            m <- if(grepl('independent',method)) mi else moments
            K <- if(grepl('6',method)) 6L else 4L
            if(startsWith(method,'mom')) {
              w <- moment_weights(m[1:K],raw);residual <- w$residual;fmg_tail(q,w$weights)
            } else {
              value <- cycle_cgf_tail(q,m[1:K]);bound <- is.na(value);value
            }
          })
        if(!bound && (!is.finite(p)||p<0||p>1)) stop('Invalid tail probability')
        list(p=p,reject=if(bound) FALSE else p<.05,residual=residual,bound=bound)
      },error=function(e)e)
      if(inherits(result,'error')) ans$error[j] <- conditionMessage(result) else {
        ans$p_value[j] <- result$p;ans$reject[j] <- result$reject
        ans$moment_residual[j] <- result$residual;ans$below_mean_bound[j] <- result$bound
      }
    }
  };finish()
}
started <- Sys.time(); all <- list()
for(i in seq_len(nrow(grid))) {
  cellrows <- list()
  for(r in seq_len(opt$reps)) {
    cellrows[[r]] <- one(grid[i,],r)
    if(r%%25==0) {cat(sprintf('Cell %d/12, rep %d/%d\n',i,r,opt$reps));flush.console()}
  }
  all[[i]] <- do.call(rbind,cellrows)
  write.table(all[[i]],file.path(outdir,'replicates.csv'),sep=',',row.names=FALSE,
              col.names=i==1,append=i>1,na='')
  cat(sprintf('Cell %d/12 complete; elapsed %.1fs\n',i,as.numeric(difftime(Sys.time(),started,units='secs'))));flush.console()
}
raw <- do.call(rbind,all)
summary <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$statistic_kind,raw$method,drop=TRUE)),function(x) {
  ok <- !nzchar(x$error)&!is.na(x$reject); n <- sum(ok); hits <- sum(x$reject[ok])
  ci <- if(n) binom.test(hits,n)$conf.int else c(NA,NA)
  data.frame(x[1,c('cell','p','n','distribution','statistic_kind','method','df')],
    d_over_n=x$df[1]/x$n[1],attempted=nrow(x),valid=n,failed=sum(!ok),
    rejection=if(n)hits/n else NA_real_,lower=ci[1],upper=ci[2],
    below_mean_bounds=sum(x$below_mean_bound),median_moment_residual=median(x$moment_residual,na.rm=TRUE))
}))
write_out(summary,'summary.csv')
# Paired contrasts include only jointly valid decisions; failures remain above.
paired <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$statistic_kind,drop=TRUE)),function(x) {
  ref <- x[x$method=='peba4',]
  do.call(rbind,lapply(setdiff(methods,'peba4'),function(m) {
    z <- x[x$method==m,]; idx <- match(z$rep,ref$rep)
    ok <- !nzchar(z$error)&!nzchar(ref$error[idx])&!is.na(z$reject)&!is.na(ref$reject[idx])
    delta <- as.numeric(z$reject[ok])-as.numeric(ref$reject[idx][ok])
    data.frame(z[1,c('cell','statistic_kind','method')],paired=sum(ok),
      difference=mean(delta),mcse=if(length(delta)>1)sd(delta)/sqrt(length(delta)) else NA_real_)
  }))
}))
write_out(paired,'paired.csv')
cat('Wrote ',outdir,'/{metadata,design,replicates,summary,paired}.csv\n',sep='')
