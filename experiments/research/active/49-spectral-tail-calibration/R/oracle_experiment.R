# Replay the saved fitted statistics; no refitting or new SEM datasets.
if('--help'%in%args) {
  cat('Usage: Rscript run_experiment.R --oracle [--smoke] [--input NAME] [--output NAME]\n',
      ' [--benchmark-reps N] [--seed-base N]\n',
      'Defaults: input mv_pilot500, output oracle_pilot500, 100000 Gaussian draws per population.\n',
      'Adds analytic population spectrum and constrained MV to expected-score/LRT results.\n',
      '--smoke: first two saved replications per cell; 5000 Gaussian draws. New output required.\n')
} else {
opt <- list(input='mv_pilot500',output='oracle_pilot500',benchmark_reps=100000L,seed_base=20260928L)
smoke <- '--smoke'%in%args;original_args <- args;args <- args[args!='--smoke']
if(smoke)opt$benchmark_reps <- 5000L
i <- 1L
while(i<=length(args)) {
  key <- c('--input'='input','--output'='output','--benchmark-reps'='benchmark_reps','--seed-base'='seed_base')[args[i]]
  if(is.na(key)||i==length(args))stop('Unknown or incomplete option')
  i <- i+1L;opt[[key]] <- if(key%in%c('input','output'))args[i] else as.integer(args[i]);i <- i+1L
}
stopifnot(all(grepl('^[a-zA-Z0-9_-]+$',c(opt$input,opt$output))),
          opt$benchmark_reps>=1000,opt$benchmark_reps<=1e7,opt$seed_base>0,opt$seed_base<1e9)
source(file.path(base,'R','spectral.R'));source(file.path(base,'R','mv.R'));source(file.path(base,'R','oracle.R'))
indir <- file.path(base,'results',opt$input);outdir <- file.path(base,'results',opt$output)
input_files <- file.path(indir,c('replicates.csv','design.csv','metadata.csv','source_hashes.csv'))
if(!all(file.exists(input_files)))stop('Run --mv with the requested input output name first')
if(dir.exists(outdir))stop('Output exists; choose a new --output')
x <- read.csv(input_files[1],na.strings='',stringsAsFactors=FALSE);x$error[is.na(x$error)] <- ''
x <- x[x$statistic_kind%in%c('score','lrt'),]
if(smoke)x <- x[x$rep<=2,]
stopifnot(all(c('mv','mv_tau2','sb','peba4','all')%in%x$method))
dir.create(outdir,recursive=TRUE)
write_out <- function(x,name)write.csv(x,file.path(outdir,name),row.names=FALSE,na='')
root <- normalizePath(file.path(base,'../../../..'))
writeLines(capture.output(sessionInfo()),file.path(outdir,'session.txt'))
writeLines(system2('git',c('-C',shQuote(root),'status','--short'),stdout=TRUE),file.path(outdir,'worktree-status.txt'))
src <- file.path(base,c('run_experiment.R','R/oracle_experiment.R','R/oracle.R','R/mv.R','R/spectral.R'))
write_out(data.frame(path=basename(src),md5=unname(tools::md5sum(src))),'source_hashes.csv')
write_out(data.frame(path=basename(input_files),md5=unname(tools::md5sum(input_files))),'input_hashes.csv')
file.copy(file.path(indir,'metadata.csv'),file.path(outdir,'input_metadata.csv'))
write_out(data.frame(key=c('args','input','benchmark_seed','benchmark_reps','magmaanlab','git','package_so_md5'),value=c(
 paste(original_args,collapse=' '),opt$input,opt$seed_base,opt$benchmark_reps,as.character(packageVersion('magmaanlab')),
 system2('git',c('-C',shQuote(root),'rev-parse','HEAD'),stdout=TRUE),
 paste(tools::md5sum(list.files(find.package('magmaanlab'),pattern='\\.so$',recursive=TRUE,full.names=TRUE)),collapse=';'))),'metadata.csv')
grid <- read.csv(input_files[2]);write_out(grid,'design.csv')
pop <- unique(grid[,c('model','p','distribution')]);oracles <- list();spectra <- list();bench <- list()
for(i in seq_len(nrow(pop))) {
  z <- pop[i,];key <- paste(z$model,z$p,z$distribution,sep='_');o <- cfa_oracle(z$model,z$p,z$distribution)
  oracles[[key]] <- o
  spectra[[i]] <- data.frame(z,eigen_index=seq_len(o$df),lambda=o$lambda,
                             tau1=o$tau1,tau2=o$tau2,projected_excess=o$projected_excess,row.names=NULL)
  critical <- uniroot(function(q)fmg_tail(q,o$lambda)-.05,c(.01,10*o$tau1),tol=1e-9)$root
  # Direct Gaussian quadratic-form generation independently checks the numerical tail.
  set.seed(opt$seed_base+i);remaining <- opt$benchmark_reps;values <- numeric(remaining);offset <- 0L
  while(remaining>0) {
    m <- min(remaining,10000L);g <- matrix(rnorm(m*o$df),m,o$df)
    values[offset+seq_len(m)] <- as.numeric(g^2%*%o$lambda)
    offset <- offset+m;remaining <- remaining-m
  }
  rate <- mean(values>critical)
  tails <- vapply(c(.5,.9,.95,.99),function(a)fmg_tail(qchisq(a,o$df),o$lambda),numeric(1))
  flat_error <- if(max(abs(o$lambda-1))<1e-9)max(abs(tails-c(.5,.1,.05,.01))) else NA_real_
  bench[[i]] <- data.frame(z,draws=length(values),critical=critical,rejection=rate,
    rejection_mcse=sqrt(rate*(1-rate)/length(values)),mean=mean(values),mean_mcse=sd(values)/sqrt(length(values)),
    oracle_mean=o$tau1,variance=var(values),oracle_variance=2*o$tau2,flat_tail_error=flat_error)
  if(abs(rate-.05)>6*sqrt(.05*.95/length(values)))stop('Gaussian benchmark rejection check failed')
  if(is.finite(flat_error)&&flat_error>1e-7)stop('Flat-spectrum numerical tail check failed')
  cat(sprintf('Oracle population %d/%d; df=%d; eigenvalue range %.10f--%.10f\n',i,nrow(pop),o$df,min(o$lambda),max(o$lambda)))
}
write_out(do.call(rbind,spectra),'oracle_spectra.csv');write_out(do.call(rbind,bench),'gaussian_benchmark.csv')
x$bound_active <- NA
new <- list()
for(i in seq_len(nrow(grid))) {
  cell <- grid[i,];o <- oracles[[paste(cell$model,cell$p,cell$distribution,sep='_')]]
  src <- x[x$cell==cell$cell & x$method=='mv_tau2',]
  ok <- !nzchar(src$error)&is.finite(src$statistic)
  stopifnot(all(src$df[ok]==o$df))
  for(method in c('mv_constrained','oracle_all','oracle_mv')) {
    z <- src;z$method <- method
    if(method=='mv_constrained') {
      z$bound_active[ok] <- z$tau2[ok]<z$tau1[ok]^2/o$df
      z$tau2[ok] <- mapply(mv_constrained_moment,z$tau1[ok],z$tau2[ok],o$df)
    } else {z$tau1[ok] <- o$tau1;z$tau2[ok] <- o$tau2}
    z$effective_df[ok] <- z$tau1[ok]^2/z$tau2[ok]
    z$moment_infeasible[ok] <- z$tau2[ok]<z$tau1[ok]^2/o$df-1e-10
    z$p_value[ok] <- if(method=='oracle_all')vapply(z$statistic[ok],fmg_tail,numeric(1),lambda=o$lambda) else
      mapply(mv_moment_tail,z$statistic[ok],z$tau1[ok],z$tau2[ok])
    if(any(!is.finite(z$p_value[ok])))stop('Invalid new tail probability')
    new[[length(new)+1L]] <- z
  }
  cat(sprintf('Oracle replay cell %d/%d complete\n',i,nrow(grid)));flush.console()
}
raw <- rbind(x,do.call(rbind,new));write_out(raw,'replicates.csv')
summary <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$statistic_kind,raw$method,drop=TRUE)),function(z) {
  ok <- !nzchar(z$error)&is.finite(z$p_value);n <- sum(ok);hits <- sum(z$p_value[ok]<.05)
  ci <- if(n)binom.test(hits,n)$conf.int else c(NA,NA)
  data.frame(z[1,c('cell','model','p','n','distribution','statistic_kind','method')],
    attempted=nrow(z),valid=n,failed=sum(!ok),rejection=if(n)hits/n else NA,lower=ci[1],upper=ci[2],
    bound_fraction=if(any(!is.na(z$bound_active[ok])))mean(z$bound_active[ok]) else NA)
}))
write_out(summary,'summary.csv')
paired <- do.call(rbind,lapply(split(raw,interaction(raw$cell,raw$statistic_kind,drop=TRUE)),function(z) {
  do.call(rbind,lapply(c('mv_tau2','sb','peba4','oracle_all'),function(reference) {
    ref <- z[z$method==reference,]
    do.call(rbind,lapply(c('mv_constrained','oracle_all','oracle_mv'),function(method) {
      a <- z[z$method==method,];b <- ref[match(a$rep,ref$rep),]
      ok <- !nzchar(a$error)&!nzchar(b$error)&is.finite(a$p_value)&is.finite(b$p_value)
      delta <- as.numeric(a$p_value[ok]<.05)-as.numeric(b$p_value[ok]<.05)
      data.frame(a[1,c('cell','model','p','n','distribution','statistic_kind','method')],
        reference=reference,paired=sum(ok),difference=mean(delta),mcse=sd(delta)/sqrt(length(delta)),
        decision_disagreement=mean((a$p_value[ok]<.05)!=(b$p_value[ok]<.05)))
    }))
  }))
}))
write_out(paired,'paired.csv')
means <- do.call(rbind,lapply(split(raw[raw$method=='mv_tau2',],interaction(raw$cell[raw$method=='mv_tau2'],raw$statistic_kind[raw$method=='mv_tau2'],drop=TRUE)),function(z) {
  o <- oracles[[paste(z$model[1],z$p[1],z$distribution[1],sep='_')]]
  z <- z[!nzchar(z$error)&is.finite(z$statistic),];n <- nrow(z)
  data.frame(z[1,c('cell','model','p','n','distribution','statistic_kind')],valid=n,
    oracle_tau1=o$tau1,oracle_tau2=o$tau2,mean_statistic=mean(z$statistic),
    mean_mcse=sd(z$statistic)/sqrt(n),mean_ratio=mean(z$statistic)/o$tau1,
    variance_ratio=var(z$statistic)/(2*o$tau2),
    fitted_tau1_ratio=mean(z$tau1)/o$tau1,
    plugin_tau2_ratio=mean(z$tau2_plugin)/o$tau2,corrected_tau2_ratio=mean(z$tau2)/o$tau2)
}))
write_out(means,'statistic_moments.csv')
# On bound-active rows, the constrained gamma must coincide with saved SB.
a <- raw[raw$method=='mv_constrained'&!is.na(raw$bound_active)&raw$bound_active,]
b <- raw[raw$method=='sb',];key <- function(z)paste(z$cell,z$rep,z$statistic_kind)
stopifnot(max(abs(a$p_value-b$p_value[match(key(a),key(b))]),na.rm=TRUE)<1e-10)
cat('Wrote oracle and constrained-MV results to ',outdir,'\n',sep='')
}
