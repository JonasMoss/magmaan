#!/usr/bin/env Rscript
# Independent Demo 9.1 data-reader gate. Synthetic data and Demo outputs stay
# in the cache; only derived comparisons are frozen. No translator imports.
suppressPackageStartupMessages(library(magmaanlab))
suppressPackageStartupMessages(library(jsonlite))
Sys.setenv(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
root <- normalizePath('.')
scratch <- path.expand('~/.cache/magmaan-logs/task-55-data')
dir.create(scratch,recursive=TRUE,showWarnings=FALSE)
mpdemo <- '/home/jonas/mplusdemo/mpdemo'
x <- rbind(c(1,2,3),c(2,5,4),c(3,4,7),c(4,3,6),c(5,8,9))
colnames(x) <- paste0('y',1:3)
raw <- apply(x,1,paste,collapse=' ')
S <- crossprod(scale(x,scale=FALSE))/nrow(x)
mu <- colMeans(x)
tri_lines <- function(S) vapply(1:3,function(i) paste(S[i,1:i],collapse=' '),'')
full_lines <- function(S) apply(S,1,paste,collapse=' ')
cases <- list()
add <- function(id,lines,data='',variable='',groups=NULL) {
  cases[[id]] <<- list(lines=lines,data=data,variable=variable,groups=groups)
}
add('free',raw)
add('free_extra',paste(raw,99))
add('free_wrapped',unlist(lapply(1:5,function(i) c(paste(x[i,1:2],collapse=','),x[i,3]))))
add('free_nobservations',c(raw,'99 99 99'),data='NOBSERVATIONS=5;')
add('fixed_repeat',apply(x*10,1,function(r) paste(sprintf('%02d',r),collapse='x')),data='FORMAT=2(F2.1,1X),F2.1;')
add('fixed_tab',apply(x*10,1,function(r) paste0(sprintf('%02d',r[1]),'xx',paste(sprintf('%02d',r[2:3]),collapse=''))),data='FORMAT=F2.1,T5,2F2.1;')
add('fixed_records',unlist(lapply(1:5,function(i) c(sprintf('%02d',10*x[i,1]),paste(sprintf('%02d',10*x[i,2:3]),collapse='')))),data='FORMAT=(F2.1,/,2F2.1);')
add('fixed_explicit',apply(x,1,function(r) paste(sprintf('%3.1f',r),collapse='')),data='FORMAT=3F3.1;')
for (sym in c('*','.')) add(paste0('missing_',if(sym=='*')'star' else 'dot'),c(paste(rep(sym,3),collapse=' '),raw),variable=paste0('MISSING=',sym,';'))
add('missing_blank',c('      ',apply(x,1,function(r) paste(sprintf('%02d',r),collapse=''))),data='FORMAT=3F2.0;',variable='MISSING=BLANK;')
add('missing_numeric',c('-9 -9.0 -9',raw),variable='MISSING=ALL (-9);')
add('missing_scaled',c('999999',apply(x*10,1,function(r) paste(sprintf('%02d',r),collapse=''))),data='FORMAT=3F2.1;',variable='MISSING=ALL (9.9);')
add('missing_range',c('-7 -6 -5',raw),variable='MISSING=y1-y3 (-9--5);')
for(full in c(FALSE,TRUE)) {
  matrix_lines <- if(full) full_lines else tri_lines
  for(means in c(FALSE,TRUE)) add(paste0('cov_',full,'_',means),c(if(means)paste(mu,collapse=' '),matrix_lines(S)),
    data=paste0('TYPE=',if(full)'FULLCOV' else 'COVA',if(means)' MEANS' else '', '; NOBSERVATIONS=5;'))
  for(sd in c(FALSE,TRUE)) add(paste0('corr_',full,'_',sd),c(if(sd)paste(sqrt(diag(S)),collapse=' '),matrix_lines(cov2cor(S))),
    data=paste0('TYPE=',if(full)'FULLCORR' else 'CORR',if(sd)' STD' else '', '; NOBSERVATIONS=5;'))
}
add('summary_groups',rep(c(paste(mu,collapse=' '),tri_lines(S)),2),data='TYPE=COVA MEANS; NGROUPS=2; NOBSERVATIONS=5 5;')
add('file_groups',list(raw,raw),data='FILE (b) = first.dat; FILE (a) = second.dat;',groups=c('b','a'))
add('group_codes',c(paste(raw,1),paste(raw,2),'99 99 99 9'),variable='GROUPING=g(1=a 2=b);')
number <- '-?[0-9]+(?:\\.[0-9]+)?(?:[EeDd][+-]?[0-9]+)?'
nums <- function(line) as.numeric(gsub('[dD]','e',regmatches(line,gregexpr(number,line,perl=TRUE))[[1]]))
results <- list()
for(id in names(cases)) {
  c <- cases[[id]];dir <- file.path(scratch,id);dir.create(dir,showWarnings=FALSE)
  if(is.list(c$lines)) {writeLines(c$lines[[1]],file.path(dir,'first.dat'));writeLines(c$lines[[2]],file.path(dir,'second.dat'))}
  else writeLines(c$lines,file.path(dir,'data.dat'))
  input <- c(paste0('DATA: ',if(is.null(c$groups))'FILE=data.dat;' else '',gsub(';',';\n',c$data,fixed=TRUE)),
    paste0('VARIABLE: NAMES=y1 y2 y3',if(id=='group_codes')' g' else '', ';'),c$variable,
    'MODEL: y1 y2 y3; y1 WITH y2 y3; y2 WITH y3;', 'OUTPUT: SAMPSTAT;')
  # For BASIC summaries Mplus omits means if not supplied. Reader constructs
  # the same mean-free model; BASIC itself needs no MODEL statement.
  demo_input <- if(grepl('TYPE=(COVA|CORR|FULL)',c$data)) input else c(input[!grepl('^MODEL:',input)],'ANALYSIS: TYPE=BASIC;')
  writeLines(demo_input,file.path(dir,'probe.inp'))
  old <- getwd();setwd(dir)
  status <- system2(mpdemo,c('probe.inp','probe.out'),stdout='console.log',stderr='console.err',timeout=30)
  setwd(old)
  output <- readLines(file.path(dir,'probe.out'),warn=FALSE)
  if(status!=0 || any(grepl('*** ERROR',output,fixed=TRUE))) stop('Demo error: ',id)
  spec <- mplus_model(paste(input,collapse='\n'))
  spec$mplus_input_dir <- dir
  data <- mplus_data(spec)
  if(is.data.frame(data)) {
    group <- if(nzchar(spec$group_var)) data[[spec$group_var]] else rep('single',nrow(data))
    labels <- if(nzchar(spec$group_var)) spec$group_labels else 'single'
    moments <- lapply(labels,function(g) {
      d <- as.matrix(data[group==g,c('y1','y2','y3')]);d <- d[complete.cases(d),,drop=FALSE]
      # Explicitly reproduce DA05 for all-missing rows in these synthetic
      # cases; reader intentionally retains those rows and reports the policy.
      list(S=crossprod(scale(d,scale=FALSE))/nrow(d),mean=colMeans(d),n=nrow(d))
    })
  } else moments <- lapply(seq_along(data$S),function(g) list(S=attr(data,"mplus_data_report")$input_covariance[[g]],mean=if(spec$requested_meanstructure)data$mean[[g]] else NULL,n=data$nobs[g]))
  nlines <- grep('Number of observations',output)
  if(length(moments)==1) ns <- tail(nums(output[nlines[1]]),1) else {
    ns <- vapply(seq_along(moments),function(g) tail(nums(output[nlines[1]+g]),1),0.0)
  }
  cov_heads <- grep('^           Covariances(/Correlations/Residual Correlations)?\\s*$',output)
  mean_heads <- grep('^           Means(/Intercepts/Thresholds)?\\s*$',output)
  if(length(cov_heads)!=length(moments)) stop('Demo covariance block count: ',id)
  for(g in seq_along(moments)) {
    m <- moments[[g]]
    if(ns[g]!=m$n) stop('Demo N mismatch: ',id,' ',ns[g],' vs ',m$n)
    h <- cov_heads[g];lines <- output[(h+1):min(length(output),h+12)]
    lines <- head(lines[grepl('^ Y[123]\\s',lines)],3L)
    demo <- unlist(lapply(lines,function(line) nums(sub('^ Y[123]','',line))))
    target <- unlist(lapply(1:3,function(j)m$S[j,1:j]))
    if(length(demo)!=6 || any(abs(demo-target)>.00050001)) stop('Demo covariance mismatch: ',id)
    mean_error <- NA_real_
    if(!is.null(m$mean)) {
      h <- mean_heads[g];values <- nums(output[h+3L])
      if(length(values)!=3 || any(abs(values-m$mean)>.00050001)) stop('Demo mean mismatch: ',id)
      mean_error <- max(abs(values-m$mean))
    }
    ml_error <- lavaan_error <- NA_real_
    if (!is.data.frame(data)) {
      fit <- fit_model(spec, data, estimator='ML')
      pt <- fit$partable
      gp <- if ('group' %in% names(pt)) pt$group == g else rep(TRUE,nrow(pt))
      cov_rows <- pt[gp & pt$op == '~~',]
      expected_cov <- m$S * (m$n-1)/m$n
      expected_est <- expected_cov[cbind(match(cov_rows$lhs,colnames(expected_cov)),match(cov_rows$rhs,colnames(expected_cov)))]
      stopifnot(max(abs(cov_rows$est-expected_est)) < 1e-6)
      # Parse the independent printed saturated MODEL RESULTS by group and
      # section; compare each variance/covariance and supplied mean.
      section <- ''; lhs <- ''; group <- 1L; demo_est <- numeric(); target_est <- numeric()
      start <- grep('^MODEL RESULTS$',output)[1]
      for (line in output[(start+1):length(output)]) {
        if (grepl('^QUALITY OF',line)) break
        if (grepl('^Group ',trimws(line))) { group <- match(tolower(sub('^Group +','',trimws(line))),tolower(spec$group_labels)); next }
        text <- trimws(line)
        if (text %in% c('Means','Intercepts','Variances','Residual Variances')) { section <- text; next }
        if (grepl('^Y[123] +WITH$',text)) {section <- 'WITH';lhs <- strsplit(text,' +')[[1]][1];next}
        if (group != g || !grepl('^Y[123] +[-0-9]',text)) next
        rhs <- tolower(strsplit(text,' +')[[1]][1]); value <- nums(sub('^Y[123]','',text))[1]
        estimate_target <- if(section %in% c('Means','Intercepts')) m$mean[match(rhs,colnames(expected_cov))] else if(section=='WITH') expected_cov[tolower(lhs),rhs] else expected_cov[rhs,rhs]
        demo_est <- c(demo_est,value);target_est <- c(target_est,estimate_target)
      }
      stopifnot(length(demo_est)==6L+if(is.null(m$mean))0L else 3L)
      ml_error <- max(abs(demo_est-target_est))
      stopifnot(ml_error <= .00050001)
      ref <- lavaan::sem('y1 ~~ y1+y2+y3; y2 ~~ y2+y3; y3 ~~ y3',
        sample.cov=m$S,sample.mean=m$mean,sample.nobs=m$n,meanstructure=!is.null(m$mean),fixed.x=FALSE)
      rpt <- lavaan::parTable(ref); key <- function(p) paste(p$lhs,p$op,p$rhs)
      selected <- pt[gp,]
      lavaan_error <- max(abs(selected$est-rpt$est[match(key(selected),key(rpt))]))
      stopifnot(is.finite(lavaan_error),lavaan_error < 1e-6)
    }
    results[[length(results)+1L]] <- data.frame(case=id,group=g,n=m$n,max_mean_error=mean_error,max_covariance_error=max(abs(demo-target)),max_ml_error=ml_error,max_lavaan_error=lavaan_error)
  }
}
result <- do.call(rbind,results)
writeLines(toJSON(list(version='Mplus 9.1 Demo',covariance_divisor='N-1 input, rescaled to N for ML',printed_tolerance=.00050001,comparisons=result),auto_unbox=TRUE,pretty=TRUE,digits=NA,na='null'),file.path(root,'cpp/tests/fixtures/mplus/data_summary.json'))
cat(length(cases),'Demo data cases;',nrow(result),'group comparisons; covariance/mean printed tolerance 0.00050001\n')

# Fit round trips use the same independent corpus inputs as the end-to-end gate.
args <- commandArgs(TRUE)
corpus <- if(length(args)) normalizePath(args[1]) else file.path(root,'external/textbook-corpus')
for(case in c('mplus_users_guide_v8_ch5_ex5_1','mplus_users_guide_v8_ch5_ex5_14')) {
  folder <- file.path(corpus,'cases/mplus_users_guide_v8',case)
  spec <- mplus_model(file=file.path(folder,'source/original.inp'))
  d <- read.csv(file.path(folder,'data/raw.csv'),check.names=FALSE)
  if(nzchar(spec$group_var)) {
    g <- as.character(d[[spec$group_var]])
    for(i in seq_len(nrow(spec$mplus_groups))) g[toupper(g)==toupper(spec$mplus_groups$label[i])] <- spec$mplus_groups$code[i]
    d[[spec$group_var]] <- as.numeric(g)
  }
  baseline <- fit_model(spec,d,estimator='ML')
  path <- file.path(scratch,paste0(case,'.dat'))
  write.table(d[,spec$mplus_data_plan$names,drop=FALSE],path,row.names=FALSE,col.names=FALSE,quote=FALSE)
  read <- mplus_data(spec,file=path)
  fit <- fit_model(spec,read,estimator='ML')
  stopifnot(isTRUE(all.equal(fit$partable$est,baseline$partable$est,tolerance=1e-8)))
  cat(case,'fit round trip passed\n')
}
