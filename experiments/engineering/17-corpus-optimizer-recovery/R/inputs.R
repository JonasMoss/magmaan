# Consume the optional corpus directly; never borrow another experiment's inputs.
read_case <- function(root, relative, estimator) {
  directory <- file.path(root, relative)
  meta <- jsonlite::fromJSON(file.path(directory,"meta.json"), simplifyVector=TRUE)
  mo <- meta$model_options
  if (isTRUE(meta$out_of_scope)) stop("excluded: corpus out_of_scope")
  if (length(mo$ordered)) stop("excluded: categorical model")
  if (identical(mo$missing,"fiml")) stop("excluded: FIML requires a different objective")
  if (length(mo$mimic) && mo$mimic != "lavaan") stop("excluded: non-lavaan mimic")
  if (length(mo$parameterization) && mo$parameterization != "delta") stop("excluded: categorical parameterization")
  syntax <- paste(readLines(file.path(directory,"model.lav"),warn=FALSE),collapse="\n")
  if (grepl("<~|level:",syntax)) stop("excluded: composite or multilevel model")
  args <- list(model=syntax, estimator=estimator, meanstructure=isTRUE(mo$meanstructure),
               fixed.x=isTRUE(mo$fixed_x), se="none", test="standard")
  for (key in c("group_equal","group_partial"))
    if (length(mo[[key]])) args[[gsub("_",".",key)]] <- mo[[key]]
  if (length(mo$sample_cov_rescale)) args$sample.cov.rescale <- mo$sample_cov_rescale
  if (identical(meta$data$kind,"raw")) {
    args$data <- read.csv(file.path(directory,meta$data$files$raw),check.names=FALSE)
    if (length(meta$data$group_var)) args$group <- meta$data$group_var
    args$missing <- "listwise"
  } else {
    args$sample.cov <- lapply(meta$data$files$sample_cov,function(f)
      as.matrix(read.csv(file.path(directory,f),row.names=1L,check.names=FALSE)))
    args$sample.nobs <- as.integer(meta$data$n_obs)
    if (length(meta$data$files$sample_mean)) args$sample.mean <-
      lapply(meta$data$files$sample_mean,function(f) {
        z<-read.csv(file.path(directory,f),check.names=FALSE);setNames(z[[2]],z[[1]])
      })
    if (length(args$sample.cov)==1L) {
      args$sample.cov <- args$sample.cov[[1]]
      if(length(args$sample.mean)) args$sample.mean <- args$sample.mean[[1]]
    }
  }
  fun <- getExportedValue("lavaan",meta$lavaan_function %||% "sem")
  pre <- suppressWarnings(do.call(fun,c(args,list(do.fit=FALSE))))
  ss <- lavaan::lavInspect(pre,"sampstat")
  ng <- lavaan::lavInspect(pre,"ngroups")
  if(ng==1L) ss<-list(ss)
  n <- as.integer(lavaan::lavInspect(pre,"nobs"))
  if(any(vapply(ss,function(s) min(eigen(s$cov,symmetric=TRUE,only.values=TRUE)$values)<=0,logical(1))))
    stop("excluded: sample covariance is not positive definite")
  sp <- list(syntax=syntax,meanstructure=isTRUE(mo$meanstructure),fixed_x=isTRUE(mo$fixed_x),
             auto_cov_y=meta$lavaan_function %in% c("sem","cfa","growth"))
  if(meta$lavaan_function=="growth") sp$model_type<-"growth"
  if(ng>1L) {sp$group<-"group";sp$group_labels<-as.character(seq_len(ng))}
  for(key in c("group_equal","group_partial")) if(length(mo[[key]])) sp[[key]]<-mo[[key]]
  model <- do.call(magmaanlab::model_spec,sp)
  sample <- list(S=lapply(ss,`[[`,"cov"),nobs=n,
                 mean=if(isTRUE(mo$meanstructure)) lapply(ss,`[[`,"mean") else NULL)
  # Compare the estimable row pattern before attributing differences to solvers.
  mp <- model$partable; lp <- lavaan::parTable(pre)
  formula <- mp$op %in% c("=~","~~","~","~1")
  idx <- match(row_key(mp[formula,]),row_key(lp))
  if(anyNA(idx) || any((mp$free[formula]>0)!=(lp$free[idx]>0)))
    stop("excluded: model parameter freedom differs from lavaan")
  list(model=model,sample=sample,args=args,fun=fun,pre=pre,meta=meta,directory=directory)
}
row_key <- function(p) {
  left<-p$lhs;right<-p$rhs; cov<-p$op=="~~"
  left[cov]<-pmin(p$lhs[cov],p$rhs[cov]);right[cov]<-pmax(p$lhs[cov],p$rhs[cov])
  paste(left,p$op,right,p$group,sep="|")
}
map_theta <- function(model, reference) {
  p<-model$partable; q<-lavaan::parTable(reference)
  rows<-which(p$free>0); idx<-match(row_key(p[rows,]),row_key(q))
  if(anyNA(idx)) stop("reference parameter mapping failed")
  ans<-numeric(max(p$free));ans[p$free[rows]]<-q$est[idx];ans
}
reference_fit <- function(case) {
  args<-case$args
  if(length(case$meta$model_options$start_values)) {
    z<-read.csv(file.path(case$directory,case$meta$model_options$start_values),check.names=FALSE)
    p<-lavaan::parTable(case$pre); st<-p$start[p$free>0]
    idx<-match(row_key(p[p$free>0,]),row_key(z))
    if(anyNA(idx)) stop("verified-start parameter mapping failed")
    st[]<-z$value[idx];args$start<-st
  }
  suppressWarnings(do.call(case$fun,args))
}
