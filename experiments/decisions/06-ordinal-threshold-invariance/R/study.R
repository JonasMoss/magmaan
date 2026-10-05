threshold_cells <- function() {
  x <- expand.grid(factors=c(1L,2L), categories=c(4L,5L,7L,3L),
    skew=c(FALSE,TRUE), n=c(250L,500L,1000L), parameterization=c("theta","delta"),
    misspecified=c(FALSE,TRUE), role=c("null","power"), stringsAsFactors=FALSE)
  x$cell_id <- seq_len(nrow(x)); x$production_reps <- ifelse(x$role=="null",2000L,1000L)
  x
}
threshold_draw <- function(cell, seed) {
  set.seed(seed); p <- 6L*cell$factors
  loading <- matrix(0,p,cell$factors)
  for(j in seq_len(p)) loading[j,ceiling(j/6)] <- .7
  if(cell$misspecified) {
    # For one factor, an omitted shared residual component replaces a cross-loading.
    if(cell$factors==2L) loading[2,2] <- .3
  }
  latent_cov <- matrix(.3,cell$factors,cell$factors); diag(latent_cov) <- 1
  cuts <- qnorm(seq_len(cell$categories-1L)/cell$categories)
  if(cell$skew) cuts <- cuts+.65
  groups <- lapply(1:2,function(g) {
    f <- matrix(rnorm(cell$n*cell$factors),cell$n)%*%chol(latent_cov)
    residual_var <- 1-diag(loading%*%latent_cov%*%t(loading))
    e <- matrix(rnorm(cell$n*p),cell$n)%*%diag(sqrt(residual_var))
    if(cell$misspecified && cell$factors==1L) {
      shared <- rnorm(cell$n); e[,1:2] <- e[,1:2]*sqrt(.7)+sqrt(.3*residual_var[1])*shared
    }
    y <- f%*%t(loading)+e
    out <- as.data.frame(lapply(seq_len(p),function(j) {
      th <- cuts
      if(g==2L && j==2L && cell$role=="power") th[2] <- th[2]+.3
      ordered(findInterval(y[,j],th)+1L,levels=seq_len(cell$categories))
    }))
    names(out) <- paste0("x",seq_len(p)); out$group <- factor(g,levels=1:2); out
  })
  do.call(rbind,groups)
}
threshold_fit <- function(cell,data,equal=NULL, parameterization=cell$parameterization) {
  syntax <- paste(vapply(seq_len(cell$factors),function(f)
    paste0("f",f," =~ ",paste0("x",((f-1)*6+1):(f*6),collapse=" + ")),""),collapse="\n")
  spec <- magmaanlab::model_spec(syntax,ordered=paste0("x",seq_len(6*cell$factors)),
    group="group",group_labels=c("1","2"),parameterization=parameterization,group_equal=equal)
  magmaanlab::fit_model(spec,data,estimator="DWLS")
}
threshold_replicate <- function(cell,replicate,seed_base) {
  seed <- seed_base+10000L*cell$cell_id+replicate; start <- proc.time()
  conv1 <- conv0 <- NA; spectrum <- numeric(); rows <- list()
  add <- function(arm,p=NA_real_,statistic=NA_real_,df=NA_real_,reason="available",detail="") {
    rows[[length(rows)+1L]] <<- data.frame(arm,p,statistic,df,reason,detail)
  }
  tryCatch({
    d <- threshold_draw(cell,seed)
    h1 <- threshold_fit(cell,d); conv1 <- h1$converged
    h0 <- threshold_fit(cell,d,"thresholds"); conv0 <- h0$converged
    if(!isTRUE(conv1) || !isTRUE(conv0)) stop("not_converged")
    policy <- magmaanlab::policy_nested(h1,h0)$lr
    if(isTRUE(policy$available)) {
      spectrum <- policy$eigenvalues
      for(a in c("sb","peba4")) add(paste0("policy_",a),policy[[paste0("p_",a)]],policy$statistic,policy$df)
      exact <- magmaanlab::magmaan_core$robust_fmg_test(policy$statistic,policy$df,spectrum,
        "all",0,truncate_negative=TRUE)
      add("all",exact$p_value,policy$statistic,policy$df,
        if(is.finite(exact$p_value)) "available" else "numeric_failure")
    } else for(a in c("policy_sb","policy_peba4","all"))
      add(a,reason=policy$reason,detail=policy$detail %||% "")
    # The comparator is always evaluated in delta coordinates, on the same draw.
    tryCatch({
      if(cell$parameterization=="delta") { a <- h1; b <- h0 } else {
        a <- threshold_fit(cell,d,parameterization="delta")
        b <- threshold_fit(cell,d,"thresholds",parameterization="delta")
      }
      test <- magmaanlab::convention_nested(a,b,"WLSMV")$test
      add("delta_wlsmv",test$p_value %||% NA_real_,test$statistic %||% NA_real_,
        test$df %||% NA_real_,test$reason %||% "available",test$detail %||% "")
    },error=function(e) add("delta_wlsmv",reason="error",detail=conditionMessage(e)))
  },error=function(e) {
    rows <<- list()
    for(a in c("policy_sb","policy_peba4","all","delta_wlsmv")) add(a,reason="error",detail=conditionMessage(e))
  })
  z <- do.call(rbind,rows); elapsed <- proc.time()-start
  z$cell_id <- cell$cell_id; z$replicate <- replicate; z$seed <- seed
  z$h1_converged <- conv1; z$h0_converged <- conv0
  z$spectrum_size <- length(spectrum)
  z$elapsed_seconds <- unname(elapsed["elapsed"])
  z$cpu_seconds <- unname(sum(elapsed[c("user.self","sys.self")]))
  z$eigenvalues <- c(list(spectrum),rep(list(NULL),nrow(z)-1L)); z
}
threshold_wilson <- function(k,n) {
  if(!n) return(c(NA_real_,NA_real_))
  z <- qnorm(.975); center <- (k/n+z*z/(2*n))/(1+z*z/n)
  half <- z*sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
  c(center-half,center+half)
}
threshold_summarize <- function(raw,cells,out) {
  raw <- raw[order(raw$cell_id,raw$replicate,raw$arm),]; rownames(raw) <- NULL
  summaries <- list(); timing <- list()
  for(i in seq_len(nrow(cells))) {
    cell <- cells[i,]; x <- raw[raw$cell_id==cell$cell_id,]
    attempts <- x[!duplicated(x$replicate),]
    timing[[i]] <- data.frame(cell_id=cell$cell_id,attempted=nrow(attempts),
      mean_seconds=mean(attempts$elapsed_seconds),mean_cpu_seconds=mean(attempts$cpu_seconds),
      production_reps=cell$production_reps)
    for(arm in c("policy_sb","policy_peba4","all","delta_wlsmv")) {
      a <- x[x$arm==arm,]; valid <- is.finite(a$p) & a$reason=="available"
      n <- sum(valid); k <- sum(a$p[valid]<.05); ci <- threshold_wilson(k,n)
      adjusted <- NA_real_; null_n <- 0L
      if(cell$role=="power") {
        matched <- cells$role=="null"
        for(key in c("factors","categories","skew","n","parameterization","misspecified"))
          matched <- matched & cells[[key]]==cell[[key]]
        nx <- raw[raw$cell_id %in% cells$cell_id[matched] & raw$arm==arm &
          raw$reason=="available" & is.finite(raw$p),]
        null_n <- nrow(nx)
        if(n && null_n) adjusted <- mean(a$p[valid]<quantile(nx$p,.05,type=8))
      }
      rate <- if(n) k/n else NA_real_
      summaries[[length(summaries)+1L]] <- data.frame(cell_id=cell$cell_id,arm,
        attempted=nrow(a),successful=n,failed=nrow(a)-n,rejected=k,rate,
        wilson_low=ci[1],wilson_high=ci[2],size_adjusted_power=adjusted,matched_null_successful=null_n,
        registered_flag=cell$role=="null" && cell$n>=500 && startsWith(arm,"policy_") &&
          is.finite(rate) && (rate<.03 || rate>.07))
    }
  }
  timing <- do.call(rbind,timing)
  write_csv(do.call(rbind,summaries),file.path(out,"summary.csv"))
  write_csv(timing,file.path(out,"timing.csv"))
  write_csv(raw[raw$reason!="available",setdiff(names(raw),"eigenvalues")],file.path(out,"failures.csv"))
  # Extrapolate each design family from its observed timing; expose missing families.
  full <- threshold_cells(); predicted <- numeric(nrow(full))
  for(i in seq_len(nrow(full))) {
    match <- cells$factors==full$factors[i] & cells$parameterization==full$parameterization[i]
    predicted[i] <- if(any(match)) mean(timing$mean_cpu_seconds[match])*full$production_reps[i]/3600 else NA_real_
  }
  write_csv(data.frame(observed_cells=nrow(cells),full_cells=nrow(full),
    production_cpu_hours=if(all(is.finite(predicted))) sum(predicted) else NA_real_,
    unpriced_cells=sum(!is.finite(predicted)),estimation="family mean CPU time; excludes cloud build/startup"),file.path(out,"cost.csv"))
}
