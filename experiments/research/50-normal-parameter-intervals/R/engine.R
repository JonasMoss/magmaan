# Experiment orchestration only; fitting, information and score projection are
# magmaanlab primitives. No statistical code is imported from another leaf.
model_text <- paste('f1 =~ x1 + b*x2 + x3 + x4',
                    'f2 =~ x5 + x6 + x7 + x8', 'f1 ~~ rho*f2', sep = '\n')
labels <- c(loading = 'b', correlation = 'rho')
truths <- c(loading = .7, correlation = .4)
control_primary <- list(max_iter = 2000L, ftol = 1e-12, gtol = 1e-8)
cutoff <- qchisq(.95, 1)
core <- magmaanlab::magmaan_core

population_sigma <- function() {
  L <- matrix(0, 8, 2); L[1:4, 1] <- L[5:8, 2] <- c(.8, .7, .9, .6)
  L %*% matrix(c(1,.4,.4,1),2) %*% t(L) + diag(1-rowSums(L^2))
}
draw_sample <- function(n, seed) {
  set.seed(seed)
  z <- matrix(rnorm(n*10), n, 10)
  f <- cbind(z[,1], .4*z[,1]+sqrt(.84)*z[,2])
  lam <- rep(c(.8,.7,.9,.6),2)
  X <- sweep(f[,rep(1:2,each=4)],2,lam,'*') +
    sweep(z[,3:10],2,sqrt(1-lam^2),'*')
  colnames(X) <- paste0('x',1:8)
  unclass(scale(X, scale=FALSE))
}
free_id <- function(fit, target) {
  p <- fit$partable
  id <- p$free[p$label == labels[[target]] & p$op %in% c('=~','~~')]
  stopifnot(length(id)==1L, id>0L)
  id
}
admissible <- function(fit) isTRUE(fit$diagnostics$admissibility$admissible)
check_fit <- function(fit, target=NULL, value=NULL) {
  if (!isTRUE(fit$converged)) stop('library convergence verdict: ',fit$verdict$status)
  if (any(!is.finite(fit$theta)) || !is.finite(fit$fmin)) stop('nonfinite fit')
  p <- fit$partable
  anchors <- p$free[p$op=='=~' & p$rhs %in% c('x1','x5')]
  if (any(fit$theta[anchors] <= 0)) stop('factor orientation changed')
  if (!is.null(target) && abs(fit$theta[free_id(fit,target)]-value)>1e-6)
    stop('equality constraint residual exceeds 1e-6')
  fit
}
fit_sample <- function(X, target=NULL, value=NULL, start=NULL) {
  syntax <- if (is.null(target)) model_text else
    paste(model_text,sprintf('%s == %.17g',labels[[target]],value),sep='\n')
  # Ambient unpenalized ML, no bounds; identical solver for every method.
  # Primary restricted start is the unrestricted fit, independent of root order.
  errors <- character()
  for (attempt in 1:2) {
    ctl <- control_primary
    if (!is.null(start)) ctl$start <- as.numeric(start)
    if (attempt==2L) { ctl$start <- 'scaled-fabin'; ctl$max_iter <- 5000L }
    ans <- tryCatch(suppressWarnings(magmaanlab::fit_model(
      syntax,as.data.frame(X),std_lv=TRUE,meanstructure=FALSE,
      optimizer='nlopt-slsqp',control=ctl,bounds=NULL)),error=identity)
    if (!inherits(ans,'error')) ans <- tryCatch(check_fit(ans,target,value),error=identity)
    if (!inherits(ans,'error')) {
      attr(ans,'fit_attempts') <- attempt
      return(ans)
    }
    errors <- c(errors,conditionMessage(ans))
  }
  stop(paste(errors,collapse='; fallback: '),call.=FALSE)
}
ordinary_lr <- function(u,r,n) {
  value <- 2*n*(r$fmin-u$fmin)
  if (!is.finite(value) || value < -1e-6) stop('negative/nonfinite profile LR: ',value)
  max(0,value) # Only roundoff within the declared 1e-6 tolerance.
}
model_covariances <- function(u,X) {
  ctx <- magmaanlab::prepare_inference(u,X)
  lapply(c('expected','observed'),function(kind) {
    tryCatch({
      H <- magmaanlab::inference_information(ctx,kind)
      magmaanlab::parameter_covariance(ctx,H)
    },error=identity)
  }) |> setNames(c('expected','observed'))
}
new_engine <- function(u,X,target,method,B=0L,boot_seed=1L) {
  e <- new.env(parent=emptyenv())
  e$cache <- new.env(parent=emptyenv()); e$rows <- list(); e$boot <- list()
  e$fits <- 0L
  counted_fit <- function(...) {
    value <- tryCatch(fit_sample(...),error=identity)
    e$fits <- e$fits + if(inherits(value,'error')) 2L else attr(value,'fit_attempts')
    if(inherits(value,'error')) stop(conditionMessage(value),call.=FALSE)
    value
  }
  Z <- NULL
  if (method=='lr_bartlett') {
    set.seed(boot_seed)
    # The first B matrices are a prefix when B increases. Reused at every b.
    Z <- lapply(seq_len(B),function(k) matrix(rnorm(nrow(X)*ncol(X)),nrow(X)))
  }
  evaluate <- function(b, refresh=FALSE) {
    key <- sprintf('%.17g',b)
    if (!refresh && exists(key,e$cache,inherits=FALSE)) return(get(key,e$cache))
    t0 <- proc.time()[['elapsed']]
    out <- tryCatch({
      r <- counted_fit(X,target,b,u$theta)
      lr <- ordinary_lr(u,r,nrow(X)); stat <- lr; correction <- 1; mcse <- 0
      if (method=='score_expected') {
        comp <- magmaanlab::score_components(r,X,H1=u,sensitivity='expected',metric='expected')
        projected <- magmaanlab::project_scores(comp)
        if (projected$df!=1L) stop('score restriction rank is not one')
        stat <- projected$statistic
      }
      if (method=='lr_bartlett') {
        sigma <- core$model_implied(r)$sigma[[1]]
        C <- chol(sigma); Ts <- rep(NA_real_,B); boot_errors <- character(B)
        for (k in seq_len(B)) {
          xb <- Z[[k]] %*% C
          colnames(xb) <- colnames(X); xb <- unclass(scale(xb,scale=FALSE))
          rr <- tryCatch({
            ub <- counted_fit(xb,start=r$theta)
            rb <- counted_fit(xb,target,b,ub$theta)
            ordinary_lr(ub,rb,nrow(X))
          },error=identity)
          if (inherits(rr,'error')) boot_errors[k] <- conditionMessage(rr) else Ts[k] <- rr
        }
        valid <- is.finite(Ts)
        diagnostic_mean <- if(any(valid)) mean(Ts[valid]) else NA_real_
        diagnostic_mcse <- if(sum(valid)>1) sd(Ts[valid])/sqrt(sum(valid)) else NA_real_
        e$boot[[length(e$boot)+1L]] <- data.frame(candidate=b,B_attempted=B,
          B_valid=sum(valid),correction=if(all(valid)) diagnostic_mean else NA_real_,
          mcse=if(all(valid)) diagnostic_mcse else NA_real_,
          diagnostic_valid_mean=diagnostic_mean,diagnostic_valid_mcse=diagnostic_mcse,
          failures=paste(unique(boot_errors[!valid]),collapse=' | '),seed=boot_seed)
        if (!all(valid)) stop(sprintf('bootstrap fits failed: %d/%d',sum(!valid),B))
        correction <- mean(Ts); mcse <- sd(Ts)/sqrt(B)
        if (!is.finite(correction) || correction<=0) stop('invalid Bartlett correction')
        stat <- lr/correction
      }
      if (!is.finite(stat) || stat<0) stop('invalid statistic')
      list(statistic=stat,lr=lr,correction=correction,mcse=mcse,
           constraint=abs(r$theta[free_id(r,target)]-b),admissible=admissible(r))
    },error=identity)
    e$rows[[length(e$rows)+1L]] <- data.frame(candidate=b,
      statistic=if(inherits(out,'error')) NA_real_ else out$statistic,
      lr=if(inherits(out,'error')) NA_real_ else out$lr,
      correction=if(inherits(out,'error')) NA_real_ else out$correction,
      correction_mcse=if(inherits(out,'error')) NA_real_ else out$mcse,
      constraint_residual=if(inherits(out,'error')) NA_real_ else out$constraint,
      admissible=if(inherits(out,'error')) NA else out$admissible,
      refreshed=refresh,seconds=proc.time()[['elapsed']]-t0,
      error=if(inherits(out,'error')) conditionMessage(out) else '')
    if (inherits(out,'error')) stop(conditionMessage(out),call.=FALSE)
    assign(key,out,e$cache); out
  }
  list(evaluate=evaluate,state=e)
}
invert <- function(engine,estimate,se) {
  f <- function(b) engine$evaluate(b)$statistic-cutoff
  if (f(estimate)>0) stop('estimate rejected by its own test')
  ends <- numeric(2)
  for (side in 1:2) {
    direction <- c(-1,1)[side]; step <- 1.96*se
    outer <- estimate+direction*step
    for (k in seq_len(20)) {
      if (f(outer)>0) break
      step <- step*1.5; outer <- estimate+direction*step
    }
    if (f(outer)<=0) stop('unresolved bracket after 20 expansions')
    bounds <- sort(c(estimate,outer))
    root <- uniroot(f,bounds,tol=1e-7,maxiter=100L,check.conv=TRUE)$root
    # Refit independently rather than trusting the root solver's cached value.
    gap <- engine$evaluate(root,refresh=TRUE)$statistic-cutoff
    if(abs(gap)>1e-4) stop('endpoint statistic tolerance failed: ',gap)
    ends[side] <- root
  }
  width <- diff(ends)
  # A finite wider-grid audit, not a proof of connectedness over the real line.
  grid <- sort(unique(c(seq(ends[1]-.5*width,ends[2]+.5*width,length.out=9),estimate)))
  accept <- vapply(grid,function(b) f(b)<=0,logical(1))
  if (sum(abs(diff(as.integer(accept))))!=2L || accept[1] || tail(accept,1))
    stop('disconnected or unresolved acceptance set on wider audit grid')
  list(lower=ends[1],upper=ends[2],audit_lower=min(grid),audit_upper=max(grid))
}
empty_interval <- function(target,method,estimate=NA_real_) {
  data.frame(target=target,method=method,estimate=estimate,truth=truths[[target]],
    lower=NA_real_,upper=NA_real_,valid=FALSE,error='',covered=NA,
    lower_miss=NA,upper_miss=NA,width=NA_real_,asymmetry=NA_real_,
    truth_rejected=NA,truth_agreement=NA,endpoint_gap=NA_real_,
    audit_lower=NA_real_,audit_upper=NA_real_,unrestricted_admissible=NA,
    candidate_inadmissible=0L,refits=0L,seconds=NA_real_)
}
run_dataset <- function(n,replicate,seed_base,mode,B=399L,stream=0L) {
  seed <- as.integer(seed_base+n*1000L+replicate)
  X <- draw_sample(n,seed); t0 <- proc.time()[['elapsed']]
  u <- tryCatch(fit_sample(X),error=identity)
  shared_seconds <- proc.time()[['elapsed']]-t0
  targets <- if(mode=='bartlett') 'loading' else names(truths)
  methods <- if(mode=='bartlett') c('wald_expected','wald_observed','score_expected','lr','lr_bartlett') else
    c('wald_expected','wald_observed','score_expected','lr')
  intervals <- candidates <- bootstrap <- list()
  Vs <- if(inherits(u,'error')) u else tryCatch(model_covariances(u,X),error=identity)
  for(target in targets) for(method in methods) {
    est <- if(inherits(u,'error')) NA_real_ else u$theta[free_id(u,target)]
    row <- empty_interval(target,method,est); engine <- NULL
    t1 <- proc.time()[['elapsed']]
    result <- tryCatch({
      if(inherits(u,'error')) stop(conditionMessage(u))
      if(inherits(Vs,'error')) stop(conditionMessage(Vs))
      if(inherits(Vs$expected,'error')) stop(conditionMessage(Vs$expected))
      id <- free_id(u,target); se <- sqrt(Vs$expected[id,id])
      row$unrestricted_admissible <- admissible(u)
      if(grepl('^wald_',method)) {
        kind <- sub('wald_','',method)
        if(inherits(Vs[[kind]],'error')) stop(conditionMessage(Vs[[kind]]))
        ses <- sqrt(Vs[[kind]][id,id])
        row$lower <- est-qnorm(.975)*ses; row$upper <- est+qnorm(.975)*ses
        row$truth_rejected <- abs(est-truths[[target]])/ses>qnorm(.975)
      } else {
        boot_seed <- as.integer(100000L+n*10000L+replicate*10L+stream)
        engine <- new_engine(u,X,target,method,B,boot_seed)
        ci <- invert(engine,est,se)
        row$lower <- ci$lower; row$upper <- ci$upper
        row$audit_lower <- ci$audit_lower; row$audit_upper <- ci$audit_upper
        row$truth_rejected <- engine$evaluate(truths[[target]])$statistic>cutoff
        row$endpoint_gap <- max(abs(c(engine$evaluate(ci$lower)$statistic,
                                      engine$evaluate(ci$upper)$statistic)-cutoff))
      }
      row$covered <- row$lower<=truths[[target]] && row$upper>=truths[[target]]
      row$lower_miss <- truths[[target]]<row$lower
      row$upper_miss <- truths[[target]]>row$upper
      row$truth_agreement <- row$covered==!row$truth_rejected
      if(!row$truth_agreement) stop('truth test/inversion disagreement')
      row$width <- row$upper-row$lower
      row$asymmetry <- (row$upper+row$lower-2*est)/row$width
      row$valid <- TRUE
      TRUE
    },error=identity)
    if(inherits(result,'error')) row$error <- conditionMessage(result)
    row$seconds <- proc.time()[['elapsed']]-t1
    if(!is.null(engine)) {
      row$refits <- engine$state$fits
      if(length(engine$state$rows)) {
        cc <- do.call(rbind,engine$state$rows)
        row$candidate_inadmissible <- sum(!cc$admissible,na.rm=TRUE)
        cc$target <- target; cc$method <- method; candidates[[length(candidates)+1L]] <- cc
      }
      if(length(engine$state$boot)) {
        bb <- do.call(rbind,engine$state$boot)
        bb$target <- target; bb$method <- method; bootstrap[[length(bootstrap)+1L]] <- bb
      }
    }
    intervals[[length(intervals)+1L]] <- row
  }
  attach_ids <- function(x) {
    if(!length(x)) return(data.frame())
    x <- do.call(rbind,x); x$n <- n; x$replicate <- replicate; x$seed <- seed
    x$B <- if(mode=='bartlett') B else 0L; x$stream <- stream; x
  }
  list(intervals=attach_ids(intervals),candidates=attach_ids(candidates),
       bootstrap=attach_ids(bootstrap),timing=data.frame(n=n,replicate=replicate,
       B=if(mode=='bartlett') B else 0L,stream=stream,
       shared_fit_seconds=shared_seconds,total_seconds=proc.time()[['elapsed']]-t0))
}
