# Literal populations, independent seeds and pinned live oracle. No leaf inputs.
.hard_key <- function(p) paste(p$lhs, p$op, p$rhs, p$group, sep = "\r")
.hard_text <- function(x) paste(capture.output(dput(x)), collapse = " ")
.hard_close <- function(x, y, tol) length(x) == length(y) &&
  all(is.finite(c(x, y))) && all(abs(x-y) <= tol * (1+pmax(abs(x), abs(y))))
.hard_capture <- function(f) {
  warnings <- character()
  value <- tryCatch(withCallingHandlers(f(), warning = function(w) {
    warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
  }), error = identity)
  list(value = value, warnings = warnings,
       error = if (inherits(value, "error")) conditionMessage(value) else "")
}
.hard_cells <- function() {
  design <- c(rep("D1",4),rep("D2",2),rep("D3",2),rep("D3b",2),
              rep("D4",2),rep("D5",2),rep("D6",2),"D6b","D7")
  data.frame(design = design, n = c(10,15,20,30,20,50,30,60,30,60,
                                  20,50,20,40,50,100,100,30))
}
.hard_design <- function(design, n) {
  d1 <- "Y =~ y1+y2+y3\nX =~ x1+x2+x3\nY ~ X"
  load <- c(if (design == "D2") .15 else 1,.8,.6)
  draw <- function(n) {
    x <- rnorm(n); y <- .25*x+rnorm(n)
    out <- cbind(outer(y,load)+matrix(rnorm(n*3),n),
                 outer(x,load)+matrix(rnorm(n*3),n))
    colnames(out) <- c(paste0("y",1:3),paste0("x",1:3))
    as.data.frame(out)
  }
  data <- draw(n); model <- d1; ordered <- NULL; estimator <- "ML"
  if (design %in% c("D3","D3b")) {
    times <- if (design == "D3") 0:3 else c(0,.5,1,3)
    # Literal intercept/slope means 1/.2, variances 1/1, covariance .2;
    # independent unit residuals. The basis population satisfies l2+l3=1.5.
    i <- 1+rnorm(n); s <- .2+.2*(i-1)+sqrt(.96)*rnorm(n)
    data <- as.data.frame(outer(i,rep(1,4))+outer(s,times)+matrix(rnorm(n*4),n))
    names(data) <- paste0("t",1:4)
    model <- paste("i =~ 1*t1+1*t2+1*t3+1*t4",
      if (design == "D3") "s =~ 0*t1+1*t2+2*t3+3*t4" else
        "s =~ 0*t1+l2*t2+l3*t3+3*t4\nl2+l3 == 1.5",
      "i ~ 1\ns ~ 1\nt1 ~ 0*1\nt2 ~ 0*1\nt3 ~ 0*1\nt4 ~ 0*1",
      "t1 ~~ e*t1\nt2 ~~ e*t2\nt3 ~~ e*t3\nt4 ~~ e*t4",sep="\n")
  }
  if (design == "D4") {
    z1 <- rnorm(n); z2 <- rnorm(n); f <- .25*z1+.25*z2+rnorm(n)
    data <- as.data.frame(outer(f,c(1,.8,.6,.7))+matrix(rnorm(n*4),n))
    names(data) <- paste0("x",1:4); data$z1 <- z1; data$z2 <- z2
    model <- "f =~ x1+x2+x3+x4\nf ~ z1+z2"
  }
  if (design == "D5") {
    estimator <- "FIML"
    for (v in names(data)) data[runif(n)<.2,v] <- NA_real_
  }
  if (design %in% c("D6","D6b")) {
    estimator <- "DWLS"
    ordered <- if (design == "D6") names(data) else paste0("y",1:3)
    for (v in ordered) {
      variance <- 1+load[as.integer(substr(v,2,2))]^2 *
        if (substr(v,1,1)=="y") 1.0625 else 1
      data[[v]] <- as.integer(cut(data[[v]],c(-Inf,sqrt(variance)*qnorm(c(.25,.5,.75)),Inf)))
    }
  }
  if (design == "D7") {
    # Measurement-only population: independent unit-variance factors.
    independent <- function() {
      out <- cbind(outer(rnorm(n),load)+matrix(rnorm(n*3),n),
                   outer(rnorm(n),load)+matrix(rnorm(n*3),n))
      colnames(out) <- names(data); as.data.frame(out)
    }
    data <- rbind(independent(),independent()); data$g <- rep(c("a","b"),each=n)
    model <- "Y =~ y1+y2+y3\nX =~ x1+x2+x3"
  }
  list(model=model,data=data,ordered=ordered,estimator=estimator)
}
.hard_endpoint <- function(actual, oracle) {
  result <- list(ok=FALSE, estimate_gap=NA_real_, chisq_gap=NA_real_,
                 endpoint_gradient=NA_real_, se_units=NA_real_, reason="parameter keys differ")
  mp <- actual$partable; lp <- lavaan::parTable(oracle)
  # Align structural rows; compare free estimates and fixed marker loadings.
  mp <- mp[mp$op %in% c("=~","~","~~","~1","|","~*~"),]
  lp <- lp[lp$op %in% c("=~","~","~~","~1","|","~*~"),]
  mk <- .hard_key(mp); lk <- .hard_key(lp)
  if (!setequal(mk,lk) || anyDuplicated(mk) || anyDuplicated(lk)) return(result)
  lp <- lp[match(mk,lk),]; delta <- abs(mp$est-lp$est)
  free <- lp$free>0
  result$estimate_gap <- max(delta[free])
  if (any(!free & mp$op=="=~" & delta>1e-5*(1+pmax(abs(mp$est),abs(lp$est))))) {
    result$reason <- "fixed markers differ"; return(result)
  }
  if (.hard_close(mp$est[free],lp$est[free],1e-5)) {
    result$ok <- TRUE; result$reason <- "relative estimates"; return(result)
  }
  # The task-59 contract measures parameter displacement in oracle SE units,
  # not the difference between two independently computed SEs.
  theta <- oracle@optim$x; theta[lp$free[free]] <- mp$est[free]
  endpoint <- lavaan:::lav_model_set_parameters(oracle@Model,theta)
  objective <- as.numeric(lavaan:::lav_model_objective(endpoint,endpoint@GLIST,
                                                     oracle@SampleStats,oracle@Data))
  gradient <- lavaan:::lav_model_grad(endpoint,endpoint@GLIST,oracle@SampleStats,oracle@Data)
  result$chisq_gap <- 2*lavaan::lavInspect(oracle,"ntotal")*abs(objective-oracle@optim$fx)
  result$endpoint_gradient <- max(abs(c(gradient,oracle@optim$dx)))
  result$se_units <- if (all(is.finite(lp$se[free]) & lp$se[free]>0))
    max(delta[free]/lp$se[free]) else NA_real_
  result$ok <- all(is.finite(unlist(result[c("chisq_gap","endpoint_gradient","se_units")]))) &&
    result$chisq_gap<=1e-6 && result$endpoint_gradient<=1e-3 && result$se_units<=1e-3
  result$reason <- if (result$ok) "endpoint contract" else "endpoint difference"
  result
}
.hard_replicate <- function(design,n,seed,trace_state,final) {
  set.seed(seed); case <- .hard_design(design,n)
  args <- list(model=case$model,data=case$data,meanstructure=TRUE,fixed_x=FALSE,
    estimator=case$estimator,control=list(fitting_options=list(preset="lavaan-0.7.2")))
  la <- list(model=case$model,data=case$data,meanstructure=TRUE,fixed.x=FALSE,
             estimator=if(case$estimator=="FIML") "ML" else case$estimator)
  if (case$estimator=="FIML") la$missing <- "ml"
  if (!is.null(case$ordered)) args$ordered <- la$ordered <- case$ordered
  if (design=="D7") {
    args$groups <- la$group <- "g"; args$group_equal <- la$group.equal <- "loadings"
  }
  trace_state$attempts <- list()
  oracle <- .hard_capture(function() do.call(lavaan::sem,la))
  attempts <- trace_state$attempts
  actual <- .hard_capture(function() do.call(magmaanlab::fit_model,args))
  m <- actual$value; l <- oracle$value
  me <- nzchar(actual$error); le <- nzchar(oracle$error)
  ma <- if (!me) m$fitting$attempts else list()
  selected <- if (!me) m$fitting$selected_attempt else NA_integer_
  ms <- if(length(ma)) ma[[selected]] else list()
  ls <- if(length(attempts)) attempts[[length(attempts)]] else list()
  mc <- if(me) NA else m$converged
  lc <- if(le) NA else lavaan::lavInspect(l,"converged")
  mg <- if(me) NA_real_ else ms$gradient_max
  lg <- if(le || !length(ls$gradient)) NA_real_ else max(abs(ls$gradient))
  post_m <- if(!me && !is.null(m$fitting$post_check)) m$fitting$post_check else NA
  post_l <- if(le) NA else suppressWarnings(lavaan::lavInspect(l,"post.check"))
  switch_l <- if(le) NA else any(grepl("marker",oracle$warnings,ignore.case=TRUE))
  if (!le && design %in% c("D1","D2","D5","D6","D6b","D7","D4")) {
    pt <- lavaan::parTable(l); markers <- pt[pt$op=="=~" & pt$rhs %in% c("y1","x1"),]
    switch_l <- switch_l || any(markers$free>0 | abs(markers$est-1)>1e-8)
  }
  switch_m <- if(!me && !is.null(m$fitting$marker_switch)) m$fitting$marker_switch else NA
  # Missing baseline marker metadata corresponds to main's keep-marker behavior.
  decision_m <- if(is.na(switch_m)) FALSE else isTRUE(switch_m)
  starts_m <- if(length(ma)) ma[[1]]$start else numeric()
  starts_l <- if(length(attempts)) attempts[[1]]$theta_start else numeric()
  # Compare original free parameters by partable identity, since ordinal and
  # constrained optimizer coordinates have different storage orders.
  starts_agree <- FALSE
  if (!me && length(attempts) && isTRUE(attempts[[1]]$constrained)) {
    starts_agree <- .hard_close(ma[[1]]$optimizer_start,attempts[[1]]$start,1e-9)
  } else if (!me && length(attempts) && length(starts_m)>0) {
    mp <- m$partable[m$partable$free>0,]
    lp <- attempts[[1]]$partable
    lk <- .hard_key(lp); mk <- .hard_key(mp)
    if(setequal(mk,lk) && !anyDuplicated(mk) && !anyDuplicated(lk))
      starts_agree <- .hard_close(starts_m[mp$free],
        starts_l[lp$free[match(mk,lk)]],1e-9)
  }
  endpoint <- list(ok=FALSE,estimate_gap=NA_real_,chisq_gap=NA_real_,
                   endpoint_gradient=NA_real_,se_units=NA_real_,reason="fit error")
  if (!me && !le) endpoint <- tryCatch(.hard_endpoint(m,l),error=function(e) {
    endpoint$reason <- conditionMessage(e); endpoint
  })
  reason <- endpoint$reason; class <- "rule_difference"
  # No exemption without both the documented refusal and an observed retry.
  defect <- me && !le && isTRUE(lc) && design=="D3b" &&
    grepl("standardized|affine RHS|constraint surface",actual$error) &&
    any(vapply(attempts,function(a) identical(a$parscale,"standardized"),logical(1)))
  switch_diff <- !is.na(switch_l) && decision_m!=switch_l
  post_diff <- !is.na(post_m) && !is.na(post_l) && post_m!=post_l
  own_rule <- function(conv,status,grad) !is.na(conv) && !is.null(status) &&
    is.finite(grad) && identical(isTRUE(conv),status %in% 3:6 && grad<=1e-3)
  if (defect) { class <- "known_defect"; reason <- actual$error
  } else if (!final && switch_diff && is.na(switch_m)) {
    class <- "pending_feature"; reason <- "baseline marker switch absent"
  } else if (!me && !le && starts_agree && !switch_diff && !post_diff && endpoint$ok &&
             identical(mc,lc) && length(ma)==length(attempts)) {
    class <- "agree"
  } else if (!me && !le && starts_agree && !switch_diff && !post_diff && !endpoint$ok &&
             !identical(mc,lc) && own_rule(mc,ms$raw_status,mg) && own_rule(lc,ls$status,lg)) {
    class <- "path_divergence"
  } else {
    reason <- paste(c(if(me) actual$error,if(le) oracle$error,
      if(!starts_agree) "first starts differ/unavailable",if(switch_diff) "marker decision differs",
      if(post_diff) "post.check differs",if(!identical(mc,lc)) "convergence differs",
      if(length(ma)!=length(attempts)) "attempt counts differ",endpoint$reason),collapse="; ")
  }
  if (final && class=="agree" && (is.na(post_m)||is.na(switch_m))) {
    class <- "pending_feature"; reason <- "required final metadata absent"
  }
  data.frame(design=design,n=n,seed=seed,class=class,reason=reason,
    m_converged=mc,l_converged=lc,m_error=actual$error,l_error=oracle$error,
    m_iterations=if(me) NA else m$iterations,l_iterations=if(le) NA else l@optim$iterations,
    m_fmin=if(me) NA else m$fmin,l_fmin=if(le) NA else as.numeric(l@optim$fx),
    m_gradient=mg,l_gradient=lg,m_status=if(is.null(ms$raw_status)) NA else ms$raw_status,
    l_status=if(is.null(ls$status)) NA else ls$status,
    m_post_check=post_m,l_post_check=post_l,m_marker_switch=switch_m,l_marker_switch=switch_l,
    m_attempt_count=length(ma),l_attempt_count=length(attempts),m_selected=selected,
    l_selected=if(length(attempts)) length(attempts) else NA_integer_,
    starts_agree=starts_agree,m_first_start=.hard_text(starts_m),l_first_start=.hard_text(starts_l),
    m_estimates=.hard_text(if(me) NULL else m$partable),
    l_estimates=.hard_text(if(le) NULL else lavaan::parTable(l)),
    m_attempts=.hard_text(ma),l_attempts=.hard_text(attempts),
    m_warnings=paste(actual$warnings,collapse="; "),l_warnings=paste(oracle$warnings,collapse="; "),
    estimate_gap=endpoint$estimate_gap,chisq_gap=endpoint$chisq_gap,
    endpoint_gradient=endpoint$endpoint_gradient,se_units=endpoint$se_units)
}

test_that("hard cases retain the pinned live lavaan preset rules", {
  skip_if(Sys.getenv("MAGMAAN_PARITY")!="1","opt-in hard preset parity")
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan"))!="0.7.2","requires lavaan 0.7.2")
  final <- Sys.getenv("MAGMAAN_HARD_PARITY_FINAL")=="1"
  reps <- as.integer(Sys.getenv("MAGMAAN_HARD_PARITY_REPS","100"))
  stopifnot(is.finite(reps),reps>0,reps<=100)
  directory <- path.expand(Sys.getenv("MAGMAAN_HARD_PARITY_OUTPUT",
                                     "~/.cache/magmaan-logs/task-129.4"))
  dir.create(directory,recursive=TRUE,showWarnings=FALSE)
  started <- proc.time()[["elapsed"]]; rows <- list(); cells <- .hard_cells()
  state <- new.env(parent=emptyenv()); state$attempts <- list()
  old <- options(magmaan.hard.trace=state)
  # Read-only exit instrumentation records actual optimizer calls (including
  # retries and marker refits), rather than reconstructing starts after fitting.
  trace("lav_model_est",where=asNamespace("lavaan"),print=FALSE,exit=quote({
    if (exists("optim_out",inherits=FALSE) && exists("start_x",inherits=FALSE) &&
        any(lavpartable$op == "=~")) {
      message <- optim_out$message
      status <- if(length(message) && grepl("\\([0-9]+\\)",message))
        as.integer(sub(".*\\(([0-9]+)\\).*","\\1",message)) else NA_integer_
      recorder <- getOption("magmaan.hard.trace")
      recorder$attempts[[length(recorder$attempts)+1L]] <- list(
        start=start_x,constrained=lavmodel@eq.constraints || lavmodel@ceq.simple.only,
        theta_start=as.numeric(if(lavmodel@eq.constraints)
          lavmodel@eq.constraints.K %*% start_x + lavmodel@eq.constraints.k0 else start_x)/parscale,
        partable=as.data.frame(lavpartable)[lavpartable$free>0,c("lhs","op","rhs","group","free")],
        status=status,message=message,parscale=lavoptions$optim.parscale,
        converged=attr(returnValue(),"converged"),
        gradient=if(is.function(gradient)) gradient(optim_out$par) else numeric(),
        fmin=attr(returnValue(),"fx"),iterations=attr(returnValue(),"iterations"))
    }
  }))
  on.exit({untrace("lav_model_est",where=asNamespace("lavaan")); options(old)},add=TRUE)
  # Round-robin cells ensures every design is represented if the time budget
  # cuts replication short. A single worker leaves the aggregate lane budget free.
  exhausted <- FALSE
  for (replicate in seq_len(reps)) {
    for (cell in seq_len(nrow(cells))) {
      if (proc.time()[["elapsed"]]-started>1740) {exhausted <- TRUE; break}
      row <- .hard_replicate(cells$design[cell],cells$n[cell],12940000L+1000L*cell+replicate,state,final)
      rows[[length(rows)+1L]] <- row
      utils::write.table(row,file=file.path(directory,"replicates.csv"),sep=",",
        row.names=FALSE,col.names=length(rows)==1L,append=length(rows)>1L,qmethod="double")
    }
    cat(sprintf("\nhard parity round %d: %.1f seconds\n",replicate,proc.time()[["elapsed"]]-started))
    if(exhausted) break
  }
  results <- do.call(rbind,rows)
  classes <- c("agree","path_divergence","known_defect","pending_feature","rule_difference")
  summary <- as.data.frame(table(factor(paste(results$design,results$n),
    levels=paste(cells$design,cells$n)),factor(results$class,levels=classes)))
  names(summary) <- c("cell","class","count")
  utils::write.csv(summary,file.path(directory,"summary.csv"),row.names=FALSE)
  elapsed <- proc.time()[["elapsed"]]-started
  writeLines(c(sprintf("elapsed_seconds=%.3f",elapsed),paste0("final=",final),
    paste0("requested_replicates=",reps),paste0("budget_exhausted=",exhausted),
    paste0("revision=",Sys.getenv("MAGMAAN_HARD_PARITY_REVISION","unspecified")),
    paste0("lavaan=",utils::packageVersion("lavaan")),
    paste0("magmaanlab=",utils::packageVersion("magmaanlab"))),file.path(directory,"run.txt"))
  print(summary[summary$count>0,],row.names=FALSE)
  print(head(results[results$class=="rule_difference",c("design","n","seed","reason")],5),row.names=FALSE)
  expect_equal(sum(results$class=="rule_difference"),0L)
  if(final) {
    expect_equal(sum(results$class=="pending_feature"),0L)
    rates <- tapply(results$class=="path_divergence",paste(results$design,results$n),mean)
    expect_true(all(rates<=.05),info=paste(names(rates)[rates>.05],collapse=", "))
  }
})
