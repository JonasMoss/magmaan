geometry_cells <- function() {
  x <- expand.grid(n = c(100L, 300L), role = c('null', 'power'),
    distribution = c('normal', 'skewed'), larger = c('correct', 'mild', 'strong'),
    stringsAsFactors = FALSE)
  x$cell_id <- seq_len(nrow(x)); x$production_reps <- ifelse(x$role == 'null', 2000L, 1000L)
  x
}

population_syntax <- function(cell, group) {
  loading <- rep(c(.7, .8, .75, .65), 2)
  if (cell$role == 'power' && group == 'b') loading[3] <- loading[3] - .15
  residual <- 1 - loading^2
  if (cell$larger == 'strong') residual[5] <- residual[5] - .4^2 - 2*.4*loading[5]*.4
  paste(c(paste0('f1 =~ ', paste0(loading[1:4], '*x', 1:4, collapse = ' + ')),
    paste0('f2 =~ ', paste0(loading[5:8], '*x', 5:8, collapse = ' + ')),
    if (cell$larger == 'strong') 'f1 =~ .4*x5',
    'f1 ~~ 1*f1', 'f2 ~~ 1*f2', 'f1 ~~ .4*f2',
    paste0('x', 1:8, ' ~~ ', residual, '*x', 1:8),
    if (cell$larger != 'correct') paste0('x1 ~~ ', .3 * sqrt(residual[1]*residual[5]), '*x5')),
    collapse = '\n')
}

geometry_draw <- function(cell, rep, seed) {
  started <- proc.time()[['elapsed']]
  rows <- expand.grid(geometry = c('expected', 'observed'), test = c('score', 'lr'),
    stringsAsFactors = FALSE)
  rows <- cbind(cell[rep(1, nrow(rows)), ], rep = rep, seed = seed, rows)
  rows$converged_h0 <- rows$converged_h1 <- FALSE
  for (k in c('statistic','df','p_sb','p_peba4','policy_p_sb','policy_p_peba4','policy_gap')) rows[[k]] <- NA_real_
  rows$error <- ''; rows$elapsed_s <- NA_real_
  finish <- function() { rows$elapsed_s <- proc.time()[['elapsed']] - started; rows }
  tryCatch({
    set.seed(seed)
    data <- do.call(rbind, lapply(c('a', 'b'), function(g) {
      x <- lavaan::simulateData(population_syntax(cell, g), sample.nobs = cell$n,
        skewness = if (cell$distribution == 'skewed') rep(2, 8) else NULL,
        kurtosis = if (cell$distribution == 'skewed') rep(7, 8) else NULL)
      x$group <- g; x
    }))
    syntax <- 'f1 =~ x1 + x2 + x3 + x4\nf2 =~ x5 + x6 + x7 + x8'
    fit <- function(equal = NULL) magmaan::magmaan(magmaan::magmaan_model(syntax,
      prototype = data, group = 'group', group.equal = equal), data, estimator = 'ML')
    h1 <- fit(); h0 <- fit('loadings')
    f1 <- magmaan::as_lab_fit(h1); f0 <- magmaan::as_lab_fit(h0)
    rows$converged_h0 <- isTRUE(f0$converged); rows$converged_h1 <- isTRUE(f1$converged)
    if (!all(rows$converged_h0 & rows$converged_h1)) stop('library convergence verdict failed')
    shared <- magmaanlab::prepare_inference_data(f1)
    hyp <- magmaanlab::prepare_hypothesis(magmaanlab::prepare_inference(f0, shared),
                                        magmaanlab::prepare_inference(f1, shared))
    policy <- anova(h1, h0)
    for (i in seq_len(nrow(rows))) {
      tryCatch({
        q <- magmaanlab::inference_quadratic(hyp, rows$test[i], rows$geometry[i])
        p <- magmaanlab::calibrate_quadratic(q, c('sb', 'peba4'))
        rows$statistic[i] <- q$statistic; rows$df[i] <- q$df
        if (q$df != 6) stop('restriction df differs from registered six')
        rows$p_sb[i] <- p$p_value[1]; rows$p_peba4[i] <- p$p_value[2]
        at <- if (rows$test[i] == 'score') which(policy$test == 'score') else which(policy$test != 'score')
        rows$policy_p_sb[i] <- policy$p.sb[at]; rows$policy_p_peba4[i] <- policy$p.peba4[at]
        if (rows$geometry[i] == 'observed') {
          rows$policy_gap[i] <- max(abs(c(rows$p_sb[i]-rows$policy_p_sb[i],
            rows$p_peba4[i]-rows$policy_p_peba4[i])))
          if (!is.finite(rows$policy_gap[i]) || rows$policy_gap[i] > 1e-7) stop('observed arm disagrees with anova policy')
        }
        if (any(!is.finite(c(rows$statistic[i], rows$p_sb[i], rows$p_peba4[i])))) stop('nonfinite quadratic calibration')
      }, error = function(e) rows$error[i] <<- conditionMessage(e))
    }
  }, error = function(e) rows$error <<- conditionMessage(e))
  finish()
}

geometry_population <- function() {
  do.call(rbind, lapply(c('correct', 'mild', 'strong'), function(level) {
    cell <- data.frame(role='null', larger=level)
    loading <- rep(c(.7,.8,.75,.65),2)
    lambda <- matrix(0,8,2); lambda[1:4,1] <- loading[1:4]
    lambda[5:8,2] <- loading[5:8]
    if (level=='strong') lambda[5,1] <- .4
    phi <- matrix(c(1,.4,.4,1),2)
    common <- lambda %*% phi %*% t(lambda)
    residual <- diag(1-diag(common))
    if (level!='correct') residual[1,5] <- residual[5,1] <-
      .3*sqrt(residual[1,1]*residual[5,5])
    sigma <- common+residual; dimnames(sigma) <- list(paste0('x',1:8),paste0('x',1:8))
    stopifnot(min(eigen(sigma,symmetric=TRUE,only.values=TRUE)$values)>0)
    fit <- lavaan::cfa('f1 =~ x1 + x2 + x3 + x4\nf2 =~ x5 + x6 + x7 + x8',
      sample.cov=list(a=sigma,b=sigma), sample.nobs=c(1e7,1e7),
      sample.mean=list(a=rep(0,8),b=rep(0,8)), meanstructure=TRUE,
      sample.cov.rescale=FALSE)
    if (!lavaan::lavInspect(fit,'converged')) stop('Population fit failed: ',level)
    implied <- lavaan::lavInspect(fit,'cov.ov')
    f <- unname(lavaan::fitMeasures(fit,'fmin'))*2
    df <- unname(lavaan::fitMeasures(fit,'df'))
    data.frame(larger=level, groups=2L, n_per_group=1e7, df=df,
      ml_discrepancy=f, population_rmsea=sqrt(max(0,2*f/df)),
      largest_standardized_residual=max(vapply(implied,function(x)
        max(abs((sigma-x)/sqrt(outer(diag(sigma),diag(sigma))))),numeric(1))))
  }))
}
