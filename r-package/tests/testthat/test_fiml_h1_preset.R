# Literal D5 population; keep the frozen C++ fixture and live gate independent.
test_that("preset FIML H1 and first starts match lavaan 0.7.2 at small N", {
  skip_if_not_installed("lavaan")
  skip_if(as.character(utils::packageVersion("lavaan")) != "0.7.2")
  model <- "Y =~ y1+y2+y3\nX =~ x1+x2+x3\nY ~ X"
  probe <- getFromNamespace("prepared_fiml_h1_impl", "magmaanlab")
  key <- function(pt) paste(pt$lhs, pt$op, pt$rhs, pt$group, sep = "\r")
  close <- function(x, y) {
    expect_equal(length(x), length(y))
    expect_true(all(abs(x-y) <= 1e-10*pmax(1,abs(y))))
  }
  ns <- asNamespace("lavaan")
  em_result <- new.env(parent = emptyenv())
  assign(".magmaan_h1_em_probe", em_result, envir = .GlobalEnv)
  trace("lav_em_squarem", where = ns, print = FALSE,
        exit = quote(.GlobalEnv$.magmaan_h1_em_probe$result <- returnValue()))
  on.exit({
    untrace("lav_em_squarem", where = ns)
    rm(".magmaan_h1_em_probe", envir = .GlobalEnv)
  }, add = TRUE)
  for (seed in c(12953003L, 12953002L, 12954001L)) {
    n <- if (seed == 12954001L) 40L else 20L
    set.seed(seed)
    x <- rnorm(n); y <- .25*x+rnorm(n)
    d <- as.data.frame(cbind(outer(y,c(1,.8,.6))+matrix(rnorm(n*3),n),
                             outer(x,c(1,.8,.6))+matrix(rnorm(n*3),n)))
    names(d) <- c(paste0("y",1:3),paste0("x",1:3))
    for (v in names(d)) d[runif(n)<.2,v] <- NA_real_
    prepared <- prepare_model(model, meanstructure = TRUE, fixed_x = FALSE)
    data <- prepare_data(prepared, d, kind = "raw")
    actual <- suppressWarnings(probe(prepared$native, data$native))
    lv <- suppressWarnings(lavaan::sem(model,d,missing="ml",meanstructure=TRUE,
      fixed.x=FALSE,se="none",test="none",do.fit=FALSE))
    oracle <- lavaan::lavInspect(lv,"h1")
    oracle_converged <- em_result$result$converged
    expect_identical(as.logical(actual$converged), oracle_converged)
    expect_true(actual$lavaan_covariance_ridge)
    expect_gt(actual$iterations[[1]], 0L)
    if (oracle_converged) {
      close(actual$mean[[1]], oracle$mean)
      close(actual$cov[[1]], oracle$cov)
    } else {
      # Observed covariance gap 1.1e-7; stalled lavaan call forms differ 1.8e-4.
      expect_gt(actual$covariance_repairs[[1]], 0L)
    }
    fit <- suppressWarnings(fit_model(model,d,estimator="FIML",meanstructure=TRUE,
      fixed_x=FALSE,options=list(preset="lavaan-0.7.2")))
    expect_equal(fit$fitting$h1, actual[c("converged", "iterations",
      "covariance_repairs", "lavaan_covariance_ridge")])
    if (!oracle_converged) next
    mp <- fit$partable; lp <- lavaan::parTable(lv)
    free <- mp$free > 0L
    expected <- lp$start[match(key(mp[free,]),key(lp))]
    close(fit$fitting$attempts[[1]]$start[mp$free[free]], expected)
  }
})
