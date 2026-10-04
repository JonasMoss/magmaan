test_that("Mplus model families preserve portable partables and refit in a fresh process", {
  set.seed(560071)
  n <- 400L
  eta <- rnorm(n)
  continuous <- data.frame(y1=eta+rnorm(n,sd=.6),
    y2=.8*eta+rnorm(n,sd=.7),y3=.7*eta+rnorm(n,sd=.8))
  grouped <- continuous; grouped$g <- rep(1:2,each=n/2)
  categorical <- as.data.frame(lapply(continuous,function(x)
    as.integer(cut(x,c(-Inf,-.5,.5,Inf)))))
  names(categorical) <- paste0("u",1:3)
  i <- rnorm(n,2,.8); s <- rnorm(n,.4,.3)
  growth <- as.data.frame(sapply(0:3,function(t) i+t*s+rnorm(n,sd=.5)))
  names(growth) <- paste0("y",1:4)
  x <- rnorm(n); m <- .5*x+rnorm(n); y <- .3*x+.6*m+rnorm(n)
  paths <- data.frame(x=x,m=m,y=y)
  base <- function(names, model, variable="", extra="")
    paste0("DATA: FILE=x;\nVARIABLE: NAMES=",names,";\n",variable,
      "\nMODEL: ",model,"\n",extra)
  cases <- list(
    single=list(input=base("y1-y3","f BY y1-y3;"),data=continuous,estimator="ML"),
    groups=list(input=base("y1-y3 g","f BY y1-y3;","GROUPING=g(1=a 2=b);"),
      data=grouped,estimator="ML"),
    categorical=list(input=base("u1-u3","f BY u1-u3;","CATEGORICAL=u1-u3;"),
      data=categorical,estimator="DWLS"),
    growth=list(input=base("y1-y4","i s | y1@0 y2@1 y3@2 y4@3;"),
      data=growth,estimator="ML"),
    constraint=list(input=base("x m y","m ON x (a);\ny ON x (b);\ny ON m;",
      extra="MODEL CONSTRAINT: NEW(c); b=a+c;"),data=paths,estimator="ML"),
    indirect=list(input=base("x m y","m ON x; y ON m x;",
      extra="MODEL INDIRECT: y IND x;"),data=paths,estimator="ML"))
  directory <- tempfile("mplus_roundtrip_");dir.create(directory)
  on.exit(unlink(directory,recursive=TRUE),add=TRUE)
  datafile <- file.path(directory,"observations.dat")
  write.table(continuous,datafile,row.names=FALSE,col.names=FALSE,quote=FALSE)
  inputfile <- file.path(directory,"model.inp")
  writeLines(sub("FILE=x", "FILE=observations.dat",cases$single$input,fixed=TRUE),inputfile)
  file_spec <- mplus_model(file=inputfile)
  cases$data_file <- list(spec=file_spec,data=mplus_data(file_spec),estimator="ML")
  saved <- list()
  for (id in names(cases)) {
    case <- cases[[id]]
    spec <- if(is.null(case$spec)) mplus_model(case$input) else case$spec
    fit <- fit_model(spec,case$data,estimator=case$estimator)
    expect_true(fit$converged,info=id)
    # The portable table is a projection; rebuilding must keep source defaults,
    # category metadata and auxiliary NEW rows as well as the parameter values.
    portable <- as_magmaan_model_spec(spec$partable)
    expect_equal(portable$partable,spec$partable,info=id)
    # The row projection carries restrictions; expression text is also needed
    # for derived-parameter reporting. Compare keys after category completion,
    # since a generic partable route can append thresholds in another order.
    portable$syntax <- spec$syntax
    from_table <- fit_model(portable,case$data,estimator=case$estimator)
    key <- function(pt) paste(pt$group,pt$lhs,pt$op,pt$rhs)
    index <- match(key(fit$partable),key(from_table$partable))
    expect_false(anyNA(index),info=id)
    expect_equal(from_table$partable$est[index],fit$partable$est,tolerance=1e-5,info=id)
    rebuilt <- magmaanlab:::.rebuild_model_spec(spec)
    expect_identical(rebuilt$mplus_source,spec$mplus_source,info=id)
    expect_identical(rebuilt$partable,spec$partable,info=id)
    refit <- fit_model(rebuilt,case$data,estimator=case$estimator)
    expect_equal(refit$partable$est,fit$partable$est,tolerance=1e-5,info=id)
    prepared <- prepare_model(spec,prototype=case$data)
    reused <- estimate(prepared,prepare_data(prepared,case$data),estimator=case$estimator)
    expect_equal(reused$partable$est,fit$partable$est,tolerance=1e-5,info=id)
    saved[[id]] <- list(spec=spec,fit=fit,data=case$data,estimator=case$estimator)
  }
  bundle <- file.path(directory,"bundle.rds");saveRDS(saved,bundle)
  result <- file.path(directory,"result.rds")
  script <- file.path(directory,"worker.R")
  writeLines(c("library(magmaanlab)",sprintf("cases <- readRDS(%s)",deparse(bundle)),
    "results <- lapply(cases,function(x) {",
    "  rebuilt <- magmaanlab:::.rebuild_model_spec(x$spec)",
    "  data <- if (!is.null(x$spec$mplus_input_dir)) mplus_data(x$spec) else x$data",
    "  a <- fit_model(rebuilt,data,estimator=x$estimator)",
    "  b <- fit_model(x$fit$model,data,estimator=x$estimator)",
    "  list(partable=rebuilt$partable,a=a$partable$est,b=b$partable$est,",
    "       converged=a$converged && b$converged)",
    "})",sprintf("saveRDS(results,%s)",deparse(result))),script)
  output <- system2(file.path(R.home("bin"),"Rscript"),shQuote(script),stdout=TRUE,stderr=TRUE)
  expect_null(attr(output,"status"),info=paste(output,collapse="\n"))
  if(!file.exists(result)) return(invisible(NULL))
  restored <- readRDS(result)
  for(id in names(saved)) {
    expect_true(restored[[id]]$converged,info=id)
    expect_identical(restored[[id]]$partable,saved[[id]]$spec$partable,info=id)
    expect_equal(restored[[id]]$a,saved[[id]]$fit$partable$est,tolerance=1e-5,info=id)
    expect_equal(restored[[id]]$b,saved[[id]]$fit$partable$est,tolerance=1e-5,info=id)
  }
})
