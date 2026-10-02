#!/usr/bin/env Rscript
# FIML fitting component oracle, generated from formulas and synthetic data.
# Run from the repository root; installed lavaan 0.7.2 is required.
stopifnot(as.character(utils::packageVersion("lavaan")) == "0.7.2")
stopifnot(gsub("-", ".", trimws(readLines("cpp/tests/fixtures/lavaan_version.txt"))) == "0.7.2")
.fitting_attempts <- list()
trace("lav_model_est", where = asNamespace("lavaan"), print = FALSE,
  exit = quote({
    out <- returnValue()
    .GlobalEnv$.fitting_attempts[[length(.GlobalEnv$.fitting_attempts) + 1L]] <- list(
      simple = start == "simple", standardized = lavoptions$optim.parscale != "none",
      start = as.numeric(attr(out, "start")), parameter_scale = as.numeric(attr(out, "parscale")),
      port_scale = if (exists("scale_1", inherits = FALSE)) as.numeric(scale_1) else numeric(),
      gradient = as.numeric(attr(out, "dx")), theta = as.numeric(out),
      accepted = isTRUE(attr(out, "converged")), iterations = attr(out, "iterations"))
  }))
set.seed(10072)
n <- 160
lambda <- c(1, .8, .6, .9)
x <- outer(rnorm(n), lambda) + matrix(rnorm(n*4, sd = sqrt(.7)), n, 4)
x <- sweep(x, 2, c(.2,.4,.6,.8), "+")
colnames(x) <- paste0("x",1:4)
cases <- list()
for (mar in c(FALSE, TRUE)) for (groups in 1:2) {
  d <- as.data.frame(x)
  for (j in 2:4) {
    p <- if (mar) plogis(-1.2 + .6*d$x1) else rep(.25,n)
    d[runif(n) < p,j] <- NA_real_
  }
  if (groups == 2) d$g <- rep(c("a","b"),each=n/2)
  cases[[length(cases)+1L]] <- list(model="f =~ x1+x2+x3+x4", mechanism=if(mar) "MAR" else "MCAR", data=d, groups=groups)
}
cases <- c(cases, list(
  modifyList(cases[[1]],list(model="f =~ x1+a*x2+a*x3+x4")),
  modifyList(cases[[2]],list(group_equal=c("loadings","intercepts"))),
  modifyList(cases[[1]],list(invalid_start=TRUE)),
  modifyList(cases[[1]],list(rescale=100))))
for (i in seq_along(cases)) {
  case <- cases[[i]]
  d <- case$data
  if (!is.null(case$rescale)) { d$x1 <- d$x1*case$rescale; d$x3 <- d$x3/case$rescale }
  args <- list(model=case$model,data=d,missing="ml",fixed.x=FALSE,meanstructure=TRUE,se="none",test="none")
  if (case$groups == 2) args$group <- "g"
  if (!is.null(case$group_equal)) args$group.equal <- case$group_equal
  if (isTRUE(case$invalid_start)) {
    initial <- do.call(lavaan::sem,c(args,list(do.fit=FALSE)))
    args$start <- rep(0,max(lavaan::parTable(initial)$free))
  }
  .fitting_attempts <- list()
  lv <- suppressWarnings(do.call(lavaan::sem,args))
  pt <- lavaan::parTable(lv)
  blocks <- if (case$groups == 1) list(d) else split(d,d$g)
  case$data <- NULL
  cases[[i]] <- c(case,list(raw=lapply(blocks,function(b) unname(as.matrix(b[paste0("x",1:4)]))),
    parameters=pt[pt$free>0,c("lhs","op","rhs","group","start","est")],
    h1_cov=lapply(lv@SampleStats@missing.h1,function(h) unname(h$sigma)),
    h1_mean=lapply(lv@SampleStats@missing.h1,function(h) unname(h$mu)),
    fmin=as.numeric(lv@optim$fx),converged=isTRUE(lv@optim$converged),
    basis=unname(lv@Model@eq.constraints.K), offset=as.numeric(lv@Model@eq.constraints.k0),
    attempts=.fitting_attempts))
}
untrace("lav_model_est",where=asNamespace("lavaan"))
jsonlite::write_json(list(source="regen_fiml_fitting_options.R; seed 10072; synthetic Gaussian CFA; MCAR/MAR",
  lavaan_version="0.7.2",cases=cases),"cpp/tests/fixtures/fitting/lavaan_fiml_0_7_2.json",
  auto_unbox=TRUE,digits=17,pretty=TRUE,na="null")
