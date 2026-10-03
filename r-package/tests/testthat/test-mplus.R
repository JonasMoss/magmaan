mplus_cases <- list(
  list(names = "x1 x2 x3", model = "F BY X1 X2 X3;",
       lavaan = "F =~ 1*x1 + x2 + x3; F ~~ F; F ~ 0*1"),
  list(names = "x1 x2 x3 ageyr", model = "F BY x1-x3; F ON ageyr;",
       lavaan = "F =~ 1*x1 + x2 + x3; F ~ ageyr; F ~~ F; F ~ 0*1"),
  list(names = "x1 x2 x3", model = "x2 x3 ON x1;",
       lavaan = "x2 ~ x1; x3 ~ x1; x2 ~~ x3"),
  list(names = "x1 x2 x3", model = "F BY x1 x2*0.8 x3 (1);\nx2 x3 (v);",
       lavaan = "F =~ 1*x1 + start(.8)*x2 + .eq1.*x2 + .eq1.*x3; F ~~ F; F ~ 0*1; x2 ~~ v*x2; x3 ~~ v*x3")
)
mplus_input <- function(case) paste0("TITLE: lab test\nDATA: FILE=x;\nVARIABLE: NAMES=", case$names,
    ";\nMODEL: ", case$model, "\nOUTPUT: TECH1;\n")
mplus_reference <- function(case, data, ...) {
  observed <- strsplit(case$names, " ")[[1]]
  # Explicit residual variances and means; x moments are supplied by fixed.x.
  ys <- if (grepl("age", case$names)) observed[1:3] else if (grepl("x2 x3 ON", case$model)) observed[2:3] else observed
  syntax <- paste(case$lavaan, paste(ys, "~~", ys, collapse="; "),
                  paste(ys, "~ 1", collapse="; "), sep="; ")
  lavaan::lavaan(syntax, data=data, fixed.x=TRUE, meanstructure=TRUE,
    auto.var=FALSE, auto.cov.lv.x=FALSE, auto.cov.y=FALSE,
    auto.fix.first=FALSE, auto.fix.single=FALSE, information="expected", ...)
}
mplus_key <- function(pt) {
  lhs <- pt$lhs; rhs <- pt$rhs
  swap <- pt$op == "~~" & lhs > rhs
  lhs[swap] <- pt$rhs[swap]; rhs[swap] <- pt$lhs[swap]
  paste(lhs, pt$op, rhs)
}

test_that("Mplus rows and ML inference match independent live lavaan models", {
  skip_if_not_installed("lavaan")
  d <- lavaan::HolzingerSwineford1939
  for (case in mplus_cases) {
    m <- mplus_model(mplus_input(case))
    ref <- mplus_reference(case, d)
    pt <- lavaan::parTable(ref)
    # Generated equality rows use row-number plabels, which depend on ordering.
    expect_equal(sum(m$partable$op == "=="), sum(pt$op == "=="))
    pt <- pt[pt$op != "==", ]
    rows <- m$partable[m$partable$op != "==", ]
    idx <- match(mplus_key(rows), mplus_key(pt))
    expect_false(anyNA(idx))
    expect_equal(rows$free > 0, pt$free[idx] > 0)
    expect_equal(rows$ustart, pt$ustart[idx], tolerance=1e-12)
    fit <- fit_model(m, d, estimator="ML")
    expect_true(fit$converged)
    idx <- match(mplus_key(pt), mplus_key(fit$partable))
    expect_false(anyNA(idx))
    # Conditioned x moments are sample quantities, not fitted parameters.
    fitted <- pt$free > 0 | is.finite(pt$ustart)
    expect_equal(fit$partable$est[idx[fitted]], pt$est[fitted], tolerance=1e-5)
    se <- magmaan_core$inference_se(magmaan_core$inference_vcov(magmaan_core$inference_information_expected(fit), fit))
    free <- pt$free > 0
    expect_equal(se[fit$partable$free[idx[free]]], pt$se[free], tolerance=1e-5, ignore_attr=TRUE)
    measures <- fit_measures(fit)
    expect_equal(measures$chisq, unname(lavaan::fitMeasures(ref,"chisq")), tolerance=1e-5)
    expect_equal(measures$df, unname(lavaan::fitMeasures(ref,"df")))
    prepared <- prepare_model(m)
    reused <- estimate(prepared, prepare_data(prepared,d))
    expect_equal(reused$partable$est, fit$partable$est, tolerance=1e-5)
    rebuilt <- magmaanlab:::.rebuild_model_spec(m)
    expect_identical(rebuilt$mplus_source, m$mplus_source)
    expect_equal(fit_model(rebuilt,d)$partable$est, fit$partable$est, tolerance=1e-5)
  }
})

test_that("Mplus inputs retain notes, spelling, starts and serialize across processes", {
  skip_if_not_installed("lavaan")
  case <- mplus_cases[[1]]
  case$names <- "X1 X2 X3"
  m <- mplus_model(mplus_input(case))
  d <- lavaan::HolzingerSwineford1939
  names(d)[match(c("x1","x2","x3"), names(d))] <- c("X1","X2","X3")
  fit <- fit_model(m,d)
  expect_true(all(c("X1","X2","X3") %in% m$partable$lhs))
  expect_named(m$mplus_notes, c("class","rule","line","col","message"))
  expect_output(print(m), "reported but not imported; see \\$mplus_notes")
  inp <- tempfile(fileext=".inp"); writeLines(m$mplus_source, inp)
  expect_equal(mplus_model(file=inp)$partable, m$partable)
  expect_identical(mplus_model(file=inp)$mplus_source, paste0(m$mplus_source, "\n"))
  saved <- tempfile(fileext=".rds"); result <- tempfile(fileext=".rds")
  saveRDS(list(spec=m, fit=fit, data=d), saved)
  script <- tempfile(fileext=".R")
  writeLines(c("library(magmaanlab)",
    sprintf("x <- readRDS(%s)", deparse(saved)),
    "a <- fit_model(x$spec,x$data)",
    "b <- fit_model(x$fit$model,x$data)",
    sprintf("saveRDS(list(a=a$partable$est,b=b$partable$est),%s)", deparse(result))), script)
  status <- system2(file.path(R.home("bin"),"Rscript"), shQuote(script), stdout=TRUE, stderr=TRUE)
  expect_null(attr(status,"status"))
  got <- readRDS(result)
  expect_equal(got$a,fit$partable$est,tolerance=1e-5)
  expect_equal(got$b,fit$partable$est,tolerance=1e-5)
})

test_that("Mplus rejects unsupported inputs and construction overrides explicitly", {
  base <- mplus_input(mplus_cases[[1]])
  for (item in list(c(paste0(base,"DEFINE: x1=2;\n"),"CL16"),
                    c(sub("NAMES=", "GROUPING=g(1=a 2=b); NAMES=",base),"MG03"),
                    c(sub("F BY X1 X2 X3;", "x2 ON x1; x1;",base),"MS08"),
                    c(paste0(base,"ANALYSIS: MODEL=NOMEANSTRUCTURE;\n"),"MS11"),
                    c(sub("x1 x2 x3", "a1b-a3b",base),"NM02"))) {
    expect_error(mplus_model(item[1]), item[2])
    expect_error(mplus_model(item[1]), "instead|remove|write|use", ignore.case=TRUE)
  }
  expect_error(mplus_model(), "exactly one")
  expect_error(mplus_model(base,file="x"), "exactly one")
  m <- mplus_model(base)
  expect_error(magmaanlab:::.rebuild_model_spec(m, overrides=list(fixed_x=FALSE)), "cannot be overridden")
  expect_error(magmaanlab:::.rebuild_model_spec(m, group="g"), "edit mplus_source")
})

test_that("Mplus multiple groups preserve numeric order, topology and live lavaan fits", {
  skip_if_not_installed("lavaan")
  set.seed(5202)
  n <- 600L; f <- rnorm(2*n)
  d <- data.frame(y1=f+.8*rnorm(2*n),y2=.8*f+.8*rnorm(2*n),
    y3=.7*f+.8*rnorm(2*n),y4=.9*f+.8*rnorm(2*n),G=rep(c(2,1),each=n))
  input <- paste("DATA: FILE=x;", "VARIABLE: NAMES=y1 y2 y3 y4 G;",
    "GROUPING=g(2=b 1=a);", "MODEL: f BY y1-y4;",
    "MODEL b: f BY y2*0.8;", "[y3]; y1 WITH y2;",sep="\n")
  s <- mplus_model(input)
  expect_identical(s$group_var,"G")
  expect_identical(s$group_labels,c("1","2"))
  expect_equal(s$mplus_groups,data.frame(label=c("a","b"),code=c("1","2")))
  expect_output(print(s),"a = 1, b = 2")
  zero <- s$partable$group==1 & s$partable$lhs=="y1" & s$partable$op=="~~" & s$partable$rhs=="y2"
  if (!any(zero)) zero <- s$partable$group==1 & s$partable$lhs=="y2" & s$partable$op=="~~" & s$partable$rhs=="y1"
  expect_equal(s$partable$user[zero],0L)
  expect_equal(s$partable$ustart[zero],0)
  expect_equal(fit_model(s,d,groups="G")$partable$group,s$partable$group)
  projection <- lavaan::lavaanify(s$syntax, ngroups=2, meanstructure=TRUE,
    auto.var=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,auto.fix.first=FALSE)
  expect_equal(projection$ustart[projection$lhs=="f" & projection$op=="=~" & projection$rhs=="y2" & projection$group==2],.8)
  expect_true(projection$free[projection$lhs=="f" & projection$op=="=~" & projection$rhs=="y2" & projection$group==2]>0)
  syntax <- paste("f =~ c(1,1)*y1 + c(l2,NA)*y2 + c(l3,l3)*y3 + c(l4,l4)*y4",
    "f ~~ c(NA,NA)*f; f ~ c(0,NA)*1",
    "y1 ~~ c(NA,NA)*y1 + c(0,NA)*y2; y2 ~~ c(NA,NA)*y2",
    "y3 ~~ c(NA,NA)*y3; y4 ~~ c(NA,NA)*y4",
    "y1 ~ c(i1,i1)*1; y2 ~ c(i2,i2)*1; y3 ~ c(i3,NA)*1; y4 ~ c(i4,i4)*1",sep="\n")
  ref <- lavaan::lavaan(syntax,data=d,group="G",group.label=c("1","2"),
    meanstructure=TRUE,fixed.x=TRUE,auto.var=FALSE,auto.cov.lv.x=FALSE,
    auto.cov.y=FALSE,auto.fix.first=FALSE,auto.fix.single=FALSE,information="expected")
  pt <- lavaan::parTable(ref); a <- fit_model(s,d)
  keep <- a$partable$op!="=="
  key <- function(x) paste(x$group,mplus_key(x))
  idx <- match(key(a$partable[keep,]),key(pt))
  expect_false(anyNA(idx))
  expect_equal(a$partable$free[keep]>0,pt$free[idx]>0)
  expect_equal(a$partable$est[keep],pt$est[idx],tolerance=1e-5)
  got <- magmaan_core$inference_se(magmaan_core$inference_vcov(magmaan_core$inference_information_expected(a), a))
  free <- a$partable$free[keep]>0
  expect_equal(got[a$partable$free[keep][free]],pt$se[idx][free],tolerance=1e-5,ignore_attr=TRUE)
  expect_equal(fit_measures(a)$chisq,unname(lavaan::fitMeasures(ref,"chisq")),tolerance=1e-5)
  b <- fit_model(s,d[nrow(d):1,])
  expect_equal(a$partable$est[keep],b$partable$est[keep],tolerance=1e-5)
  rebuilt <- magmaanlab:::.rebuild_model_spec(s,group="G",group_labels=c("1","2"))
  expect_equal(rebuilt$partable,s$partable)
  path <- tempfile(fileext=".rds");on.exit(unlink(path),add=TRUE);saveRDS(s,path)
  c <- fit_model(readRDS(path),d)
  expect_equal(c$partable$est[keep],a$partable$est[keep],tolerance=1e-5)
  expect_error(magmaanlab:::.rebuild_model_spec(s,group="other"),"MG03.*edit mplus_source")
  expect_error(magmaanlab:::.rebuild_model_spec(s,group_labels=c("2","1")),"MG03.*edit mplus_source")
  d$G[1:3] <- 3
  expect_error(fit_model(s,d),"MG03.*3 \\(3 rows\\).*Mplus drops.*filter")
})

test_that("categorical Mplus models preserve data-driven thresholds and live lavaan inference", {
  skip_if_not_installed("lavaan")
  data <- lavaan::HolzingerSwineford1939
  variables <- paste0("x",1:6)
  for (categories in c(2L,3L)) for (parameterization in c("delta","theta")) {
    d <- data
    for(v in variables) d[[v]] <- as.integer(cut(d[[v]],categories))
    input <- paste0("DATA: FILE=x;\nVARIABLE: NAMES=x1-x6; CATEGORICAL=x1-x6;\n",
      "ANALYSIS: PARAMETERIZATION=",toupper(parameterization),";\nMODEL: f1 BY x1-x3; f2 BY x4-x6;")
    spec <- mplus_model(input)
    expect_identical(spec$ordered,variables)
    expect_identical(spec$parameterization,parameterization)
    expect_identical(magmaanlab:::.rebuild_model_spec(spec)$partable,spec$partable)
    file <- tempfile();saveRDS(spec,file);expect_identical(readRDS(file),spec);unlink(file)
    expect_output(print(spec),"Categorical")
    syntax <- c("f1 =~ 1*x1+x2+x3; f2 =~ 1*x4+x5+x6",
      "f1 ~~ f1; f2 ~~ f2; f1 ~~ f2; f1 ~ 0*1; f2 ~ 0*1")
    for(v in variables) syntax <- c(syntax,paste0(v," | ",paste0("t",seq_len(categories-1L),collapse="+")),
      paste0(v," ~ 0*1"),paste0(v,if(parameterization=="theta") " ~~ 1*" else " ~*~ 1*",v))
    actual <- fit_model(spec,d,estimator="DWLS")
    oracle <- lavaan::lavaan(paste(syntax,collapse="\n"),data=d,ordered=variables,
      parameterization=parameterization,estimator="WLSMV",meanstructure=TRUE,
      auto.var=FALSE,auto.fix.first=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE)
    expect_true(actual$converged);expect_true(lavaan::lavInspect(oracle,"converged"))
    p <- actual$partable;q <- lavaan::parTable(oracle)
    key <- function(p) paste(p$lhs,p$op,p$rhs,p$group)
    index <- match(key(p),key(q));use <- !is.na(index)&p$op %in% c("=~","|","~1","~*~")
    expect_equal(p$est[use],q$est[index[use]],tolerance=1e-5)
    inference <- convention_inference(actual,"WLSMV")
    expect_true(inference$covariance_available);expect_true(inference$test$available)
    expect_equal(inference$test$df,unname(lavaan::fitMeasures(oracle,"df")))
    expect_equal(inference$test$statistic,unname(lavaan::fitMeasures(oracle,"chisq.scaled")),tolerance=1e-5)
    use <- !is.na(index)&p$free>0&p$op %in% c("=~","|")
    expect_equal(sqrt(diag(inference$covariance))[p$free[use]],q$se[index[use]],tolerance=1e-5)
    expect_error(fit_model(spec,d,estimator="ML"),"categorical Mplus fit route.*DWLS")
  }
})

test_that("categorical Mplus boundaries identify the fit route and category rule", {
  single <- mplus_model("DATA: FILE=x;\nVARIABLE: NAMES=u1-u3; CATEGORICAL=u1-u3;\nMODEL: f BY u1-u3;")
  d <- data.frame(u1=rep(1:11,3),u2=rep(1:3,11),u3=rep(1:3,11))
  expect_error(fit_model(single,d,estimator="DWLS"),"CT01.*ten categories")
  mixed <- mplus_model("DATA: FILE=x;\nVARIABLE: NAMES=u1-u3 x; CATEGORICAL=u1-u3;\nMODEL: f BY u1-u3 x;")
  d$u1 <- rep(1:3,11);d$x <- seq_len(nrow(d))
  expect_error(fit_model(mixed,d,estimator="DWLS"),"mixed categorical Mplus fit route.*mixed WLSMV")
  conditional <- mplus_model("DATA: FILE=x;\nVARIABLE: NAMES=u1-u3 x; CATEGORICAL=u1-u3;\nMODEL: f BY u1-u3; f ON x;")
  expect_error(fit_model(conditional,d,estimator="DWLS"),"conditional categorical Mplus fit route.*conditional WLSMV")
  grouped <- mplus_model("DATA: FILE=x;\nVARIABLE: NAMES=u1-u3 g; CATEGORICAL=u1-u3; GROUPING=g(1=a 2=b);\nMODEL: f BY u1-u3;")
  d$g <- rep(1:2,length.out=nrow(d));d$u1[d$g==2]<-1
  expect_error(fit_model(grouped,d,estimator="DWLS"),"CT07.*lacks a category")
})
