data_spec <- function(data = "", variable = "", model = "f BY y1-y3;") {
  mplus_model(paste0("DATA: ", gsub(";", ";\n", data, fixed=TRUE), "\nVARIABLE: NAMES=y1 y2 y3;\n",
    variable,"\nMODEL: ",model,"\n"))
}
with_dat <- function(lines, spec, check) {
  path <- tempfile(fileext=".dat"); on.exit(unlink(path))
  writeLines(lines,path); check(mplus_data(spec,file=path))
}
test_that("free format wrapping, extra fields and sample reports follow Mplus", {
  spec <- data_spec("NOBSERVATIONS=2;")
  with_dat(c("1,2", "3,99", "4\t5 6 77", "7 8 9"),spec,function(x) {
    expect_equal(x[,1:3],data.frame(y1=c(1,4),y2=c(2,5),y3=c(3,6)),ignore_attr=TRUE)
    expect_equal(attr(x,"mplus_data_report")$read_n,2L)
  })
  path <- tempfile();on.exit(unlink(path));writeLines("1,,3",path)
  expect_error(mplus_data(data_spec(),file=path),"empty comma")
  writeLines("1 2",path);expect_error(mplus_data(data_spec(),file=path),"incomplete")
  writeLines("1 q 3",path);expect_error(mplus_data(data_spec(),file=path),"non-numeric")
})
test_that("fixed fields implement decimals, skips, tabs and record breaks", {
  spec <- data_spec("FORMAT=(2(F2.1,1X),T9,3.0);")
  with_dat(c("12x34x  567", "99x-9x  123"),spec,function(x)
    expect_equal(unname(as.matrix(x)),rbind(c(1.2,3.4,567),c(9.9,-.9,123)),ignore_attr=TRUE))
  spec <- data_spec("FORMAT=(F3.1,/,2F3.1);")
  with_dat(c("1.2", "034056", "078", "9.0012"),spec,function(x)
    expect_equal(unname(as.matrix(x)),rbind(c(1.2,3.4,5.6),c(7.8,9,1.2)),ignore_attr=TRUE))
})
test_that("missing symbols and numeric rules compare scaled values", {
  for(symbol in c("*",".")) with_dat(c(paste(symbol,"2 3"),"4 5 6"),
    data_spec(variable=paste0("MISSING=",symbol,";")), function(x) expect_true(is.na(x$y1[1])))
  with_dat(c("   0203","040506"),data_spec("FORMAT=3F2.0;", "MISSING=BLANK;"),
    function(x) expect_true(is.na(x$y1[1])))
  with_dat(c("99-910", "111213"),data_spec("FORMAT=3F2.1;", "MISSING=y1 (9.9) y2 (-.9);"),
    function(x) {expect_true(is.na(x$y1[1]));expect_true(is.na(x$y2[1]));expect_equal(x$y3[1],1)})
  with_dat(c("-9 -9.0 -7", "1 2 3"),data_spec(variable="MISSING=ALL (-9,-7--5);"),
    function(x) expect_true(all(is.na(x[1,]))))
})
test_that("summary formats retain covariance and means and convert correlation", {
  S <- matrix(c(4,1,2,1,9,3,2,3,16),3,3,dimnames=list(paste0("y",1:3),paste0("y",1:3)))
  for(type in c("COVA","FULLCOV")) {
    lines <- if(type=="COVA") c("4","1 9","2 3 16") else c("4 1 2","1 9 3","2 3 16")
    with_dat(c("1 2 3",lines),data_spec(paste0("TYPE=",type," MEANS;NOBSERVATIONS=50;")),function(x) {
      expect_equal(x$S[[1]],S);expect_equal(unname(x$mean[[1]]),c(1,2,3));expect_equal(x$nobs,50L)
    })
  }
  for(type in c("CORR","FULLCORR")) {
    lines <- if(type=="CORR") c("1","0.5 1","0.25 0.5 1") else c("1 .5 .25",".5 1 .5",".25 .5 1")
    with_dat(c("2 3 4",lines),data_spec(paste0("TYPE=",type," STD;NOBSERVATIONS=50;")),function(x) {
      expect_equal(unname(x$S[[1]]),matrix(c(4,3,2,3,9,6,2,6,16),3,3));expect_false(any(x$mean[[1]]!=0))
    })
    with_dat(lines,data_spec(paste0("TYPE=",type,";NOBSERVATIONS=50;")),function(x) {
      expect_equal(diag(x$S[[1]]),setNames(rep(1,3),paste0("y",1:3)))
      expect_match(tail(attr(x,"mplus_data_report")$notes,1),"unit variances")
    })
  }
  with_dat(c("4","1 9","2 3 16","4","1 9","2 3 16"),data_spec("TYPE=COVA;NGROUPS=2;NOBSERVATIONS=50 60;"),function(x) {
    expect_equal(x$S,list(S,S));expect_equal(x$nobs,c(50L,60L))
  })
})
test_that("FILE groups and GROUPING codes preserve order and report exclusions", {
  paths <- c(tempfile(),tempfile());on.exit(unlink(paths))
  writeLines(c("1 2 3","4 5 6"),paths[1]);writeLines(c("7 8 9","10 11 12"),paths[2])
  spec <- data_spec("FILE (b)=b.dat; FILE (a)=a.dat;")
  x <- mplus_data(spec,file=paths)
  expect_equal(x$.mplus_group,c("b","b","a","a"));expect_equal(spec$group_labels,c("b","a"))
  expect_error(mplus_data(spec,file=paths[1]),"one data file")
  spec <- mplus_model("DATA: FILE=x;\nVARIABLE: NAMES=y1 y2 y3 g;\nGROUPING=g(1=a 2=b);\nMODEL: f BY y1-y3;\n")
  writeLines(c("1 2 3 2","4 5 6 9","7 8 9 1"),paths[1])
  x <- mplus_data(spec,file=paths[1]);expect_equal(x$g,c(2,1))
  expect_equal(unname(attr(x,"mplus_data_report")$dropped_group_codes),structure(1L,dim=1L,dimnames=list("9"),class="table"),ignore_attr=TRUE)
})
test_that("input-directory path fallback and USEVARIABLES follow the plan", {
  dir <- tempfile();dir.create(dir);on.exit(unlink(dir,recursive=TRUE))
  writeLines("1 2 3",file.path(dir,"data.dat"))
  inp <- "DATA: FILE=data.dat;\nVARIABLE: NAMES=y1 y2 y3;\nUSEVARIABLES=y1 y3;\nMODEL: y3 ON y1;\n"
  writeLines(inp,file.path(dir,"input.inp"))
  x <- mplus_data(mplus_model(file=file.path(dir,"input.inp")))
  expect_equal(names(x),c("y1","y3"));expect_equal(x$y3,3)
})

test_that("summary moments fit on Mplus N scaling against explicit lavaan covariance", {
  skip_if_not_installed("lavaan")
  S <- matrix(c(4,1,2,1,9,3,2,3,16),3,3,dimnames=list(paste0("y",1:3),paste0("y",1:3)))
  spec <- data_spec("TYPE=COVA;NOBSERVATIONS=200;")
  with_dat(c("4","1 9","2 3 16"),spec,function(data) {
    fit <- fit_model(spec,data,estimator="ML")
    ref <- lavaan::lavaan("f =~ 1*y1+y2+y3; f ~~ f; y1 ~~ y1; y2 ~~ y2; y3 ~~ y3",
      sample.cov=S,sample.nobs=200,sample.cov.rescale=FALSE,meanstructure=FALSE,
      auto.var=FALSE,auto.cov.lv.x=FALSE,auto.fix.first=FALSE)
    expected <- lavaan::parTable(ref)
    key <- function(pt) paste(pt$lhs,pt$op,pt$rhs)
    expect_equal(fit$partable$est,expected$est[match(key(fit$partable),key(expected))],tolerance=1e-5)
    expect_equal(lavaan::lavInspect(ref,"sampstat")$cov,S,ignore_attr=TRUE)
  })
})

test_that("reader rejects file and summary shape errors and reports LISTWISE", {
  expect_error(mplus_data(data_spec(),file=tempfile()),"file not found")
  path <- tempfile();on.exit(unlink(path));writeLines("1 2",path)
  expect_error(mplus_data(data_spec("TYPE=COVA;NOBSERVATIONS=20;"),file=path),"wrong number")
  writeLines(c("* 2 3","4 5 6"),path)
  x <- mplus_data(data_spec("LISTWISE=ON;", "MISSING=*;"),file=path)
  expect_equal(nrow(x),2L)
  expect_true(any(grepl("LISTWISE = ON",attr(x,"mplus_data_report")$notes,fixed=TRUE)))
  with_dat("99-910",data_spec("FORMAT=3F2.1;", "MISSING=ALL (99);"),function(x) expect_equal(x$y1,9.9))
  with_dat("99 98 97",data_spec(variable="MISSING=ALL (98 99);"),function(x) expect_true(all(is.na(x[1,1:2]))))
})
