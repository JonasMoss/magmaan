mplus_growth_key <- function(pt) {
  lhs<-pt$lhs;rhs<-pt$rhs;swap<-pt$op=="~~" & lhs>rhs
  lhs[swap]<-pt$rhs[swap];rhs[swap]<-pt$lhs[swap]
  paste(lhs,pt$op,rhs)
}
mplus_growth_cases <- list(
  linear=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 y4 y5;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: i s | y1@0 y2@1 y3@2 y4@3 y5@4;", reference="i =~ 1*y1+1*y2+1*y3+1*y4+1*y5\ns =~ 0*y1+1*y2+2*y3+3*y4+4*y5\ny1 ~ 0*1;y2 ~ 0*1;y3 ~ 0*1;y4 ~ 0*1;y5 ~ 0*1\ny1 ~~ y1;y2 ~~ y2;y3 ~~ y3;y4 ~~ y4;y5 ~~ y5\ni ~ 1;s ~ 1\ni ~~ i+s;s ~~ s"),
  quadratic=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 y4 y5;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: i s q | y1@0 y2@1 y3@2 y4@3 y5@4;", reference="i =~ 1*y1+1*y2+1*y3+1*y4+1*y5\ns =~ 0*y1+1*y2+2*y3+3*y4+4*y5\nq =~ 0*y1+1*y2+4*y3+9*y4+16*y5\ny1 ~ 0*1;y2 ~ 0*1;y3 ~ 0*1;y4 ~ 0*1;y5 ~ 0*1\ny1 ~~ y1;y2 ~~ y2;y3 ~~ y3;y4 ~~ y4;y5 ~~ y5\ni ~ 1;s ~ 1;q ~ 1\ni ~~ i+s+q;s ~~ s+q;q ~~ q"),
  free_times=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 y4 y5;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: i s | y1@0 y2@1 y3 y4 y5@4;", reference="i =~ 1*y1+1*y2+1*y3+1*y4+1*y5\ns =~ 0*y1+1*y2+t3*y3+t4*y4+4*y5\ny1 ~ 0*1;y2 ~ 0*1;y3 ~ 0*1;y4 ~ 0*1;y5 ~ 0*1\ny1 ~~ y1;y2 ~~ y2;y3 ~~ y3;y4 ~~ y4;y5 ~~ y5\ni ~ 1;s ~ 1\ni ~~ i+s;s ~~ s"),
  piecewise=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 y4 y5;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: i s1 | y1@0 y2@1 y3@2 y4@2 y5@2;\ni s2 | y1@0 y2@0 y3@0 y4@1 y5@2;", reference="i =~ 1*y1+1*y2+1*y3+1*y4+1*y5\ns1 =~ 0*y1+1*y2+2*y3+2*y4+2*y5\ns2 =~ 0*y1+0*y2+0*y3+1*y4+2*y5\ny1 ~ 0*1;y2 ~ 0*1;y3 ~ 0*1;y4 ~ 0*1;y5 ~ 0*1\ny1 ~~ y1;y2 ~~ y2;y3 ~~ y3;y4 ~~ y4;y5 ~~ y5\ni ~ 1;s1 ~ 1;s2 ~ 1\ni ~~ i+s1+s2;s1 ~~ s1+s2;s2 ~~ s2"),
  cubic=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 y4 y5;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: i s q c | y1@0 y2@1 y3@2 y4@3 y5@4;", reference="i =~ 1*y1+1*y2+1*y3+1*y4+1*y5\ns =~ 0*y1+1*y2+2*y3+3*y4+4*y5\nq =~ 0*y1+1*y2+4*y3+9*y4+16*y5\nc =~ 0*y1+1*y2+8*y3+27*y4+64*y5\ny1 ~ 0*1;y2 ~ 0*1;y3 ~ 0*1;y4 ~ 0*1;y5 ~ 0*1\ny1 ~~ y1;y2 ~~ y2;y3 ~~ y3;y4 ~~ y4;y5 ~~ y5\ni ~ 1;s ~ 1;q ~ 1;c ~ 1\ni ~~ i+s+q+c;s ~~ s+q+c;q ~~ q+c;c ~~ c"),
  nonlinear_explicit=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 x1;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: y1 ON x1 (p1);\ny2 ON x1 (p2);\ny3 ON x1 (p3);\nMODEL CONSTRAINT: p1=p2**2+p3**2;", reference="y1 ~ p1*x1; y2 ~ p2*x1; y3 ~ p3*x1;\ny1 ~~ y1+y2+y3; y2 ~~ y2+y3; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1\np1==p2^2+p3^2"),
  nonlinear_implicit=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 x1;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: y1 ON x1 (p1);\ny2 ON x1 (p2);\ny3 ON x1 (p3);\nMODEL CONSTRAINT: 0=p1-p2**2-p3**2;", reference="y1 ~ p1*x1; y2 ~ p2*x1; y3 ~ p3*x1;\ny1 ~~ y1+y2+y3; y2 ~~ y2+y3; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1\n0==p1-p2^2-p3^2"),
  new_free=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 x1;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: y1 ON x1 (p1);\ny2 ON x1 (p2);\ny3 ON x1 (p3);\nMODEL CONSTRAINT: NEW(c); p2=p1+c;", reference="y1 ~ p1*x1; y2 ~ p2*x1; y3 ~ p3*x1;\ny1 ~~ y1+y2+y3; y2 ~~ y2+y3; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1\naux =~ 0*y1; aux ~~ c*aux; aux ~ 0*1; p2==p1+c"),
  linear_equality=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 x1;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: y1 ON x1 (p1);\ny2 ON x1 (p2);\ny3 ON x1 (p3);\nMODEL CONSTRAINT: p1=p2;", reference="y1 ~ p1*x1; y2 ~ p2*x1; y3 ~ p3*x1;\ny1 ~~ y1+y2+y3; y2 ~~ y2+y3; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1\np1==p2"),
  derived=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 x1;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: y1 ON x1 (p1);\ny2 ON x1 (q1);\nMODEL CONSTRAINT: NEW(r); r=p1/q1;", reference="y1 ~ p1*x1; y2 ~ q1*x1; y1 ~~ y1+y2; y2 ~~ y2; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1; r:=p1/q1"),
  functions_loops=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 x1;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: y1 ON x1 (p1);\ny2 ON x1 (p2);\ny3 ON x1 (p3);\nMODEL CONSTRAINT: NEW(r1-r3 h); DO(1,3) r#=p#**2;\nh=PHI(p1)+SQRT(p2**2)+LOG10(10);", reference="y1 ~ p1*x1; y2 ~ p2*x1; y3 ~ p3*x1;\ny1 ~~ y1+y2+y3; y2 ~~ y2+y3; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1\nr1:=p1^2; r2:=p2^2; r3:=p3^2; h:=pnorm(p1)+sqrt(p2^2)+log10(10)"),
  indirect=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 x1;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: y1 ON x1 (a);\ny2 ON y1 (b);\ny3 ON y2 (c);\ny3 ON y1 (d);\nMODEL INDIRECT: y3 IND x1; y3 IND y2 y1 x1; y3 VIA y2 x1; y3 IND y1 y2 x1;", reference="y1~a*x1; y2~b*y1; y3~c*y2+d*y1; y1~~y1; y2~~y2; y3~~y3; y1~1; y2~1; y3~1;\nind_g1_y3_ind_x1:=a*b*c+a*d; ind_g1_y3_ind_y2_y1_x1:=a*b*c; ind_g1_y3_via_y2_x1:=a*b*c; ind_g1_y3_ind_y1_y2_x1:=0"),
  factor_indirect=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 y x;\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: f BY y1-y3; f ON x (a); y ON f (b); y ON x (c);\nMODEL INDIRECT: y IND f x;", reference="f=~1*y1+y2+y3; f~a*x; y~b*f+c*x; f~~f; y~~y; f~0*1; y~1; y1~~y1; y2~~y2; y3~~y3; y1~1; y2~1; y3~1; ind_g1_y_ind_f_x:=a*b"),
  group_growth=list(input="DATA: FILE=golden.dat;\nVARIABLE: NAMES=y1 y2 y3 y4 y5 g; GROUPING=g(1=a 2=b);\nANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: i s | y1@0 y2@1 y3@2 y4@3 y5@4;", reference="i=~c(1,1)*y1+c(1,1)*y2+c(1,1)*y3+c(1,1)*y4+c(1,1)*y5\ns=~c(0,0)*y1+c(1,1)*y2+c(2,2)*y3+c(3,3)*y4+c(4,4)*y5\ny1~c(0,0)*1;y2~c(0,0)*1;y3~c(0,0)*1;y4~c(0,0)*1;y5~c(0,0)*1\ny1~~c(NA,NA)*y1;y2~~c(NA,NA)*y2;y3~~c(NA,NA)*y3;y4~~c(NA,NA)*y4;y5~~c(NA,NA)*y5\ni~c(NA,NA)*1; s~c(NA,NA)*1; i~~c(NA,NA)*i+c(NA,NA)*s; s~~c(NA,NA)*s")
)
test_that("Mplus growth, constraints and indirect effects fit live independent lavaan models", {
  skip_if_not_installed("lavaan")
  set.seed(540073);n<-600L
  for (id in names(mplus_growth_cases)) {
    c <- mplus_growth_cases[[id]]
    if(id %in% c("linear","quadratic","cubic","free_times","piecewise","group_growth")) {
      time<-0:4
      load<-switch(id,quadratic=cbind(1,time,time^2),cubic=cbind(1,time,time^2,time^3),
        free_times=cbind(1,c(0,1,1.8,2.9,4)),piecewise=cbind(1,pmin(time,2),pmax(time-2,0)),cbind(1,time))
      sd_eta<-c(.7,.3,.1,.02)[seq_len(ncol(load))]
      # Match a separately specified interior population covariance exactly,
      # so cubic growth checks exercise parity without sample Heywood warnings.
      population<-load%*%diag(sd_eta^2)%*%t(load)+diag(.36,5)
      Q<-qr.Q(qr(scale(matrix(rnorm(n*5),n),center=TRUE,scale=FALSE)))
      mu<-as.vector(load%*%c(.5,.2,.03,.002)[seq_len(ncol(load))])
      d<-as.data.frame(sweep(sqrt(n)*Q%*%chol(population),2,mu,"+"));names(d)<-paste0("y",1:5)
    } else if(id=="factor_indirect") {
      x<-rnorm(n);f<-.4*x+rnorm(n)
      d<-data.frame(y1=f+rnorm(n,sd=.7),y2=.8*f+rnorm(n,sd=.7),y3=.7*f+rnorm(n,sd=.7),y=.5*f+.1*x+rnorm(n),x=x)
    } else if(id=="indirect") {
      x<-rnorm(n);y1<-.4*x+rnorm(n);y2<-.5*y1+rnorm(n)
      d<-data.frame(y1=y1,y2=y2,y3=.6*y2+.2*y1+rnorm(n),x1=x)
    } else {
      x<-rnorm(n)
      d<-data.frame(y1=.25*x+rnorm(n),y2=.4*x+rnorm(n),y3=.3*x+rnorm(n),x1=x)
    }
    grouped<-id=="group_growth"
    if(grouped) d$g<-rep(1:2,each=n/2)
    ref <- lavaan::lavaan(c$reference,data=d,group=if(grouped) "g" else NULL,
      group.equal=if(grouped) "none" else "",meanstructure=TRUE,fixed.x=TRUE,
      auto.var=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,auto.fix.first=FALSE,
      auto.fix.single=FALSE,information="expected",control=list(iter.max=10000))
    expect_true(lavaan::lavInspect(ref,"converged"),info=id)
    spec <- mplus_model(c$input)
    fit <- fit_model(spec,d,estimator="ML")
    expect_true(fit$converged,info=id)
    pt <- lavaan::parTable(ref)
    if(id=="new_free") {
      aux <- pt$op=="~~" & pt$label=="c"
      pt$lhs[aux]<-"c";pt$rhs[aux]<-"";pt$op[aux]<-"new"
      pt<-pt[pt$lhs!="aux",]
    }
    V <- magmaan_core$inference_vcov(magmaan_core$inference_information_expected(fit),fit)
    se <- magmaan_core$inference_se(V)
    keep <- pt$free>0 & !pt$op %in% c("==",":=")
    idx <- match(paste(pt$group[keep],mplus_growth_key(pt[keep,])),paste(fit$partable$group,mplus_growth_key(fit$partable)))
    expect_false(anyNA(idx),info=id)
    expect_equal(fit$partable$est[idx],pt$est[keep],tolerance=1e-5,info=id)
    expect_equal(unname(se[fit$partable$free[idx]]),pt$se[keep],tolerance=1e-5,info=id)
    defs <- compute_defined(spec,fit,V)
    expected <- pt[pt$op==":=",]
    expect_equal(nrow(defs),nrow(expected),info=id)
    if(nrow(expected)) {
      idx <- match(expected$lhs,defs$lhs)
      expect_equal(defs$est[idx],expected$est,tolerance=1e-5,info=id)
      expect_equal(defs$se[idx],expected$se,tolerance=1e-5,info=id)
    }
    fm <- fit_measures(fit)
    expect_equal(fm$df,unname(lavaan::fitMeasures(ref,"df")),info=id)
    expect_equal(fm$chisq,unname(lavaan::fitMeasures(ref,"chisq")),tolerance=1e-5,info=id)
    restored <- tempfile(fileext=".rds");saveRDS(spec,restored)
    rebuilt <- magmaanlab:::.rebuild_model_spec(readRDS(restored))
    expect_equal(fit_model(rebuilt,d)$partable$est,fit$partable$est,tolerance=1e-5,info=id)
  }
})

test_that("Mplus deliberate inequality and pending ordinal nonlinear boundaries are explicit", {
  base <- "DATA: FILE=x;\nVARIABLE: NAMES=u1-u4; CATEGORICAL=u1-u4;\nMODEL: f BY u1; f BY u2 (a);\nf BY u3 (b); f BY u4;\nMODEL CONSTRAINT: "
  expect_error(mplus_model(paste0(base,"a=b**2;")),"TASK-54.2")
  pending <- tryCatch(mplus_model(paste0(base,"a=b**2;")),error=identity)
  expect_s3_class(pending,"error")
  message <- conditionMessage(pending)
  expect_match(message,"[1-9][0-9]*:[1-9][0-9]* \\[CN01\\]")
  for(part in c("found ordinal nonlinear", "Mplus enforces", "magmaan", "instead"))
    expect_match(message,part,fixed=TRUE)
  expect_error(mplus_model(paste0(base,"a>0;")),"CN01.*boundary asymptotics")
})

test_that("Mplus equalities remain active on continuous LS and missing-data paths", {
  skip_if_not_installed("lavaan")
  set.seed(540074);n<-400L;x<-rnorm(n)
  d<-data.frame(y1=.25*x+rnorm(n),y2=.4*x+rnorm(n),y3=.3*x+rnorm(n),x1=x)
  input<-paste("DATA: FILE=x;\nVARIABLE: NAMES=y1 y2 y3 x1;",
    "MODEL: y1 ON x1 (p1);", "y2 ON x1 (p2);", "y3 ON x1 (p3);",
    "MODEL CONSTRAINT: NEW(c); p2=p1+c;",sep="\n")
  spec<-mplus_model(input)
  get_values<-function(fit) setNames(fit$partable$est[fit$partable$label %in% c("p1","p2","c")],fit$partable$label[fit$partable$label %in% c("p1","p2","c")])
  for(estimator in c("ULS","FIML")) {
    data<-d
    if(estimator=="FIML") data$y2[seq(5,n,by=13)]<-NA_real_
    fit<-fit_model(spec,data,estimator=estimator)
    expect_true(fit$converged)
    v<-get_values(fit)
    expect_equal(unname(v["p2"]-v["p1"]-v["c"]),0,tolerance=1e-8)
  }
  nonlinear<-mplus_model(sub("NEW(c); p2=p1+c;","p1=p2**2+p3**2;",input,fixed=TRUE))
  for(estimator in c("ULS","FIML")) {
    fit<-fit_model(nonlinear,d,estimator=estimator,optimizer="nlopt-slsqp")
    expect_true(fit$converged)
    vals<-setNames(fit$partable$est,fit$partable$label)
    expect_equal(unname(vals["p1"]-vals["p2"]^2-vals["p3"]^2),0,tolerance=1e-8)
  }
})
