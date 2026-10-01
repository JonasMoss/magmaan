#!/usr/bin/env Rscript
# Independent lavaan models gate the supported EQS lowering. No EQS executable
# is used; this fixture makes no claim about EQS runtime/estimator parity.
suppressPackageStartupMessages({ library(lavaan); library(jsonlite) })
script <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)[1])
fixture_dir <- file.path(dirname(normalizePath(script)), "..", "fixtures")
pin <- trimws(readLines(file.path(fixture_dir, "lavaan_version.txt"))[1])
version <- as.character(packageVersion("lavaan"))
stopifnot(gsub("-", ".", pin, fixed = TRUE) == version)

one_eqs <- function(marker) paste(
  "/EQUATIONS", paste0("V1 = ", if (marker) "1" else "1*", "F1 + E7;"),
  "V2 = .8*F1 + E8; V3 = .65*F1 + E9; V4 = .9*F1 + E10;",
  "/VARIANCES", paste0("F1 = ", if (marker) ".8*" else "1", ";"),
  "E7-E10 = .5*; /END", sep = "\n")
one_lavaan <- function(marker) paste(
  paste0("F1 =~ ", if (marker) "1" else "start(1)", "*V1 + start(.8)*V2 + start(.65)*V3 + start(.9)*V4"),
  paste0("F1 ~~ ", if (marker) "start(.8)" else "1", "*F1"),
  "V1 ~~ start(.5)*V1; V2 ~~ start(.5)*V2; V3 ~~ start(.5)*V3; V4 ~~ start(.5)*V4",
  sep = "\n")
s1 <- .8 * tcrossprod(c(1, .8, .65, .9)) + diag(c(.5, .6, .7, .45))
s1[1,2] <- s1[2,1] <- s1[1,2] + .08
lambda <- matrix(0, 6, 2)
lambda[1:3,1] <- c(1,.8,.7); lambda[4:6,2] <- c(1,.9,.75)
s2 <- lambda %*% matrix(c(1,.3,.3,.9),2) %*% t(lambda) + diag(.5,6)
# F2 = .6 F1 + .3 V7 + D8; F1 and V7 covary, errors E2/E5 covary.
h <- matrix(0,7,3)
h[1:3,1] <- c(1,.8,.7)
h[4:6,] <- outer(c(1,.9,.75),c(.6,1,.3))
h[7,3] <- 1
q <- matrix(c(1,0,.25,0,.7,0,.25,0,1.3),3)
s3 <- h %*% q %*% t(h) + diag(c(rep(.5,6),0))
s3[2,5] <- s3[5,2] <- s3[2,5] + .05
cases <- list(
  list(id = "marker", eqs = one_eqs(TRUE), lavaan = one_lavaan(TRUE), S = s1),
  list(id = "unit_variance", eqs = one_eqs(FALSE), lavaan = one_lavaan(FALSE), S = s1),
  list(id = "zero_factor_covariance",
       eqs = paste("/EQU V1=F1+E1; V2=.8*F1+E2; V3=.7*F1+E3;",
                   "V4=F2+E4; V5=.9*F2+E5; V6=.75*F2+E6; /VAR F1,F2=1*; E1-E6=.5*;"),
       lavaan = paste("F1 =~ 1*V1 + start(.8)*V2 + start(.7)*V3",
                      "F2 =~ 1*V4 + start(.9)*V5 + start(.75)*V6",
                      "F1 ~~ start(1)*F1; F2 ~~ start(1)*F2",
                      paste(paste0("V",1:6," ~~ start(.5)*V",1:6),collapse="; "),sep="\n"), S=s2),
  list(id = "structural_disturbance",
       eqs = paste("/EQU V1=F1+E1; V2=.8*F1+E2; V3=.7*F1+E3;",
                   "V4=F2+E4; V5=.9*F2+E5; V6=.75*F2+E6; F2=.4*F1+.2*V7+D8;",
                   "/VAR F1=1; D8=.7*; V7=1.3*; E1-E6=.5*; /COV F1,V7=.25*; E2,E5=.05*;"),
       lavaan = paste("F1 =~ 1*V1 + start(.8)*V2 + start(.7)*V3",
                      "F2 =~ 1*V4 + start(.9)*V5 + start(.75)*V6",
                      "F2 ~ start(.4)*F1 + start(.2)*V7",
                      "F1 ~~ 1*F1; F2 ~~ start(.7)*F2; V7 ~~ start(1.3)*V7",
                      "F1 ~~ start(.25)*V7; V2 ~~ start(.05)*V5",
                      paste(paste0("V",1:6," ~~ start(.5)*V",1:6),collapse="; "),sep="\n"), S=s3)
)
result <- lapply(cases, function(case) {
  S <- case$S
  dimnames(S) <- list(paste0("V",seq_len(nrow(S))),paste0("V",seq_len(ncol(S))))
  fit <- lavaan(case$lavaan, sample.cov=S, sample.nobs=500L,
                sample.cov.rescale=FALSE, estimator="ML", information="expected",
                auto.var=FALSE, auto.cov.lv.x=FALSE, auto.cov.y=FALSE,
                auto.fix.first=FALSE, auto.fix.single=FALSE, fixed.x=FALSE,
                meanstructure=FALSE, control=list(iter.max=2000))
  stopifnot(lavInspect(fit,"converged"))
  pt <- parTable(fit)
  rows <- lapply(seq_len(nrow(pt)),function(i) list(
    lhs=pt$lhs[i],op=pt$op[i],rhs=pt$rhs[i],free=pt$free[i]>0,
    ustart=pt$ustart[i],est=pt$est[i],se=pt$se[i]))
  list(id=case$id, eqs=case$eqs, lavaan=case$lavaan, rows=rows,
       sample_cov=unname(S), n=500L, implied=unname(fitted(fit)$cov),
       chisq=unname(fitMeasures(fit,"chisq")), df=unname(fitMeasures(fit,"df")))
})
write_json(list(meta=list(lavaan_version=version,
                          generator="cpp/tests/tools/regen_oracle_eqs.R",
                          provenance="Hand-specified paired EQS/lavaan models; synthetic covariance summaries; no EQS runtime oracle",
                          conventions="ML, divisor-N sample covariance, expected-information SEs",
                          tolerances=list(parameters=1e-5,se=1e-5,chisq=1e-5,implied=1e-5)),
                cases=result),file.path(fixture_dir,"eqs.json"),
           auto_unbox=TRUE,pretty=TRUE,digits=16,na="null")
cat("Wrote EQS lowering fixtures against lavaan", version, "\n")
