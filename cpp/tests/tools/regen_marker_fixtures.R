# Literal covariance inputs; installed lavaan owns only the expected outputs.
stopifnot(as.character(packageVersion("lavaan")) == "0.7.2")
library(jsonlite)
weak <- matrix(c(1,.02,.01,.02,1,.6,.01,.6,1),3,3)
strong <- matrix(c(1,.4,.3,.4,1,.6,.3,.6,1),3,3)
small <- diag(3); small[row(small)!=col(small)] <- .01
tie <- weak; tie[1,3] <- tie[3,1] <- .02
zero <- weak; zero[1,] <- zero[,1] <- 0
negative <- weak; negative[1,2:3] <- negative[2:3,1] <- c(-.02,-.01)
make <- function(name, syntax="f =~ x1+x2+x3", cov=list(weak), groups=1L, equal=NULL, marker=NULL) {
  args <- list(model=syntax, auto.fix.first=TRUE, auto.var=TRUE, auto.cov.lv.x=TRUE,
               ngroups=groups, group.equal=equal, marker=marker)
  pt <- do.call(lavaan::lavaanify,args)
  ov <- lavaan::lavNames(pt,"ov")
  cov <- lapply(cov,function(x) {dimnames(x)<-list(ov,ov);x})
  res <- lavaan:::lav_pt_marker_adapt(pt,list(implied=list(cov=cov)),threshold=.1)
  list(name=name,syntax=syntax,n_groups=groups,group_equal=equal,marker=if(is.null(marker)) NULL else as.list(marker),
       cov=lapply(cov,function(x) lapply(unname(split(x,row(x))),as.list)),names=rep(list(as.list(ov)),length(cov)),
       info=if(is.null(res)) list() else lapply(seq_len(nrow(res$info)),function(i)
         list(lv=res$info$lv[i],old=res$info$old[i],new=res$info$new[i],r_old=res$info$r.old[i],r_new=res$info$r.new[i])),
       pt=pt[,c("lhs","op","rhs","free","ustart","label","plabel")])
}
rule <- list(make("switch"),make("strong",cov=list(strong)),make("no_better",cov=list(small)),
 make("tie",cov=list(tie)),make("zero",cov=list(zero)),
 make("groups",cov=list(weak,negative),groups=2L),
 make("groups_na",cov=list(zero,weak),groups=2L),
 make("explicit_one",syntax="f =~ 1*x1+x2+x3"),
 make("explicit_second",syntax="f =~ NA*x1+1*x2+x3"),
 make("single",syntax="f =~ x1",cov=list(matrix(1))),
 make("second_order",syntax="f =~ x1+x2+x3\ng =~ f",cov=list(weak)))
build <- list(make("one",marker=c(f="x2")),
 make("two",syntax="f =~ x1+x2+x3\ng =~ y1+y2+y3\ng ~ f",cov=list(diag(6)),marker=c(f="x2",g="y3")),
 make("equal_groups",groups=2L,equal="loadings",cov=list(weak,weak),marker=c(f="x2")),
 make("labels",syntax="f =~ a*x1+b*x2+c*x3",marker=c(f="x2")),
 make("constraint",syntax="f =~ a*x1+b*x2+c*x3\na == c",marker=c(f="x2")))
write_json(list(lavaan_version="0.7.2",rule=rule,build=build),
 "cpp/tests/fixtures/fitting/lavaan_marker_0_7_2.json",auto_unbox=TRUE,pretty=TRUE,na="null",null="null",digits=16)
