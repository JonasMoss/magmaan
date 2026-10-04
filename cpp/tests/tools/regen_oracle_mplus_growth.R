#!/usr/bin/env Rscript
# Independently authored explicit lavaan models; Demo checks model meaning.
suppressPackageStartupMessages({library(lavaan);library(jsonlite)})
Sys.setenv(OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1)
stopifnot(as.character(packageVersion('lavaan'))==gsub('-','.',trimws(readLines('cpp/tests/fixtures/lavaan_version.txt')),fixed=TRUE))
set.seed(540071); n<-600L
cases<-list()
add<-function(id,vars,mplus,syntax,data) {
 input<-paste('DATA: FILE=golden.dat;',paste0('VARIABLE: NAMES=',paste(vars,collapse=' '),';'),
 'ANALYSIS: ESTIMATOR=ML; CONVERGENCE=0.00000001; ITERATIONS=10000;',paste0('MODEL: ',mplus),sep='\n')
 cases[[id]]<<-list(input=input,syntax=syntax,data=round(data,8),vars=vars)
}
yvars<-paste0('y',1:5)
time<-0:4
eta<-cbind(rnorm(n,.5,.7),rnorm(n,.2,.3),rnorm(n,.03,.06))
make_growth<-function(load) eta[,seq_len(ncol(load)),drop=FALSE]%*%t(load)+matrix(rnorm(n*nrow(load),sd=.6),n)
growth_syntax<-function(load,factors) {
 c(vapply(seq_along(factors),function(k) paste0(factors[k],' =~ ',paste(paste0(load[,k],'*',yvars),collapse='+')),character(1)),
 paste(paste0(yvars,' ~ 0*1'),collapse=';'),paste(paste0(yvars,' ~~ ',yvars),collapse=';'),
 paste(paste0(factors,' ~ 1'),collapse=';'),paste(vapply(seq_along(factors),function(k) paste0(factors[k],' ~~ ',paste(factors[k:length(factors)],collapse='+')),character(1)),collapse=';'))
}
L<-cbind(1,time);d<-make_growth(L);colnames(d)<-yvars
add('linear',yvars,'i s | y1@0 y2@1 y3@2 y4@3 y5@4;',paste(growth_syntax(L,c('i','s')),collapse='\n'),d)
L<-cbind(1,time,time^2);d<-make_growth(L);colnames(d)<-yvars
add('quadratic',yvars,'i s q | y1@0 y2@1 y3@2 y4@3 y5@4;',paste(growth_syntax(L,c('i','s','q')),collapse='\n'),d)
L<-cbind(1,c(0,1,1.8,2.9,4));d<-make_growth(L);colnames(d)<-yvars
syn<-growth_syntax(cbind(1,time),c('i','s'));syn[2]<-'s =~ 0*y1+1*y2+t3*y3+t4*y4+4*y5'
add('free_times',yvars,'i s | y1@0 y2@1 y3 y4 y5@4;',paste(syn,collapse='\n'),d)
L<-cbind(1,pmin(time,2),pmax(time-2,0));d<-make_growth(L);colnames(d)<-yvars
add('piecewise',yvars,'i s1 | y1@0 y2@1 y3@2 y4@2 y5@2;\ni s2 | y1@0 y2@0 y3@0 y4@1 y5@2;',paste(growth_syntax(L,c('i','s1','s2')),collapse='\n'),d)
L<-cbind(1,time,time^2,time^3)
d<-cbind(eta,rnorm(n,.002,.005))%*%t(L)+matrix(rnorm(n*5,sd=.6),n,5);colnames(d)<-yvars
add('cubic',yvars,'i s q c | y1@0 y2@1 y3@2 y4@3 y5@4;',paste(growth_syntax(L,c('i','s','q','c')),collapse='\n'),d)
x<-rnorm(n);e<-matrix(rnorm(n*3,sd=.8),n,3);e[,2]<-.2*e[,1]+e[,2];e[,3]<-.15*e[,1]+.1*e[,2]+e[,3]
d<-cbind(y1=.25*x+e[,1],y2=.4*x+e[,2],y3=.3*x+e[,3],x1=x)
base<-'y1 ~ p1*x1; y2 ~ p2*x1; y3 ~ p3*x1;\ny1 ~~ y1+y2+y3; y2 ~~ y2+y3; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1'
model<-'y1 ON x1 (p1);\ny2 ON x1 (p2);\ny3 ON x1 (p3);'
add('nonlinear_explicit',colnames(d),paste0(model,'\nMODEL CONSTRAINT: p1=p2**2+p3**2;'),paste(base,'p1==p2^2+p3^2',sep='\n'),d)
add('nonlinear_implicit',colnames(d),paste0(model,'\nMODEL CONSTRAINT: 0=p1-p2**2-p3**2;'),paste(base,'0==p1-p2^2-p3^2',sep='\n'),d)
# A disconnected zero-loading phantom is the independent lavaan free coordinate.
add('new_free',colnames(d),paste0(model,'\nMODEL CONSTRAINT: NEW(c); p2=p1+c;'),paste(base,'aux =~ 0*y1; aux ~~ c*aux; aux ~ 0*1; p2==p1+c',sep='\n'),d)
add('linear_equality',colnames(d),paste0(model,'\nMODEL CONSTRAINT: p1=p2;'),paste(base,'p1==p2',sep='\n'),d)
add('derived',colnames(d),paste0('y1 ON x1 (p1);\ny2 ON x1 (q1);\nMODEL CONSTRAINT: NEW(r); r=p1/q1;'),
 'y1 ~ p1*x1; y2 ~ q1*x1; y1 ~~ y1+y2; y2 ~~ y2; y3 ~~ y3; y1 ~ 1; y2 ~ 1; y3 ~ 1; r:=p1/q1',d)
add('functions_loops',colnames(d),paste0(model,'\nMODEL CONSTRAINT: NEW(r1-r3 h); DO(1,3) r#=p#**2;\nh=PHI(p1)+SQRT(p2**2)+LOG10(10);'),
 paste(base,'r1:=p1^2; r2:=p2^2; r3:=p3^2; h:=pnorm(p1)+sqrt(p2^2)+log10(10)',sep='\n'),d)
x<-rnorm(n);y1<-.4*x+rnorm(n);y2<-.5*y1+rnorm(n);y3<-.6*y2+.2*y1+rnorm(n);d<-cbind(y1,y2,y3,x1=x)
add('indirect',colnames(d),'y1 ON x1 (a);\ny2 ON y1 (b);\ny3 ON y2 (c);\ny3 ON y1 (d);\nMODEL INDIRECT: y3 IND x1; y3 IND y2 y1 x1; y3 VIA y2 x1; y3 IND y1 y2 x1;',
 'y1~a*x1; y2~b*y1; y3~c*y2+d*y1; y1~~y1; y2~~y2; y3~~y3; y1~1; y2~1; y3~1;\nind_g1_y3_ind_x1:=a*b*c+a*d; ind_g1_y3_ind_y2_y1_x1:=a*b*c; ind_g1_y3_via_y2_x1:=a*b*c; ind_g1_y3_ind_y1_y2_x1:=0',d)
x<-rnorm(n);f<-.4*x+rnorm(n);y<-.5*f+.1*x+rnorm(n)
d<-cbind(y1=f+rnorm(n,sd=.7),y2=.8*f+rnorm(n,sd=.7),y3=.7*f+rnorm(n,sd=.7),y,x)
add('factor_indirect',colnames(d),'f BY y1-y3; f ON x (a); y ON f (b); y ON x (c);\nMODEL INDIRECT: y IND f x;',
 'f=~1*y1+y2+y3; f~a*x; y~b*f+c*x; f~~f; y~~y; f~0*1; y~1; y1~~y1; y2~~y2; y3~~y3; y1~1; y2~1; y3~1; ind_g1_y_ind_f_x:=a*b',d)
L<-cbind(1,0:4);d<-make_growth(L);colnames(d)<-yvars
syn<-c('i=~c(1,1)*y1+c(1,1)*y2+c(1,1)*y3+c(1,1)*y4+c(1,1)*y5',
 's=~c(0,0)*y1+c(1,1)*y2+c(2,2)*y3+c(3,3)*y4+c(4,4)*y5',
 paste(paste0(yvars,'~c(0,0)*1'),collapse=';'),paste(paste0(yvars,'~~c(NA,NA)*',yvars),collapse=';'),
 'i~c(NA,NA)*1; s~c(NA,NA)*1; i~~c(NA,NA)*i+c(NA,NA)*s; s~~c(NA,NA)*s')
add('group_growth',yvars,'i s | y1@0 y2@1 y3@2 y4@3 y5@4;',paste(syn,collapse='\n'),d)
cases$group_growth$groups<-rep(1:2,each=n/2)
cases$group_growth$input<-sub('NAMES=y1 y2 y3 y4 y5;','NAMES=y1 y2 y3 y4 y5 g; GROUPING=g(1=a 2=b);',cases$group_growth$input,fixed=TRUE)
printed_rows<-function(lines) {
 result<-list();active<-FALSE;lhs<-op<-'';group<-1L
 for(line in lines) {
  text<-trimws(line)
  if(text=='MODEL RESULTS') {active<-TRUE;next}
  if(grepl('^(QUALITY OF|STANDARDIZED|R-SQUARE|TECHNICAL|TOTAL, TOTAL)',text)) active<-FALSE
  if(!active) next
  if(startsWith(text,'Group ')) {group<-match(tolower(sub('Group ','',text)),c('a','b'));next}
  if(grepl('^[A-Za-z][A-Za-z0-9_]* +(BY|ON|WITH|[|])$',text)) {
   words<-strsplit(text,' +')[[1]];lhs<-tolower(words[1]);op<-if(words[2]=='|') '=~' else switch(words[2],BY='=~',ON='~',WITH='~~');next
  }
  if(text %in% c('Means','Intercepts','Variances','Residual Variances','New/Additional Parameters')) {
   lhs<-'';op<-if(text %in% c('Means','Intercepts')) '~1' else if(text=='New/Additional Parameters') ':=' else '~~';next
  }
  m<-regmatches(text,regexec('^([A-Za-z][A-Za-z0-9_]*) +(-?[0-9]+[.][0-9]+)(?: +|$)',text))[[1]]
  if(length(m) && nzchar(op)) {l<-if(nzchar(lhs)) lhs else tolower(m[2]);r<-if(op %in% c('~1',':=')) '' else if(nzchar(lhs)) tolower(m[2]) else l;result[[length(result)+1]]<-data.frame(lhs=l,op=op,rhs=r,group=group,est=as.numeric(m[3]))}
 }
 if(length(result)) do.call(rbind,result) else data.frame()
}
key<-function(p) {l<-p$lhs;r<-p$rhs;swap<-p$op=='~~' & l>r;tmp<-l[swap];l[swap]<-r[swap];r[swap]<-tmp;paste(l,p$op,r,p$group)}
results<-list()
for(id in names(cases)) {
 case<-cases[[id]];d<-as.data.frame(case$data);names(d)<-case$vars
 if(!is.null(case$groups)) d$g<-case$groups
 lv<-lavaan(case$syntax,data=d,group=if(!is.null(case$groups)) 'g' else NULL,group.equal=if(!is.null(case$groups)) 'none' else '',meanstructure=TRUE,fixed.x=TRUE,auto.var=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,auto.fix.first=FALSE,auto.fix.single=FALSE,information='expected',control=list(iter.max=10000))
 stopifnot(lavInspect(lv,'converged'));pt<-parTable(lv);fm<-fitMeasures(lv,c('npar','df','chisq'))
 dir<-file.path(path.expand('~/.cache/magmaan-logs/mplus-growth-golden'),id);dir.create(dir,recursive=TRUE,showWarnings=FALSE)
 write.table(d,file.path(dir,'golden.dat'),quote=FALSE,row.names=FALSE,col.names=FALSE);writeLines(case$input,file.path(dir,'golden.inp'))
 old<-getwd();setwd(dir);status<-system2('/home/jonas/mplusdemo/mpdemo','golden.inp',stdout='run.log',stderr='run.log');setwd(old)
 stopifnot(status==0);lines<-readLines(file.path(dir,'golden.out'));stopifnot(!any(grepl('ERROR in|DID NOT TERMINATE NORMALLY|NO CONVERGENCE',lines)))
 free<-as.integer(sub('.* +([0-9]+) *$','\\1',grep('Number of Free Parameters',lines,value=TRUE)[1]))
 chi<-grep('^\\s*Chi-Square Test of Model Fit\\s*$',lines)[1];window<-lines[(chi+1):(chi+8)]
 df<-as.integer(sub('.*Degrees of Freedom +([0-9]+).*','\\1',grep('Degrees of Freedom',window,value=TRUE)[1]))
 chisq<-as.numeric(sub('.*Value +([0-9.]+).*','\\1',grep('Value',window,value=TRUE)[1]))
 stopifnot(free==fm['npar'],df==fm['df'],abs(chisq-fm['chisq'])<.002+1e-5*fm['chisq'])
 printed<-printed_rows(lines);idx<-match(key(printed),key(pt));idx[printed$op==':=']<-match(printed$lhs[printed$op==':='],pt$lhs)
 # NEW free coordinates print in Additional Parameters, mapped independently.
 if(id=='new_free') idx[printed$lhs=='c' & printed$op==':=']<-which(pt$label=='c' & pt$op=='~~')
 stopifnot(!anyNA(idx));err<-abs(printed$est-pt$est[idx]);stopifnot(all(err<=.0005+1e-5*abs(pt$est[idx])))
 if(id=='new_free') {aux<-pt$label=='c' & pt$op=='~~';pt$lhs[aux]<-'c';pt$rhs[aux]<-'';pt$op[aux]<-'new';pt<-pt[pt$lhs!='aux',]}
 rows<-lapply(seq_len(nrow(pt)),function(j) as.list(pt[j,c('lhs','op','rhs','group','free','label','est','se')]))
 blocks<-if(is.null(case$groups)) list(case$data) else unname(split(as.data.frame(case$data),case$groups))
 block_cov<-lapply(blocks,function(z) {z<-as.matrix(z);centered<-sweep(z,2,colMeans(z),'-');unname(crossprod(centered)/nrow(z))});block_mean<-lapply(blocks,function(z) unname(colMeans(z)))
 centered<-sweep(case$data,2,colMeans(case$data),'-')
 indirect_demo<-NULL
 if(id %in% c('indirect','factor_indirect')) {
  lines<-lines[grep('^TOTAL, TOTAL INDIRECT',lines)[1]:length(lines)]
  nums<-function(line) as.numeric(regmatches(line,regexpr('-?[0-9]+[.][0-9]+',line)))
  if(id=='indirect') {
   indirect_demo<-c(total=nums(grep('^  Total indirect ',lines,value=TRUE)[1]),specific=nums(grep('^    X1 +',lines,value=TRUE)[3]),reverse=nums(grep('^    X1 +',lines,value=TRUE)[4]),via=nums(tail(grep('^    X1 +',lines,value=TRUE),1)))
   wanted<-pt$est[match(c('ind_g1_y3_ind_x1','ind_g1_y3_ind_y2_y1_x1','ind_g1_y3_ind_y1_y2_x1','ind_g1_y3_via_y2_x1'),pt$lhs)]
  } else {
   indirect_demo<-c(specific=nums(tail(grep('^    X +',lines,value=TRUE),1)))
   wanted<-pt$est[match('ind_g1_y_ind_f_x',pt$lhs)]
  }
  stopifnot(all(abs(indirect_demo-wanted)<=.0005+1e-5*abs(wanted)))
 }
 results[[id]]<-list(input=case$input,reference=case$syntax,variables=case$vars,n=n,sample_cov=unname(crossprod(centered)/n),sample_mean=unname(colMeans(case$data)),rows=rows,npar=unname(fm['npar']),df=unname(fm['df']),chisq=unname(fm['chisq']),group_sample_cov=block_cov,group_sample_mean=block_mean,group_n=as.list(vapply(blocks,nrow,integer(1))),printed_indirect=as.list(indirect_demo),demo=list(npar=free,df=df,chisq=chisq,printed=printed,maximum_estimate_error=max(err)))
 cat(id,': npar',free,'df',df,'max Demo error',max(err),'\n')
}
write_json(list(lavaan_version=as.character(packageVersion('lavaan')),mplus_version='9.1 Demo',generator='cpp/tests/tools/regen_oracle_mplus_growth.R',seed=540071,cases=results),'cpp/tests/fixtures/mplus/golden_growth.json',auto_unbox=TRUE,pretty=TRUE,digits=17,na='null')
