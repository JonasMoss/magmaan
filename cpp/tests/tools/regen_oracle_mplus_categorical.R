#!/usr/bin/env Rscript
# Explicit reference syntax is authored independently of frontend projections.
# --demo gates model meaning: parameter count, df, estimates and scaled test.
# Printed Demo SEs are retained as convention observations, not meaning assertions.
suppressPackageStartupMessages({library(lavaan);library(jsonlite)})
stopifnot(as.character(packageVersion('lavaan'))==gsub('-','.',trimws(readLines('cpp/tests/fixtures/lavaan_version.txt')),fixed=TRUE))
set.seed(533072);n<-600L
eta<-matrix(rnorm(2*n),n,2);eta[,2]<-.4*eta[,1]+sqrt(.84)*eta[,2]
x<-cbind(outer(eta[,1],c(.8,.7,.65)),outer(eta[,2],c(.85,.75,.6)))+matrix(rnorm(6*n,sd=.7),n,6)
printed_rows <- function(lines) {
 rows<-list();active<-FALSE;group<-1L;lhs<-op<-''
 for(line in lines) {
  text<-trimws(line)
  if(text=='MODEL RESULTS') {active<-TRUE;next}
  if(grepl('^(QUALITY OF|STANDARDIZED|R-SQUARE|TECHNICAL)',text)) active<-FALSE
  if(!active) next
  if(startsWith(text,'Group ')) {group<-match(tolower(sub('Group ','',text)),c('a','b'));next}
  m<-regmatches(text,regexec('^([A-Za-z][A-Za-z0-9_]*) +(?:BY|ON|WITH)$',text,perl=TRUE))[[1]]
  if(length(m)) {lhs<-tolower(m[2]);op<-if(grepl('BY$',text)) '=~' else if(grepl('WITH$',text)) '~~' else '~';next}
  if(text %in% c('Means','Intercepts','Variances','Residual Variances','Thresholds','Scales')) {lhs<-'';op<-if(text %in% c('Means','Intercepts')) '~1' else if(text=='Thresholds') '|' else if(text=='Scales') '~*~' else '~~';next}
  tokens<-strsplit(text,' +')[[1]]
  if(length(tokens)<3 || !grepl('^[A-Za-z][A-Za-z0-9_$]*$',tokens[1]) || !grepl('^-?[0-9]+[.][0-9]+$',tokens[2]) || !nzchar(op)) next
  l<-if(nzchar(lhs)) lhs else tolower(tokens[1]);r<-if(op=='~1') '' else if(op=='|') paste0('t',sub('.*[$]','',l)) else if(nzchar(lhs)) tolower(tokens[1]) else l
  if(op=='|') l<-sub('[$].*','',l)
  rows[[length(rows)+1]]<-data.frame(lhs=l,op=op,rhs=r,group=group,est=as.numeric(tokens[2]),se=as.numeric(tokens[3]))
 }
 do.call(rbind,rows)
}
out<-list(version=as.character(packageVersion('lavaan')),seed=533072,cases=list())
check_demo <- '--demo' %in% commandArgs(TRUE)
for(par in c('delta','theta')) for(kind in c('binary','ordinal','configural','scalar','default')) {
 k<-if(kind=='binary') 2L else 3L;grouped<-kind %in% c('configural','scalar','default');shared<-kind %in% c('scalar','default')
 d<-as.data.frame(apply(x,2,function(z) as.integer(cut(z,if(k==2) c(-Inf,0,Inf) else c(-Inf,-.5,.5,Inf)))));names(d)<-paste0('u',1:6);d$g<-rep(1:2,each=n/2)
 load<-function(j) if(shared) paste0('c(l',j,',l',j,')') else if(grouped) 'c(NA,NA)' else 'NA'
 marker<-if(grouped) 'c(1,1)' else '1'
 syntax<-c(paste0('f1 =~ ',marker,'*u1+',load(2),'*u2+',load(3),'*u3'),paste0('f2 =~ ',marker,'*u4+',load(5),'*u5+',load(6),'*u6'),
  if(grouped) 'f1 ~~ c(NA,NA)*f1; f2 ~~ c(NA,NA)*f2; f1 ~~ c(NA,NA)*f2' else 'f1 ~~ f1; f2 ~~ f2; f1 ~~ f2',
  if(shared) 'f1 ~ c(0,NA)*1; f2 ~ c(0,NA)*1' else if(grouped) 'f1 ~ c(0,0)*1; f2 ~ c(0,0)*1' else 'f1 ~ 0*1; f2 ~ 0*1')
 for(j in 1:6) {
  terms<-vapply(seq_len(k-1),function(t) paste0(if(shared) paste0('c(t',j,t,',t',j,t,')') else if(grouped) 'c(NA,NA)' else 'NA','*t',t),character(1))
  syntax<-c(syntax,paste0('u',j,' | ',paste(terms,collapse='+')),paste0('u',j,' ~ ',if(grouped) 'c(0,0)' else '0','*1'),paste0('u',j,if(par=='delta') ' ~*~ ' else ' ~~ ',if(shared) 'c(1,NA)' else if(grouped) 'c(1,1)' else '1','*u',j))
 }
 input<-paste('DATA: FILE=golden.dat;',paste0('VARIABLE: NAMES=u1-u6',if(grouped) ' g' else '', '; CATEGORICAL=u1-u6;'),if(grouped) 'GROUPING=g(1=a 2=b);' else '',paste0('ANALYSIS: ESTIMATOR=WLSMV; PARAMETERIZATION=',toupper(par),';'),if(kind %in% c('configural','scalar')) paste0('MODEL=',toupper(kind),';') else '', 'MODEL: f1 BY u1-u3; f2 BY u4-u6;',sep='\n')
 args<-list(model=paste(syntax,collapse='\n'),data=d,ordered=paste0('u',1:6),parameterization=par,estimator='WLSMV',meanstructure=TRUE,auto.var=FALSE,auto.fix.first=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE)
 # Group restrictions are fully specified in the independent syntax.
 if(grouped) {args$group<-'g';args$group.equal<-'none'}
 lv<-do.call(lavaan,args);stopifnot(lavInspect(lv,'converged'))
 pt<-parTable(lv)
 demo<-NULL
 if(check_demo) {
 scratch<-file.path(path.expand('~/.cache/magmaan-logs/mplus-categorical-golden'),paste(par,kind,sep='_'));dir.create(scratch,recursive=TRUE,showWarnings=FALSE)
 write.table(d[if(grouped) names(d) else paste0('u',1:6)],file.path(scratch,'golden.dat'),row.names=FALSE,col.names=FALSE,quote=FALSE)
 demo_input<-sub('MODEL: f1 BY','CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: f1 BY',input,fixed=TRUE)
 writeLines(demo_input,file.path(scratch,'golden.inp'))
 old<-getwd();setwd(scratch);status<-system2(path.expand('~/mplusdemo/mpdemo'),'golden.inp',stdout='run.log',stderr='run.log');setwd(old);stopifnot(status==0)
 lines<-readLines(file.path(scratch,'golden.out'));stopifnot(!any(grepl('THE MODEL ESTIMATION DID NOT TERMINATE NORMALLY|NO CONVERGENCE|ERROR in',lines)))
 printed<-printed_rows(lines);stopifnot(nrow(printed)>0)
 mimic<-do.call(lavaan,c(args,list(mimic='Mplus')));stopifnot(lavInspect(mimic,'converged'))
 qp<-parTable(mimic);key<-function(p) {l<-p$lhs;r<-p$rhs;swap<-p$op=='~~'&l>r;tmp<-l[swap];l[swap]<-r[swap];r[swap]<-tmp;paste(l,p$op,r,p$group)}
 index<-match(key(printed),key(qp));stopifnot(!anyNA(index))
 allowance<-function(expected) .0005+1e-5*abs(expected)
 bad<-abs(printed$est-qp$est[index])>allowance(printed$est)
 if(any(bad)) print(cbind(printed[bad,],oracle_est=qp$est[index[bad]],oracle_se=qp$se[index[bad]]))
 stopifnot(!any(bad))
 count_line<-grep('Number of Free Parameters',lines,value=TRUE)[1];demo_npar<-as.integer(sub('.* +([0-9]+) *$','\\1',count_line))
 chi<-grep('Chi-Square Test of Model Fit',lines)[1];window<-lines[(chi+1):(chi+8)]
 demo_chisq<-as.numeric(sub('.*Value +([0-9.]+).*','\\1',grep('Value',window,value=TRUE)[1]))
 demo_df<-as.integer(sub('.*Degrees of Freedom +([0-9]+).*','\\1',grep('Degrees of Freedom',window,value=TRUE)[1]))
 stopifnot(demo_npar==fitMeasures(mimic,'npar'),demo_df==fitMeasures(mimic,'df'),abs(demo_chisq-fitMeasures(mimic,'chisq.scaled'))<=allowance(demo_chisq))
 demo<-list(npar=demo_npar,df=demo_df,chisq=demo_chisq,rows=printed,se_role="observations_only",mimic_se=as.list(qp$se[index]))
 }
 out$cases[[paste(par,kind,sep='_')]]<-list(input=input,model=args$model,parameterization=par,category_counts=replicate(if(grouped) 2 else 1,as.list(rep(k,6)),simplify=FALSE),
  partable=pt[,c('lhs','op','rhs','group','free','label','est','se')],R=lapply(lv@SampleStats@cov,unname),thresholds=lapply(lv@SampleStats@th,as.numeric),n_obs=as.list(as.integer(unlist(lv@SampleStats@nobs))),weight=lapply(lv@SampleStats@WLS.VD,as.numeric),nacov=lapply(lv@SampleStats@NACOV,unname),test=as.list(fitMeasures(lv,c('chisq','chisq.scaled','chisq.scaling.factor','df'))),shift=as.numeric(lavInspect(lv,'test')$scaled.shifted$shift.parameter),demo=demo)
 cat(par,kind,'df',fitMeasures(lv,'df'),'\n')
}
write_json(out,'cpp/tests/fixtures/mplus/golden_categorical.json',auto_unbox=TRUE,pretty=TRUE,digits=17,na='null')
