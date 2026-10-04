#!/usr/bin/env Rscript
# Explicit reference syntax is authored independently of frontend projections.
# --demo gates model meaning: parameter count, df, estimates and scaled test.
# Printed Demo SEs are retained as convention observations, not meaning assertions.
suppressPackageStartupMessages({library(lavaan);library(jsonlite)})
stopifnot(as.character(packageVersion('lavaan'))==gsub('-','.',trimws(readLines('cpp/tests/fixtures/lavaan_version.txt')),fixed=TRUE))
set.seed(540072);n<-800L
eta<-cbind(rnorm(n,0,.6),rnorm(n,.18,.3))
x<-eta%*%t(cbind(1,0:3))+matrix(rnorm(4*n,sd=.65),n,4)
printed_rows <- function(lines) {
 rows<-list();active<-FALSE;group<-1L;lhs<-op<-''
 for(line in lines) {
  text<-trimws(line)
  if(text=='MODEL RESULTS') {active<-TRUE;next}
  if(grepl('^(QUALITY OF|STANDARDIZED|R-SQUARE|TECHNICAL)',text)) active<-FALSE
  if(!active) next
  if(startsWith(text,'Group ')) {group<-match(tolower(sub('Group ','',text)),c('a','b'));next}
  m<-regmatches(text,regexec('^([A-Za-z][A-Za-z0-9_]*) +(?:BY|ON|WITH|[|])$',text,perl=TRUE))[[1]]
  if(length(m)) {lhs<-tolower(m[2]);op<-if(grepl('BY$|[|]$',text)) '=~' else if(grepl('WITH$',text)) '~~' else '~';next}
  if(text %in% c('Means','Intercepts','Variances','Residual Variances','Thresholds','Scales')) {lhs<-'';op<-if(text %in% c('Means','Intercepts')) '~1' else if(text=='Thresholds') '|' else if(text=='Scales') '~*~' else '~~';next}
  tokens<-strsplit(text,' +')[[1]]
  if(length(tokens)<3 || !grepl('^[A-Za-z][A-Za-z0-9_$]*$',tokens[1]) || !grepl('^-?[0-9]+[.][0-9]+$',tokens[2]) || !nzchar(op)) next
  l<-if(nzchar(lhs)) lhs else tolower(tokens[1]);r<-if(op=='~1') '' else if(op=='|') paste0('t',sub('.*[$]','',l)) else if(nzchar(lhs)) tolower(tokens[1]) else l
  if(op=='|') l<-sub('[$].*','',l)
  rows[[length(rows)+1]]<-data.frame(lhs=l,op=op,rhs=r,group=group,est=as.numeric(tokens[2]),se=as.numeric(tokens[3]))
 }
 do.call(rbind,rows)
}
out<-list(version=as.character(packageVersion('lavaan')),seed=540072,cases=list())
check_demo <- TRUE
for(par in c('delta','theta')) for(kind in c('single','groups')) {
 k<-3L;grouped<-kind=='groups'
 d<-as.data.frame(apply(x,2,function(z) as.integer(cut(z,c(-Inf,-.5,.5,Inf)))));names(d)<-paste0('u',1:4);d$g<-rep(1:2,each=n/2)
 fixed<-function(v) if(grouped) paste0('c(',v,',',v,')') else as.character(v)
 syntax<-c(paste0('i =~ ',paste(paste0(fixed(1),'*u',1:4),collapse='+')),
  paste0('s =~ ',paste(vapply(0:3,function(t) paste0(fixed(t),'*u',t+1),character(1)),collapse='+')),
  if(grouped) 'i ~~ c(NA,NA)*i; s ~~ c(NA,NA)*s; i ~~ c(NA,NA)*s; i ~ c(0,NA)*1; s ~ c(NA,NA)*1' else 'i ~~ i+s; s ~~ s; i ~ 0*1; s ~ 1')
 for(j in 1:4) {
  terms<-vapply(1:2,function(t) paste0(if(grouped) paste0('c(t',t,',t',t,')') else paste0('t',t),'*t',t),character(1))
  syntax<-c(syntax,paste0('u',j,' | ',paste(terms,collapse='+')),paste0('u',j,' ~ ',fixed(0),'*1'),paste0('u',j,if(par=='delta') ' ~*~ ' else ' ~~ ',if(j==1) {if(grouped) 'c(1,NA)' else '1'} else {if(grouped) 'c(NA,NA)' else 'NA'},'*u',j))
 }
 input<-paste('DATA: FILE=golden.dat;',paste0('VARIABLE: NAMES=u1-u4',if(grouped) ' g' else '', '; CATEGORICAL=u1-u4;'),if(grouped) 'GROUPING=g(1=a 2=b);' else '',paste0('ANALYSIS: ESTIMATOR=WLSMV; PARAMETERIZATION=',toupper(par),';'), 'MODEL: i s | u1@0 u2@1 u3@2 u4@3;',sep='\n')
 args<-list(model=paste(syntax,collapse='\n'),data=d,ordered=paste0('u',1:4),parameterization=par,estimator='WLSMV',meanstructure=TRUE,auto.var=FALSE,auto.fix.first=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE)
 # Group restrictions are fully specified in the independent syntax.
 if(grouped) {args$group<-'g';args$group.equal<-'none'}
 lv<-do.call(lavaan,args);stopifnot(lavInspect(lv,'converged'))
 pt<-parTable(lv)
 demo<-NULL
 if(check_demo) {
 scratch<-file.path(path.expand('~/.cache/magmaan-logs/mplus-growth-categorical-golden'),paste(par,kind,sep='_'));dir.create(scratch,recursive=TRUE,showWarnings=FALSE)
 write.table(d[if(grouped) names(d) else paste0('u',1:4)],file.path(scratch,'golden.dat'),row.names=FALSE,col.names=FALSE,quote=FALSE)
 demo_input<-sub('MODEL: i s','CONVERGENCE=0.00000001; ITERATIONS=10000;\nMODEL: i s',input,fixed=TRUE)
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
 out$cases[[paste(par,kind,sep='_')]]<-list(input=input,model=args$model,parameterization=par,category_counts=replicate(if(grouped) 2 else 1,as.list(rep(k,4)),simplify=FALSE),
  partable=pt[,c('lhs','op','rhs','group','free','label','est','se')],R=lapply(lv@SampleStats@cov,unname),thresholds=lapply(lv@SampleStats@th,as.numeric),n_obs=as.list(as.integer(unlist(lv@SampleStats@nobs))),weight=lapply(lv@SampleStats@WLS.VD,as.numeric),nacov=lapply(lv@SampleStats@NACOV,unname),test=as.list(fitMeasures(lv,c('chisq','chisq.scaled','chisq.scaling.factor','df'))),shift=as.numeric(lavInspect(lv,'test')$scaled.shifted$shift.parameter),demo=demo)
 cat(par,kind,'df',fitMeasures(lv,'df'),'\n')
}
write_json(out,'cpp/tests/fixtures/mplus/golden_growth_categorical.json',auto_unbox=TRUE,pretty=TRUE,digits=17,na='null')
