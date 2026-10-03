#!/usr/bin/env Rscript
# Independent continuous multigroup models. No frontend-generated syntax is used.
suppressPackageStartupMessages({library(lavaan); library(jsonlite)})
Sys.setenv(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
script <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
root <- normalizePath(file.path(dirname(script), '../../..'))
pin <- trimws(readLines(file.path(root,'cpp/tests/fixtures/lavaan_version.txt'))[1])
stopifnot(gsub('-','.',pin,fixed=TRUE)==as.character(packageVersion('lavaan')))
scratch <- path.expand('~/.cache/magmaan-logs/mplus-groups-golden')
dir.create(scratch,recursive=TRUE,showWarnings=FALSE)
cases <- list()
add <- function(id, model, syntax, vars=paste0('y',1:4), codes=1:2,
                sections='', analysis='', population='cfa') {
  labels <- letters[seq_along(codes)]
  inp <- paste('DATA: FILE=golden.dat;', 'VARIABLE:',
    paste0('NAMES=',paste(c(vars,'g'),collapse=' '),';'),
    paste0('USEVARIABLES=',paste(vars,collapse=' '),';'),
    paste0('GROUPING=g(',paste(paste(codes,'=',labels),collapse=' '),');'),
    'ANALYSIS: ESTIMATOR=ML; INFORMATION=EXPECTED;',analysis,
    'MODEL:',model,sections,'OUTPUT: TECH1;',sep='\n')
  stopifnot(all(nchar(strsplit(inp,'\n')[[1]])<=90))
  cases[[length(cases)+1]] <<- list(id=id,input=inp,syntax=syntax,vars=vars,
    codes=sort(codes),labels=labels[order(codes)],population=population)
}
# Explicit reference syntax with independent labels and group-specific values.
cfa <- paste('f =~ c(1,1)*y1 + c(l2,l2)*y2 + c(l3,l3)*y3 + c(l4,l4)*y4',
 'f ~~ c(NA,NA)*f; f ~ c(0,NA)*1',
 'y1 ~~ c(NA,NA)*y1; y2 ~~ c(NA,NA)*y2; y3 ~~ c(NA,NA)*y3; y4 ~~ c(NA,NA)*y4',
 'y1 ~ c(i1,i1)*1; y2 ~ c(i2,i2)*1; y3 ~ c(i3,i3)*1; y4 ~ c(i4,i4)*1',sep='\n')
add('mg_default','f BY y1-y4;',cfa)
add('mg_order','f BY y1-y4;',cfa,codes=c(2,1))
add('mg_three','f BY y1-y4;',paste(
 'f =~ c(1,1,1)*y1 + c(l2,NA,l2)*y2 + c(l3,l3,l3)*y3 + c(l4,l4,l4)*y4',
 'f ~~ c(NA,NA,NA)*f; f ~ c(0,NA,NA)*1',
 'y1 ~~ c(NA,NA,NA)*y1; y2 ~~ c(NA,NA,NA)*y2; y3 ~~ c(NA,NA,NA)*y3; y4 ~~ c(NA,NA,NA)*y4',
 'y1 ~ c(i1,i1,i1)*1; y2 ~ c(i2,i2,i2)*1; y3 ~ c(i3,i3,NA)*1; y4 ~ c(i4,i4,i4)*1',sep='\n'),
 codes=1:3,sections='MODEL b: f BY y2;\nMODEL c: [y3];')
add('mg_labels','f BY y1-y4;\ny1 (v);\ny2 (1);',paste(cfa,
 'y1 ~~ c(v,v)*y1; y2 ~~ c(.eq1.,.eq1.)*y2; y3 ~~ c(v3,v)*y3; y4 ~~ c(v4,.eq1.)*y4',sep='\n'),
 sections='MODEL b: y3 (v);\ny4 (1);')
add('mg_first_mean','f BY y1-y4;',sub('c(0,NA)','c(NA,0)',cfa,fixed=TRUE),
 sections='MODEL a: [f];\nMODEL b: [f@0];')
add('mg_asymmetric','f BY y1-y4;',paste(cfa,'y1 ~~ c(0,NA)*y2',sep='\n'),
 sections='MODEL b: y1 WITH y2;')
# Two-factor invariance shortcuts, marker and variance identification.
for (variance in c(FALSE,TRUE)) for (method in c('CONFIGURAL','METRIC','SCALAR')) {
 shared <- method!='CONFIGURAL'; intercepts <- method=='SCALAR'
 l <- function(n) if(shared) paste0('c(',n,',',n,')') else 'c(NA,NA)'
 ints <- vapply(1:6,function(j) paste0('y',j,' ~ ',if(intercepts) paste0('c(i',j,',i',j,')') else 'c(NA,NA)','*1'),character(1))
 syntax <- paste(paste0('f1 =~ ',if(variance) l('l1') else 'c(1,1)','*y1 + ',l('l2'),'*y2 + ',l('l3'),'*y3'),
 paste0('f2 =~ ',if(variance) l('l4') else 'c(1,1)','*y4 + ',l('l5'),'*y5 + ',l('l6'),'*y6'),
 paste0('f1 ~~ ',if(variance) if(shared) 'c(1,NA)' else 'c(1,1)' else 'c(NA,NA)','*f1'),
 paste0('f2 ~~ ',if(variance) if(shared) 'c(1,NA)' else 'c(1,1)' else 'c(NA,NA)','*f2'),
 'f1 ~~ c(NA,NA)*f2',paste0('f1 ~ ',if(intercepts) 'c(0,NA)' else 'c(0,0)','*1'),
 paste0('f2 ~ ',if(intercepts) 'c(0,NA)' else 'c(0,0)','*1'),paste(ints,collapse='; '),
 paste(paste0('y',1:6,' ~~ c(NA,NA)*y',1:6),collapse='; '),sep='\n')
 add(paste0('iv_',tolower(method),if(variance) '_variance' else '_marker'),
 if(variance) 'f1 BY y1* y2-y3; f2 BY y4* y5-y6; f1@1 f2@1;' else 'f1 BY y1-y3; f2 BY y4-y6;',
 syntax,vars=paste0('y',1:6),analysis=paste0('MODEL=',method,';'),population='two_factor')
}
add('mg_structural','f BY y1-y4; y4 ON x1; y5 ON f x1;',paste(cfa,
 'y4 ~ c(NA,NA)*x1; y5 ~ c(NA,NA)*f + c(NA,NA)*x1; y5 ~~ c(NA,NA)*y5; y5 ~ c(NA,NA)*1',sep='\n'),
 vars=c(paste0('y',1:5),'x1'),population='structural')
add('mg_second_order','f1 BY y1-y2; f2 BY y3-y4; f3 BY y5-y6; h BY f1-f3; [f1-f3@0];',paste(
 'f1 =~ c(1,1)*y1 + c(l2,l2)*y2; f2 =~ c(1,1)*y3 + c(l4,l4)*y4; f3 =~ c(1,1)*y5 + c(l6,l6)*y6',
 'h =~ c(1,1)*f1 + c(NA,NA)*f2 + c(NA,NA)*f3',
 'h ~~ c(NA,NA)*h; f1 ~~ c(NA,NA)*f1; f2 ~~ c(NA,NA)*f2; f3 ~~ c(NA,NA)*f3',
 'h ~ c(0,NA)*1; f1 ~ c(0,0)*1; f2 ~ c(0,0)*1; f3 ~ c(0,0)*1',
 paste(paste0('y',1:6,' ~~ c(NA,NA)*y',1:6),collapse='; '),
 paste(paste0('y',1:6,' ~ c(i',1:6,',i',1:6,')*1'),collapse='; '),sep='\n'),
 vars=paste0('y',1:6),population='second_order')
# Parse only the printed TECH1 parameter specification, preserving group cells.
tech1 <- function(lines) {
 out<-list(); group<-''; mat<-''; cols<-character(); active<-FALSE
 for(line in lines) {
  s<-trimws(line)
  if(startsWith(s,'PARAMETER SPECIFICATION FOR ')) {active<-TRUE;group<-sub('PARAMETER SPECIFICATION FOR ','',s);next}
  if(startsWith(s,'STARTING VALUES')) active<-FALSE
  if(!active || !nzchar(s)) next
  if(s %in% c('NU','LAMBDA','THETA','ALPHA','BETA','PSI')) {mat<-s;cols<-character();next}
  t<-strsplit(s,' +')[[1]]
  if(all(grepl('^[A-Z][A-Z0-9]*$',t))) {cols<-t;next}
  if(!length(cols) || !nzchar(mat)) next
  row<-'vector'; nums<-t
  if(!all(grepl('^[0-9]+$',t))) {row<-t[1];nums<-t[-1]}
  if(length(nums)>length(cols) || !all(grepl('^[0-9]+$',nums))) next
  for(j in seq_along(nums)) out[[length(out)+1]]<-list(group=group,matrix=mat,row=row,col=cols[j],number=as.integer(nums[j]))
 }
 out
}
printed_estimates <- function(lines, labels) {
 active<-FALSE;group<-1L;lhs<-op<-'';out<-list()
 for(line in lines) {
  s<-trimws(line)
  if(s=='MODEL RESULTS') {active<-TRUE;next}
  if(grepl('^(QUALITY OF|STANDARDIZED|TECHNICAL)',s)) active<-FALSE
  if(!active) next
  if(startsWith(s,'Group ')) {group<-match(tolower(sub('Group ','',s)),tolower(labels));next}
  relation<-regmatches(s,regexec('^([A-Za-z][A-Za-z0-9_]*) +(?: +)?(BY|ON|WITH)$',s))[[1]]
  if(length(relation)) {lhs<-tolower(relation[2]);op<-c(BY='=~',ON='~',WITH='~~')[[relation[3]]];next}
  if(s %in% c('Means','Intercepts','Variances','Residual Variances')) {lhs<-'';op<-if(s %in% c('Means','Intercepts')) '~1' else '~~';next}
  m<-regmatches(s,regexec('^([A-Za-z][A-Za-z0-9_]*) +(-?[0-9]+\\.[0-9]+) +',s))[[1]]
  if(length(m) && nzchar(op)) {
   l<-if(nzchar(lhs)) lhs else tolower(m[2]);r<-if(op=='~1') '' else if(nzchar(lhs)) tolower(m[2]) else l
   out[[length(out)+1]]<-list(group=group,lhs=l,op=op,rhs=r,est=as.numeric(m[3]))
  }
 }
 out
}
opts<-list(estimator='ML',information='expected',fixed.x=TRUE,meanstructure=TRUE,
 auto.var=FALSE,auto.cov.lv.x=FALSE,auto.cov.y=FALSE,auto.fix.first=FALSE,auto.fix.single=FALSE)
results<-list()
for(i in seq_along(cases)) {
 case<-cases[[i]];set.seed(52500+i);n<-600L;blocks<-list()
 for(code in case$codes) {
  z<-matrix(rnorm(n*15),n,15); f<-z[,1]; f2<-.3*f+sqrt(.91)*z[,2]
  y<-outer(f,c(1,.8,.7,.9,.85,.75))+.8*z[,3:8]
  if(case$population=='two_factor') y[,4:6]<-outer(f2,c(1,.85,.75))+.8*z[,6:8]
  if(case$population=='second_order') {fs<-outer(f,c(1,.8,.7))+.7*z[,9:11];for(j in 1:3) y[,(2*j-1):(2*j)]<-outer(fs[,j],c(1,.8))+.8*z[,(2*j+1):(2*j+2)]}
  if(case$population=='structural') {y[,4]<-y[,4]+.2*z[,12];y[,5]<-.5*f+.2*z[,12]+.8*z[,7]}
  colnames(y)<-paste0('y',1:6)
  d<-data.frame(y,x1=z[,12]);d<-d[,case$vars,drop=FALSE]
  d<-sweep(as.matrix(d),2,seq_len(ncol(d))*.1,'+')
  blocks[[as.character(code)]]<-data.frame(round(d,8),g=code,check.names=FALSE)
 }
 # Higher codes occur first in the raw file, even for a reversed declaration.
 data<-do.call(rbind,rev(blocks));rownames(data)<-NULL
 fit<-do.call(lavaan,c(list(model=case$syntax,data=data,group='g',group.label=as.character(case$codes)),opts,list(control=list(iter.max=3000))))
 pt<-parTable(fit);fm<-fitMeasures(fit,c('npar','df','chisq','pvalue'))
 stopifnot(lavInspect(fit,'converged'),all(pt$est[pt$op=='~~' & pt$lhs==pt$rhs]>=0),fm['df']==0 || fm['pvalue']>=.001)
 dir<-file.path(scratch,case$id);dir.create(dir,showWarnings=FALSE)
 write.table(data,file.path(dir,'golden.dat'),row.names=FALSE,col.names=FALSE,quote=FALSE)
 writeLines(case$input,file.path(dir,'golden.inp'))
 old<-getwd();setwd(dir);status<-system2('/home/jonas/mplusdemo/mpdemo',c('golden.inp','golden.out'),stdout='console.log',stderr='console.err',timeout=120);setwd(old)
 lines<-readLines(file.path(dir,'golden.out'),warn=FALSE)
 if(status!=0 || any(grepl('\\*\\*\\* ERROR',lines)) || !any(grepl('TERMINATED NORMALLY',lines))) stop(case$id,': Demo failed')
 free<-as.numeric(sub('.*Number of Free Parameters +','',grep('Number of Free Parameters',lines,value=TRUE)))
 stopifnot(length(free)==1,free==fm['npar'])
 chi_start<-grep('^ *Chi-Square Test of Model Fit *$',lines);cb<-lines[chi_start:(chi_start+8)]
 chi<-as.numeric(sub('.*Value +([0-9.]+).*','\\1',grep('^ *Value ',cb,value=TRUE)[1]))
 df<-as.numeric(sub('.*Degrees of Freedom +','',grep('Degrees of Freedom',cb,value=TRUE)[1]))
 if(df!=fm['df'] || abs(chi-fm['chisq'])>.002+1e-5*fm['chisq']) stop(case$id,': Demo/lavaan fit mismatch')
 demo_estimates<-printed_estimates(lines,case$labels)
 if(!length(demo_estimates)) stop(case$id,': no printed estimates')
 for(j in seq_along(demo_estimates)) {
  r<-demo_estimates[[j]]
  hit<-which(pt$group==r$group & pt$lhs==r$lhs & pt$op==r$op & pt$rhs==r$rhs)
  if(!length(hit) && r$op=='~~') hit<-which(pt$group==r$group & pt$lhs==r$rhs & pt$op=='~~' & pt$rhs==r$lhs)
  if(length(hit)!=1) stop(case$id,': unmatched Demo estimate ',r$lhs,r$op,r$rhs)
  error<-abs(r$est-pt$est[hit]);tolerance<-.001+2e-4*abs(pt$est[hit])
  if(error>tolerance) stop(case$id,': Demo/lavaan estimate mismatch ',r$lhs,r$op,r$rhs)
  demo_estimates[[j]]$absolute_error<-error
  demo_estimates[[j]]$tolerance<-tolerance
 }
 implied<-fitted(fit)
 summaries<-lapply(case$codes,function(code) {d<-as.matrix(data[data$g==code,case$vars,drop=FALSE]);centered<-sweep(d,2,colMeans(d),'-');list(n=n,cov=unname(crossprod(centered)/n),mean=unname(colMeans(d)))})
 rows<-lapply(seq_len(nrow(pt)),function(j) list(lhs=pt$lhs[j],op=pt$op[j],rhs=pt$rhs[j],group=pt$group[j],free=pt$free[j],label=if(is.na(pt$label[j])) '' else pt$label[j],ustart=pt$ustart[j],est=pt$est[j],se=pt$se[j]))
 results[[i]]<-list(id=case$id,mplus_input=case$input,lavaan_syntax=case$syntax,variables=case$vars,codes=as.character(case$codes),labels=case$labels,sample=summaries,rows=rows,
 implied=unname(lapply(implied,function(g) list(variables=colnames(g$cov),cov=unname(g$cov),mean=unname(g$mean)))),
 npar=unname(fm['npar']),df=unname(fm['df']),chisq=unname(fm['chisq']),pvalue=unname(fm['pvalue']),demo=list(npar=free,df=df,chisq=chi,tech1=tech1(lines),estimates=demo_estimates))
 cat(case$id,': npar',free,'df',df,'chi error',abs(chi-fm['chisq']),'\n')
}
output<-file.path(root,'cpp/tests/fixtures/mplus/golden.json')
fixture<-list()
fixture$multigroup_meta<-list(lavaan_version=as.character(packageVersion('lavaan')),mplus_version='9.1 Demo',generator='cpp/tests/tools/regen_oracle_mplus_groups.R',provenance='Independent explicit grouped lavaan models; synthetic complete observations; derived summaries only')
fixture$multigroup_cases<-results
# Preserve the single-group fixture's bytes and numeric spelling on extension.
base<-paste(readLines(output,warn=FALSE),collapse='\n')
if(grepl('\"multigroup_meta\"',base,fixed=TRUE)) {
 base<-sub(',\n  "multigroup_meta":.*$','',base)
} else base<-sub('\n?}[[:space:]]*$','',base)
extra<-toJSON(fixture,auto_unbox=TRUE,pretty=TRUE,digits=16,na='null',null='null')
writeLines(paste0(base,',',substr(extra,2,nchar(extra)-1),'\n}'),output)
stopifnot(file.info(output)$size<1000000)
