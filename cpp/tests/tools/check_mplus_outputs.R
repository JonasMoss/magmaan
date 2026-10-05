#!/usr/bin/env Rscript
# Local, data-free output meaning gate. Originals remain in the corpus/ZIPs.
# TASK-85 draft: nontrivial constraint rank and missing threshold rows fail.
# Exit failure is intentional until every accepted case is resolved.
suppressPackageStartupMessages(library(magmaanlab))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(digest))
args <- commandArgs(TRUE)
corpus <- normalizePath(if(length(args)) args[1] else 'external/textbook-corpus')
report <- if(length(args)>1) args[2] else 'cpp/tests/fixtures/mplus/out_meaning_summary.json'
number <- function(lines, pattern) {
  hit <- grep(pattern,lines,value=TRUE,ignore.case=TRUE)
  if(!length(hit)) return(NA_integer_)
  as.integer(sub('.*?([0-9]+)\\s*$', '\\1', hit[1],perl=TRUE))
}
# Threshold counts come only from unstandardized MODEL RESULTS, per group.
# Missing threshold rows fail explicitly; no category count is guessed.
threshold_counts <- function(lines, spec, ng) {
  counts <- matrix(NA_integer_,ng,length(spec$ordered),dimnames=list(NULL,spec$ordered))
  start <- grep('^MODEL RESULTS\\s*$',lines)
  if(length(start)) {
    group <- 1L; active <- FALSE
    for(line in lines[seq.int(start[1]+1L,length(lines))]) {
      text <- trimws(line)
      if(grepl('^(STANDARDIZED|QUALITY OF|R-SQUARE|CONFIDENCE)',text)) break
      if(startsWith(text,'Group ')) {
        group <- match(tolower(sub('^Group ','',text)),tolower(spec$group_labels));active<-FALSE
      }
      if(text=='Thresholds') {active<-TRUE;next}
      if(active) {
        m<-regmatches(text,regexec('^([A-Za-z0-9_]+)[$]([0-9]+) +[-0-9.]',text))[[1]]
        if(length(m) && !is.na(group)) {
          v<-match(tolower(m[2]),tolower(spec$ordered))
          if(!is.na(v)) counts[group,v]<-max(c(counts[group,v],as.integer(m[3])),na.rm=TRUE)
        } else if(nzchar(text) && !grepl('^[0-9 .-]+$',text)) active<-FALSE
      }
    }
  }
  counts
}
model_dimension <- function(spec, lines) {
  pt <- spec$partable
  ng <- max(c(1L,pt$group))
  if(any(pt$op %in% c('<','>','new'))) stop('constraint rank needs independent counting')
  equalities<-pt[pt$op=='==',,drop=FALSE]
  for(i in seq_len(nrow(equalities))) {
    lhs<-match(equalities$lhs[i],pt$plabel);rhs<-match(equalities$rhs[i],pt$plabel)
    if(is.na(lhs)||is.na(rhs)||!nzchar(pt$label[lhs])||pt$label[lhs]!=pt$label[rhs])
      stop('constraint rank needs independent counting')
  }
  # Growth thresholds are completed from categories after lowering. The same
  # index is shared across the outcomes of each | time-score statement (GR05).
  source<-gsub('!.*?(\\n|$)',' ',spec$mplus_source,perl=TRUE)
  statements<-strsplit(source,';',fixed=TRUE)[[1]]
  growth<-list()
  for(statement in statements[grepl('[|].*@',statements)]) {
    rhs<-sub('.*[|]','',statement)
    tokens<-regmatches(rhs,gregexpr('[A-Za-z][A-Za-z0-9_]*(?=\\s*@)',rhs,perl=TRUE))[[1]]
    if(length(tokens)) growth[[length(growth)+1L]]<-tolower(tokens)
  }
  free <- pt$free>0L & pt$op!='|'
  coordinates <- ifelse(nzchar(pt$label[free]),paste0('label:',pt$label[free]),paste0('free:',pt$free[free]))
  thresholds <- threshold_counts(lines,spec,ng)
  if(anyNA(thresholds)) stop('threshold counts absent from MODEL RESULTS')
  indicators <- unique(pt$rhs[pt$op=='=~'])
  for(g in seq_len(ng)) for(v in spec$ordered) for(k in seq_len(thresholds[g,v])) {
    row<-which(pt$lhs==v & pt$op=='|' & pt$rhs==paste0('t',k) & pt$group==g)
    explicit<-length(row)>0 && pt$user[row[1]]!=0L
    if(explicit && pt$free[row[1]]==0L) next
    coordinate<-if(explicit && nzchar(pt$label[row[1]])) paste0('label:',pt$label[row[1]]) else
      if(explicit || !v %in% indicators) paste('threshold',g,v,k,sep=':') else {
          bundle<-which(vapply(growth,function(vars) tolower(v) %in% vars,logical(1)))
          paste('threshold',if(length(bundle)) paste0('growth',bundle[1]) else v,k,sep=':')
        }
    coordinates<-c(coordinates,coordinate)
  }
  npar<-length(unique(coordinates))
  x<-spec$mplus_observed_x
  outcomes<-setdiff(spec$mplus_data_plan$analysis,x)
  q<-length(setdiff(outcomes,spec$ordered))
  means<-any(pt$op=='~1' & pt$lhs %in% outcomes)
  moments<-sum(thresholds)+ng*(q+as.integer(means)*q+choose(length(outcomes),2L)+length(outcomes)*length(x))
  list(groups=ng,npar=npar,df=moments-npar)
}
entries <- list()
for(path in sort(list.files(corpus,'\\.out$',recursive=TRUE,full.names=TRUE,ignore.case=TRUE)))
  entries[[length(entries)+1L]]<-list(path=substring(path,nchar(corpus)+2L),file=path)
for(path in sort(list.files(corpus,'\\.zip$',recursive=TRUE,full.names=TRUE,ignore.case=TRUE))) {
  members<-tryCatch(unzip(path,list=TRUE)$Name,error=function(e) character())
  for(member in members[grepl('\\.out$',members,ignore.case=TRUE)])
    entries[[length(entries)+1L]]<-list(path=paste0(substring(path,nchar(corpus)+2L),'!',member),file=path,member=member)
}
records<-list(); seen<-new.env(parent=emptyenv()); mplus<-errors<-missing_input<-0L
for(entry in entries) {
  con<-if(is.null(entry$member)) file(entry$file,encoding='latin1') else unz(entry$file,entry$member,encoding='latin1')
  lines<-tryCatch(readLines(con,warn=FALSE),finally=close(con))
  lines<-iconv(lines,from='latin1',to='UTF-8',sub='byte')
  header<-grep('^Mplus VERSION',lines,value=TRUE)
  if(!length(header)) next
  mplus<-mplus+1L
  version<-trimws(sub('^Mplus VERSION','',header[1]))
  start<-grep('^INPUT INSTRUCTIONS\\s*$',lines)
  end<-grep('^INPUT READING TERMINATED NORMALLY|^SUMMARY OF ANALYSIS|^\\s*\\*\\*\\* (ERROR|WARNING)',lines)
  if(any(grepl('\\*\\*\\* ERROR',lines))) {errors<-errors+1L;next}
  end<-end[end>start[1]]
  if(!length(start)||!length(end)) {missing_input<-missing_input+1L;next}
  input<-paste(sub('^  ','',lines[seq.int(start[1]+1L,end[1]-1L)]),collapse='\n')
  normalized<-tolower(gsub('\\s+',' ',trimws(input)))
  hash<-substr(digest(normalized,algo='sha256',serialize=FALSE),1,16)
  expected<-list(groups=number(lines,'^Number of groups'),npar=number(lines,'^Number of Free Parameters'))
  fit<-grep('^Chi-Square Test of Model Fit\\s*$',lines)
  expected$df<-if(length(fit)) number(lines[seq.int(fit[1]+1L,min(length(lines),fit[1]+12L))],'Degrees of Freedom') else NA_integer_
  # Keep separate observations of the same input: versions can change meaning.
  if(exists(hash,seen,inherits=FALSE)) {
    index<-get(hash,seen); records[[index]]$versions<-unique(c(records[[index]]$versions,version))
    records[[index]]$outputs<-c(records[[index]]$outputs,list(list(path=entry$path,version=version,expected=expected)))
    next
  }
  spec<-tryCatch(mplus_model(input=input),error=function(e)e)
  record<-list(hash=hash,versions=version,outputs=list(list(path=entry$path,version=version,expected=expected)))
  if(inherits(spec,'error')) {
    codes<-unique(regmatches(conditionMessage(spec),gregexpr('\\[[A-Z]+[0-9]+[a-z]?\\]',conditionMessage(spec)))[[1]])
    record$status<-'rejected';record$rules<-codes
  } else {
    record$status<-'accepted'
    dimension<-tryCatch(model_dimension(spec,lines),error=function(e)e)
    if(inherits(dimension,'error')) {record$classification<-'unresolved_counting';record$detail<-conditionMessage(dimension)} else record$actual<-dimension
  }
  records[[length(records)+1L]]<-record;assign(hash,length(records),seen)
}
for(i in seq_along(records)) {
  r<-records[[i]]
  if(r$status!='accepted' || is.null(r$actual)) next
  for(j in seq_along(r$outputs)) {
    o<-r$outputs[[j]]
    fields<-c('groups','npar','df')
    o$comparison<-setNames(lapply(fields,function(f) if(is.na(o$expected[[f]])) 'not_printed' else if(o$expected[[f]]==r$actual[[f]]) 'match' else 'mismatch'),fields)
    r$outputs[[j]]<-o
  }
  r$classification<-if(any(vapply(r$outputs,function(o) any(unlist(o$comparison)=='mismatch'),logical(1)))) 'unclassified_mismatch' else 'match'
  records[[i]]<-r
}
totals<-list(outputs_scanned=length(entries),mplus_outputs=mplus,mplus_error_outputs=errors,missing_input=missing_input,
             distinct_inputs=length(records),status=as.list(table(vapply(records,function(r)r$status,character(1)))))
accepted<-Filter(function(r)r$status=='accepted',records)
totals$rejected_by_rule<-as.list(table(unlist(lapply(Filter(function(r)r$status=='rejected',records),function(r)r$rules))))
totals$classification<-as.list(table(vapply(accepted,function(r)r$classification,character(1))))
for(f in c('groups','npar','df')) totals[[paste0(f,'_match')]]<-sum(vapply(accepted,function(r) !is.null(r$actual) && all(vapply(r$outputs,function(o) identical(o$comparison[[f]],'match'),logical(1))),logical(1)))
write_json(list(totals=totals,inputs=records),report,pretty=TRUE,auto_unbox=TRUE,na='null')
print(totals)
if(any(vapply(accepted,function(r)r$classification!='match',logical(1)))) stop('Unresolved meaning/counting cases; see derived summary')
