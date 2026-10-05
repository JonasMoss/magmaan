#!/usr/bin/env Rscript
# Local, data-free output meaning gate. Originals remain in the corpus/ZIPs.
# Dimensions use an independent symbolic constraint Jacobian and printed categories.
suppressPackageStartupMessages(library(magmaanlab))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(digest))
args <- commandArgs(TRUE)
report <- if(length(args)>1) args[2] else 'cpp/tests/fixtures/mplus/out_meaning_summary.json'
number <- function(lines, pattern) {
  hit <- grep(pattern,trimws(lines),value=TRUE,ignore.case=TRUE)
  if(!length(hit)) return(NA_integer_)
  as.integer(sub('.*?([0-9]+)\\s*$', '\\1', hit[1],perl=TRUE))
}
# Threshold counts come only from unstandardized MODEL RESULTS, per group.
# Missing rows fall back to positive-count categories in the printed proportions.
threshold_counts <- function(lines, spec, ng) {
  counts <- matrix(NA_integer_,ng,length(spec$ordered),dimnames=list(NULL,spec$ordered))
  start <- grep('^MODEL RESULTS\\s*$',lines)
  if(length(start)) {
    group <- 1L; active <- FALSE
    for(line in lines[seq.int(start[1]+1L,length(lines))]) {
      text <- trimws(line)
      if(grepl('^(STANDARDIZED|QUALITY OF|R-SQUARE|CONFIDENCE)',text)) break
      if(startsWith(text,'Group ')) {
        group <- match(tolower(sub('^Group ','',text)),tolower(spec$mplus_groups$label));active<-FALSE
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
  category_counts <- matrix(0L,ng,length(spec$ordered))
  start <- grep('^UNIVARIATE PROPORTIONS AND COUNTS FOR CATEGORICAL VARIABLES',lines)
  if(length(start)) {
    group <- 1L; variable <- NA_integer_
    for(line in lines[seq.int(start[1]+1L,length(lines))]) {
      text <- trimws(line)
      if(grepl('^(UNIVARIATE SAMPLE|MODEL FIT|SUMMARY OF|THE MODEL)',text)) break
      if(startsWith(text,'Group ')) {
        group <- match(tolower(sub('^Group ','',text)),tolower(spec$mplus_groups$label))
        variable <- NA_integer_
      }
      if(tolower(text) %in% tolower(spec$ordered)) variable <- match(tolower(text),tolower(spec$ordered))
      m <- regmatches(text,regexec('^Category +[0-9]+ +([0-9.]+) +([0-9.]+)',text))[[1]]
      if(length(m) && !is.na(group) && !is.na(variable) && as.numeric(m[3])>0)
        category_counts[group,variable] <- category_counts[group,variable]+1L
    }
  }
  fallback <- is.na(counts) & category_counts>1L
  counts[fallback] <- category_counts[fallback]-1L
  attr(counts,'fallback_cells') <- sum(fallback)
  counts
}
# Replace parameter aliases by independent coordinate symbols, then differentiate
# the equality residuals. Derived := quantities are substitutions, not restrictions.
constraint_rank <- function(pt, coordinates, row_coordinates) {
  equalities <- pt[pt$op=='==',,drop=FALSE]
  if(!nrow(equalities)) return(0L)
  symbols <- paste0('q',seq_along(coordinates))
  aliases <- list()
  for(i in seq_len(nrow(pt))) if(!pt$op[i] %in% c('==',':=','<','>')) {
    if(pt$op[i]=='new' && pt$free[i]==0L) next # declaration metadata, not a fixed coordinate
    value <- if(pt$free[i]>0L) as.name(symbols[match(row_coordinates[i],coordinates)]) else pt$ustart[i]
    for(name in c(pt$plabel[i],pt$label[i],if(pt$op[i]=='new') pt$lhs[i] else ''))
      if(nzchar(name)) aliases[[name]] <- value
  }
  definitions <- pt[pt$op==':=',,drop=FALSE]
  substitute_alias <- function(expr, visiting=character()) {
    if(is.name(expr)) {
      name <- as.character(expr)
      if(name %in% names(aliases)) return(aliases[[name]])
      hit <- match(name,definitions$lhs)
      if(!is.na(hit)) {
        if(name %in% visiting) stop('cyclic derived quantity')
        return(substitute_alias(str2lang(definitions$rhs[hit]),c(visiting,name)))
      }
      stop('unknown constraint parameter: ',name)
    }
    if(is.call(expr)) {
      for(i in seq.int(2L,length(expr))) expr[[i]] <- substitute_alias(expr[[i]],visiting)
    }
    expr
  }
  residuals <- lapply(seq_len(nrow(equalities)),function(i)
    substitute_alias(call('-',str2lang(equalities$lhs[i]),str2lang(equalities$rhs[i]))))
  derivatives <- lapply(residuals,function(expr) lapply(symbols,function(v) D(expr,v)))
  ranks <- vapply(1:3,function(point) {
    values <- as.list(1.2+sin(seq_along(symbols)*point)*.3); names(values) <- symbols
    jacobian <- t(vapply(derivatives,function(row) vapply(row,function(expr)
      as.numeric(eval(expr,values)),numeric(1)),numeric(length(symbols))))
    if(any(!is.finite(jacobian))) stop('nonfinite generic constraint Jacobian')
    singular <- svd(jacobian,nu=0,nv=0)$d
    # Absolute floor handles exact zero label equalities after substitution.
    sum(singular>max(1e-9,max(c(0,singular))*1e-8))
  },integer(1))
  if(length(unique(ranks))!=1L) stop('unstable generic constraint rank')
  ranks[1]
}
model_dimension <- function(spec, lines) {
  pt <- spec$partable
  ng <- max(c(1L,pt$group))
  if(any(pt$op %in% c('<','>'))) stop('inequality dimension requires an interior regime')
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
  row_coordinates <- ifelse(nzchar(pt$label),paste0('label:',pt$label),paste0('free:',pt$free))
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
    if(length(row)) row_coordinates[row] <- coordinate
    coordinates<-c(coordinates,coordinate)
  }
  rank <- constraint_rank(pt,unique(coordinates),row_coordinates)
  npar<-length(unique(coordinates))-rank
  x<-spec$mplus_observed_x
  outcomes<-setdiff(spec$mplus_data_plan$analysis,x)
  q<-length(setdiff(outcomes,spec$ordered))
  means<-any(pt$op=='~1' & pt$lhs %in% outcomes)
  moments<-sum(thresholds)+ng*(q+as.integer(means)*q+choose(length(outcomes),2L)+length(outcomes)*length(x))
  list(groups=ng,npar=npar,df=moments-npar,constraint_rank=rank,
       threshold_fallback_cells=attr(thresholds,'fallback_cells'))
}
if('--self-test' %in% args) {
  stopifnot(number('          Number of Free Parameters             21','^Number of Free Parameters')==21L)
  input <- function(model) paste('DATA: FILE=x;\nVARIABLE: NAMES=y1 y2 y3;\nMODEL:',model)
  base <- 'f BY y1* (l1)\ny2 (l2)\ny3 (l3); f@1;'
  examples <- list(
    list(model=base,rank=0L,npar=9L),
    list(model=paste(base,'\nMODEL CONSTRAINT: NEW(d); d=l1*l2;'),rank=0L,npar=9L),
    list(model=paste(base,'\nMODEL CONSTRAINT: l1=l2*l3; 2*l1=2*l2*l3;'),rank=1L,npar=8L),
    list(model=paste(base,'\nMODEL CONSTRAINT: NEW(c); l1=c*l2;'),rank=1L,npar=9L),
    list(model=paste(base,'\nMODEL CONSTRAINT: l1=l2; l2=l3;'),rank=2L,npar=7L),
    list(model=paste(base,'\nMODEL CONSTRAINT: NEW(d); d=l1-l2; d=0;'),rank=1L,npar=8L))
  for(example in examples) {
    actual <- model_dimension(mplus_model(input=input(example$model)),character())
    stopifnot(actual$constraint_rank==example$rank,actual$npar==example$npar)
  }
  cat('Independent dimension examples: 6 passed\n')
  quit(status=0)
}
corpus <- normalizePath(if(length(args)) args[1] else 'external/textbook-corpus')
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
  # Physical line boundaries carry label semantics (LB02/LB03).
  normalized<-tolower(paste(trimws(strsplit(gsub('[ \t]+',' ',input),'\n',fixed=TRUE)[[1]]),collapse='\n'))
  hash<-substr(digest(normalized,algo='sha256',serialize=FALSE),1,16)
  expected<-list(groups=number(lines,'^Number of groups'),npar=number(lines,'^Number of Free Parameters'),
    npar_format=if(any(grepl('^\\s+Number of Free Parameters',lines))) 'indented' else 'unindented')
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
# Cases diagnosed with the old frontend before the bracket fix. The six
# constraint cases had missing aliases; the last two had wrong printed counts.
fixed_bracket_inputs <- c('f99385dddd126fa1','772cdc7a02a0ff2e','2cd02d04b1c14118',
  'c758f64bac0d0647','11654563391d771e','8c3d405408061b13',
  '9bf71b28ab3693ad','776ae6ac019c541b')
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
  if(r$hash %in% fixed_bracket_inputs && r$classification=='match')
    r$resolution <- list(class='fixed_magmaan_bug',rule='LB02',
      detail='later bracket-group labels were silently dropped; canonical regression and P-LB7')
  if(is.null(r$resolution) && r$classification=='match' &&
     any(vapply(r$outputs,function(o) o$expected$npar_format=='indented',logical(1))))
    r$resolution <- list(class='gate_artifact',rule='output_format',
      detail='Mplus 5.1/5.2 indents the printed free-parameter heading; trim before extraction')
  records[[i]]<-r
}
totals<-list(outputs_scanned=length(entries),mplus_outputs=mplus,mplus_error_outputs=errors,missing_input=missing_input,
             distinct_inputs=length(records),status=as.list(table(vapply(records,function(r)r$status,character(1)))))
accepted<-Filter(function(r)r$status=='accepted',records)
totals$rejected_by_rule<-as.list(table(unlist(lapply(Filter(function(r)r$status=='rejected',records),function(r)r$rules))))
totals$resolved_by_class<-as.list(table(vapply(accepted,function(r) if(is.null(r$resolution)) 'unchanged_match' else r$resolution$class,character(1))))
totals$classification<-as.list(table(vapply(accepted,function(r)r$classification,character(1))))
for(f in c('groups','npar','df')) totals[[paste0(f,'_match')]]<-sum(vapply(accepted,function(r) !is.null(r$actual) && all(vapply(r$outputs,function(o) identical(o$comparison[[f]],'match'),logical(1))),logical(1)))
write_json(list(totals=totals,inputs=records),report,pretty=TRUE,auto_unbox=TRUE,na='null')
print(totals)
if(any(vapply(accepted,function(r)r$classification!='match',logical(1)))) stop('Unresolved meaning/counting cases; see derived summary')
