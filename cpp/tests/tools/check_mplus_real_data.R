#!/usr/bin/env Rscript
# Local real-file reader gate. Third-party originals remain outside Git.
# Analysis-sample selection belongs here, never in mplus_data().
suppressPackageStartupMessages(library(magmaanlab))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(digest))
args <- commandArgs(TRUE)
corpus <- normalizePath(if(length(args)) args[1] else '/home/jonas/Files/research/magmaan/external/textbook-corpus')
report <- if(length(args)>1) args[2] else 'cpp/tests/fixtures/mplus/real_data_summary.json'
scratch <- path.expand('~/.cache/magmaan-logs/task-108-real-data')
dir.create(scratch,recursive=TRUE,showWarnings=FALSE)
read_text <- function(path) iconv(readLines(path,warn=FALSE,encoding='latin1'),from='latin1',to='UTF-8',sub='byte')
short_hash <- function(path) substr(digest(file=path,algo='sha256'),1,16)
entries <- list(); archive_errors <- list()
# Keep archive namespaces separate: identical basenames in different books are
# never interchangeable references. No corpus translator is imported.
for(zip in sort(list.files(file.path(corpus,'raw'),'\\.zip$',recursive=TRUE,full.names=TRUE,ignore.case=TRUE))) {
  members <- tryCatch(unzip(zip,list=TRUE)$Name,error=function(e) character())
  if(!length(members)) {archive_errors[[length(archive_errors)+1L]] <- substring(zip,nchar(corpus)+2L); next}
  members <- members[!grepl('/$',members)]
  if(any(grepl('(^/|(^|/)\\.\\.(/|$))',members))) stop('Unsafe archive member: ',zip)
  dir <- file.path(scratch,short_hash(zip));dir.create(dir,showWarnings=FALSE)
  missing_members <- members[!file.exists(file.path(dir,members))]
  if(length(missing_members)) unzip(zip,files=missing_members,exdir=dir)
  paths <- file.path(dir,members)
  for(i in which(grepl('\\.inp$',members,ignore.case=TRUE)))
    entries[[length(entries)+1L]] <- list(path=paste0(substring(zip,nchar(corpus)+2L),'!',members[i]),file=paths[i],pool=paths,archive=zip,base=dir)
}
for(path in sort(list.files(corpus,'\\.inp$',recursive=TRUE,full.names=TRUE,ignore.case=TRUE)))
  entries[[length(entries)+1L]] <- list(path=substring(path,nchar(corpus)+2L),file=path,pool=list.files(dirname(path),full.names=TRUE),archive=NULL)
resolve <- function(name,entry) {
  name <- gsub('\\\\','/',name)
  local <- file.path(dirname(entry$file),name)
  hit <- entry$pool[tolower(entry$pool)==tolower(local)]
  if(!length(hit)) hit <- entry$pool[tolower(basename(entry$pool))==tolower(basename(name))]
  if(length(hit)==1L) hit else NA_character_
}
analysis_sample <- function(d,spec) {
  analysis <- spec$mplus_data_plan$analysis
  x <- intersect(spec$mplus_observed_x,analysis)
  outcomes <- setdiff(analysis,x)
  keep <- if(length(x)) complete.cases(d[,x,drop=FALSE]) else rep(TRUE,nrow(d))
  if(length(outcomes)) keep <- keep & rowSums(!is.na(d[,outcomes,drop=FALSE]))>0L
  if(spec$mplus_data_plan$listwise) keep <- keep & complete.cases(d[,analysis,drop=FALSE])
  d[keep,,drop=FALSE]
}
blocks <- function(d,spec) {
  if(!nzchar(spec$group_var)) return(list(single=d))
  setNames(lapply(spec$group_labels,function(g) d[as.character(d[[spec$group_var]])==g,,drop=FALSE]),spec$group_labels)
}
printed_n <- function(lines,ng) {
  h <- grep('^Number of observations(\\s|$)',trimws(lines))
  if(!length(h)) return(NULL)
  text <- trimws(lines[h[1]])
  if(ng==1L && grepl('[0-9]+$',text)) return(as.integer(sub('.*?([0-9]+)$','\\1',text,perl=TRUE)))
  following <- trimws(lines[seq.int(h[1]+1L,min(length(lines),h[1]+ng+4L))])
  values <- following[grepl('^(Group +)?[A-Za-z0-9_\\-]+\\s+[0-9]+$',following)]
  if(length(values)<ng) stop('Cannot parse grouped observation counts')
  as.integer(sub('.*?([0-9]+)$','\\1',head(values,ng),perl=TRUE))
}
# Parse labelled, paginated sample-statistic tables, retaining each token's
# precision. Model results and univariate higher moments are not SAMPSTAT.
printed_moments <- function(lines,spec) {
  start <- grep('^SAMPLE STATISTICS$',trimws(lines))
  if(!length(start)) return(list())
  end <- grep('^(UNIVARIATE|THE MODEL|MODEL FIT|SUMMARY OF DATA|QUALITY OF)',trimws(lines))
  end <- end[end>start[1]]
  section <- lines[seq.int(start[1]+1L,if(length(end)) end[1]-1L else length(lines))]
  mode <- ''; group <- 1L; columns <- character(); out <- list()
  for(line in section) {
    t <- trimws(line)
    if(grepl('^(Group |SAMPLE STATISTICS FOR )',t)) {
      label<-sub('^(Group |SAMPLE STATISTICS FOR )','',t)
      group<-match(tolower(label),tolower(spec$mplus_groups$label))
      if(is.na(group)) stop('Unknown printed sample-statistics group: ',label)
      columns<-character();mode<-'';next
    }
    if(grepl('^Means(/Intercepts/Thresholds)?$',t)) {mode<-'mean';columns<-character();next}
    if(grepl('^Covariances(/Correlations/Residual Correlations)?$',t)) {mode<-'cov';columns<-character();next}
    if(grepl('^(Correlations|Skewness|Kurtosis|Minimum|Maximum)',t)) {mode<-'';next}
    if(!nzchar(mode)||!nzchar(t)||grepl('^_+$',t)) next
    tokens <- strsplit(t,'\\s+')[[1]]
    if(all(grepl('^[A-Za-z][A-Za-z0-9_]*$',tokens))) {columns<-tolower(tokens);next}
    if(!length(columns)) next
    numeric_token <- '^[+-]?[0-9]*[.]?[0-9]+([EeDd][+-]?[0-9]+)?$'
    if(mode=='mean' && length(tokens)==length(columns)+1L && tokens[1]=='1') tokens<-tokens[-1]
    if(mode=='mean' && length(tokens)==length(columns) && all(grepl(numeric_token,tokens))) {
      for(j in seq_along(tokens)) out[[length(out)+1L]]<-list(group=group,kind='mean',lhs=columns[j],rhs='',printed=tokens[j])
    } else if(mode=='cov' && length(tokens)>1L && grepl('^[A-Za-z]',tokens[1]) && all(grepl(numeric_token,tokens[-1]))) {
      for(j in seq_along(tokens[-1])) out[[length(out)+1L]]<-list(group=group,kind='cov',lhs=tolower(tokens[1]),rhs=columns[j],printed=tokens[j+1L])
    }
  }
  out
}
# Independent BASIC references for the two real legacy EOF files. The Demo
# stays optional for other pairs; these diagnosed reader regressions require it.
demo_eof_reference <- function(files, spec, hashes) {
  known <- c('1907b41d217bbcdf', 'ddbc84b438fc77c2')
  if(length(files) != 1L || !hashes[1L] %in% known) return(NA_character_)
  binary <- hashes[1L] == known[1L]
  dir <- file.path(scratch, paste0('demo-eof-', hashes[1L]))
  dir.create(dir, showWarnings=FALSE)
  file.copy(files[1L], file.path(dir,'data.dat'), overwrite=TRUE)
  writeLines(c('DATA: FILE=data.dat;', if(binary) 'FORMAT=6F1;',
    paste0('VARIABLE: NAMES=', paste(spec$mplus_data_plan$names,collapse=' '), ';'),
    'ANALYSIS: TYPE=BASIC;', 'OUTPUT: SAMPSTAT;'), file.path(dir,'probe.inp'))
  old <- getwd(); on.exit(setwd(old)); setwd(dir)
  status <- system2('/home/jonas/mplusdemo/mpdemo', c('probe.inp','probe.out'),
    stdout='console.log',stderr='console.err',timeout=120)
  if(status != 0L || !file.exists('probe.out')) stop('EOF real-file Demo probe failed')
  normalizePath('probe.out')
}
records <- list()
for(entry in entries) {
  if(length(records) %% 100L == 0L) {cat('Inputs processed:',length(records),'/',length(entries),'\n');flush.console()}
  r <- list(path=entry$path,input_hash=short_hash(entry$file))
  spec <- tryCatch(mplus_model(file=entry$file),error=function(e)e)
  if(inherits(spec,'error')) {r$status<-'rejected';records[[length(records)+1L]]<-r;next}
  r$status <- 'accepted'
  files <- vapply(spec$mplus_data_plan$files,function(f) resolve(f$path,entry),'')
  if(!length(files)||anyNA(files)) {r$classification<-'unresolved_file';records[[length(records)+1L]]<-r;next}
  r$data_hash <- vapply(files,short_hash,'')
  r$data_path <- vapply(files,function(f) if(is.null(entry$archive)) substring(f,nchar(corpus)+2L) else paste0(substring(entry$archive,nchar(corpus)+2L),'!',substring(f,nchar(entry$base)+2L)),'')
  result <- tryCatch({
    d <- mplus_data(spec,file=files)
    summary <- !is.data.frame(d)
    if(summary) {
      moments <- Map(function(S,mu,n) list(S=S,mean=mu,n=n),attr(d,'mplus_data_report')$input_covariance,d$mean,d$nobs)
      ns <- d$nobs
    } else {
      sample <- analysis_sample(d,spec); b <- blocks(sample,spec);ns <- vapply(b,nrow,integer(1))
      moments <- lapply(b,function(v) {
        v <- as.matrix(v[,spec$mplus_data_plan$analysis,drop=FALSE])
        if(anyNA(v)) return(NULL) # incomplete-data H1 moments require estimation
        list(S=crossprod(scale(v,scale=FALSE))/nrow(v),mean=colMeans(v),n=nrow(v))
      })
    }
    r$reader_n <- if(summary) d$nobs else as.integer(attr(d,'mplus_data_report')$nobs)
    r$analysis_n <- unname(ns)
    # Verified cases resolve to the corresponding archived original input,
    # not to the translator's generated syntax or data.
    raw_match <- NULL
    if(grepl('raw/(mplususerguid|MPLUS)/chapter',entry$path)) {
      ex <- sub('\\.inp$','',basename(entry$file),ignore.case=TRUE)
      folders <- list.dirs(file.path(corpus,'cases/mplus_users_guide_v8'),recursive=FALSE,full.names=TRUE)
      folder <- folders[endsWith(folders,paste0('_',gsub('[.]','_',ex)))]
      if(length(folder)==1L && file.exists(file.path(folder,'data/raw.csv'))) {
        ref <- read.csv(file.path(folder,'data/raw.csv'),check.names=FALSE)
        names(ref)<-tolower(names(ref));names(sample)<-tolower(names(sample))
        if(nzchar(spec$group_var)) {
          g<-tolower(spec$group_var)
          for(i in seq_len(nrow(spec$mplus_groups))) ref[[g]][toupper(as.character(ref[[g]]))==toupper(spec$mplus_groups$label[i])]<-spec$mplus_groups$code[i]
          ref[[g]]<-as.numeric(ref[[g]])
        }
        expected_columns <- tolower(unique(c(spec$mplus_data_plan$analysis,if(nzchar(spec$group_var)) spec$group_var)))
        ref <- ref[,expected_columns,drop=FALSE]
        r$raw_column_match <- identical(names(sample),names(ref))
        same_na <- identical(dim(sample),dim(ref)) && all(is.na(as.matrix(sample))==is.na(as.matrix(ref)))
        raw_match <- r$raw_column_match && nrow(sample)==nrow(ref) && same_na && isTRUE(all.equal(as.matrix(sample),as.matrix(ref),tolerance=1e-10,check.attributes=FALSE))
        r$raw_reference <- substring(folder,nchar(corpus)+2L);r$raw_n<-nrow(ref);r$raw_na_match<-same_na;r$raw_match<-raw_match
      }
    }
    out <- resolve(sub('\\.inp$','.out',basename(entry$file),ignore.case=TRUE),entry)
    if(is.na(out)) {
      out <- demo_eof_reference(files,spec,r$data_hash)
      if(!is.na(out)) r$reference_kind <- 'Mplus_9.1_Demo_BASIC_real_file'
    }
    n_match <- pattern_match <- moment_match <- NULL
    if(!is.na(out)) {
      lines <- read_text(out)
      if(any(grepl('*** ERROR',lines,fixed=TRUE))) r$output<-'mplus_error' else {
        expected_n <- printed_n(lines,length(ns));r$output_n<-expected_n
        if(!is.null(expected_n)) n_match<-identical(as.integer(ns),expected_n)
        h<-grep('^Number of missing data patterns\\s+[0-9]+$',trimws(lines),ignore.case=TRUE)
        if(length(h) && !summary && length(b)==1L) {
          expected_pattern<-as.integer(sub('.*?([0-9]+)$','\\1',trimws(lines[h[1]]),perl=TRUE))
          actual_pattern<-length(unique(apply(is.na(sample[,spec$mplus_data_plan$analysis,drop=FALSE]),1,paste,collapse='')))
          r$output_patterns<-expected_pattern;r$actual_patterns<-actual_pattern;pattern_match<-actual_pattern==expected_pattern
        }
        cells <- printed_moments(lines,spec);checks<-logical();max_error<-0
        for(cell in cells) {
          if(cell$group>length(moments)) stop('Sample statistics group outside reader groups')
          m<-moments[[cell$group]]
          if(is.null(m)) next
          # Fixed-X sample statistics may contain X-only blocks; use names,
          # never rely on a model/reader column-position coincidence.
          if(!cell$lhs %in% tolower(colnames(m$S))) next
          lhs<-match(cell$lhs,tolower(colnames(m$S)))
          if(cell$kind=='cov' && !cell$rhs %in% tolower(colnames(m$S))) next
          target<-if(cell$kind=='mean') m$mean[lhs] else m$S[lhs,match(cell$rhs,tolower(colnames(m$S)))]
          if(is.null(target)||!length(target)) next
          token<-gsub('[dD]','e',cell$printed); expected<-as.numeric(token)
          mantissa<-sub('[eE].*','',token);digits<-if(grepl('.',mantissa,fixed=TRUE)) nchar(sub('.*[.]','',mantissa)) else 0L
          exponent<-if(grepl('[eE]',token)) as.numeric(sub('.*[eE]','',token)) else 0
          tolerance<-.5*10^(exponent-digits)+1e-8
          checks<-c(checks,is.finite(target)&&abs(target-expected)<=tolerance);max_error<-max(max_error,abs(target-expected))
        }
        r$moment_cells<-length(checks);r$moment_max_error<-if(length(checks)) max_error else NULL
        r$moment_status<-if(!length(checks)) if(length(cells)) 'incomplete_data_requires_H1' else 'not_printed' else if(all(checks)) 'match' else 'mismatch'
        if(length(checks)) moment_match<-all(checks)
      }
    }
    r$n_match<-n_match;r$pattern_match<-pattern_match
    checks<-c(raw_match,n_match,pattern_match,moment_match)
    r$classification<-if(length(checks) && any(!checks)) 'unclassified_mismatch' else if(length(checks)) 'match' else 'read_without_reference'
    r
  },error=function(e) {r$classification<-'unclassified_error';r$detail<-conditionMessage(e)
    r})
  records[[length(records)+1L]]<-result
}
accepted<-Filter(function(r)r$status=='accepted',records)
totals<-list(inputs=length(entries),accepted=length(accepted),archives_unreadable=archive_errors,
  classifications=as.list(table(vapply(accepted,function(r)r$classification,''))),
  raw_comparisons=sum(vapply(accepted,function(r)!is.null(r$raw_match),TRUE)),
  n_comparisons=sum(vapply(accepted,function(r)!is.null(r$n_match),TRUE)),
  pattern_comparisons=sum(vapply(accepted,function(r)!is.null(r$pattern_match),TRUE)),
  moment_comparisons=sum(vapply(accepted,function(r)!is.null(r$moment_cells)&&r$moment_cells>0L,TRUE)))
# Explicitly account for all verified UG samples, including rejected originals.
verified <- lapply(sort(list.dirs(file.path(corpus,'cases/mplus_users_guide_v8'), recursive=FALSE)), function(folder) {
  input <- file.path(folder,'source/original.inp')
  spec <- tryCatch(mplus_model(file=input), error=function(e)e)
  if (inherits(spec,'error')) return(list(case=basename(folder), status='input_rejected', rules=unique(regmatches(conditionMessage(spec), gregexpr('\\[[A-Z]+[0-9]+\\]',conditionMessage(spec)))[[1L]])))
  hits <- Filter(function(r) identical(r$raw_reference, substring(folder,nchar(corpus)+2L)), records)
  list(case=basename(folder), status=if(length(hits)) 'compared' else 'reference_unresolved', comparisons=length(hits))
})
saveRDS(records,file.path(scratch,'records.rds'))
write_json(list(sample_rules='Gate only: missing fixed-X; all dependent variables missing; declared LISTWISE. Reader retains these rows.',totals=totals,verified_cases=verified,inputs=records),report,pretty=TRUE,auto_unbox=TRUE,na='null',digits=NA)
print(totals)
if(any(vapply(accepted,function(r)r$classification %in% c('unclassified_error','unclassified_mismatch','reader_bug_needs_decision'),TRUE))) quit(status=1)
