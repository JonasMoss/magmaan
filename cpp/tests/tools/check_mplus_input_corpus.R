#!/usr/bin/env Rscript
# Optional corpus sweep of original inputs, including ZIP members. No translator.
args<-commandArgs(TRUE)
corpus<-if(length(args)) normalizePath(args[1]) else normalizePath('external/textbook-corpus')
script<-sub('^--file=','',grep('^--file=',commandArgs(),value=TRUE)[1])
root<-normalizePath(file.path(dirname(script),'../../..'))
scratch<-path.expand('~/.cache/magmaan-logs/mplus-input-sweep')
dir.create(scratch,recursive=TRUE,showWarnings=FALSE)
files<-list.files(corpus,'\\.inp$',recursive=TRUE,full.names=TRUE,ignore.case=TRUE)
archives<-list.files(file.path(corpus,'raw'),'\\.zip$',recursive=TRUE,full.names=TRUE,ignore.case=TRUE)
for(i in seq_along(archives)) {
  entries<-tryCatch(unzip(archives[i],list=TRUE)$Name,error=function(e) {cat('Unreadable ZIP:',basename(archives[i]),'\n');character()})
  entries<-entries[grepl('\\.inp$',entries,ignore.case=TRUE)]
  if(!length(entries)) next
  dir<-file.path(scratch,paste0('archive_',i));dir.create(dir,showWarnings=FALSE)
  unzip(archives[i],files=entries,exdir=dir)
  files<-c(files,file.path(dir,entries))
}
files<-files[file.exists(files)]
compiler<-sub('^CMAKE_CXX_COMPILER:[^=]*=','',grep('^CMAKE_CXX_COMPILER:',readLines(file.path(root,'cpp/build/opt/CMakeCache.txt')),value=TRUE))
binary<-file.path(scratch,'driver')
status<-system2(compiler,c('-std=c++23','-fno-exceptions','-fno-rtti',paste0('-I',shQuote(file.path(root,'cpp/include'))),
 shQuote(file.path(root,'cpp/tests/tools/mplus_input_sweep.cpp')),shQuote(file.path(root,'cpp/build/opt/libmagmaan.a')),'-o',shQuote(binary)))
stopifnot(status==0)
manifest<-file.path(scratch,'manifest.txt');writeLines(files,manifest)
report<-file.path(scratch,'results.tsv');status<-system2(binary,stdin=manifest,stdout=report)
stopifnot(status==0)
out<-read.delim(report,header=FALSE,col.names=c('file','reader','parser'),quote='',stringsAsFactors=FALSE)
contents<-vapply(files,function(f) paste(readLines(f,warn=FALSE),collapse='\n'),character(1))
cat('Files:',nrow(out),'distinct inputs:',length(unique(contents)),'\n')
cat('Reader accepted:',sum(out$reader=='accepted'),'parser accepted:',sum(out$parser=='accepted'),'\n')
print(table(out$reader));print(table(out$parser))
if(any(out$reader=='unclassified' | out$parser=='unclassified')) stop('Unclassified rejection')
