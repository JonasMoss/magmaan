#!/usr/bin/env Rscript
args<-commandArgs(TRUE)
if('--help' %in% args){cat('Usage: Rscript R/classify_problems.R CORPUS RESULTS
Tags every prepared case with its problem class, read from the model syntax: nonlinear-constraints when an
equality constraint is nonlinear in the parameters (powers, quotients, products of labels, functions), else
standard. Standard cases form the main comparison; other classes are reported as their own lanes.
Writes RESULTS/problem_classes.csv. Seconds.\n');quit(save='no')}
root<-normalizePath(args[1]);out<-normalizePath(args[2])
d<-read.csv(file.path(out,'comparison.csv'),stringsAsFactors=FALSE)
manifest<-read.csv(file.path(root,'manifest.csv'),stringsAsFactors=FALSE)
cases<-unique(d$case)
nonlinear<-function(line){
  if(!grepl('==',line,fixed=TRUE))return(FALSE)
  side<-sub('#.*','',line)
  grepl('\\^|/|(exp|log|sqrt|abs|sin|cos|tan)\\(',side) ||
    grepl('[A-Za-z_.][A-Za-z0-9_.]*\\s*\\*\\s*[A-Za-z_.(]',side)
}
class_of<-function(case){
  f<-file.path(root,manifest$case_dir[match(case,manifest$case_id)],'model.lav')
  if(any(vapply(readLines(f,warn=FALSE),nonlinear,logical(1))))'nonlinear-constraints' else 'standard'
}
z<-data.frame(case=cases,class=vapply(cases,class_of,''),stringsAsFactors=FALSE)
write.csv(z,file.path(out,'problem_classes.csv'),row.names=FALSE)
print(table(z$class));print(z[z$class!='standard',],row.names=FALSE)
