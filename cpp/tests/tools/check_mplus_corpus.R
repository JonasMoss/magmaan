#!/usr/bin/env Rscript
# Local end-to-end gate. Reads original inputs/output and derived analysis data;
# no dependency on the corpus translator or its translated model.
suppressPackageStartupMessages(library(magmaanlab))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(TRUE)
corpus <- if (length(args)) args[1] else "external/textbook-corpus"
report <- path.expand("~/.cache/magmaan-logs/mplus-corpus-gate.csv")
books <- c("mplus_users_guide_v8", "muthen_2017", "brown_2015")
folders <- unlist(lapply(books, function(book) list.dirs(file.path(corpus,"cases",book), recursive=FALSE)))
key <- function(lhs, op, rhs) {
  lhs <- tolower(lhs); rhs <- tolower(rhs)
  swap <- op == "~~" & lhs > rhs
  tmp <- lhs[swap]; lhs[swap] <- rhs[swap]; rhs[swap] <- tmp
  paste(lhs,op,rhs)
}
printed_parameters <- function(lines, groups) {
  start <- grep("^MODEL RESULTS\\s*$", lines)
  if (!length(start)) return(data.frame())
  lines <- lines[(start[1]+1):length(lines)]
  lhs <- op <- ""; out <- list(); group <- 1L
  for (line in lines) {
    if (grepl("^QUALITY OF|^STANDARDIZED|^R-SQUARE|^CONFIDENCE",line)) break
    text <- trimws(line)
    if (grepl("^Group ", text) && nrow(groups)) {
      group <- match(tolower(sub("^Group ", "", text)), tolower(groups$label)); next
    }
    relation <- regexec("^([A-Za-z][A-Za-z0-9_]*)\\s+(BY|ON|WITH)$",text)
    m <- regmatches(text,relation)[[1]]
    if (length(m)) { lhs <- m[2]; op <- switch(m[3],BY="=~",ON="~",WITH="~~"); next }
    if (text %in% c("Means","Intercepts","Variances","Residual Variances")) {
      lhs <- ""; op <- if (text %in% c("Means","Intercepts")) "~1" else "~~"; next
    }
    m <- regmatches(text,regexec("^([A-Za-z][A-Za-z0-9_]*)\\s+(-?[0-9]+\\.[0-9]+)\\s+",text))[[1]]
    if (length(m) && nzchar(op)) {
      l <- if (nzchar(lhs)) lhs else m[2]
      r <- if (op == "~1") "" else if (nzchar(lhs)) m[2] else l
      out[[length(out)+1]] <- data.frame(key=paste(group,key(l,op,r)),est=as.numeric(m[3]),printed=m[3])
    }
  }
  if (length(out)) do.call(rbind,out) else data.frame()
}
decimals <- function(value) {
  if (!grepl(".", value, fixed=TRUE)) return(0L)
  nchar(sub("^[^.]*\\.", "", value))
}
json_printed <- function(text, field) {
  pattern <- paste0('"', field, '"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
  match <- regmatches(text, regexec(pattern, text, perl=TRUE))[[1]]
  if (length(match)) match[2] else NULL
}
rows <- list()
for (folder in folders) {
  inp <- file.path(folder,"source/original.inp")
  raw <- file.path(folder,"data/raw.csv")
  bookfile <- file.path(folder,"expected/book.json")
  if (!all(file.exists(c(inp,raw,bookfile)))) next
  case <- basename(folder)
  spec <- tryCatch(mplus_model(file=inp),error=identity)
  if (inherits(spec,"error")) {
    msg <- conditionMessage(spec)
    rule <- regmatches(msg,regexpr("\\[[A-Z]+[0-9]+\\]",msg))
    rows[[length(rows)+1]] <- data.frame(case=case,status="rejected",rule=if(length(rule)) rule else "unclassified",quantity="",got=NA,expected=NA,deviation=NA,tolerance=NA,decimals=NA_integer_,detail=msg)
    next
  }
  result <- tryCatch({
    data <- read.csv(raw,check.names=FALSE)
    # The corpus stores human group labels in raw.csv for these examples.
    # Restore the original integer coding from the explicit GROUPING schema;
    # never change the sample or allow the adapter to accept undeclared codes.
    if (nzchar(spec$group_var) && spec$group_var %in% names(data)) {
      g <- as.character(data[[spec$group_var]])
      if (!all(g %in% spec$group_labels) && all(tolower(g) %in% tolower(spec$mplus_groups$label))) {
        data[[spec$group_var]] <- as.integer(spec$mplus_groups$code[match(tolower(g),tolower(spec$mplus_groups$label))])
        cat(case, ": restored corpus group labels to declared integer codes\n", sep="")
      }
    }
    meta <- fromJSON(file.path(folder,"meta.json"))
    book <- fromJSON(bookfile)
    book_text <- paste(readLines(bookfile,warn=FALSE),collapse="\n")
    estimator <- if(anyNA(data)) "FIML" else "ML"
    fit <- fit_model(spec,data,estimator=estimator)
    fm <- fit_measures(fit)
    checks <- list()
    add <- function(quantity,got,expected,tolerance=0,printed=NULL) {
      if (is.null(expected) || !length(expected)) return()
      d <- if (is.null(printed)) NA_integer_ else decimals(printed)
      if (!is.na(d)) tolerance <- max(tolerance, .5*10^(-d)+1e-5*abs(expected))
      checks[[length(checks)+1]] <<- data.frame(case=case,
        status=if(is.finite(got) && abs(got-expected)<=tolerance) "matched" else "failing",
        rule="",quantity=quantity,got=got,expected=expected,deviation=got-expected,tolerance=tolerance,decimals=d,detail=estimator)
    }
    add("converged", as.integer(isTRUE(fit$converged)), 1)
    add("N",sum(fit$nobs),sum(meta$data$n_obs))
    add("npar",fm$npar,book$fit$npar)
    add("df",fm$df,book$fit$df)
    if (!identical(meta$estimator_default,"MLR")) add("chisq",fm$chisq,book$fit$chisq,.002+1e-5*book$fit$chisq,json_printed(book_text,"chisq"))
    outfile <- file.path(folder,"source/original.out")
    if (file.exists(outfile)) {
      lines <- readLines(outfile,warn=FALSE)
      h0 <- grep("^\\s*H0 Value\\s+",lines,value=TRUE)
      if(length(h0)) {
        expected <- as.numeric(sub(".*H0 Value\\s+(-?[0-9.]+).*","\\1",h0[1]))
        add("H0 loglik",fm$logl,expected,.002+1e-6*abs(expected),
            sub(".*H0 Value\\s+(-?[0-9.]+).*","\\1",h0[1]))
      }
      params <- printed_parameters(lines,spec$mplus_groups)
      if(nrow(params)) {
        idx <- match(params$key,paste(fit$partable$group,key(fit$partable$lhs,fit$partable$op,fit$partable$rhs)))
        for(i in seq_len(nrow(params))) add(paste("estimate",params$key[i]),
          if(is.na(idx[i])) NA_real_ else fit$partable$est[idx[i]],params$est[i],.001+2e-4*abs(params$est[i]),params$printed[i])
      }
    }
    do.call(rbind,checks)
  },error=function(e) data.frame(case=case,status="failing",rule="",quantity="fit",got=NA,expected=NA,deviation=NA,tolerance=NA,decimals=NA_integer_,detail=conditionMessage(e)))
  rows[[length(rows)+1]] <- result
}
if (!length(rows)) stop("No eligible corpus cases found at ",corpus)
out <- do.call(rbind,rows)
write.csv(out,report,row.names=FALSE)
accepted <- unique(out$case[out$status != "rejected"])
failed <- unique(out$case[out$status == "failing"])
cat("Cases:",length(unique(out$case)),"accepted:",length(accepted),"matched:",length(setdiff(accepted,failed)),"failing:",length(failed),"\n")
print(table(out$rule[out$status == "rejected"]))
if(length(failed)) { print(out[out$status=="failing",]); quit(status=1) }
