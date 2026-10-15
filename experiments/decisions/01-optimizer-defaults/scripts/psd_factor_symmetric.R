#!/usr/bin/env Rscript
script <- sub("^--file=", "", commandArgs()[grepl("^--file=", commandArgs())][1])
symmetric_lane <- TRUE
source(file.path(dirname(script), "../R/psd_factor_run.R"))
