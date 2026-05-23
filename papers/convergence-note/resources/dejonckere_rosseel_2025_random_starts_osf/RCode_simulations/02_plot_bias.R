## ---------------------------
##
## Script name: 
##
## Purpose of script:         plot to discuss bias
##
## Author: De Jonckere J.
##
## Date Created: 2024-02-01
##
## Copyright (c) Julie De Jonckere, 2024
## Email: Julie.DeJonckere@UGent.be
##
## ---------------------------
##
## ---------------------------

library("tidyverse")
library("hrbrthemes")
library("viridis")
library("reshape2")
library("ggplot2")
library("patchwork")

## STUDY 1
load("RESULTS_STUDY1_RSTART_NEW.RData")
## STUDY 2
load("RESULTS_STUDY2_RSTART_NEW.RData")


SETS_IDX_C <- as.matrix(SETS_IDX_C[1:341,])

## -------------------
## STUDY 1
## -------------------

## Converged Standard ML 
converged_all_ML <- which(OPTIM.NEW[, "optim"]==1 & OPTIM.NEW[,"se"]==1)
converged_all_ML07 <- which(OPTIM.NEW07[, "optim"]==1 & OPTIM.NEW07[,"se"]==1)

DATA_all_ML <- OPTIM.NEW[converged_all_ML,] ; nrow(DATA_all_ML) # 307
DATA_all_ML07 <- OPTIM.NEW07[converged_all_ML07,] ; nrow(DATA_all_ML07) # 350

## Random Start ------
converged_all_simple <- which(RESULTS_STUDY1_0[, "optim"]==1 & RESULTS_STUDY1_0[,"se"]==1)
converged_all_simple07 <- which(RESULTS_STUDY07_0[, "optim"]==1 & RESULTS_STUDY07_0[,"se"]==1)

DATA_all_s <- RESULTS_STUDY1_0[converged_all_simple,] ; nrow(DATA_all_s) # 353
DATA_all_s07 <- RESULTS_STUDY07_0[converged_all_simple07,] ; nrow(DATA_all_s07) # 325 (should be 331)

## Bounds ------------
converged_all_bounds <- which(BOUNDS_RES[, "optim"]==1 & BOUNDS_RES[,"se"]==1)
converged_all_bounds07 <- which(BOUNDS_RES07[, "optim"]==1 & BOUNDS_RES07[,"se"]==1)

DATA_all_bounds <- BOUNDS_RES[converged_all_bounds,] ; nrow(DATA_all_bounds) # 366
DATA_all_bounds07 <- BOUNDS_RES07[converged_all_bounds07,] ; nrow(DATA_all_bounds07) # 327

## -------------------
## STUDY 2
## -------------------

## Converged Standard ML 
converged_cross_ML <- which(ML_COMPLEX[, "optim"]==1 & ML_COMPLEX[,"se"]==1)

DATA_cross_ML <- ML_COMPLEX[converged_cross_ML,] ; nrow(DATA_cross_ML) # 659

## Random Start ------
converged_all_cross <- which(RESULTS_STUDY3[, "optim"]==1 & RESULTS_STUDY3[,"se"]==1)

DATA_all_cross <- RESULTS_STUDY3[converged_all_cross,] ; nrow(DATA_all_cross) # 636

## Bounds ------------
converged_cross_bounds <- which(BOUNDS_RES_CROSS[, "optim"]==1 & BOUNDS_RES_CROSS[,"se"]==1)
DATA_all_bounds_cross <- BOUNDS_RES_CROSS[converged_cross_bounds,] ; nrow(BOUNDS_RES_CROSS)


## -------------------------------------------------------------------------- ##

## -------------------
## PLOT FOR R START
## -------------------
converged_all_simple <- which(RESULTS_STUDY1_RSTART_NEW[, 14]>=1)
SETS_IDX_0.complete  <- as.matrix(RESULTS_STUDY1_RSTART_NEW[converged_all_simple,14])

converged_all_cross  <- which(RESULTS_STUDY2_RSTART_NEW[, 24]>=1)
SETS_IDX_C.complete  <- as.matrix(RESULTS_STUDY2_RSTART_NEW[converged_all_cross,24])

method.start <- rbind(matrix(replicate("S1_RandomStart_01", n=318)),
                      matrix(replicate("S2_RandomStart", n = 326)))
estimate.start <- rbind(matrix(RESULTS_STUDY1_RSTART_NEW[converged_all_simple,"FY~FX"]),
                        matrix(RESULTS_STUDY2_RSTART_NEW[converged_all_cross,"eta2~eta1"]))
sets.start <- rbind(SETS_IDX_0.complete,
                    SETS_IDX_C.complete) 

DATA_BIAS_PLOT.start <- cbind(method.start, estimate.start, sets.start)
DATA_BIAS_PLOT.start <- data.frame(DATA_BIAS_PLOT.start)

## now we want to categorize the number of sets as:
## "0-15", "16-30", "31-50", "51-100", "101-150"
NEW.DATA.PLOT<- DATA_BIAS_PLOT.start %>%
  mutate(sets.factor = case_when(
    between(sets.start, 0, 15) ~ "0-15",
    between(sets.start, 16, 30) ~ "16-30",
    between(sets.start, 31, 50) ~ "31-50",
    between(sets.start, 51, 100) ~ "51-100",
    between(sets.start, 101, 150) ~ "101-150")) %>%
  mutate(sets.factor = factor(sets.factor, 
                              levels = c("0-15","16-30",
                                         "31-50","51-100",
                                         "101-150")))

group1 <- subset(NEW.DATA.PLOT, sets.factor %in% c("0-15"))
group2 <- subset(NEW.DATA.PLOT, sets.factor %in% c("16-30"))
group3 <- subset(NEW.DATA.PLOT, sets.factor %in% c("31-50"))
group4 <- subset(NEW.DATA.PLOT, sets.factor %in% c("51-100"))
group5 <- subset(NEW.DATA.PLOT, sets.factor %in% c("101-150"))

colnames(NEW.DATA.PLOT) <- c("method", "estimate", "sets","sets.factor")
NEW.DATA.PLOT$estimate <- as.numeric(NEW.DATA.PLOT$estimate)

##  ACTUAL PLOT
box1 <- ggplot(NEW.DATA.PLOT, aes(x=method, y=estimate, group=method)) +
  geom_boxplot() +
  geom_jitter(aes(colour = sets.factor)) +
  coord_cartesian(ylim = c(-5, 90)) +
  scale_y_continuous(breaks = seq(-5, 90, by = 10)) +
  geom_hline(yintercept = 0.1, linetype="dashed", color = "black") 


box2 <- ggplot(NEW.DATA.PLOT, aes(x=method, y=estimate, group=method)) +
  geom_boxplot() +
  geom_jitter(aes(colour = sets.factor)) +
  coord_cartesian(ylim = c(-1, 3)) +
  scale_y_continuous(breaks = seq(-1, 3, by = 1)) +
  geom_hline(yintercept = 0.1, linetype="dashed", color = "black") 

box1/box2
