## ---------------------------
##
## Script name:         RES_Targets_M1.R 
##
## Purpose of script:   script to analyse simulation results of SIM_Targets_M1
##                      
##
## Authors:             Julie De Jonckere and Yves Rosseel
##                      Department of Data Analysis, Ghent University
##
##
## ---------------------------
## ---------------------------


SAMPLE.SIZES <- c(10, 20, 30, 40, 50, 100)
NSIZES <- length(SAMPLE.SIZES)
REP <- 1000

RFILES <- list.files(path = ".", pattern = "\\.RData")
N <- length(RFILES)

OUT_Targets_M1 <- vector("list", length=N)
#names(OUT) <- TOTAL$NAME
for(r in 1:N) {
  
  CONV <- integer(NSIZES)
  BAD    <- integer(NSIZES)
  MEAN   <- numeric(NSIZES)
  MEAN2  <- numeric(NSIZES)
  MEDIAN <- numeric(NSIZES)
  SD     <- numeric(NSIZES)
  MSE    <- numeric(NSIZES)
  VARX   <- numeric(NSIZES)
  VARY   <- numeric(NSIZES)
  #MEAN_L <- numeric(NSIZES)
  
  Name <- load(RFILES[r])
  RES <- get(Name)
  NAME <- rep(Name, NSIZES)
  RES <- get(Name)
  
  cat("r = ", r, "NAME = ", NAME[1], " ... ")
  
  if(nrow(RES) != REP*NSIZES) {
    cat(" SKIP.\n")
    next
  }
  
  for(i in seq_len(NSIZES)) {
    ss.idx <- (i-1)*REP + seq_len(REP)
    
    # converged
    idx <- ss.idx[RES[ss.idx, "optim"] == 1L & RES[ss.idx, "se"] == 1L ] # Add "&  RES[ss.idx, "lambda"] > 0" to calculate mean value for lambda 
    CONV[i] <- length(idx)
    
    BAD[i] <- REP - CONV[i]
    
    #MEAN_L[i] <- mean(RES[idx, "lambda"], na.rm = TRUE)
    
    MEAN[i] <-  mean(RES[idx, "Y~X"], na.rm = TRUE)
    
    # trimmed mean
    MEAN2[i] <- mean(RES[idx, "Y~X"], trim = 0.005, na.rm = TRUE)
    
    # median
    MEDIAN[i] <- median(RES[idx, "Y~X"], na.rm = TRUE)
    
    # sd voor L > 0
    SD[i] <- sd(RES[idx, "Y~X"], na.rm = TRUE)
    
    # mse
    MSE[i] <- SD[i]^2 + (MEAN[i] - 0.1)^2
    
    # varx
    VARX[i] <- mean(RES[idx, "X~~X"], trim = 0.005)  
    
    # vary
    VARY[i] <- mean(RES[idx, "Y~~Y"], trim = 0.005)
    
    
    
  }
  OUT_Targets_M1[[r]] <- data.frame(NAME = NAME, 
                                    BAD = BAD,
                                    SAMPLE.SIZE = SAMPLE.SIZES,
                                    CONV = CONV, 
                                    #MEAN_L = MEAN_L,
                                    MEAN = MEAN,
                                    MEAN2 = MEAN2, 
                                    MEDIAN = MEDIAN, 
                                    SD = SD, 
                                    MSE = MSE, 
                                    VARX = VARX, 
                                    VARY = VARY)
  cat("OK.\n")
}

names(OUT_Targets_M1) <- c("ConstantC", "CommonV", "Identity", "MB1", "MB2", "ML")


## ----------------------------------
##  Table: convergence  
## ----------------------------------
Table_convergence <- cbind(SAMPLE.SIZES,
                           OUT_Targets_M1$MB1$BAD,
                           OUT_Targets_M1$MB2$BAD,
                           OUT_Targets_M1$Identity$BAD,
                           OUT_Targets_M1$CommonV$BAD,
                           OUT_Targets_M1$ConstantC$BAD,
                           OUT_Targets_M1$ML$BAD)
colnames(Table_convergence) <- c("N", "MB1", "MB2", "Identity", 
                                 "CommonV", "ConstantC", "ML")
Table_convergence


## ----------------------------------
##  Table: mean bias
## ----------------------------------
Table_bias <- cbind(SAMPLE.SIZES,
                    (OUT_Targets_M1$MB1$MEAN  - 0.1)*100,
                    (OUT_Targets_M1$MB2$MEAN  - 0.1)*100,
                    (OUT_Targets_M1$Identity$MEAN  - 0.1)*100,
                    (OUT_Targets_M1$CommonV$MEAN  - 0.1)*100,
                    (OUT_Targets_M1$ConstantC$MEAN  - 0.1)*100,
                    (OUT_Targets_M1$ML$MEAN  - 0.1)*100)
colnames(Table_bias) <- c("N", "MB1", "MB2", "Identity", 
                          "CommonV", "ConstantC", "ML")
round(Table_bias)


## ----------------------------------
##  Table: MSE
## ----------------------------------
Table_mse <- cbind(SAMPLE.SIZES,
                   OUT_Targets_M1$MB1$MSE*100,
                   OUT_Targets_M1$MB2$MSE*100,
                   OUT_Targets_M1$Identity$MSE*100,
                   OUT_Targets_M1$CommonV$MSE*100,
                   OUT_Targets_M1$ConstantC$MSE*100,
                   OUT_Targets_M1$ML$MSE*100)
colnames(Table_mse) <- c("N", "MB1", "MB2", "Identity", 
                         "CommonV", "ConstantC", "ML")
round(Table_mse)


## ----------------------------------
##  Table: mean lambda
## ----------------------------------
## add &RES[ss.idx, "lambda"] > 0 when defining idx and re-run
Table_meanlambda <- cbind(SAMPLE.SIZES,
                          OUT_Targets_M1$MB1$MEAN_L,
                          OUT_Targets_M1$MB2$MEAN_L,
                          OUT_Targets_M1$Identity$MEAN_L,
                          OUT_Targets_M1$CommonV$MEAN_L,
                          OUT_Targets_M1$ConstantC$MEAN_L)
colnames(Table_meanlambda) <- c("N", "MB1", "MB2", "Identity", "CV", "CC")
round(Table_meanlambda, digits = 2)



## ----------------------------------
##  Histogram: non zero lambda's
## ----------------------------------

## ------------FOR SS30 ONLY-----------------
par(mfrow=c(3,2))

SS30_MB1 <- RES_MB1_target_M1[2001:3000,]
nonzero_rows1 <- SS30_MB1[SS30_MB1[, "lambda"]>0,]
lambda_mb1_nz1 <- nonzero_rows1[,"lambda"]
# basic R
hist(lambda_mb1_nz1, 
     main = "Model-based target | user-defined cor",
     xlab = "lambda",
     xlim = c(0,1),
     ylim = c(0,110))

## ----------------------------------
SS30_MB2 <- RES_MB2_target_M1[2001:3000,]
nonzero_rows2 <- SS30_MB2[SS30_MB2[, "lambda"]>0,]
lambda_mb2_nz2 <- nonzero_rows2[,"lambda"]
hist(lambda_mb2_nz2, 
     main = "Model-based target | zero cor",
     xlab = "lambda",
     xlim = c(0,1),
     ylim = c(0,110))

## ----------------------------------
SS30_ID <- RES_id_target_M1[2001:3000,]
nonzero_rows3 <- SS30_ID[SS30_ID[, "lambda"]>0,]
lambda_id_nz3 <- nonzero_rows3[,"lambda"]
hist(lambda_id_nz3, 
     main = "Identity target",
     xlab = "lambda",
     xlim = c(0,1),
     ylim = c(0,110))

## ----------------------------------
SS30_CV <- RES_cv_target_M1[2001:3000,]
nonzero_rows4 <- SS30_CV[SS30_CV[, "lambda"]>0,]
lambda_cv_nz4 <- nonzero_rows4[,"lambda"]
hist(lambda_cv_nz4, 
     main = "Common variance target",
     xlab = "lambda",
     xlim = c(0,1),
     ylim = c(0,110))

## ----------------------------------
SS30_CC      <- RES_cc_target_M1[2001:3000,]
nonzero_rows5 <- SS30_CC[SS30_CC[,"lambda"]>0,]
lambda_cc_nz5 <- nonzero_rows5[,"lambda"]
hist(lambda_cc_nz5, 
     main = "Constant correlation target",
     xlab = "lambda",
     xlim = c(0,1),
     ylim = c(0,110))



