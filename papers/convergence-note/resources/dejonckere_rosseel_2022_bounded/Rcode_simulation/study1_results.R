 SAMPLE.SIZES <- c(10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
NSIZES <- length(SAMPLE.SIZES)
REP <- 10000

RFILES <- list.files(path = ".", pattern = "\\.RData")
N <- length(RFILES)

OUT <- vector("list", length=N)
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
        idx <- ss.idx[RES[ss.idx, "optim"] == 1L & RES[ss.idx, "se"] == 1L]
        CONV[i] <- length(idx)

        BAD[i] <- REP - CONV[i]

        # mean
        MEAN[i] <- mean(RES[idx, "Y~X"])

        # trimmed mean
        MEAN2[i] <- mean(RES[idx, "Y~X"], trim = 0.005)

        # median
        MEDIAN[i] <- median(RES[idx, "Y~X"])

        # sd
        SD[i] <- sd(RES[idx, "Y~X"])

        # mse
        MSE[i] <- SD[i]^2 + (MEAN[i] - 0.25)^2

        # varx
        VARX[i] <- mean(RES[idx, "X~~X"], trim = 0.005)

        # vary
        VARY[i] <- mean(RES[idx, "Y~~Y"], trim = 0.005)
    }
    OUT[[r]] <- data.frame(NAME = NAME, SAMPLE.SIZE = SAMPLE.SIZES, 
                           CONV = CONV, MEAN = MEAN, MEAN2 = MEAN2,
                           MEDIAN = MEDIAN,
                           SD = SD, MSE = MSE, 
                           VARX = VARX, VARY = VARY, BAD = BAD)
    cat("OK.\n")
}

names(OUT) <- c("ML", "OVAR", "STAND", "WIDE")


# Table 3
Table3 <- cbind(SAMPLE.SIZES, OUT$ML$BAD, OUT$OVAR$BAD,
                OUT$STAND$BAD, OUT$WIDE$BAD)
colnames(Table3) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
Table3

# Table 5a -- mean
Table5a <- cbind(SAMPLE.SIZES,
                 OUT$ML$MEAN,
                 OUT$OVAR$MEAN,
                 OUT$STAND$MEAN,
                 OUT$WIDE$MEAN)
colnames(Table5a) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
round(Table5a, 2)

# Table 5b -- sd
Table5b <- cbind(SAMPLE.SIZES,
                 OUT$ML$SD,   
                 OUT$OVAR$SD, 
                 OUT$STAND$SD,
                 OUT$WIDE$SD) 
colnames(Table5b) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
round(Table5b, 2)


# Table 6a -- relative mean bias
Table6a <- cbind(SAMPLE.SIZES,
                 (OUT$ML$MEAN    - 0.25)*100, 
                 (OUT$OVAR$MEAN  - 0.25)*100,
                 (OUT$STAND$MEAN - 0.25)*100, 
                 (OUT$WIDE$MEAN  - 0.25)*100)
colnames(Table6a) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
round(Table6a)

# Table 6b -- relative mean bias
Table6b <- cbind(SAMPLE.SIZES, 
                 OUT$ML$MSE*100, OUT$OVAR$MSE*100, 
                 OUT$STAND$MSE*100, OUT$WIDE$MSE*100)
colnames(Table6b) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
round(Table6b)

# Table 7a -- trimmed means
Table7a <- cbind(SAMPLE.SIZES,
                 OUT$ML$MEAN2,   
                 OUT$OVAR$MEAN2, 
                 OUT$STAND$MEAN2,
                 OUT$WIDE$MEAN2) 
colnames(Table7a) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
round(Table7a, 2)

# Table 7b -- median
Table7b <- cbind(SAMPLE.SIZES,
                 OUT$ML$MEDIAN,     
                 OUT$OVAR$MEDIAN, 
                 OUT$STAND$MEDIAN,
                 OUT$WIDE$MEDIAN) 
colnames(Table7b) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
round(Table7b, 2)

