 library(lavaan)

SAMPLE.SIZES <- c(10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)
NSIZES <- length(SAMPLE.SIZES)
REP <- 10000
RFILES <- list.files(path = ".", pattern = "\\.RData")
#RFILES <- list.files(path = ".", pattern = "DEF.REL.*\\.RData")

N <- length(RFILES)

OUT <- vector("list", length=N)
names(OUT) <- RFILES
for(r in 1:N) {

    CONV <- integer(NSIZES)
    BAD  <- integer(NSIZES)

    Name <- load(RFILES[r])
    RES <- get(Name)
    NAME <- rep(Name, NSIZES)

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

    }
    OUT[[r]] <- data.frame(NAME = NAME, SAMPLE.SIZE = SAMPLE.SIZES, 
                           CONV = CONV, BAD = BAD)
    class(OUT[[r]]) <- c("lavaan.data.frame", "data.frame")
    cat("OK.\n")
}

# table 4

# M0 part
M0 <- cbind(SAMPLE.SIZES, OUT$M0ML.RData$BAD, OUT$M0OVVAR.RData$BAD, 
            OUT$M0STAND.RData$BAD, OUT$M0WIDE.RData$BAD)
colnames(M0) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
M0[1:5,]

# M1 part
M1 <- cbind(SAMPLE.SIZES, OUT$M1ML.RData$BAD, OUT$M1OVVAR.RData$BAD, 
            OUT$M1STAND.RData$BAD, OUT$M1WIDE.RData$BAD)
colnames(M1) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
M1[1:5,]

# M2 part
M2 <- cbind(SAMPLE.SIZES, OUT$M2ML.RData$BAD, OUT$M2OVVAR.RData$BAD, 
            OUT$M2STAND.RData$BAD, OUT$M2WIDE.RData$BAD)
colnames(M2) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
M2[1:5,]

# M3 part
M3 <- cbind(SAMPLE.SIZES, OUT$M3ML.RData$BAD, OUT$M3OVVAR.RData$BAD, 
            OUT$M3STAND.RData$BAD, OUT$M3WIDE.RData$BAD)
colnames(M3) <- c("sample size", "no-bounds", "ov.var", "standard", "wide")
M3[1:5,]

