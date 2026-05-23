## ---------------------------
##
## Script name:          New plot study 1 - 2 - 3
##
## Purpose of script:    
##
##
## ---------------------------
## ---------------------------



load("RESULTS_STUDY1_RSTART_NEW.RData")
load("RESULTS_STUDY2_RSTART_NEW.RData")
load("RESULTS_MSST_RSTART_NEW.RData")

library("patchwork")

## -------------------
## STUDY ONE
## -------------------

table_sets <- table(RESULTS_STUDY1_RSTART_NEW[,14])
names_table <- names(table_sets)
nr_sets <- as.matrix(table_sets)

OUTPUT <- as.matrix(cbind(as.numeric(names_table), 
                          nr_sets, 
                          (366-cumsum(nr_sets))))
colnames(OUTPUT) <- c("Number of random sets", 
                      "Number of cases that converged", 
                      "Number nonconvergences")
OUTPUT <- data.frame(OUTPUT)


OUTPUT2 <- as.matrix(cbind(as.numeric(names_table), 
                           cumsum(nr_sets)))
colnames(OUTPUT2) <- c("Number of random sets", 
                       "Number of cases that converged")
OUTPUT2 <- data.frame(OUTPUT2)


p1 <- ggplot(data=OUTPUT, aes(x=OUTPUT[,1], y=OUTPUT[,3], group=1)) +
  geom_line()+
  ylim(0,260) +
  scale_x_continuous(breaks = seq(0, 150, by = 15)) +
  labs(title="Number of nonconvergences per starting value set, 
       simple model β = 0.1",
       x="Number of random starting value sets", 
       y = "Number of nonconvergences")

p2 <- ggplot(data=data.frame(OUTPUT2), aes(x=OUTPUT2[,2], y=OUTPUT2[,1])) +
  geom_line()+
  ylim(0,150) +
  scale_x_continuous(breaks = seq(100, 366, by = 25)) +
  geom_label(aes(x = OUTPUT2[1,2], y = OUTPUT2[1,1], 
                 label = round(OUTPUT2[1,2], 2)),
             size = 3) +
  geom_label(aes(x = OUTPUT2[83,2], y = OUTPUT2[83,1], 
                 label = round(OUTPUT2[83,2], 2)),
             size = 3) +
  labs(title="Cumulative convergences per starting value set, 
       simple model β = 0.1",
       x="Cumulative number of convergences", 
       y = "Number of sets")


## -------------------
## STUDY TWO
## -------------------
SETS_IDX_C <- SETS_IDX_C[1:341,]


table_setsC <- table(RESULTS_STUDY2_RSTART_NEW[,24])
names_tableC <- names(table_setsC)
nr_setsC <- as.matrix(table_setsC)

OUTPUT_C <- as.matrix(cbind(as.numeric(names_tableC),nr_setsC, 
                            (341-cumsum(nr_setsC))))
colnames(OUTPUT_C) <- c("Number of random sets", 
                        "Number of cases that converged", 
                        "Number nonconverged")
OUTPUT_C <- data.frame(OUTPUT_C)


OUTPUT2_C <- as.matrix(cbind(as.numeric(names_tableC),cumsum(nr_setsC)))
colnames(OUTPUT2_C) <- c("Number of random sets", 
                         "Number of cases that converged")
OUTPUT2_C <- data.frame(OUTPUT2_C)

p1_C <- ggplot(data=OUTPUT_C, aes(x=OUTPUT_C[,1], y=OUTPUT_C[,3], group=1)) +
  geom_line()+
  ylim(0,260) +
  xlim(0,150) +
  labs(title="Number of nonconvergences per starting value set, 
         crossloadings model",
       x="Number of random starting value sets", 
       y = "Number of nonconvergences")

p2_C <- ggplot(data=data.frame(OUTPUT2_C), aes(x=OUTPUT2_C[,2], 
                                               y=OUTPUT2_C[,1])) +
  geom_line()+
  ylim(0,150) +
  scale_x_continuous(breaks = seq(0, 360, by = 15)) +
  geom_label(aes(x = OUTPUT2_C[1,2], y = OUTPUT2_C[1,1], 
                 label = round(OUTPUT2_C[1,2], 2)),
             size = 3) +
  geom_label(aes(x = OUTPUT2_C[34,2], y = OUTPUT2_C[34,1], 
                 label = round(OUTPUT2_C[34,2], 2)),
             size = 3) +
  labs(title="Cumulative convergences per starting value set, 
         crossloadings model",
       x="Cumulative number of convergences", 
       y = "Number of sets")


## -------------------
## STUDY THREE
## -------------------

table_sets_MSST <- table(RESULTS_MSST_RSTART_NEW[,22])
names_table_MSST <- names(table_sets_MSST)
nr_sets_MSST <- as.matrix(table_sets_MSST)

OUTPUT_MSST <- as.matrix(cbind(as.numeric(names_table_MSST),nr_sets_MSST, 
                               (385-cumsum(nr_sets_MSST))))
colnames(OUTPUT_MSST) <- c("Number of random sets", 
                           "Number of cases that converged", 
                           "Number nonconverged")
OUTPUT_MSST <- data.frame(OUTPUT_MSST)


OUTPUT2_MSST <- as.matrix(cbind(as.numeric(names_table_MSST),cumsum(nr_sets_MSST)))
colnames(OUTPUT2_MSST) <- c("Number of random sets", 
                            "Number of cases that converged")
OUTPUT2_MSST <- data.frame(OUTPUT2_MSST)

p1_MSST <- ggplot(data=OUTPUT_MSST, aes(x=OUTPUT_MSST[,1], y=OUTPUT_MSST[,3], group=1)) +
  geom_line()+
  ylim(0,400) +
  xlim(0,150) +
  labs(title="Number of nonconvergences per starting value set, 
         MSST model",
       x="Number of random starting value sets", 
       y = "Number of nonconvergences")

p2_MSST <- ggplot(data=data.frame(OUTPUT2_MSST), aes(x=OUTPUT2_MSST[,2], 
                                                     y=OUTPUT2_MSST[,1])) +
  geom_line()+
  ylim(0,150) +
  scale_x_continuous(breaks = seq(0, 360, by = 15)) +
  geom_label(aes(x = OUTPUT2_MSST[1,2], y = OUTPUT2_MSST[1,1], 
                 label = round(OUTPUT2_MSST[1,2], 2)),
             size = 3) +
  geom_label(aes(x = OUTPUT2_MSST[98,2], y = OUTPUT2_MSST[98,1], 
                 label = round(OUTPUT2_MSST[98,2], 2)),
             size = 3) +
  labs(title="Cumulative convergences per starting value set, 
         MSST model",
       x="Cumulative number of convergences", 
       y = "Number of sets")

# ---------------------- #  
p1 / p1_C / p1_MSST
p2 / p2_C / p2_MSST
# ---------------------- #  