# fit the model with wide bounds + boostrap
set.seed(1234)
fit.bootstrap <- sem(model, data = Data, estimator = "ML",
                     bounds = "wide", verbose = TRUE,
                     se = "bootstrap", bootstrap = 1000L)
summary(fit.bootstrap)
