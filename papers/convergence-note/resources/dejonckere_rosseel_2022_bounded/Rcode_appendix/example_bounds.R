# 1. fit the model with standard bounds
model <- '
    Y =~ y1 + y2 + y3
    X =~ x1 + x2 + x3
    Y ~ X
'

# fit sem using ML with standard bounds
fit.bounds.standard <- 
    sem(model, data = Data, estimator = "ML",
        optim.bounds = list(lower = c("ov.var", "lv.var", "loadings"),
                            upper = c("ov.var", "lv.var", "loadings"),
                            min.reliability.marker = 0.1,
                            min.var.lv.endo = 0.005))
summary(fit.bounds)


# 2. fit the model with widened bounds
fit.bounds.wide <- 
    sem(model, data = Data, estimator = "ML",
        optim.bounds = list(lower = c("ov.var", "lv.var", "loadings"),
                            upper = c("ov.var", "lv.var", "loadings"),
                            lower.factor = c(1.05, 1.0, 1.1),
                            upper.factor = c(1.20, 1.3, 1.1),
                            min.reliability.marker = 0.1,
                            min.var.lv.endo = 0.005))
summary(fit.bounds2)


