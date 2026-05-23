# adding two bounds:
# - lower (zero) bound for x2 ~~ x2
# - upper (2) bound for Y =~ x2

# 1. general method -- using labels and inequality constraints
model1 <- '
    Y =~ y1 + y2 + y3
    X =~ x1 + l5*x2 + x3
    Y ~ X
    x2 ~~ rvar2*x2

    # imposing lower and upper bounds
    l5 < 5
    rvar2 > 0
'

# fit model using nlminb.contr()
fit1 <- sem(model1, data = Data, estimator = "ML")
summary(fit1)



# 2. efficient method -- using lower()/upper() modifiers
model2 <- '
    Y =~ y1 + y2 + y3
    X =~ x1 + upper(5)*x2 + x3
    Y ~ X
    x2 ~~ lower(0)*x2
'

# fit sem using nlminb()
fit2 <- sem(model2, data = Data, estimator = "ML")
summary(fit2)
