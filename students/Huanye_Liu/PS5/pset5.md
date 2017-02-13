    bd = read.csv("biden.csv",header=T)

Describe the data
-----------------

    hist(bd$biden,breaks=c(0:100),main = "Histogram of feeling thermometer",xlab = "feeling thermometer",ylab="frequency")

![](pset5_files/figure-markdown_strict/histogram-1.png)

We can see from the histogram above that most of participants of the
survey gave scores above 50, which shows a general positive feeling
towards Biden. The mode is around 50 and there are several other peaks
above 50. One interesting feature is that most scores takes one of the
following values: 0,15, 30,40,50,60,70,85,100. Therefore we can observe
9 peaks. One reason may be related to the design of the survey. Maybe
the survey provided several discrete scores for participants to choose
from. Another reason is that participants are more likely to give scores
the number value of which ends with 0 or 5.

Simple linear regression
------------------------

    sim <- lm(biden~age,data = bd)
    summary(sim)

    ## 
    ## Call:
    ## lm(formula = biden ~ age, data = bd)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -64.876 -12.318  -1.257  21.684  39.617 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 59.19736    1.64792   35.92   <2e-16 ***
    ## age          0.06241    0.03267    1.91   0.0563 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 23.44 on 1805 degrees of freedom
    ## Multiple R-squared:  0.002018,   Adjusted R-squared:  0.001465 
    ## F-statistic: 3.649 on 1 and 1805 DF,  p-value: 0.05626

The estimated intercept is
*b**e**t**a*<sub>0</sub> = 59.19736
 and its standard error is 1.64792. The estimated coefficient of
variable age is
*b**e**t**a*<sub>1</sub> = 0.06241
 and its standard error is 0.03267.

1.  From the summary of the linear model we can see there is a
    relationship between age and feeling thermometer since the p value
    of the coefficient of variable age is 0.0563, which means we accept
    that the coefficient is not zero if the significant level is above
    0.0563, and the p-value of the F test of the model is also 0.0562,
    so the relationship between variable biden and variable age exists.
2.  But the relationship between variable age and variable biden is very
    weak since the estimated coefficient of variable age is only
    0.06241, a number close to zero.
3.  The relationship is positive because the estimated coefficient of
    variable age is positive.
4.  The R squared of the model is 0.002018, and the adjusted R squared
    is 0.001465. If using the former one, only 0.2018 percent of the
    variation in variable biden can variable age alone explain, which
    means the linear model using age alone will probably be a bad model
    because the change in variable biden can be barely explained or
    predicted by the variable age alone.

<!-- -->

    predict(sim,list(age=45),interval="confidence")

    ##       fit      lwr      upr
    ## 1 62.0056 60.91177 63.09943

1.  The predicted biden value associated with age of 45 is 62.0056, and
    the associated 95% confidence interval is \[60.91177 63.09943\]. 6
    The graph is shown below.

<!-- -->

    plot(bd$age,bd$biden,pch=20,xlab='age',ylab='biden',main='Variable biden v.s. variable age')
    abline(sim,col='red')

![](pset5_files/figure-markdown_strict/sim%20plot-1.png) \#\# Multiple
linear regression

    mul1 <- lm(biden~age+female+educ,data = bd)
    summary(mul1)

    ## 
    ## Call:
    ## lm(formula = biden ~ age + female + educ, data = bd)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -67.084 -14.662   0.703  18.847  45.105 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 68.62101    3.59600  19.083  < 2e-16 ***
    ## age          0.04188    0.03249   1.289    0.198    
    ## female       6.19607    1.09670   5.650 1.86e-08 ***
    ## educ        -0.88871    0.22469  -3.955 7.94e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 23.16 on 1803 degrees of freedom
    ## Multiple R-squared:  0.02723,    Adjusted R-squared:  0.02561 
    ## F-statistic: 16.82 on 3 and 1803 DF,  p-value: 8.876e-11

The estimated intercept is
*b**e**t**a*<sub>0</sub> = 68.62101
 and its standard error is 3.59600. The estimated coefficient of
variable age is
*b**e**t**a*<sub>1</sub> = 0.04188
 and its standard error is 0.03249. The estimated coefficient of
variable female is
*b**e**t**a*<sub>2</sub> = 6.19607
 and its standard error is 1.09670. The estimated coefficient of
variable education is
*b**e**t**a*<sub>3</sub> = −0.88871
 and its standard error is 0.22469.

1.  From the p-values of those predictors we can see that there is
    significant replationship between variable biden and variable female
    or gender, and the relationship is also significant between varible
    biden and variable education , but not between varible biden and
    variable age.
2.  The estimated coefficient 6.19607 of variable female means that
    keeping other preditors fixed, on average female participants of the
    survey gave feeling thermometer score 6.19607 higher than the male
    participants did. If the sample data is representative, this means
    Biden seems more popular among the female than among the male.
3.  Teh R squared of the model is 0.02723, and the adjusted R squared
    is 0.02561. Based on the former one, only 2.723% variation in the
    variable biden can age, gender plus education explain. Compared with
    the age-only model, the adjusted R squared improves from 0.001465 to
    0.02561, so this model is better.
4.  The graph is shown below:

<!-- -->

    plot(predict(mul1),residuals(mul1),pch=20)
    dem_index = (bd$dem==1)
    rep_index = (bd$rep==1)
    indep_index = (bd$dem==0)&(bd$rep==0)
    abline(lm(residuals(mul1)[dem_index]~predict(mul1)[dem_index]),col='blue')
    abline(lm(residuals(mul1)[rep_index]~predict(mul1)[rep_index]),col='red')
    abline(lm(residuals(mul1)[indep_index]~predict(mul1)[indep_index]),col='green')
    legend(72,-40,legend = c("democratics","republicans",'independents'),col=c('blue','red','green'),lty=1,cex = 0.6,text.font=2)

![](pset5_files/figure-markdown_strict/mul%20res%20plot-1.png)

From the above graph we can see the residuals of variable biden strongly
depends on the party ID type. In terms of the residual values of varible
biden, the line fitting the data from democratics is strictly above the
one fitting the data from independents in the middle, which is then
strictly above the line fitting the data from republicans, and there is
not any crossing between any two lines, which means the party ID type is
a strong predictor of the variable biden. However, the model doesn't
include the party ID type as one of the predictors and therefore is a
less predictive model, which can also makes the estimated coefficients
of predictors included in the model less precise.

Multiple linear regression model with even more variables
---------------------------------------------------------

    mul2 <- lm(biden~age+female+educ+dem+rep,data = bd)
    summary(mul2)

    ## 
    ## Call:
    ## lm(formula = biden ~ age + female + educ + dem + rep, data = bd)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -75.546 -11.295   1.018  12.776  53.977 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  58.81126    3.12444  18.823  < 2e-16 ***
    ## age           0.04826    0.02825   1.708   0.0877 .  
    ## female        4.10323    0.94823   4.327 1.59e-05 ***
    ## educ         -0.34533    0.19478  -1.773   0.0764 .  
    ## dem          15.42426    1.06803  14.442  < 2e-16 ***
    ## rep         -15.84951    1.31136 -12.086  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 19.91 on 1801 degrees of freedom
    ## Multiple R-squared:  0.2815, Adjusted R-squared:  0.2795 
    ## F-statistic: 141.1 on 5 and 1801 DF,  p-value: < 2.2e-16

The estimated intercept is
*b**e**t**a*<sub>0</sub> = 58.81126
 and its standard error is 3.12444. The estimated coefficient of
variable age is
*b**e**t**a*<sub>1</sub> = 0.04826
 and its standard error is 0.02825. The estimated coefficient of
variable female is
*b**e**t**a*<sub>2</sub> = 4.10323
 and its standard error is 0.94823. The estimated coefficient of
variable education is
*b**e**t**a*<sub>3</sub> = −0.34533
 and its standard error is 0.19478.The estimated coefficient of variable
dem is
*b**e**t**a*<sub>4</sub> = 15.42426
 and its standard error is 1.06803.The estimated coefficient of variable
rep is
*b**e**t**a*<sub>5</sub> = −15.84951
 and its standard error is 1.31136. 1. Compared with the previous model,
the estimated coefficients of variable gender decreases from 6.19607 to
4.10323, which means after adding the party ID predictors, part of the
original effects of gender on Baiden warmth can be attributed to the
effects of party ID on Baiden warmth and the relationship between gender
and party ID. 2. The R squared of the model is 0.2815, and the adjusted
R squared is 0.2795. Based on the former one, 28.15% variation in the
variable biden can age, gender, education plus party ID explain.
Compared with the previous model, there is a great improvement in
adjusted R squared from 0.02561 to 0.2795. So this model is much better
than the previous one. 3. The graph is shown below.

    plot(predict(mul2),residuals(mul2),pch=20)
    dem_index = (bd$dem==1)
    rep_index = (bd$rep==1)
    indep_index = (bd$dem==0)&(bd$rep==0)
    abline(lm(residuals(mul2)[dem_index]~predict(mul2)[dem_index]),col='blue')
    abline(lm(residuals(mul2)[rep_index]~predict(mul2)[rep_index]),col='red')
    abline(lm(residuals(mul2)[indep_index]~predict(mul2)[indep_index]),col='green')
    legend(72,-40,legend = c("democratics","republicans",'independents'),col=c('blue','red','green'),lty=1,cex = 0.6,text.font=2)

![](pset5_files/figure-markdown_strict/mul2%20res%20plot-1.png) From the
above graph after including the party ID predictors, we can see now
three fitted lines cross each other, which means the residuals no longer
strongly depend on the party ID and seem more random compared with the
previous model, which is a sign for a good variable selection for the
model.

Interactive linear regression model
-----------------------------------

    bd = bd[bd$rep==1|bd$dem==1,]
    interactive <- lm(biden~female*dem,data = bd)
    summary(interactive)

    ## 
    ## Call:
    ## lm(formula = biden ~ female * dem, data = bd)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -75.519 -13.070   4.223  11.930  55.618 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   39.382      1.455  27.060  < 2e-16 ***
    ## female         6.395      2.018   3.169  0.00157 ** 
    ## dem           33.688      1.835  18.360  < 2e-16 ***
    ## female:dem    -3.946      2.472  -1.597  0.11065    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 19.42 on 1147 degrees of freedom
    ## Multiple R-squared:  0.3756, Adjusted R-squared:  0.374 
    ## F-statistic:   230 on 3 and 1147 DF,  p-value: < 2.2e-16

The estimated intercept is
*b**e**t**a*<sub>0</sub> = 39.382
 and its standard error is 1.455. The estimated coefficient of variable
female is
*b**e**t**a*<sub>1</sub> = 6.395
 and its standard error is 2.018. The estimated coefficient of variable
dem is
*b**e**t**a*<sub>2</sub> = 33.688
 and its standard error is 1.835. The estimated coefficient of
interaction term female\*dem is
*b**e**t**a*<sub>3</sub> = −3.946
 and its standard error is 2.472.

    predict(interactive,list(female=c(1,1,0,0),dem=c(1,0,1,0)),interval="confidence")

    ##        fit      lwr      upr
    ## 1 75.51883 73.77632 77.26133
    ## 2 45.77720 43.03494 48.51947
    ## 3 73.06954 70.87731 75.26176
    ## 4 39.38202 36.52655 42.23750

1.The predicted Biden warmth associated with female Democrats is
75.51883, and its 95% confidence interval is \[73.77632 77.26133\]. The
predicted Biden warmth associated with female Republicans is 45.77720,
and its 95% confidence interval is \[43.03494 48.51947\].The predicted
Biden warmth associated with male Democrats is 73.06954, and its 95%
confidence interval is \[70.87731 75.26176\].The predicted Biden warmth
associated with male Republicans is 39.38202, and its 95% confidence
interval is \[36.52655 42.23750\]. So we can see the relationship
between party ID and Biden warmth does differ for males and females
based on the model: when gender is female, the average effects of Party
ID on Biden warmth is 75.51883-45.77720 = 29.74163, but when gender is
male the average effects of Party ID on Biden warmth is
73.06954-39.38202=33.68752. The difference in the effects
29.74163-33.68752=-3.94589 is exactly the coefficients of the
interaction term. Similarly, the relationship between gender and Biden
warmth does differ for Democrats and Republicans: for Democrats, the
average effects of gender on Biden warmth is 75.51883-73.06954=2.44929,
but for Republicans, the average effects of gender on Biden warmth is
45.77720-39.38202=6.39518. Again, the difference in the effects
2.44929-6.39518 = -3.94589 is exactly the coefficients of the
interaction term.
