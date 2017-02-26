# Problem Set 6
MACS 30100 - Perspectives on Computational Modeling<br> Luxi Han 10449918  





## Problem 1

1. The following is the graph for the histogram of the variable 

![](ps6-glm_files/figure-html/problem 1a-1.png)<!-- -->

```
## [1] "The unconditional probablity a voter will vote is: 0.682357443551473"
```

  The unconditional probablity of voter voting is about 68.24%.
  
2. The following is the scatter plot and the smoothed regression line:

![](ps6-glm_files/figure-html/problem 1b-1.png)<!-- -->

This graph tells us that the relationship between voter turnout and mental health is negatively correlated. 
The reason why this graph is problematic is that: 1) voter turnout is a binary choice taking on values of either 0 or 1, while the predicted value is a strand of continous value between 0 and 1; 2) if we were to plot further down along the x axis, then we will get negative predicted voter turnout. This is unsenesible.

##Problem 2

 1. There is indeed a significantly negative relationship between voter turnout and meantal health. The esitmated parameter is about -0.14348 which is significant on 0.001 significance level.
 

```
## 
## Call:
## glm(formula = vote96 ~ mhealth_sum, family = binomial, data = clean_data)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.6834  -1.2977   0.7452   0.8428   1.6911  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  1.13921    0.08444  13.491  < 2e-16 ***
## mhealth_sum -0.14348    0.01969  -7.289 3.13e-13 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1672.1  on 1321  degrees of freedom
## Residual deviance: 1616.7  on 1320  degrees of freedom
## AIC: 1620.7
## 
## Number of Fisher Scoring iterations: 4
```
 
 2. When the evaluation on mental health increases by one unit, the log odds of voter voting against not voting decreases by -0.14348. 
 The following is the graph:
 
![](ps6-glm_files/figure-html/problem 2b-1.png)<!-- -->

3. The estimator on odds can be interpreted as percent change. When the evaluation on mental health increases by one unit, the odds of voter voting against not voting decreases by -0.14348 percent(%).  

![](ps6-glm_files/figure-html/problem 2c-1.png)<!-- -->

4. The interpretation of the estimator from the perspective of probablity is not certain. Since the first difference typically depend on the initial age.

![](ps6-glm_files/figure-html/problem 2d-1.png)<!-- -->

```
## [1] "first difference going from 1 to 2 is  -0.0291782383716035"
```

```
## [1] "first difference going from 5 to 6 is  -0.0347782137951934"
```

The first difference for an increase in the mental health index from 1 to 2 is -0.0292; from 5 to 6 is -0.0348.

5. 

```
## [1] 0.677761
```

```
## [1] 0.01616628
```

```
## Area under the curve: 0.6243
```
The accuracy rate is 0.6778. The prediction error reduction is 1.62%. The AUC is 0.6243. The model doesn't really explain the binary choice of voting very well. We can see that the prediction error reduction is only around 1.6%, which is a small magnitude with a 0-100% scale.

##Problem 3

1. We have the following: 
  random component is bernouli distribution: 
  $$Pr(Y_i = y_i | \pi_i) = (\pi_i)^{y_i}(1 - \pi_i)^{1 - y_i}$$
  Then we know $\pi_i$ is the population 'mean' we want to model; 
  
  linear predictor is: 
  $$\eta_i = \beta_0 + \beta_1 X_{1,i} + \beta_2 X_{2,i} + ... ...$$
  
  the link function is:
  $$\pi_i = g(\eta_i) = \frac{e^{\eta_i}}{1 + e^{\eta_i}}$$
  
2. The following is the regression result:

```
## 
## Call:
## glm(formula = vote96 ~ mhealth_sum + age + educ + black + female + 
##     married + inc10, family = binomial, data = clean_data)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.4843  -1.0258   0.5182   0.8428   2.0758  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -4.304103   0.508103  -8.471  < 2e-16 ***
## mhealth_sum -0.089102   0.023642  -3.769 0.000164 ***
## age          0.042534   0.004814   8.835  < 2e-16 ***
## educ         0.228686   0.029532   7.744 9.65e-15 ***
## black        0.272984   0.202585   1.347 0.177820    
## female      -0.016969   0.139972  -0.121 0.903507    
## married      0.296915   0.153164   1.939 0.052557 .  
## inc10        0.069614   0.026532   2.624 0.008697 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1468.3  on 1164  degrees of freedom
## Residual deviance: 1241.8  on 1157  degrees of freedom
##   (157 observations deleted due to missingness)
## AIC: 1257.8
## 
## Number of Fisher Scoring iterations: 4
```


--------------------------------------------------------------
     &nbsp;        Estimate   Std. Error   z value   Pr(>|z|) 
----------------- ---------- ------------ --------- ----------
 **mhealth_sum**   -0.0891     0.02364     -3.769   0.0001641 

     **age**       0.04253     0.004814     8.835   9.987e-19 

    **educ**        0.2287     0.02953      7.744   9.651e-15 

    **black**       0.273       0.2026      1.347     0.1778  

   **female**      -0.01697      0.14      -0.1212    0.9035  

   **married**      0.2969      0.1532      1.939    0.05256  

    **inc10**      0.06961     0.02653      2.624    0.008697 

 **(Intercept)**    -4.304      0.5081     -8.471   2.435e-17 
--------------------------------------------------------------

Table: Fitting generalized (binomial/logit) linear model: vote96 ~ mhealth_sum + age + educ + black + female + married + inc10

3. 

```
## [1] 0.1481481
```
Overall, the preformance or prediciton power of this model improves significantly relative to the model in last question. Using prediction error reduction as a criterion, we get the result that the prediction error reduces by 14.8% compared to the baseline model where we predict one individual will always vote. 
Among all of the independent variables, the mental health index, age, education and income turn out to be significant variables. Mental health, age and education are significant on the significance level of 0.001, while income is significant on the level of 0.05. Mental health index has a negative relationship with the voter turnout. On average, one level increase in the mental health index will reduce the odds, defined by the probablity of a voter voting versus not voting, by 1 percent. On the other hand, age, education and income all have positive effect on voter turnout. Specifically, one year increase in age will increase the odds of voting by 4.2%; one year increase in years of educatoin will on average increase the odds of voting by 22.86%; and every ten thousand dollar increase in income will on average increase the odds of voting by 7.0%.

##Problem 4

1.  We have the following: 
  random component is bernouli distribution: 
  $$Pr(Y_i = y_i | \mu_i) = \frac{\mu^k e^{-y_i}}{y_i!}$$
  Then we know $\pi_i$ is the population 'mean' we want to model; 
  
  linear predictor is: 
  $$\eta_i = \beta_0 + \beta_1 X_{1,i} + \beta_2 X_{2,i} + ... ...$$
  
  the link function is:
  $$\mu_i = g(\eta_i) = ln(\eta_i)$$

2. The following is the regression result:

```
## 
## Call:
## glm(formula = tvhours ~ age + childs + educ + female + grass + 
##     hrsrelax + black + social_connect + voted04 + xmovie + zodiac, 
##     family = poisson, data = tv_data)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -3.1554  -0.6695  -0.1160   0.4198   4.9059  
## 
## Coefficients:
##                    Estimate Std. Error z value Pr(>|z|)    
## (Intercept)        1.112287   0.235571   4.722 2.34e-06 ***
## age                0.001450   0.002815   0.515   0.6064    
## childs            -0.001404   0.023809  -0.059   0.9530    
## educ              -0.029077   0.012337  -2.357   0.0184 *  
## female             0.042035   0.064610   0.651   0.5153    
## grass             -0.097781   0.066750  -1.465   0.1430    
## hrsrelax           0.046556   0.010264   4.536 5.73e-06 ***
## black              0.462000   0.076397   6.047 1.47e-09 ***
## social_connect     0.043404   0.040403   1.074   0.2827    
## voted04           -0.096374   0.076641  -1.257   0.2086    
## xmovie             0.085894   0.075674   1.135   0.2564    
## zodiacAries       -0.118718   0.149049  -0.797   0.4257    
## zodiacCancer       0.007815   0.142978   0.055   0.9564    
## zodiacCapricorn   -0.232994   0.163973  -1.421   0.1553    
## zodiacGemini       0.007246   0.145343   0.050   0.9602    
## zodiacLeo         -0.177663   0.152692  -1.164   0.2446    
## zodiacLibra       -0.057168   0.135272  -0.423   0.6726    
## zodiacNaN         -0.313867   0.211348  -1.485   0.1375    
## zodiacPisces      -0.163360   0.163070  -1.002   0.3165    
## zodiacSagittarius -0.236268   0.155779  -1.517   0.1293    
## zodiacScorpio      0.032568   0.149195   0.218   0.8272    
## zodiacTaurus      -0.147495   0.162574  -0.907   0.3643    
## zodiacVirgo       -0.143673   0.154516  -0.930   0.3525    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 531.86  on 445  degrees of freedom
## Residual deviance: 432.21  on 423  degrees of freedom
##   (4064 observations deleted due to missingness)
## AIC: 1613.1
## 
## Number of Fisher Scoring iterations: 5
```
  

