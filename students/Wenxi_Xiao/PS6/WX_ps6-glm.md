Problem set \#6: Generalized linear models
================
Wenxi Xiao
**Due Monday February 20th at 11:30am**

-   [Part 1: Modeling voter turnout](#part-1-modeling-voter-turnout)
    -   [Describe the data](#describe-the-data)
    -   [Basic model](#basic-model)
    -   [Multiple variable model](#multiple-variable-model)
        -   [Three componets:](#three-componets)
        -   [Estimate the model and report results:](#estimate-the-model-and-report-results)
-   [Part 2: Modeling tv consumption](#part-2-modeling-tv-consumption)
    -   [Three componets:](#three-componets-1)
    -   [Estimate the model and report results:](#estimate-the-model-and-report-results-1)

Part 1: Modeling voter turnout
==============================

Describe the data
-----------------

![](WX_ps6-glm_files/figure-markdown_github/hist_mental_health-1.png)

1.  The unconditional probability of a given individual turning out to vote excluding missing data is 62.95904%.The unconditional probability of a given individual turning out to vote including missing data is 68.23574%.

2.  Generating a scatterplot of the relationship between mental health and observed voter turnout and overlay a linear smoothing line: Since it makes no sense to include entries with missing data for `mental health index` or missing data for `voter behavior`, we need to first modify the dataset.

![](WX_ps6-glm_files/figure-markdown_github/scatterplot-1.png)

From the scatterplot, we can see that there is a negative correlation with `voting behavior` and `mental health index` score, which means that more depressed the voter is the less likely he or she will vote. The `voting behavior` is dichotomous that only have 2 possible values, 0 or 1, but the `mental health index` score ranges from 0 to 9, which makes the smoothed regression line here is problematic. For instance, we could have potential respondents whose `mental health index` scores are high enough that they could have a negative predicted value for voting behavior, which makes no sense in our context. Thus, linear regression model seems to be an inappropriate assumption for our data.

Basic model
-----------

Estimate a logistic regression model of the relationship between mental health and voter turnout.

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum, family = binomial, data = mental_health_no_na)
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

1.  The relationship between mental health and voter turnout is statistically significant because the p-value is 3.13e-13, which is less than 0.05 significance level. For one unit increase in `mental health index` score, we expect to see the log-odds of voting decrease by 0.14348, which means that the odds ratio associated with one unit increase in `mental health index` score is 0.8663381. Thus, I think the relationship between mental health and voter turnout is substantively significant.

2.  The estimated parameter for mental health is -0.14348. For one unit increase in `mental health index` score, we expect to see the log-odds of voting decrease by 0.14348: ![](WX_ps6-glm_files/figure-markdown_github/basic_model_graph-1.png)

3.  With one unit increase on `mental health index` score, we expect to see the odds of voter voting against not voting decreases by 14.348 percent(%).

![](WX_ps6-glm_files/figure-markdown_github/basic_model_odds_graph-1.png)

1.  The estimated parameter for mental health in terms of probabilities cannot be interpreted without the value of the initial age, on which the first difference depends.

![](WX_ps6-glm_files/figure-markdown_github/basic_model_p_graph-1.png)

    ## [1] -0.02917824

    ## [1] -0.03477821

1.  The first difference for an increase in the `mental health index` from 1 to 2 is -0.02917824. The first difference for an increase in the `mental health index` from 5 to 6 is -0.03477821.

<!-- -->

    ## [1] 0.677761

    ## [1] 0.01616628

    ## Area under the curve: 0.6243

1.  Using a threshold value of 0.5, the accuracy rate is estimated to be 0.677761. The proportional reduction in error (PRE) is estimated to be 1.616628%. The AUC for this model is 0.6243. I do not consider this model to be a good model because 1)the PRE is a quite small increase over that of the useless classifier, and 2) the AUC score is also a quite small increase over that of the useless classifier.

Multiple variable model
-----------------------

### Three componets:

My multiple variable logistic regression model of voter turnout has

a bernouli distributed random component:

Pr(*Y*<sub>*i*</sub> = *y*<sub>*i*</sub>|*π*) = *π*<sub>*i*</sub><sup>*y*<sub>*i*</sub></sup> (1 − *π*<sub>*i*</sub>)<sup>1 − *y*<sub>*i*</sub></sup> ,

a linear predictor:

*η*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*m**h**e**a**l**t**h**s**u**m*<sub>*i*</sub> + *β*<sub>2</sub>*a**g**e*<sub>*i*</sub> + *β*<sub>3</sub>*e**d**u**c**a**t**i**o**n*<sub>*i*</sub> + *β*<sub>4</sub>*b**l**a**c**k*<sub>*i*</sub> + *β*<sub>5</sub>*f**e**m**a**l**e*<sub>*i*</sub> + *β*<sub>6</sub>*m**a**r**r**i**e**d*<sub>*i*</sub> + *β*<sub>7</sub>*i**n**c**o**m**e*10<sub>*i*</sub> ,

and a logit function as the link function:

*π*<sub>*i*</sub> = *e*<sup>*η*<sub>*i*</sub></sup> / (1 + *e*<sup>*η*<sub>*i*</sub></sup>).

### Estimate the model and report results:

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum + age + educ + black + female + 
    ##     married + inc10, family = binomial, data = mental_health_no_na)
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

Among the all the predictors, mental health (p-value=-0.089102), age(p-value&lt; 2e-16), education (p-value=9.65e-15), and income (p-value=0.008697) have a statistically significant relationship with voter turnout on a p&lt;0.05 significance level. Specifically, one unit increase in the mental health score will on average decrease the odds (i.e., probability) of voting by 0.89102 percent, one year increase in age will on average increase the odds of voting by 4.2534 percent, one year increase in educatoin will on average increase the odds of voting by 22.8686 precent, and every 10 thousand dollar increase in income will on average increase the odds of voting by 6.9614 precent. I think the relationship between educatoin and voting is the most substantially significant one.

    ## [1] 0.1481481

The PRE here is 14.81481%, which is a quite large increase over that of the useless classifier. Thus, this model performed better than the basic model.

    ## [1] 0.04140208

However, after conducting the chi-square goodness of fit test, we found that the p-value from the goodness of fit test is 0.04140208, which passed the p&lt;0.05 significance level. We then reject the null hypothesis that the sample data are consistent with a hypothesized distribution (i.e., the model fits the data).

Part 2: Modeling tv consumption
===============================

Three componets:
----------------

My multiple variable Poisson regression model of tv consumption has

a Poisson-distribution random component:

Pr(*Y*<sub>*i*</sub> = *y*<sub>*i*</sub>|*μ*) = *μ*<sup>*y*<sub>*i*</sub></sup> *e*<sup>−*μ*</sup> / *y*<sub>*i*</sub>! ,

a linear predictor:

*η*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*a**g**e*<sub>*i*</sub> + *β*<sub>2</sub>*c**h**i**l**d**s*<sub>*i*</sub> + *β*<sub>3</sub>*e**d**u**c**a**t**i**o**n*<sub>*i*</sub> + *β*<sub>4</sub>*f**e**m**a**l**e*<sub>*i*</sub> + *β*<sub>5</sub>*g**r**a**s**s*<sub>*i*</sub> + *β*<sub>6</sub>*h**r**s**r**e**l**a**x*<sub>*i*</sub> + *β*<sub>7</sub>*b**l**a**c**k*<sub>*i*</sub> + *β*<sub>8</sub>*s**o**c**i**a**l**c**o**n**n**e**c**t*<sub>*i*</sub> + *β*<sub>9</sub>*v**o**t**e**d*04<sub>*i*</sub> + *β*<sub>10</sub>*x**m**o**v**i**e*<sub>*i*</sub> + *β*<sub>11</sub>*z**o**d**i**a**c*<sub>*i*</sub> ,

and a log function as the link function:

log(*μ*<sub>*i*</sub>) = *η*<sub>*i*</sub>.

Estimate the model and report results:
--------------------------------------

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

Among the all the predictors, education (p-value=0.0184), hours of relax per day (p-value=5.73e-06), and black (p-value=1.47e-09) have a statistically significant relationship with tv comsumption on a a p&lt;0.05 significance level. Specifically, one year incrase in education will on average result in a 0.029077 unit decrease in log of hours of watching TV, one hour incrase in hours of relax per day will on average result in a 0.046556 unit increase in log of hours of watching TV, and being balck increase on average the log of hours of watching TV by 0.462000 unit. I think the relationship between black and tv watching is the most substantially significant one, which suggests that we may omitted variables such as SES as predictors in our model.

    ## [1] 0.3679743

After conducting the chi-square goodness of fit test, we found that the p-value from the goodness of fit test is 0.3679743, which did not passed the p&lt;0.05 significance level. We then cannot reject the null hypothesis that the sample data are consistent with a hypothesized distribution (i.e., the model fits the data).
