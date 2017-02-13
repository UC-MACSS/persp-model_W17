Perspectives on Computational Modeling PS5
================
HyungJin Cho
February 13, 2017

Describe the data
=================

![](PS5_HJ_files/figure-markdown_github/histogram-1.png) **1.In a few sentences, describe any interesting features of the graph.** The plotted histogram has a mode of 50 and a left-skewed shape. The Feeling Thermometer of 50 has the highest counts. The left-skewed shape indicates that more people feel Biden as warm rather than as cold.

Simple linear regression
========================

    ## 
    ## Call:
    ## lm(formula = biden ~ age, data = DATA)
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

**1.Is there a relationship between the predictor and the response?** The *p* − *v**a**l**u**e* of 0.05626 implies that the relationship between age and Biden's feeling thermometer is not statistically significant.

**2.How strong is the relationship between the predictor and the response?** The *β*<sub>1</sub> of 0.06241 suggests that there is a weak relationship. An increase in age by 1 unit would increase Biden's feeling thermometer by 0.06241.

**3.Is the relationship between the predictor and the response positive or negative?** The *β*<sub>1</sub> of 0.06241 suggests that there is a positive relationship.

**4.Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does `age` alone explain? Is this a good or bad model?** The *R*<sup>2</sup> value of 0.002018 means age explains 0.2018% of the variation in Biden's feeling thermometer. This is a bad model.

    ##   age .fitted   .se.fit     ymin     ymax
    ## 1  45 62.0056 0.5577123 60.91248 63.09872

**5.What is the predicted `biden` associated with an `age` of 45? What are the associated 95% confidence intervals?** The predicted Biden's feeling thermometer is 62.0056 when it is associated with the age of 45. The associated 95% confidence intervals are (60.91248, 63.09872).

**6.Plot the response and predictor. Draw the least squares regression line.** ![](PS5_HJ_files/figure-markdown_github/simple%20linear%20regression%203-1.png)

Multiple linear regression
==========================

    ## 
    ## Call:
    ## lm(formula = biden ~ age + female + educ, data = DATA)
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

**1.Is there a statistically significant relationship between the predictors and response?** The *p* − *v**a**l**u**e* of 1.86e-08 for gender and the *p* − *v**a**l**u**e* of 7.94e-05 for education implies that coefficients of gender and education are statistically significant.

**2.What does the parameter for `female` suggest?** The *β*<sub>2</sub> of 6.19607 suggests that being female would increase Biden's feeling thermometer by 6.19607 units.

**3.Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does age, gender, and education explain? Is this a better or worse model than the age-only model?** The *R*<sup>2</sup> of 0.02723 indicates that age, gender, adn education explains 2.723% of the variation in Biden's feeling thermometer. This is a better model than the age-only model.

![](PS5_HJ_files/figure-markdown_github/multiple%20linear%20regression%201-2-1.png) **4.Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type. Is there a problem with this model? If so, what?** The difference in residual distribution implies that this model doesn't take party ID type into account which appears a significant factor.

Multiple linear regression model (with even more variables)
===========================================================

    ## 
    ## Call:
    ## lm(formula = biden ~ age + female + educ + dem + rep, data = DATA)
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

**1.Did the relationship between gender and Biden warmth change?** The *β*<sub>2</sub> of 4.1023 indicates that the relationship is weaker than the previous estimate of 6.19607. However, the *p* − *v**a**l**u**e* of 1.59e-05 supports the predictor is still statistically significant.

**2.Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does age, gender, education, and party identification explain? Is this a better or worse model than the age + gender + education model?** The *R*<sup>2</sup> of 0.2815 suggests that age, gender, education, and party identification exaplains 28.15% of the variation in Biden's feeling thermometer. This is a better model than the previous model.

![](PS5_HJ_files/figure-markdown_github/multiple%20linear%20regression%202%20plot-1.png) **3.Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type. By adding variables for party ID to the regression model, did we fix the previous problem?** The differences in residual distribution of Democrat, Republican, and Independent are around zero which imply that this model fixed the previous problem.

Interactive linear regression model
===================================

    ##   female dem     pred  .fitted   .se.fit     ymin     ymax
    ## 1      0   0 39.38202 39.38202 1.4553632 36.52951 42.23453
    ## 2      0   1 73.06954 73.06954 1.1173209 70.87959 75.25949
    ## 3      1   0 45.77720 45.77720 1.3976638 43.03778 48.51662
    ## 4      1   1 75.51883 75.51883 0.8881114 73.77813 77.25953

**1.Estimate predicted Biden warmth feeling thermometer ratings and 95% confidence intervals for female Democrats, female Republicans, male Democrats, and male Republicans. Does the relationship between party ID and Biden warmth differ for males/females? Does the relationship between gender and Biden warmth differ for Democrats/Republicans?** Estimate predicted Biden warmth feeling thermometer ratings for female Democrats is 75.51883 (73.77813, 77.25953), female Republicans is 45.77720 (43.03778, 48.51662), male Democrats is 73.06954 (70.87959, 75.25949), and male Republicans is 39.38202 (36.52951, 42.23453). Party ID has greater difference in Biden warmth for males (33.7) than females (29.7). Gender has greater difference in Biden warmth differ for Republicans (6.4) than Democrats (2.4). This indicates an interaction between gender and party ID.
