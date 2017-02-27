Problem set \#5: linear regression
================
Wenxi Xiao
February 13th 2017

-   [Problem 1 - Describe the data](#problem-1---describe-the-data)
-   [Problem 2 - Simple linear regression](#problem-2---simple-linear-regression)
-   [Problem 3 - Multiple linear regression](#problem-3---multiple-linear-regression)
-   [Problem 4 - Multiple linear regression model (with even more variables!)](#problem-4---multiple-linear-regression-model-with-even-more-variables)
-   [Problem 5 - Interactive linear regression model](#problem-5---interactive-linear-regression-model)

Problem 1 - Describe the data
=============================

![](WX_ps5-linear-regression_files/figure-markdown_github/get_biden-1.png)

The histogram shows a negatively skewed distribution, suggesting that more people in the 2008 survey showed relatively high feelings of warmth towards Joe Biden. The majority of the reported feeling thermometers in this dataset is greater than 50, with the peak in 50, which suggests that people who rated Joe Biden as neither warm nor cold are the largest subgroup. Also, I noticed that people's ratings are all multiples of 5, so I speculate that feeling thermometer options were given in such a fashion in the survey.

Problem 2 - Simple linear regression
====================================

Estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub>
 where *Y* is the Joe Biden feeling thermometer and *X*<sub>1</sub> is age.

    ## 
    ## Call:
    ## lm(formula = biden ~ age, data = biden)
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

    ##   age .fitted   .se.fit     ymin     ymax
    ## 1  45 62.0056 0.5577123 60.91248 63.09872

Beta\_0 is 59.19736, with a standard error of 1.64792. Beta\_1 is 0.06241, with a standard error of 0.03267.

1.  Statistically speaking, at a 95% confidence level, there is no relationship between `age` and `Biden feeling thermometers` because the p-value is 0.0563 which is greater than 0.05. However, at a 90% confidence level, there is a relationship between `age` and `Biden feeling thermometers` because the p-value is 0.0563 which is less than 0.1.

2.  The relationship between `age` and `Biden feeling thermometers` is not strong because the absolute value of the correlation coefficient beta\_1 is 0.06241, which is small. We can say that 1-year increase in `age` will produce on average only a 0.06241 unit increase in `Biden feeling thermometer score`.

3.  The relationship between the predictor and the response is positive because beta\_1 is 0.06241, which is a positive number. The older the respondent, the higher the feeling thermometer score he or she will give.

4.  The *R*<sup>2</sup> of the model is 0.002018, which suggests that 0.2018 percent of the variation in `Biden warmth score` can be explained by `age` alone, so this model does not fit well. It'a bad model because the predictor does not explain much of the variations in `Biden warmth score`.

5.  The predicted `biden` associated with an `age` of 45 is 62.0056. The associated 95% confidence intervals are \[60.91248, 63.09872\].

6.  Plot the response and predictor and draw the least squares regression line: ![](WX_ps5-linear-regression_files/figure-markdown_github/plot_simple_linear_reg-1.png)

Problem 3 - Multiple linear regression
======================================

Estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>3</sub>

where *Y* is the Joe Biden feeling thermometer, *X*<sub>1</sub> is age, *X*<sub>2</sub> is gender, and *X*<sub>3</sub> is education.

    ## 
    ## Call:
    ## lm(formula = biden ~ age + female + educ, data = biden)
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

Beta\_0 is 68.62101, with a standard error of 3.59600. Beta\_1 is 0.04188, with a standard error of 0.03249. Beta\_2 is 6.19607, with a standard error of 1.09670. Beta\_3 is -0.88871, with a standard error of 0.22469.

1.  When we take `age`, `gender`, and `education` into account to predict `Biden Warmth`, there is not a statistically significant relationship between `age` and `Biden Warmth` at a 95% confidence level, because the corresponding p-value is 0.198 which is greater than 0.05. However, there is a statistically significant relationship between `gender` and `Biden Warmth` because the corresponding p-value is 1.86e-08 which is less than 0.05, and there is a statistically significant relationship between `education` and `Biden Warmth` because the corresponding p-value is 7.94e-05 which is less than 0.05.

2.  The parameter for `female`, beta\_2, is 6.19607, which suggests that holding `age` and `education` constant the predicted value of `Biden Warmth` is on average 6.19607 higher when the respondent is female.

3.  The *R*<sup>2</sup> of the model is 0.02723. 2.723 percent of the variation in `biden` can be explained by `age`, `gender`, and `education`. This model is a better model than the age-only model because the *R*<sup>2</sup> here is larger, which means that this model can explain more of the variations in Joe Biden feeling thermometer than the age-only model does, although not much.

4.  Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type:

![](WX_ps5-linear-regression_files/figure-markdown_github/plot_multi_linear_reg-1.png)

There is a problem with this model because the residuals are not randomly scattered around zero. We can see systematic structures in the residuals. We see that the residuals vary together with different party ID types. We can also see that the predicted warmth score for Democrats is generally larger than the data and the predicted warmth score for Republican is generally lower than the data. To solve this, we want to include party affiliation as a predictor in our model.

Problem 4 - Multiple linear regression model (with even more variables!)
========================================================================

Estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>3</sub> + *β*<sub>4</sub>*X*<sub>4</sub> + *β*<sub>5</sub>*X*<sub>5</sub>

where *Y* is the Joe Biden feeling thermometer, *X*<sub>1</sub> is age, *X*<sub>2</sub> is gender, *X*<sub>3</sub> is education, *X*<sub>4</sub> is Democrat, and *X*<sub>5</sub> is Republican

    ## 
    ## Call:
    ## lm(formula = biden ~ age + female + educ + dem + rep, data = biden)
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

Beta\_0 is 58.81126, with a standard error of 3.12444. Beta\_1 is 0.04826, with a standard error of 0.02825. Beta\_2 is 4.10323, with a standard error of 0.94823. Beta\_3 is -0.34533, with a standard error of 0.19478. Beta\_4 is 15.42426, with a standard error of 1.06803. Beta\_5 is -15.84951, with a standard error of 1.31136.

1.  The Statistic significance of the relationship between `gender` and `Biden warmth` did not change. Here, at a 95% confidence level, there is still a significant relationship between `gender` and `Biden warmth` because the corresponding p-value is 1.59e-05, which is less than 0.05. Also, the sign of the relationship did not change. However, the strength of the relationshp decreased becuase here the correlation coefficient is 4.10323, which is less than that in problem3. Thus, when holding pary affiliaction, age, and education constant, the respondent being female makes the `Biden warmth` increase to a smaller amount compared with when holding only age and education constant.

2.  The *R*<sup>2</sup> of the model is 0.2815. 28.15 percent of the variation in `biden` can be explained by age, gender, education, and party identification. This is a better model than the age + gender + education model because this model can explain much more variations in `biden` than the previous model does.

3.  Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type: ![](WX_ps5-linear-regression_files/figure-markdown_github/plot_multi_linear_reg2-1.png)

By adding variables for party ID to the regression model, we have somewhat fixed the previous problem because we see here the residuals for different party ID types are centered around zero and the smooth fit lines for different party ID types are also all hovering around zero. However, we can still see a systematic structure in residuals due to omitting other potentially important predictors such as any interactions between predictors.

Problem 5 - Interactive linear regression model
===============================================

Estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>1</sub>*X*<sub>2</sub>

where *Y* is the Joe Biden feeling thermometer, *X*<sub>1</sub> is gender, and *X*<sub>2</sub> is Democrat.

    ## 
    ## Call:
    ## lm(formula = biden ~ female * dem, data = biden)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -75.519 -13.070  -0.198  11.930  49.802 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  50.1976     0.9219  54.452  < 2e-16 ***
    ## female        5.0135     1.2943   3.874 0.000111 ***
    ## dem          22.8719     1.5079  15.168  < 2e-16 ***
    ## female:dem   -2.5642     1.9997  -1.282 0.199902    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 20.74 on 1803 degrees of freedom
    ## Multiple R-squared:  0.2201, Adjusted R-squared:  0.2188 
    ## F-statistic: 169.6 on 3 and 1803 DF,  p-value: < 2.2e-16

    ##   female dem  .fitted   .se.fit     ymin     ymax
    ## 1      1   1 75.51883 0.9484884 73.65979 77.37787
    ## 2      1   0 55.21113 0.9085045 53.43046 56.99180
    ## 3      0   1 73.06954 1.1932803 70.73071 75.40837
    ## 4      0   0 50.19763 0.9218722 48.39076 52.00450

Beta\_0 is 50.1976, with a standard error of 0.9219. Beta\_1 is 5.0135, with a standard error of 1.2943. Beta\_2 is 22.8719, with a standard error of 1.5079. Beta\_3 is -2.5642, with a standard error of 1.9997.

1.  Female Democrats on average rated Biden's warmth highest, with an average rating of 75.51883 and a 95% confidence interval of \[73.65979 77.37787\]. Female Republicans on average rated Biden's warmth with a score of 55.21113 and the 95% confidence interval is \[53.43046 56.99180\]. Male Democrats on average rated Biden's warmth second to the highest, with an average rating of 73.06954 and a 95% confidence interval of \[70.73071 75.40837\]. Female Republicans on average rated Biden's warmth the lowest, with an average rating of 50.19763 and a 95% confidence interval of \[48.39076 52.00450\].

The relationship between party ID and Biden warmth does not differ for males/females. Male Democrats generally gave Biden higher ratings than male Republicans did, and female Democrats generally gave Biden higher ratings than female Republicans did.

The relationship between gender and Biden warmth differs for Democrats/Republicans. Male Democrats generally gave Biden lower ratings than female Democrats did, while female Republicans generally gave Biden lower ratings than male Republicans did.
