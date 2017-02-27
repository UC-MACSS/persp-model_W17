Problem set \#7: resampling and nonlinearity
================
Bobae Kang
February 27, 2017

-   [Part 1: Sexy Joe Biden (redux) \[4 points\]](#part-1-sexy-joe-biden-redux-4-points)
    -   [The traditional approach](#the-traditional-approach)
    -   [The validation set approach](#the-validation-set-approach)
    -   [The validation set approach, 100 times](#the-validation-set-approach-100-times)
    -   [The leave-one-out cross-validation (LOOCV) approach](#the-leave-one-out-cross-validation-loocv-approach)
    -   [The 10-fold cross-validation approach](#the-10-fold-cross-validation-approach)
    -   [The 10-fold cross-validation approach, 100 times](#the-10-fold-cross-validation-approach-100-times)
    -   [The bootstrap approach](#the-bootstrap-approach)
-   [Part 2: College (bivariate) \[3 points\]](#part-2-college-bivariate-3-points)
    -   [Simple linear models](#simple-linear-models)
    -   [Non-linear models](#non-linear-models)
    -   [Comparison](#comparison)
-   [Part 3: College (GAM) \[3 points\]](#part-3-college-gam-3-points)
    -   [OLS](#ols)
    -   [GAM](#gam)
    -   [Comparison](#comparison-1)
    -   [Non-linearity](#non-linearity)

Part 1: Sexy Joe Biden (redux) \[4 points\]
===========================================

The traditional approach
------------------------

Here I use all observations in the `biden` dataset to fit the linear regression model wherein the dependent variable is `biden` and independent variables are `age`, `female`, `educ`, `dem`, and `rep`. The coefficients for three independent variables (namely, `female`, `dem`, and `rep`) appear statistically significant with extremely small p-values. The mean squared error (MSE) of this model is 395.2702.

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

    ## [1] "The MSE value:"

    ## [1] 395.2702

The validation set approach
---------------------------

Here I split the dataset into the training set (70% of all observations) and the validation set (30% of all observations). Only the training set is used to fit the model. Then, I use the validation set to calcaulted the MSE of the trained model, which equals 389.8694. This is smaller than the previous MSE score!

    ## 
    ## Call:
    ## lm(formula = biden ~ age + female + educ + dem + rep, data = biden_split$train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -75.04 -11.16   0.78  12.97  52.59 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  60.40648    3.79946  15.899   <2e-16 ***
    ## age           0.06003    0.03363   1.785   0.0745 .  
    ## female        4.45037    1.14027   3.903   0.0001 ***
    ## educ         -0.59412    0.23642  -2.513   0.0121 *  
    ## dem          16.21889    1.28906  12.582   <2e-16 ***
    ## rep         -13.28393    1.57054  -8.458   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 20.03 on 1259 degrees of freedom
    ## Multiple R-squared:  0.2715, Adjusted R-squared:  0.2686 
    ## F-statistic: 93.82 on 5 and 1259 DF,  p-value: < 2.2e-16

    ## [1] "The MSE value:"

    ## [1] 389.8694

The validation set approach, 100 times
--------------------------------------

Now, I repeat the cross validation for the linear model 100 times, using 100 different splits of training and validation sets. Then I calculate MSE each time. The plot shows the MSE scores of 100 trials of cross validation. The red horizontal line marks the mean of all 100 MSEs, which equals 401.0916. This suggests that I may have been lucky on my first cv attempt.

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%201%20repeat%20cv%20100%20times-1.png)

    ## [1] "The mean of 100 CV MSE values:"

    ## [1] 401.0066

The leave-one-out cross-validation (LOOCV) approach
---------------------------------------------------

Now, I employ the LOOCV appraoch to the model and calcualte the mean MSE, which is 397.9555. This seemes to be an improvement from repeating the simple 70-30 cross validations.

    ## [1] "The LOOCV MSE value:"

    ## [1] 397.9555

The 10-fold cross-validation approach
-------------------------------------

The 10-fold cross-validation MSE score for the current model is 399.0661. Although this is larger than the LOOCV MSE score, it is still an improvement from repeating simple 70-30 cross validations. Since 10-fold cross-validation is computationally much cheaper than the LOOCV, especially when the model is complex and data is large (i.e. has many, many observations), this seems to be an effective approach.

    ## [1] "The 10-fold CV MSE value:"

    ## [1] 398.8552

The 10-fold cross-validation approach, 100 times
------------------------------------------------

Now, the repeated 10-fold cross-validation produces an even smaller MSE value: 398.1019. Although this is still larger than LOOCV MSE score in this case, the 100 repeated 10-fold approach is still computationally cheaper and the difference in MSE is very small.

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%201%20repeat%2010-fold%20cv%20100%20times-1.png)

    ## [1] "The meanf of 100 10-fold CV MSE values:"

    ## [1] 398.0958

The bootstrap approach
----------------------

Finally, I compare the linear regression estimates for coefficients and standard errors beween the original model and the bootstrap model (N=1000), which are included in the following data frame. The result shows that the original and bootstrapped estimates for coefficients and standard errors are very close to each other.

    ##          term Estimate.boot Std.Error.boot     Estimate Std. Error
    ## 1 (Intercept)   58.91195482     3.12522809  58.81125899  3.1244366
    ## 2         age    0.04830238     0.02973599   0.04825892  0.0282474
    ## 3         dem   15.42472392     1.08719047  15.42425563  1.0680327
    ## 4        educ   -0.35206406     0.19556406  -0.34533479  0.1947796
    ## 5      female    4.10156987     0.92576468   4.10323009  0.9482286
    ## 6         rep  -15.85036763     1.35581688 -15.84950614  1.3113624

Part 2: College (bivariate) \[3 points\]
========================================

Simple linear models
--------------------

In this part of the assignment, I use the `College` dataset and fit three simple linear models with the same response variable (i.e. `Outstate`) but three different explanatory variables (respectively, `Top10perc`, `PhD`, and `S.F.Ratio`).
In the first model, the coefficient for `Top10perc` is statistically significant with an extremely small p-value. The coefficient suggests that a unit increase in the percent of new students from top 10% of H.S. class, on average, leads to an increase in out-of-state tuition by $128.244. In the second model, the coefficient for `PhD` is statistically significant with an extremely small p-value. The coefficient suggests that a unit increase in the percent of faculty with Ph.D.'s, on average, leads to an increase in out-of-state tuition by $94.361.
In the third model, the coefficient for `S.F.Ratio` is statistically significant with an extremely small p-value. The coefficient suggests that a unit increase in the student/faculty ratio, on average, leads to a decrease in out-of-state tuition by $563.89.

    ## 
    ## Call:
    ## lm(formula = Outstate ~ Top10perc, data = college)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -11831.1  -2418.4    211.1   2116.4  11587.4 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 6906.459    221.614   31.16   <2e-16 ***
    ## Top10perc    128.244      6.774   18.93   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3329 on 775 degrees of freedom
    ## Multiple R-squared:  0.3162, Adjusted R-squared:  0.3153 
    ## F-statistic: 358.4 on 1 and 775 DF,  p-value: < 2.2e-16

    ## 
    ## Call:
    ## lm(formula = Outstate ~ PhD, data = college)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -8443.6 -3191.9   179.3  2554.0 14813.0 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 3584.361    608.838   5.887 5.85e-09 ***
    ## PhD           94.361      8.176  11.542  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3719 on 775 degrees of freedom
    ## Multiple R-squared:  0.1467, Adjusted R-squared:  0.1456 
    ## F-statistic: 133.2 on 1 and 775 DF,  p-value: < 2.2e-16

    ## 
    ## Call:
    ## lm(formula = Outstate ~ S.F.Ratio, data = college)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -10167.6  -2636.7   -251.5   2268.5  13267.0 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 18385.65     444.50   41.36   <2e-16 ***
    ## S.F.Ratio    -563.89      30.37  -18.57   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3349 on 775 degrees of freedom
    ## Multiple R-squared:  0.3078, Adjusted R-squared:  0.3069 
    ## F-statistic: 344.7 on 1 and 775 DF,  p-value: < 2.2e-16

The following three plots illustrate the distribution of `Outstate` variable against different explanatory variables as well as the linear regression lines. The plots suggest that the simple linear fit may not sufficiently explain the variation of the `Outstate` variable.

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%202%20exploratory%20plotting-1.png)![](Kang_persp-model_ps7_files/figure-markdown_github/Part%202%20exploratory%20plotting-2.png)![](Kang_persp-model_ps7_files/figure-markdown_github/Part%202%20exploratory%20plotting-3.png)

To further examine the fit of linear models, let's check how the residuals are distributed. The below is the combined plot of residuals. For all three models, the distribution of residuals suggest the existence of some systemic variation of `Outstate` that the linear models are failing to explain.

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%202%20residual%20plotting-1.png)

Non-linear models
-----------------

Here, I fit three non-linear models. For the first model with `Top10perc` as the explanatory variable, the new model uses the cubic spline technique. For the second model with `PhD` as the explanatory variable, the new model uses the thrid-order polynomial regression technique. For the third model with `S.F.Ratio` as the explanatory variable, the new model uses the monotonic transformation of the `S.F.Ratio` by taking its square root. In the first model, the coefficient

    ## 
    ## Call:
    ## glm(formula = Outstate ~ bs(Top10perc, degree = 3), data = college)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -11029.7   -2467.1     191.3    2160.6   11556.3  
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                  6893.4      516.3  13.352  < 2e-16 ***
    ## bs(Top10perc, degree = 3)1   3860.6     1627.9   2.372    0.018 *  
    ## bs(Top10perc, degree = 3)2  10122.4     1398.2   7.240 1.09e-12 ***
    ## bs(Top10perc, degree = 3)3  10388.2     1395.5   7.444 2.61e-13 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 11053028)
    ## 
    ##     Null deviance: 1.2559e+10  on 776  degrees of freedom
    ## Residual deviance: 8.5440e+09  on 773  degrees of freedom
    ## AIC: 14813
    ## 
    ## Number of Fisher Scoring iterations: 2

    ## 
    ## Call:
    ## glm(formula = Outstate ~ poly(PhD, 3), data = college)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -13235   -2711     201    2478   12353  
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)    10440.7      126.6  82.500  < 2e-16 ***
    ## poly(PhD, 3)1  42920.2     3527.6  12.167  < 2e-16 ***
    ## poly(PhD, 3)2  30971.5     3527.6   8.780  < 2e-16 ***
    ## poly(PhD, 3)3  11771.8     3527.6   3.337 0.000887 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 12444173)
    ## 
    ##     Null deviance: 1.2559e+10  on 776  degrees of freedom
    ## Residual deviance: 9.6193e+09  on 773  degrees of freedom
    ## AIC: 14905
    ## 
    ## Number of Fisher Scoring iterations: 2

    ## 
    ## Call:
    ## glm(formula = Outstate ~ sqrt(S.F.Ratio), data = college)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -10818.1   -2582.2    -174.8    2180.4    9847.7  
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)      26317.7      843.3   31.21   <2e-16 ***
    ## sqrt(S.F.Ratio)  -4272.7      224.7  -19.02   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 11048898)
    ## 
    ##     Null deviance: 1.2559e+10  on 776  degrees of freedom
    ## Residual deviance: 8.5629e+09  on 775  degrees of freedom
    ## AIC: 14810
    ## 
    ## Number of Fisher Scoring iterations: 2

Comparison
----------

Now, I compare linear and non-linear models. First, I compare models where `Top10perc` is the explanatory variable. The plot shows that the cubic spline model does not differ much from the linear model for the most part. Indeed, the calculated MSEs suggest that the non-linear model performs better but not to a great extent: (11052575-10996127)/11052575 = 0.005107226, or about 0.5 percent.

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%202%20compare%20Top10perc-1.png)

    ## # A tibble: 2 × 3
    ##          terms predictor      mse
    ##          <chr>     <chr>    <dbl>
    ## 1       Linear Top10perc 11052575
    ## 2 Cubic spline Top10perc 10996127

Second, I compare models where `PhD` is the explanatory variable. The plot shows that the third-order polynomial model differs much from the original linear model. There is, of course, a risk of overfitting. However, comparing these two in MSE suggests the non-linear model is indeed a better model with much improved performance: (13792993-12380110)/13792993 = 0.1024348 or approximately 10.2%.

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%202%20compare%20PhD-1.png)

    ## # A tibble: 2 × 3
    ##        Model Predictor      mse
    ##        <chr>     <chr>    <dbl>
    ## 1     Linear       PhD 13792993
    ## 2 ploy(X, 3)       PhD 12380110

Finally, I compare models where `S.F.Ratio` is the explanatory variable. On the plot, the difference between linear and non-linear models seem rather insignificant. The comparison between the two in MSE suggest that the non-linear model shows slightly improved performance on the test set: (11188174-11020458)/11188174 or approximately 1.5%

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%202%20compare%20S.F.Ratio-1.png)

    ## # A tibble: 2 × 3
    ##     Model Predictor      MSE
    ##     <chr>     <chr>    <dbl>
    ## 1  Linear S.F.Ratio 11188174
    ## 2 sqrt(X) S.F.Ratio 11020458

Part 3: College (GAM) \[3 points\]
==================================

OLS
---

First, I fit the OLS model to the training data, which consists of 80% of all observations. Its response variable is, again, `Outstate` and the explanatory variables include `Private`, `Room.Board`, `PhD`, `perc.alumni`, `Expend`, and `Grad.Rate`. The summary of the model indicates that the R-squared value suggests that model as a whole explains about 75% of the variation in `Outstate`, and the coefficients for all six predictors are statistically significant with very small p-vales. These coefficients suggest the following:

-   being a private institution, on average, leads to an increase in out-of-state tutition by $2,780, holding other variables constant;
-   a unit increase in room and board costs, on average, leads to an increase in out-of-state tuition by $0.9694, holding other variables constant ;
-   a unit increase in the percent of faculty with Ph.D.'s, on average, leads to an increase in out-of-state tuition by $3.905, holding other variables constant;
-   a unit increase in the percent of alumni who donate, on average, leads to an increase in out-of-state tuition by $4.193, holding other variables constant;
-   a unit increase in the instructional expenditure per student, on average, leads to an increase in out-of-state tuition by $0.2127, holding other variables constant; and
-   a unit increase in the graduation rate, on average, leads to an increase in out-of-state tuition by \#3.358, holding other variables constant.

<!-- -->

    ## 
    ## Call:
    ## lm(formula = Outstate ~ Private + Room.Board + PhD + perc.alumni + 
    ##     Expend + Grad.Rate, data = college_split$train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -7716.6 -1321.2   -41.9  1228.3 10538.6 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -3.472e+03  4.893e+02  -7.095 3.57e-12 ***
    ## PrivateYes   2.879e+03  2.358e+02  12.211  < 2e-16 ***
    ## Room.Board   9.498e-01  9.653e-02   9.840  < 2e-16 ***
    ## PhD          3.626e+01  6.236e+00   5.814 9.81e-09 ***
    ## perc.alumni  4.640e+01  8.248e+00   5.626 2.81e-08 ***
    ## Expend       2.093e-01  2.008e-02  10.426  < 2e-16 ***
    ## Grad.Rate    2.971e+01  5.899e+00   5.036 6.26e-07 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2057 on 614 degrees of freedom
    ## Multiple R-squared:  0.7427, Adjusted R-squared:  0.7402 
    ## F-statistic: 295.4 on 6 and 614 DF,  p-value: < 2.2e-16

GAM
---

The general additive model I used to explain 'Outstate' consists of the following: a linear regression model for `Private`, cubic spline model for `perc.alumni`, and four local regression linear regression models for `Room.Board`, `PhD`, `Expend`, and `Grad.Rate`. In choosing these models, I used the Akaike Information Criterion (AIC) score for selecting the final model. That is, the final model is the one with the lowest AIC value among all the variations I have tried. The p-value for each model suggest that each of these models makes a statistically significant difference to the model as a whole.

    ## 
    ## Call: gam(formula = Outstate ~ Private + lo(Room.Board) + lo(PhD) + 
    ##     bs(perc.alumni, degree = 3) + lo(Expend) + lo(Grad.Rate), 
    ##     data = college_split$train)
    ## Deviance Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -7086.24 -1128.16    24.14  1230.81  7939.36 
    ## 
    ## (Dispersion Parameter for gaussian family taken to be 3526371)
    ## 
    ##     Null Deviance: 10101839162 on 620 degrees of freedom
    ## Residual Deviance: 2114407007 on 599.5986 degrees of freedom
    ## AIC: 11147.4 
    ## 
    ## Number of Local Scoring Iterations: 2 
    ## 
    ## Anova for Parametric Effects
    ##                                Df     Sum Sq    Mean Sq F value    Pr(>F)
    ## Private                       1.0 2852804232 2852804232 808.992 < 2.2e-16
    ## lo(Room.Board)                1.0 2018284542 2018284542 572.341 < 2.2e-16
    ## lo(PhD)                       1.0  709409276  709409276 201.173 < 2.2e-16
    ## bs(perc.alumni, degree = 3)   3.0  357030400  119010133  33.749 < 2.2e-16
    ## lo(Expend)                    1.0  708947427  708947427 201.042 < 2.2e-16
    ## lo(Grad.Rate)                 1.0  123719503  123719503  35.084 5.323e-09
    ## Residuals                   599.6 2114407007    3526371                  
    ##                                
    ## Private                     ***
    ## lo(Room.Board)              ***
    ## lo(PhD)                     ***
    ## bs(perc.alumni, degree = 3) ***
    ## lo(Expend)                  ***
    ## lo(Grad.Rate)               ***
    ## Residuals                      
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Anova for Nonparametric Effects
    ##                             Npar Df  Npar F   Pr(F)    
    ## (Intercept)                                            
    ## Private                                                
    ## lo(Room.Board)                  3.0  3.7176 0.01105 *  
    ## lo(PhD)                         2.7  1.9100 0.13286    
    ## bs(perc.alumni, degree = 3)                            
    ## lo(Expend)                      4.0 24.9532 < 2e-16 ***
    ## lo(Grad.Rate)                   2.6  3.3814 0.02272 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

The following plots show the relationship between the response variable and each of six explanatory variables, holding others constant. The interpretation of these plots are as follows:

-   Being private schools leads to the higher out-of-state tuition, holding other variables constant;
-   The higher room and board costs generally lead to the higher out-of-state tuition, holding other variables constant;
-   Overall, the percentage of faculty with Ph.D.'s is positively correlated with out-of-state tuition, holding other variables constant.However, there is negative correlation between them around 30 &lt; `PhD` &lt; 50;
-   The higher percentage of alumni who donate generally lead to higher out-of-state tuition, holding other variables constant;
-   Holding other variables constant, instructional expenditure per student is positively correlated with out-of-state tuition while `Expend` is lower than about 30000. When `Expend` value is higher, it is negatively correlated with the response variable; and
-   Holding other variables constant, the graduate rate is positively correlated with out-of-state tuition while `Grad.Rate` is lower than about 80. When `Grad.Rate` value is higher, it is negatively correlated with the response variable.

![](Kang_persp-model_ps7_files/figure-markdown_github/Part%203%20GAM%20graph-1.png)![](Kang_persp-model_ps7_files/figure-markdown_github/Part%203%20GAM%20graph-2.png)![](Kang_persp-model_ps7_files/figure-markdown_github/Part%203%20GAM%20graph-3.png)![](Kang_persp-model_ps7_files/figure-markdown_github/Part%203%20GAM%20graph-4.png)![](Kang_persp-model_ps7_files/figure-markdown_github/Part%203%20GAM%20graph-5.png)![](Kang_persp-model_ps7_files/figure-markdown_github/Part%203%20GAM%20graph-6.png)

Comparison
----------

To compare the OLS model and GAM model, I have first looked at the residual plots. In both cases, the residuals are distributed in a roughly randomly fashion althoguh there may be the linear pattern for residuals. It is difficult to determine which model is with less errors simply using the plot. When their MSE values are compared, however, GAM appears as the better model with approximately 18.6%$ less MSE.
![](Kang_persp-model_ps7_files/figure-markdown_github/Part%203%20compare%20OLS%20and%20GAM-1.png)

    ## # A tibble: 2 × 2
    ##   Model     MSE
    ##   <chr>   <dbl>
    ## 1   OLS 4090539
    ## 2   GAM 3348411

Non-linearity
-------------

In Part 2 of this assignment, I have investigated the relationship between the response variable and theree explanatory variables (`Top10perc`, `PhD`, and `S.F.Ratio`). In all three cases, non-linear models showed some improvement from their linear counterparts. In particular, the 3rd-order polynomial model for `OutState` and `PhD` made a sizable difference in MSE when compared to the simple linear model, which leads to me believe that the true relationship between these two variables is indeed non-linear. In contrast, the difference in MSE between non-linear and linear models wherein `Top10perc` is the explanatory variable seems largely insignificant. This leads me to believe that the true relationship between `Top10perc` and `OutState` may indeed be linear.
