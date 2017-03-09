Problem set \#7: resampling and nonlinearity
================
Wenxi Xiao
**Due Monday February 27th at 11:30am**

-   [Part 1: Sexy Joe Biden (redux)](#part-1-sexy-joe-biden-redux)
    -   [Estimate the test MSE of the model using the validation set approach.](#estimate-the-test-mse-of-the-model-using-the-validation-set-approach.)
    -   [Estimate the test MSE of the model using the leave-one-out cross-validation (LOOCV) approach.](#estimate-the-test-mse-of-the-model-using-the-leave-one-out-cross-validation-loocv-approach.)
    -   [Estimate the test MSE of the model using the 10-fold cross-validation approach.](#estimate-the-test-mse-of-the-model-using-the-10-fold-cross-validation-approach.)
    -   [Repeat the 10-fold cross-validation approach 100 times, using 100 different splits of the observations into 10-folds.](#repeat-the-10-fold-cross-validation-approach-100-times-using-100-different-splits-of-the-observations-into-10-folds.)
    -   [Compare the estimated parameters and standard errors from the original model in step 1 (the model estimated using all of the available data) to parameters and standard errors estimated using the bootstrap (*n* = 1000).](#compare-the-estimated-parameters-and-standard-errors-from-the-original-model-in-step-1-the-model-estimated-using-all-of-the-available-data-to-parameters-and-standard-errors-estimated-using-the-bootstrap-n-1000.)
-   [Part 2: College (bivariate)](#part-2-college-bivariate)
    -   [1. Expend](#expend)
    -   [2. PhD](#phd)
    -   [3. perc.alumni](#perc.alumni)
-   [Part 3: College (GAM)](#part-3-college-gam)
    -   [Split the data into a training set and a test set.](#split-the-data-into-a-training-set-and-a-test-set.)
    -   [Estimate an OLS model on the training data, using out-of-state tuition as the response variable and the other six variables as the predictors.](#estimate-an-ols-model-on-the-training-data-using-out-of-state-tuition-as-the-response-variable-and-the-other-six-variables-as-the-predictors.)
    -   [Estimate a GAM on the training data, using out-of-state tuition as the response variable and the other six variables as the predictors.](#estimate-a-gam-on-the-training-data-using-out-of-state-tuition-as-the-response-variable-and-the-other-six-variables-as-the-predictors.)
    -   [Use the test set to evaluate the model fit of the estimated OLS and GAM models.](#use-the-test-set-to-evaluate-the-model-fit-of-the-estimated-ols-and-gam-models.)
    -   [For which variables, if any, is there evidence of a non-linear relationship with the response?](#for-which-variables-if-any-is-there-evidence-of-a-non-linear-relationship-with-the-response)

Part 1: Sexy Joe Biden (redux)
==============================

For this exercise we consider the following functional form:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>3</sub> + *β*<sub>4</sub>*X*<sub>4</sub> + *β*<sub>5</sub>*X*<sub>5</sub> + *ϵ*

where *Y* is the Joe Biden feeling thermometer, *X*<sub>1</sub> is age, *X*<sub>2</sub> is gender, *X*<sub>3</sub> is education, *X*<sub>4</sub> is Democrat, and *X*<sub>5</sub> is Republican. \#\# Estimate the training MSE of the model using the traditional approach. \* Fit the linear regression model using the entire dataset and calculate the mean squared error for the training set.

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

Beta\_0, the intercept, is 58.81126 with a standard error of 3.12444. Beta\_1 is 0.04826 with a standard error of 0.02825. Beta\_2 is 4.10323 with a standard error of 0.94823. Beta\_3 is -0.34533 with a standard error of 0.19478. Beta\_4 is 15.42426 with a standard error of 1.06803. Beta\_5 is -15.84951, with a standard error of 1.31136.

    ## [1] 395.2702

The mean squared error for the training set is 395.2702.

Estimate the test MSE of the model using the validation set approach.
---------------------------------------------------------------------

-   Split the sample set into a training set (70%) and a validation set (30%).
-   Fit the linear regression model using only the training observations.

-   Calculate the MSE using only the test set observations:

<!-- -->

    ## [1] 403.2375

1.  Repeat the validation set approach 100 times, using 100 different splits of the observations into a training set and a validation set:

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/validation100-1.png)

We see that the validation estimate of the MSE can be variable, depending highly on which observations are included in the training set and which observations are included in the test set. Depending on the specific training/test split, our MSE varies by up to 102.

MSE\_mean:

    ## [1] 402.3668

MSE\_median:

    ## [1] 404.3352

After doing the validation set approach 100 times, I found the mean of MSE and the median of MSE are generally quite close to each other and they are all close to the MSE after only one validation. Repeating the validation approach helps to eliminate the bias introduced by only doing one validation.

Estimate the test MSE of the model using the leave-one-out cross-validation (LOOCV) approach.
---------------------------------------------------------------------------------------------

LOOCV MSE:

    ## [1] 397.9555

The LOOCV MSE is close to the mean MSE from 100-time validation set approach. The LOOCV method produces estimates of the error rate (i.e., MSEs) that have minimal bias and are relatively steady (i.e. non-varying), unlike the validation set approach where the test MSE estimate is highly dependent on the sampling process for training/test sets.

Estimate the test MSE of the model using the 10-fold cross-validation approach.
-------------------------------------------------------------------------------

MSE\_mean:

    ## [1] 397.7777

This MSE mean can be seen as the same as the LOOCV MSE, but this approach is less computationally-intensive than LOOCV. This method yields the pretty much the same results as LOOCV does.

Repeat the 10-fold cross-validation approach 100 times, using 100 different splits of the observations into 10-folds.
---------------------------------------------------------------------------------------------------------------------

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/ten_fold_crossvalidation_100times-1.png)

From the above distribution of MSE, I found that MSEs are close in their values, which suggests that the 10-fold approach is steady. The MSE mean is 398.0641646, which is pretty much the same as the MSE obtained with only one time 10-fold cross-validation approach.

Compare the estimated parameters and standard errors from the original model in step 1 (the model estimated using all of the available data) to parameters and standard errors estimated using the bootstrap (*n* = 1000).
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Parameters and standard errors estimated using the bootstrap:

    ## # A tibble: 6 <U+00D7> 3
    ##          term    est.boot    se.boot
    ##         <chr>       <dbl>      <dbl>
    ## 1 (Intercept)  58.9371766 2.97886156
    ## 2         age   0.0479520 0.02886022
    ## 3         dem  15.4378230 1.11262713
    ## 4        educ  -0.3525426 0.19119841
    ## 5      female   4.1029972 0.95732289
    ## 6         rep -15.8678205 1.44375912

The estimated parameters and standard errors from the original model:

    ##                 Estimate Std. Error    t value     Pr(>|t|)
    ## (Intercept)  58.81125899  3.1244366  18.822996 2.694143e-72
    ## age           0.04825892  0.0282474   1.708438 8.772744e-02
    ## female        4.10323009  0.9482286   4.327258 1.592601e-05
    ## educ         -0.34533479  0.1947796  -1.772952 7.640571e-02
    ## dem          15.42425563  1.0680327  14.441745 8.144928e-45
    ## rep         -15.84950614  1.3113624 -12.086290 2.157309e-32

The parameters of the two models are pretty much the same. The standard errors of `age`, `dem`, `rep`, and `female` are slightly larger than those in the step-1 model, and the standard errors of the intercept and `education` are slightly less than those in the step-1 model, which makes sense because bootstrap standard errors should be generally larger because bootstrapping does not depend on any distributional assumptions.

Part 2: College (bivariate)
===========================

1. Expend
---------

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/Expend_linear-1.png)

The above scatterplot showing the relationship between `instructional expenditure per student` and `out of state tuition` looks logarithmic.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/Expend_log-1.png)

After log transforming `instructional expenditure per student`, the model seems to fit the data pretty well. Let's take a look at the residuals.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/Expend_residual-1.png)

The residual plot shows no structure among the residuals and they seem to randomly scattered around zero. Let's justify this model using cross-validation methods. Specifically, I will use LOOCV to compare MSEs of with log transformation and of without log transformation.

The MSE without the log transformation is 8.847579410^{6}. The MSE with the log transformation after LOOCV is 6.875011110^{6}, which is significantly less than the MSE without the log transformation. Thus, log transformation is better.

    ##              Estimate Std. Error   t value      Pr(>|t|)
    ## (Intercept) -57502.04  2089.8876 -27.51442 8.347473e-117
    ## log(Expend)   7482.15   229.9153  32.54307 4.059156e-147

Statistically, the relationship between `out-of-state tuition` and `instructional expenditure per student` is significant at alpha=0.05 level. For every one percent increase in `instructional expenditure per student`, the predicted value of `out-of-state tuition` will on average increase 74.8215 dollars.

2. PhD
------

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/phd_linear-1.png)

The above scatterplot showing the relationship between `PhD` and `out of state tuition` looks non-linear. I am going to try the 3rd degree polynomial transformation on `PhD`.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/phd_3-1.png)

After 3rd degree polynomial transformation of `Percent of faculty with Ph.D.`, the model seems to fit the data pretty well. Let's take a look at the residuals.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/phd_residual-1.png)

The residual plot shows no obvious structure among the residuals and they seem to randomly scattered around zero. Let's justify this model using the cross-validation method.

    ## [1] 13792993

    ## [1] 12730141

The MSE without the 3rd degree polynominal transformation is 1.379299310^{7}. The MSE with the polynominal transformation after CV is 1.273014110^{7}, which is less than the MSE without the transformation. Thus, 3rd degree polynominal transformation is better.

    ## 
    ## Call:
    ## lm(formula = Outstate ~ poly(PhD, 3), data = college_split$train)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -10999.2  -2701.0    103.9   2510.3  12157.6 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)    10477.4      150.6  69.575  < 2e-16 ***
    ## poly(PhD, 3)1  35674.5     3512.4  10.157  < 2e-16 ***
    ## poly(PhD, 3)2  26225.1     3512.4   7.466 3.33e-13 ***
    ## poly(PhD, 3)3  11930.8     3512.4   3.397 0.000732 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3512 on 540 degrees of freedom
    ## Multiple R-squared:  0.2399, Adjusted R-squared:  0.2357 
    ## F-statistic: 56.82 on 3 and 540 DF,  p-value: < 2.2e-16

Statistically, the relationship between `out-of-state tuition` and `PhD` is significant at alpha=0.05 level as the p-value of all terms passed the p&lt;0.05 signifigance level.

3. perc.alumni
--------------

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/a_linear-1.png)

The above scatterplot showing the relationship between `Percent of alumni who donate` and `out of state tuition` looks non-linear. I will try local regression on `Percent of alumni who donate`.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/a_LOESS-1.png) After locally regressing `perc.alumni`, the model seems to fit the data pretty well. Let's take a look at the residuals.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/a_residual-1.png)

The residual plot shows no obvious structure among the residuals and they seem to randomly scattered around zero. Let's justify this model using the CV method.

    ## [1] 10980849

    ## [1] 11038406

Unfortunetly, we found that the MSE without the 3rd degree polynominal transformation is 1.098084910^{7}. The MSE with the local transformation after CV is 1.103840610^{7}, which is greater than the MSE without the transformation. Thus, the local regression is not better than no transformation in terms of minimizing MSE. However, from the plot, we can clearly see that the local regression better fits the data and given the difference in MSEs are not that great, I decided to stick with local regression.

    ##                  Estimate Std. Error  t value      Pr(>|t|)
    ## (Intercept)     6259.4803 248.919066 25.14665 1.691632e-102
    ## lo(perc.alumni)  183.8379   9.611968 19.12594  4.349522e-67

Statistically, the relationship between `out-of-state tuition` and `perc.alumni` is significant at alpha=0.05 level. For every one percent increase in `perc.alumni`, the predicted value of `out-of-state tuition` will on average increase 183.8379 dollars.

Part 3: College (GAM)
=====================

Split the data into a training set and a test set.
--------------------------------------------------

Estimate an OLS model on the training data, using out-of-state tuition as the response variable and the other six variables as the predictors.
----------------------------------------------------------------------------------------------------------------------------------------------

    ## 
    ## Call:
    ## lm(formula = Outstate ~ Private + Room.Board + PhD + perc.alumni + 
    ##     Expend + Grad.Rate, data = college_split$train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -6101.4 -1325.3   -20.8  1230.2  5757.3 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -3.784e+03  5.175e+02  -7.312 9.61e-13 ***
    ## PrivateYes   2.801e+03  2.371e+02  11.815  < 2e-16 ***
    ## Room.Board   1.012e+00  9.917e-02  10.205  < 2e-16 ***
    ## PhD          3.863e+01  6.299e+00   6.134 1.67e-09 ***
    ## perc.alumni  5.899e+01  8.710e+00   6.773 3.32e-11 ***
    ## Expend       1.881e-01  1.949e-02   9.651  < 2e-16 ***
    ## Grad.Rate    2.818e+01  6.240e+00   4.516 7.74e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1963 on 537 degrees of freedom
    ## Multiple R-squared:  0.7632, Adjusted R-squared:  0.7605 
    ## F-statistic: 288.4 on 6 and 537 DF,  p-value: < 2.2e-16

The model fits the data moderately since the 6 variables (i.e., `Private`, `Room.Board`, `PhD`, `perc.alumni`, `Expend`, and `Grad.Rate`) together can explain 76.32 percent of the variations in `out-of-state tuition` (i.e., R-square is 0.7632). All 6 variables are statistically significant at the alpha=0.01 level. The unconditional mean of `out-of-state tuition` (i.e., intercept) is -3.784e+03. Being a public university would decrease the college's `out-of-state tuition` by 2801 dollars. With one dollar increase in `room-board costs`, the `out-of-state tuition` will on average increase 1.012 dollars. With one percent increase in `percent of faculty with Ph.D`, the `out-of-state tuition` will on average increase 38.63 dollars. With one percent increase in `percent of alumni who donate`, the `out-of-state tuition` will on average increase 58.99 dollars. With one unit increase in `instructional expenditure per student`, the `out-of-state tuition` will on average increase 0.1881 dollars. With one unit increase in `graduation rate`, the `out-of-state tuition` will on average increase 28.18 dollars.

Estimate a GAM on the training data, using out-of-state tuition as the response variable and the other six variables as the predictors.
---------------------------------------------------------------------------------------------------------------------------------------

Below constructed a GAM model that regresses `out-of-state tuition` on `Private` and `Room.Board`, the LOWESS of `PhD` and `perc.alumni`, the log transformed `Expend`, and the third-degree polynomial transformed `Grad.Rate`.

    ## 
    ## Call: gam(formula = Outstate ~ Private + Room.Board + lo(PhD) + lo(perc.alumni) + 
    ##     log(Expend) + poly(Grad.Rate, 3), data = college_split$train, 
    ##     na.action = na.fail)
    ## Deviance Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -6445.9 -1173.6   -59.8  1199.1  4785.1 
    ## 
    ## (Dispersion Parameter for gaussian family taken to be 3456513)
    ## 
    ##     Null Deviance: 8737709971 on 543 degrees of freedom
    ## Residual Deviance: 1831701864 on 529.9277 degrees of freedom
    ## AIC: 9750.032 
    ## 
    ## Number of Local Scoring Iterations: 2 
    ## 
    ## Anova for Parametric Effects
    ##                        Df     Sum Sq    Mean Sq F value    Pr(>F)    
    ## Private              1.00 2801603840 2801603840 810.529 < 2.2e-16 ***
    ## Room.Board           1.00 2024142062 2024142062 585.602 < 2.2e-16 ***
    ## lo(PhD)              1.00  830241746  830241746 240.196 < 2.2e-16 ***
    ## lo(perc.alumni)      1.00  493180010  493180010 142.681 < 2.2e-16 ***
    ## log(Expend)          1.00  479709160  479709160 138.784 < 2.2e-16 ***
    ## poly(Grad.Rate, 3)   3.00  112030823   37343608  10.804 6.708e-07 ***
    ## Residuals          529.93 1831701864    3456513                      
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Anova for Nonparametric Effects
    ##                    Npar Df Npar F   Pr(F)  
    ## (Intercept)                                
    ## Private                                    
    ## Room.Board                                 
    ## lo(PhD)                2.7 1.0463 0.36695  
    ## lo(perc.alumni)        2.4 2.7738 0.05371 .
    ## log(Expend)                                
    ## poly(Grad.Rate, 3)                         
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

All 6 variables are statistically significant at the alpha=0.01 level. Below are graphs of each term:

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/GAM_plot_Private-1.png)

From the graph above we can clearly see a difference between private and public universities. Specifically, being private positively influences the `out-of-state tuition` while being public negatively influence the `out-of-state tuition` and also to a greater extent.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/GAM_plot_Room.Board-1.png)

From the graph above we can clearly see there is a positive linear relationship between `Room and board costs` and `out of state tuition`. Specifically, as `Room and board costs` increases `out of state tuition` also increases.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/GAM_plot_PhD-1.png)

From the graph above we can generally conclude that as `Percent of faculty with Ph.D.'s` increases `out of state tuition` also increases.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/GAM_plot_perc.alumni-1.png)

From the graph above we can generally conclude that as `Percent of alumni who donate` increases `out of state tuition` also increases.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/GAM_plot_Expend-1.png)

From the graph above we can generally conclude that as `Instructional expenditure per student` increases `out of state tuition` also increases. Also, the `Instructional expenditure per student`'s positive effect on `out of state tuition` is stronger in the beginning.

![](WX_ps7-resample-nonlinear_files/figure-markdown_github/GAM_plot_Grad.Rate-1.png)

From the graph above we can generally conclude that as `Graduation rate` increases `out of state tuition` also increases. Also, the `Graduation rate`'s positive effect on `out of state tuition` is the strongest at its middle values.

Use the test set to evaluate the model fit of the estimated OLS and GAM models.
-------------------------------------------------------------------------------

The MSE of the OLS model is 4.824118110^{6}. The MSE of the GAM model is 4.47419110^{6}, which is less than the MSE of the OLS model, suggesting that the GAM model fits better, which makes sense because the GAM model is more sophisticated.

For which variables, if any, is there evidence of a non-linear relationship with the response?
----------------------------------------------------------------------------------------------

There are evidence for variables `PhD`, `perc.alumni`, `Expend`, and `Grad.Rate`. From the discussion in part two, we see clearly that `PhD`, `perc.alumni`, and `Expend` have a non-linear relationship with `out of state tuition`.

    ## Analysis of Deviance Table
    ## 
    ## Model 1: Outstate ~ Private + Room.Board + lo(PhD) + lo(perc.alumni) + 
    ##     log(Expend)
    ## Model 2: Outstate ~ Private + Room.Board + lo(PhD) + lo(perc.alumni) + 
    ##     log(Expend) + Grad.Rate
    ## Model 3: Outstate ~ Private + Room.Board + lo(PhD) + lo(perc.alumni) + 
    ##     log(Expend) + poly(Grad.Rate, 3)
    ##   Resid. Df Resid. Dev Df Deviance  Pr(>Chi)    
    ## 1    532.93 1943388350                          
    ## 2    531.93 1860176378  1 83211972 9.271e-07 ***
    ## 3    529.93 1831701864  2 28474514   0.01626 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

We can see that the non-linear model is statistically significant at alpha = 0.05. Thus, we have evidence that `Grad.Rate` also has a non-linear relationship with `out of state tuition`.
