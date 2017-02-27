PS7
================
Yuqing Zhang
2/26/2017

-   [Part 1: Sexy Joe Biden (redux)](#part-1-sexy-joe-biden-redux)
    -   [1 Estimate the training MSE of the model using the traditional approach](#estimate-the-training-mse-of-the-model-using-the-traditional-approach)
    -   [2 Estimate the test MSE of the model using the validation set approach.](#estimate-the-test-mse-of-the-model-using-the-validation-set-approach.)
    -   [3 Repeat the validation set approach 100 times.](#repeat-the-validation-set-approach-100-times.)
    -   [4 Estimate the test MSE of the model using the leave-one-out cross-validation (LOOCV) approach](#estimate-the-test-mse-of-the-model-using-the-leave-one-out-cross-validation-loocv-approach)
    -   [Problem 7 Bootstrap](#problem-7-bootstrap)
-   [Part3: College(GAM)](#part3-collegegam)
    -   [Problem1.Split the data](#problem1.split-the-data)
    -   [Problem2: OLS](#problem2-ols)
    -   [Problem 3 Estimate a GAM on the training data](#problem-3-estimate-a-gam-on-the-training-data)
-   [Problem 4 Use the test set to evaluate the model fit of the estimated OLS and GAM models](#problem-4-use-the-test-set-to-evaluate-the-model-fit-of-the-estimated-ols-and-gam-models)
-   [Problem 5 Non-linear?](#problem-5-non-linear)

Part 1: Sexy Joe Biden (redux)
------------------------------

### 1 Estimate the training MSE of the model using the traditional approach

``` r
biden <- biden %>%
  tbl_df()
lm.traditional<-lm(formula = biden ~ age + female + educ + dem + rep, data=biden)
tidy(lm.traditional)
```

    ##          term estimate std.error statistic  p.value
    ## 1 (Intercept)  58.8113    3.1244     18.82 2.69e-72
    ## 2         age   0.0483    0.0282      1.71 8.77e-02
    ## 3      female   4.1032    0.9482      4.33 1.59e-05
    ## 4        educ  -0.3453    0.1948     -1.77 7.64e-02
    ## 5         dem  15.4243    1.0680     14.44 8.14e-45
    ## 6         rep -15.8495    1.3114    -12.09 2.16e-32

``` r
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}
mse.traditional = mse(lm.traditional, biden) 
mse.traditional
```

    ## [1] 395

The mean squared error for the training set is 395.27

### 2 Estimate the test MSE of the model using the validation set approach.

``` r
set.seed(1234)
biden_split <- resample_partition(biden, c(test = 0.3, train = 0.7))
train_biden <- glm(biden ~ age + female + educ + dem + rep, data = biden_split$train)
test_biden <- glm(biden ~age + female + educ + dem + rep,data = biden_split$test)
mse.test <- mse(train_biden,biden_split$test)
```

Compare to the training MSE from step 1, the MSE using only the test set observations is 399.83 and it's larger.

### 3 Repeat the validation set approach 100 times.

``` r
mse_variable <- function(biden){
  biden_split <- resample_partition(biden, c(test = 0.3, train = 0.7))
  biden_train <- biden_split$train 
  biden_test <- biden_split$test
  model <- glm(biden ~ age + female + educ + dem + rep, data = biden_train)
  mses <-mse(model, biden_test)
  return(data_frame(mse = mses))

}
rerun = rerun(100, mse_variable(biden)) %>%
  bind_rows(.id = "id")
rerun_mean = mean(rerun$mse)

rerun_mean
```

    ## [1] 402

``` r
set.seed(1234)
MSE <- replicate(1000, {
  biden_split <- resample_partition(biden, c(valid = 0.3, train = 0.7))
  biden_train <- lm(biden ~ age + female + educ + dem + rep, data = biden_split$train)
  mse(biden_train, biden_split$valid)
})
mse_100 <- mean(MSE, na.rm = TRUE)
sd_100 <- sd(MSE, na.rm = TRUE)
mse_100
```

    ## [1] 399

### 4 Estimate the test MSE of the model using the leave-one-out cross-validation (LOOCV) approach

``` r
loocv_data <- crossv_kfold(biden, k = nrow(biden))
loocv_models <- map(loocv_data$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
loocv_mse <- map2_dbl(loocv_models, loocv_data$test, mse)
mse_loocv_mean = mean(loocv_mse)
```

The mean MSE value is 397.956 and it's smaller than the value we got before from the 100-times validation approach. It makes sense because LOOCV is not influenced by the resampling process. \#\#\#5 Estimate the test MSE of the model using the 10-fold cross-validation approach

``` r
biden_kfold <- crossv_kfold(biden, k = 10)
biden_models <- map(biden_kfold$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
                                        
biden_mse_10fold <- map2_dbl(biden_models, biden_kfold$test, mse)
biden_mse_10fold_mean = mean(biden_mse_10fold, na.rm = TRUE)
biden_mse_10fold_mean
```

    ## [1] 398

The mean MSE value is `biden_mse_10fold_mean` and it's smaller than the value we got before using LOOCV approach. It is Not a large difference from the LOOCV approach, but it take much less time to compute. \#\#\#6 Repeat the 10-fold cross-validation approach 100 times

``` r
terms <- 1:100
biden_fold10 <- vector("numeric", 100)
for(i in terms){
  biden_kfold <- crossv_kfold(biden, k = 10)
  biden_models_rep <- map(biden_kfold$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
  biden_mse_rep <- map2_dbl(biden_models_rep, biden_kfold$test, mse)
  biden_fold10[[i]] <- mean(biden_mse_rep,na.rm = TRUE)
}
biden_fold10_mean = mean(biden_fold10)
biden_fold10_mean
```

    ## [1] 398

``` r
ggplot(mapping = aes(biden_fold10)) + 
   geom_histogram(color = 'black', fill = 'white') +
   labs(title = "Distribution of MSE using 10-fold Cross-Validation Approach 100 times",
        x = "MSE values",
        y = "Frequency")+
  geom_vline(aes(xintercept = biden_fold10_mean, color = '100-times 10-fold')) +
  geom_vline(aes(xintercept = biden_mse_10fold_mean, color = '1-time 10-fold')) +
  geom_vline(aes(xintercept = mse.traditional, color = 'Origin Linear Regression')) + 
  scale_color_manual(name = NULL, breaks = c("100-times 10-fold", "1-time 10-fold","Origin Linear Regression"),values = c("blue", "green", "orange")) +
  theme(legend.position = 'bottom')
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](ps7_files/figure-markdown_github/unnamed-chunk-2-1.png) The mse values for repeating the 10-fold cross-validation approach 100 times are mostly within the range of (397, 400), the mean is 398.059. The values of these 100 times are very close.

### Problem 7 Bootstrap

``` r
biden_boot <- biden %>%
  modelr::bootstrap(1000) %>%
  mutate(model = map(strap, ~ lm(biden ~ age + female + educ + dem + rep, data = .)),
         coef = map(model, tidy))

biden_boot %>%
  unnest(coef) %>%
  group_by(term) %>%
  summarize(est.boot = mean(estimate),
            se.boot = sd(estimate, na.rm = TRUE))
```

    ## # A tibble: 6 × 3
    ##          term est.boot se.boot
    ##         <chr>    <dbl>   <dbl>
    ## 1 (Intercept)  58.5133   3.067
    ## 2         age   0.0493   0.030
    ## 3         dem  15.4365   1.024
    ## 4        educ  -0.3285   0.194
    ## 5      female   4.1267   0.979
    ## 6         rep -15.8387   1.355

The estimate from the bootstrap is very similar with the estimate from the original OLS model. The standard error differs among variables, some are larger, some are smaller. \#\#Part 2: College (bivariate)

``` r
sim_linear_mod_1 <- glm(Outstate ~ Room.Board, data = college)

ggplot(college, aes(Room.Board, Outstate)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Linear model for a linear relationship",
       x = 'Room and board costs',
       y = 'Out-of-state tuition')
```

![](ps7_files/figure-markdown_github/simple%20linear1-1.png) It seems like there is a linear relationship between 'room and board cost' and tuition. We test it by using:

``` r
test_Room.Board <- lm(Outstate ~ Room.Board, data = college)
summary(test_Room.Board)
```

    ## 
    ## Call:
    ## lm(formula = Outstate ~ Room.Board, data = college)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -8781  -2071   -351   1877  11877 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -17.4453   447.7679   -0.04     0.97    
    ## Room.Board    2.4000     0.0997   24.08   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3040 on 775 degrees of freedom
    ## Multiple R-squared:  0.428,  Adjusted R-squared:  0.427 
    ## F-statistic:  580 on 1 and 775 DF,  p-value: <2e-16

``` r
test_rb_df <- add_predictions(college, test_Room.Board)
test_rb_df <- add_residuals(test_rb_df,test_Room.Board)
ggplot(test_rb_df, aes(x = pred, y = resid)) +
  geom_smooth() +
  geom_point() +
  labs(title="Linear model regression for Room.Board",  x ="Predicted Room.Board", y = "Residuals") 
```

    ## `geom_smooth()` using method = 'loess'

![](ps7_files/figure-markdown_github/rb%20linear?-1.png) From the summary and graph above, we can see that a linear relationship can predict the relationship between out-of-state tuition and room-board cost very well. The residuals seem to be randomly around 0.

``` r
sim_linear_mod_2 <- glm(Outstate ~ Grad.Rate, data = college)

ggplot(college, aes(Grad.Rate,Outstate)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Linear model for a linear relationship",
       x = 'Graduation rate',
       y = 'Out-of-state tuition')
```

![](ps7_files/figure-markdown_github/simple%20linear2-1.png) It seems like there is a linear relationship between 'room and board cost' and tuition. We test it by using:

``` r
test_Grad.Rate <- lm(Outstate ~ Grad.Rate, data = college)
summary(test_Grad.Rate)
```

    ## 
    ## Call:
    ## lm(formula = Outstate ~ Grad.Rate, data = college)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -11222  -2252   -161   2157  12659 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   1681.9      467.3     3.6  0.00034 ***
    ## Grad.Rate      133.8        6.9    19.4  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3300 on 775 degrees of freedom
    ## Multiple R-squared:  0.326,  Adjusted R-squared:  0.326 
    ## F-statistic:  375 on 1 and 775 DF,  p-value: <2e-16

``` r
test_gr_df <- add_predictions(college, test_Grad.Rate)
test_gr_df <- add_residuals(test_gr_df,test_Grad.Rate)
ggplot(test_gr_df, aes(x = pred, y = resid)) +
  geom_smooth() +
  geom_point() +
  labs(title="Linear model regression for Grad.Rate",  x ="Predicted GraduationRate", y = "Residuals") 
```

    ## `geom_smooth()` using method = 'loess'

![](ps7_files/figure-markdown_github/gr%20linear?-1.png) From the graph we can see that when x is around 10000 the model is a good fit of the data, however it is not a good fit when x goes lower or higher. So we'll try a polynomial fit. But first let's use 10-fold Cross-vakudation to find the degree that generates the lowest mse.

``` r
set.seed(1234)
tenfold_gr <- crossv_kfold(college, k = 10)

polyMSE <- function(d) {
  tenFold_models <- map(tenfold_gr$train, ~ lm(Outstate ~ poly(Grad.Rate, d), data = .))
  tenFold_mse <- map2_dbl(tenFold_models, tenfold_gr$test, mse)
  tenFold_mean_mse <- mean(tenFold_mse)
}

tenFoldDF <- data.frame(index = 1:10)
tenFoldDF$mse <- unlist(lapply(1:10, polyMSE))

ggplot(tenFoldDF, aes(index, mse)) +
  geom_line() +
  geom_point() +
  scale_y_log10() +
  labs(title="MSE vs polynomial fit degree for Expend",  x ="Degree", y = "MSE") 
```

![](ps7_files/figure-markdown_github/find%20degree-1.png) From the above graph we can see that degree of 4 gives the lowest mse. So now we can use a polynomial model of degree 4.

``` r
outstate_mod <- glm(Outstate ~ poly(Grad.Rate, 4, raw = TRUE), data = college)
# estimate the predicted values and confidence interval
outstate_pred <- augment(outstate_mod, newdata = data_grid(college, Grad.Rate)) %>%
  rename(pred = .fitted) %>%
  mutate(pred_low = pred - 1.96 * .se.fit,
         pred_high = pred + 1.96 * .se.fit) 

# plot the log-odds curve
ggplot(outstate_pred, aes(Grad.Rate, pred, ymin = pred_low, ymax = pred_high)) +
  geom_point() +
  geom_errorbar() +
  labs(title = "Polynomial regression of Outstate",
       subtitle = "With 95% confidence interval",
       x = "Graduation Rate",
       y = "Predicted log-odds of out-of-state tuition")
```

![](ps7_files/figure-markdown_github/polynomial-1.png) Still, it does not seem like a good fit. So this time to improve my model, I plan to use splines. But first let's use cross-validation to choose number of knots and degree of the piecewise polynomial.

``` r
library(gam)
gr_bs_cv <- function(college, degree, knots){
  models <- map(college$train, ~ glm(Outstate ~ bs(Grad.Rate, degree = degree,df = degree + knots), data = .))
  models_mse <- map2_dbl(models, college$test, mse)
  return(mean(models_mse, na.rm = TRUE))
}

gr_bs_kfold <- crossv_kfold(college, k = 10)

terms <- 1:10
bs_cv_mses <- data.frame(matrix(vector(), 10, 10, dimnames=list(c(), c("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"))),stringsAsFactors=F)

for(dg in terms){
  for(kn in terms){
    bs_cv_mses[dg, kn] <- gr_bs_cv(gr_bs_kfold, degree = dg, knots = kn)
  }
}
```

The minimum value 1.0610^{7} appears in the first column, third row, indicating we should use 3 degrees of polynomial and 1 knot.

``` r
Outstate_gr_bs <- glm(Outstate ~ bs(Grad.Rate, degree = 3, df = 4), data = college)

college %>%
  add_predictions(Outstate_gr_bs) %>%
  add_residuals(Outstate_gr_bs) %>%
  {.} -> grid

ggplot(college, aes(x=Grad.Rate, y=Outstate)) +
  geom_point() +
  geom_line(aes(y=pred), data = grid, color = 'red', size = 1) +
  labs(title = "Regression of Out-of-state tuition on Graduation Rate",
        x = "Graduation Rate",
        y = "Out-of-state tuition")
```

![](ps7_files/figure-markdown_github/goodfit?-1.png)

``` r
ggplot(grid, aes(x = pred)) +
  geom_point(aes(y = resid)) +
  geom_hline(yintercept = 0, color = 'orange', size = 1, linetype = 'dashed') +
  labs(title = "Predicted Value and Residuals of linear regression (Outstate vs. Grad.Rate)",
        x = "Predicted Out-of-state tuition",
        y = "Residuals")
```

![](ps7_files/figure-markdown_github/goodfit?-2.png) As we can see from the graphs, now the line fits data very well, residuals are randomly located around 0.The graduation rate and out-of-state tuition has a reversed-u shaped relationship: first it decreases when graduation rate gets higher, after the decreasing speed reaches 0, the out-of-state increases with graduation rate.

``` r
sim_linear_mod_1 <- glm(Outstate ~ perc.alumni, data = college)

ggplot(college, aes(perc.alumni, Outstate)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Linear model for a linear relationship")
```

![](ps7_files/figure-markdown_github/simple%20linear3-1.png) It seems like there is a linear relationship between 'percent of alumni donation' and tuition. We test it by using:

``` r
test_perc.alumni <- lm(Outstate ~ perc.alumni, data = college)
summary(test_perc.alumni)
```

    ## 
    ## Call:
    ## lm(formula = Outstate ~ perc.alumni, data = college)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -11273  -2251   -361   2017   9731 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  6259.48     248.92    25.1   <2e-16 ***
    ## perc.alumni   183.84       9.61    19.1   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 3320 on 775 degrees of freedom
    ## Multiple R-squared:  0.321,  Adjusted R-squared:  0.32 
    ## F-statistic:  366 on 1 and 775 DF,  p-value: <2e-16

``` r
test_pa_df <- add_predictions(college, test_perc.alumni)
test_pa_df <- add_residuals(test_rb_df,test_perc.alumni)
ggplot(test_pa_df, aes(x = pred, y = resid)) +
  geom_smooth() +
  geom_point() +
  labs(title="Linear model regression for Room.Board",  x ="Predicted percent of alumni donation", y = "Residuals") 
```

    ## `geom_smooth()` using method = 'loess'

![](ps7_files/figure-markdown_github/pa%20linear?-1.png) From the summary and graph above, we can see that a linear relationship can predict the relationship between out-of-state tuition and percent of alumni donation cost very well. The residuals seem to be randomly around 0.

Part3: College(GAM)
-------------------

### Problem1.Split the data

``` r
set.seed(1234)
college_split <- resample_partition(college, c(test = 0.3, train = 0.7))
```

### Problem2: OLS

``` r
outstate_ols <- lm(Outstate~Private+Room.Board+PhD+perc.alumni+Expend+Grad.Rate,data=college_split$train)
summary(outstate_ols)
```

    ## 
    ## Call:
    ## lm(formula = Outstate ~ Private + Room.Board + PhD + perc.alumni + 
    ##     Expend + Grad.Rate, data = college_split$train)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -7756  -1325   -113   1300  10537 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -3.60e+03   5.38e+02   -6.68  6.0e-11 ***
    ## PrivateYes   2.58e+03   2.54e+02   10.14  < 2e-16 ***
    ## Room.Board   9.93e-01   1.03e-01    9.66  < 2e-16 ***
    ## PhD          3.65e+01   6.80e+00    5.37  1.2e-07 ***
    ## perc.alumni  5.34e+01   9.05e+00    5.90  6.4e-09 ***
    ## Expend       2.07e-01   2.07e-02    9.99  < 2e-16 ***
    ## Grad.Rate    3.07e+01   6.70e+00    4.59  5.6e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2090 on 537 degrees of freedom
    ## Multiple R-squared:  0.726,  Adjusted R-squared:  0.723 
    ## F-statistic:  238 on 6 and 537 DF,  p-value: <2e-16

The 6 predictors and intercept are all significant. Being a private university would increse the tuition by 2583 dollars. Increase room-board costs by 1 dollar would increase the out-of-state tuition by 0.993 dollar. Increase percent of faculty with Ph.D.'s by 1 percent would increase the out-of-state tuition by 36.5 dollars. Increase Percent of alumni who donate by 1, the tuition would increase 53.4 dollars. The instructional expenditure per student would increase the tuition by 0.207 if it increase 1 unit. The graduation rate would increase the tuition by 30.7 dollars if the graduation rate increases by 1.Together *R*<sup>2</sup> of the model is 0.726, meaning that the model could explain 72.6% of the variance in the training set.

### Problem 3 Estimate a GAM on the training data

``` r
library(gam)

# estimate model for splines on age and education plus dichotomous female
college_gam <- gam(Outstate ~ lo(PhD)+perc.alumni +Room.Board + lo(Expend)+bs(Grad.Rate, degree=3,df = 4) + Private , data = college)
summary(college_gam)
```

    ## 
    ## Call: gam(formula = Outstate ~ lo(PhD) + perc.alumni + Room.Board + 
    ##     lo(Expend) + bs(Grad.Rate, degree = 3, df = 4) + Private, 
    ##     data = college)
    ## Deviance Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -7042.3 -1113.0    11.5  1306.1  8066.6 
    ## 
    ## (Dispersion Parameter for gaussian family taken to be 3488425)
    ## 
    ##     Null Deviance: 1.26e+10 on 776 degrees of freedom
    ## Residual Deviance: 2.65e+09 on 760 degrees of freedom
    ## AIC: 13929 
    ## 
    ## Number of Local Scoring Iterations: 2 
    ## 
    ## Anova for Parametric Effects
    ##                                    Df   Sum Sq  Mean Sq F value Pr(>F)    
    ## lo(PhD)                             1 1.45e+09 1.45e+09   415.6 <2e-16 ***
    ## perc.alumni                         1 2.66e+09 2.66e+09   761.4 <2e-16 ***
    ## Room.Board                          1 2.34e+09 2.34e+09   670.1 <2e-16 ***
    ## lo(Expend)                          1 1.09e+09 1.09e+09   311.3 <2e-16 ***
    ## bs(Grad.Rate, degree = 3, df = 4)   4 3.41e+08 8.52e+07    24.4 <2e-16 ***
    ## Private                             1 6.32e+08 6.32e+08   181.1 <2e-16 ***
    ## Residuals                         760 2.65e+09 3.49e+06                   
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Anova for Nonparametric Effects
    ##                                   Npar Df Npar F  Pr(F)    
    ## (Intercept)                                                
    ## lo(PhD)                               2.6   3.33  0.024 *  
    ## perc.alumni                                                
    ## Room.Board                                                 
    ## lo(Expend)                            4.0  29.22 <2e-16 ***
    ## bs(Grad.Rate, degree = 3, df = 4)                          
    ## Private                                                    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

I used linear regression on Room.Board and percent of alumni donation and Private, local regression on Expend and percent of faculty with Phd, and spline with 3 degrees of freedom and 2 degrees polynomial on Grad.Rate. From the graph above we can tell that all of the predictors are significant. Let's look at some individual components:

``` r
college_gam_terms <- preplot(college_gam, se = TRUE, rug = FALSE)

data_frame(x = college_gam_terms$`lo(PhD)`$x,
           y = college_gam_terms$`lo(PhD)`$y,
           se.fit = college_gam_terms$`lo(PhD)`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of out-of-state tuition",
       x = "Percentage of falculty with PhD's",
       y = expression(f[1](PhD)))
```

![](ps7_files/figure-markdown_github/Phd-1.png) For percent of faculty with Ph.D.'s, overall there's a positive relationship. However, when the percentage is low (less than 50%), the relationship seems weaker (the 95% confidence interval is wide).

``` r
college_gam_terms <- preplot(college_gam, se = TRUE, rug = FALSE)

data_frame(x = college_gam_terms$`perc.alumni`$x,
           y = college_gam_terms$`perc.alumni`$y,
           se.fit = college_gam_terms$`perc.alumni`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of out-of-state tuition",
       x = "Percentage of alumni donation",
       y = expression(f[2](perc.alumni)))
```

![](ps7_files/figure-markdown_github/percent%20of%20alumni-1.png)

For percent of alumnis who denote, there's a positive relationship, and its slope doesn't change too much, indicating this predictor has a nearly steadily increasing influence on tuition.

``` r
# get graphs of each term
college_gam_terms <- preplot(college_gam, se = TRUE, rug = FALSE)

## roomboard
data_frame(x = college_gam_terms$`Room.Board`$x,
           y = college_gam_terms$`Room.Board`$y,
           se.fit = college_gam_terms$`Room.Board`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state tuition",
       subtitle = "Linear",
       x = "Room and board costs",
       y = expression(f[3](Room.Board)))
```

![](ps7_files/figure-markdown_github/roomboard-1.png) For Room and board costs, the effect appears substantial and statistically significant; as Room and board costs increases, predicted out-of-state tuition increases, meaning there is a positive relationship.

``` r
##expend
data_frame(x = college_gam_terms$`lo(Expend)`$x,
           y = college_gam_terms$`lo(Expend)`$y,
           se.fit = college_gam_terms$`lo(Expend`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state tuition",
       subtitle = "Local regression",
       x = "Instructional expenditure per student",
       y = expression(f[4](Expend)))
```

![](ps7_files/figure-markdown_github/expend-1.png) For instructional expenditure per student, the effect appears substantial and statistically significant until approximately 25000 dollars of instructional expenditure; as Room and board costs increases, predicted out-of-state tuition first increase until approximately 25000 dollars of instructional expenditure per student and then the effect remains flat.

``` r
##grad.rate
data_frame(x = college_gam_terms$`bs(Grad.Rate, degree = 3, df = 4)`$x,
           y = college_gam_terms$`bs(Grad.Rate, degree = 3, df = 4)`$y,
           se.fit = college_gam_terms$`bs(Grad.Rate, degree = 3, df = 4)`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state Tuition",
       subtitle = "Splines",
       x = "Grad.Rate",
       y = expression(f[5](Grad.Rate)))
```

![](ps7_files/figure-markdown_github/grad.rate-1.png) For Graduation rate, the effect appears substantial and statistically significant; as graduation rate increases, predicted out-of-state tuition first increase until approximately 90% of graduation rate, then decrease again.

``` r
##private
data_frame(x = college_gam_terms$Private$x,
           y =college_gam_terms$Private$y,
           se.fit = college_gam_terms$Private$se.y) %>%
  unique %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y, ymin = y_low, ymax = y_high)) +
  geom_errorbar() +
  geom_point() +
  labs(title = "GAM of Out-of-state tuition",
       x = 'Is Private or Public',
       y = expression(f[6](Private)))
```

![](ps7_files/figure-markdown_github/private-1.png)

For whether the school is a private or public school, the tuition difference is significant and substantial.From the graph we can tell that clearly private school requires more out of state tuition than public school.

Problem 4 Use the test set to evaluate the model fit of the estimated OLS and GAM models
----------------------------------------------------------------------------------------

``` r
mse_ols <- mse(outstate_ols, college_split$test)
mse_gam <- mse(college_gam, college_split$test)
mse_ols
```

    ## [1] 3595260

``` r
mse_gam
```

    ## [1] 3267057

GAM's MSE,3.26710^{6} is smaller than ols's mse, 3.59510^{6}, meaning that GAM fits the data better. This is because instead of using linearality for all predictors, we included various non-linear relationship in the model, which is closer to reality. This makes GAM's prediction more accurate.

Problem 5 Non-linear?
---------------------

From the discussion above we may say that percent of faculty with PhD's, the instructional expenditure per student and graduation rate has non-linear relationship with out-of-state tuition.

Looking at the ANOVA test though, only lo(Expend) has a statistically significant result from the Nonparametric Effect. This might mean that the relationship between it and outstate is nonlinear. Thus, we performed an ANOVA test between the original model and a GAM with linear Expend. In the test, the original model is more significant.This shows that Expend is indeed nonlinear.
