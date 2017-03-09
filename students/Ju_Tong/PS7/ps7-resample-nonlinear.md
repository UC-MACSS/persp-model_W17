Problem set \#7: resampling and nonlinearity
================
Tong Ju
**20170227**

-   [Part 1: Sexy Joe Biden](#part-1-sexy-joe-biden)
-   [Part 2: College (bivariate) \[3 points\]](#part-2-college-bivariate-3-points)
    -   [Instructional expenditure per student as predictor:](#instructional-expenditure-per-student-as-predictor)
    -   [Graduation rate as predictor:](#graduation-rate-as-predictor)
    -   [Room and board costs as predictor:](#room-and-board-costs-as-predictor)
-   [Part 3: College (GAM) \[3 points\]](#part-3-college-gam-3-points)

Part 1: Sexy Joe Biden
======================

Given the following functional form:

\[Y = \beta_0 + \beta_{1}X_1 + \beta_{2}X_2 + \beta_{3}X_3 + \beta_{4}X_4 + \beta_{5}X_5 + \epsilon\]

where \(Y\) is the Joe Biden feeling thermometer, \(X_1\) is age, \(X_2\) is gender, \(X_3\) is education, \(X_4\) is Democrat, and \(X_5\) is Republican.

1)Estimate the training MSE of the model using the traditional approach.

``` r
# linear regression model using the entire dataset
biden_all <- lm(biden ~ age + female + educ + dem + rep, data = biden)

# calculate the mean squared error for the training set.
calc_mse <- function(model, data){
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}

train_only <- calc_mse(biden_all, biden)

pander (biden_all)
```

<table style="width:86%;">
<caption>Fitting linear model: biden ~ age + female + educ + dem + rep</caption>
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>age</strong></td>
<td align="center">0.04826</td>
<td align="center">0.02825</td>
<td align="center">1.708</td>
<td align="center">0.08773</td>
</tr>
<tr class="even">
<td align="center"><strong>female</strong></td>
<td align="center">4.103</td>
<td align="center">0.9482</td>
<td align="center">4.327</td>
<td align="center">1.593e-05</td>
</tr>
<tr class="odd">
<td align="center"><strong>educ</strong></td>
<td align="center">-0.3453</td>
<td align="center">0.1948</td>
<td align="center">-1.773</td>
<td align="center">0.07641</td>
</tr>
<tr class="even">
<td align="center"><strong>dem</strong></td>
<td align="center">15.42</td>
<td align="center">1.068</td>
<td align="center">14.44</td>
<td align="center">8.145e-45</td>
</tr>
<tr class="odd">
<td align="center"><strong>rep</strong></td>
<td align="center">-15.85</td>
<td align="center">1.311</td>
<td align="center">-12.09</td>
<td align="center">2.157e-32</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">58.81</td>
<td align="center">3.124</td>
<td align="center">18.82</td>
<td align="center">2.694e-72</td>
</tr>
</tbody>
</table>

Based on the linear regression model, the estimated parameters and p-values are reported in the above table. Using the entire dataset as training and testing set, the mean squared error is 395.27.

2)Estimate the test MSE of the model using the validation set approach.

``` r
set.seed(1234)
# split the dataset
biden_split <- resample_partition(biden, c(valid = 0.3, train = 0.7))
# modelin of the train data set
biden_train <- lm(biden ~ age + female + educ + dem + rep, data = biden_split$train)

mse1 <- calc_mse(biden_train, biden_split$valid)
```

I fit the linear regression model using only the training observations. The mean squared error for the test data on this model is 399.83 , which is larger than the previous MSE. Because the model built upon the training dataset cannot perfectly generalize the remaining 70% observation in test set, this model is less accurate than the one in section 1.1.

3)Repeat the validation set approach 100 times, using 100 different splits of the observations into a training set and a validation set. Comment on the results obtained.

``` r
set.seed(1234)
# replicate the validation set approach for 100 times
mse100 <- replicate(100, {
  biden_split <- resample_partition(biden, c(valid = 0.3, train = 0.7))
  biden_train <- lm(biden ~ age + female + educ + dem + rep, data = biden_split$train)
  calc_mse(biden_train, biden_split$valid)
})
mse100_mean <- mean(mse100)


mse1000 <- replicate(1000, {
  biden_split <- resample_partition(biden, c(valid = 0.3, train = 0.7))
  biden_train <- lm(biden ~ age + female + educ + dem + rep, data = biden_split$train)
  calc_mse(biden_train, biden_split$valid)
})
mse1000_mean <- mean(mse1000)

# histogram of the MSE values

ggplot(mapping = aes(mse100)) + 
  geom_histogram(color = 'black', fill = 'blue') +
  theme_bw()+
  geom_vline(aes(xintercept = mse100_mean, color = 'MSE for 100-times Validation')) +
  geom_vline(aes(xintercept = mse1000_mean, color = 'MSE for 1000-times Validation')) +
  geom_vline(aes(xintercept = mse1, color = 'MSE for 1-time Validation')) +
  geom_vline(aes(xintercept = train_only, color = 'MSE for all data model')) + 
  labs(title = "Distribution of MSE using Validation Set Approach 100 times and 1000 times",
        x = "MSE values",
        y = "Frequency") 
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

<img src="ps7-resample-nonlinear_files/figure-markdown_github/1.3-1.png" style="display: block; margin: auto;" /> Based on the histogram above, repeating the validation set approach for 100 times did not really improve the MSE (the MSE value is 401.66, which is even a little bit larger than the one-time validation). However, after repeating for 1000 times, the MSE (399.3) is much closer to the MSE for all-data model. This result may suggest when validation set approach is deployed for more times, the avarage value of MSE will be much closer to the MSE for the model based on all the observations. More importantly, the distribution of MSE ranges from 330 to 450, indicating that the validation set approach is not so steady and vulnerable to the split of the training and test sets.

4)Estimate the test MSE of the model using the leave-one-out cross-validation (LOOCV) approach. Comment on the results obtained.

``` r
# LOOCV method
biden_loocv <- crossv_kfold(biden, k = nrow(biden))
biden_models <- map(biden_loocv$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
biden_mses <- map2_dbl(biden_models, biden_loocv$test, calc_mse)
mse_loocv <- mean(biden_mses)
```

The estimated test MSE of the model using the LOOCV approach is 397.9555046. This value is much closer to the training MSE of the model (395.27) we obtained in the section 1.1. Given that the LOOCV method doesn't depend on the sampling process for training/test sets, it is a much more steady methods. However, this method is rather time consuming.

5)Estimate the test MSE of the model using the \(10\)-fold cross-validation approach. Comment on the results obtained.

``` r
set.seed(1234)
# 10-fold cross-validation
biden_10fold <- crossv_kfold(biden, k = 10)
biden_10models <- map(biden_10fold$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
biden_10mses <- map2_dbl(biden_10models, biden_10fold$test, calc_mse)
mse_10fold <- mean(biden_10mses, na.rm = TRUE)
```

Using 10-fold cross-validation approach, 397.88 is gained, which is slightly smaller than leave-one-out approach. Since this approach repeats the validation approach for 10 times rather than the count of observations, the flexibility decreases. However, the computational efficiency increases.

6)Repeat the \(10\)-fold cross-validation approach 100 times, using 100 different splits of the observations into \(10\)-folds. Comment on the results obtained.

``` r
set.seed(1234)
# 10-fold cross-validation for 100 times
mse_10fold_100 <- replicate(100, {
  biden_10fold <- crossv_kfold(biden, k = 10)
  biden_10models <- map(biden_10fold$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
  biden_10mses <- map2_dbl(biden_10models,
                           biden_10fold$test, calc_mse)
  mse_10fold <- mean(biden_10mses)
})

mean_mse_10fold<- mean(mse_10fold_100)

ggplot(mapping = aes(mse_10fold_100)) + 
  geom_histogram(color = 'black', fill = 'blue') +
  theme_bw()+
  geom_vline(aes(xintercept = mean_mse_10fold, color = 'MSE for 100-times Validation')) +
  geom_vline(aes(xintercept = mse_10fold, color = 'MSE for 1-time Validation')) +
  geom_vline(aes(xintercept = train_only, color = 'MSE for all data model')) + 
  labs(title = "Distribution of MSE using 10-fold Cross-validation Approach for 100 Times ",
        x = "MSE values",
        y = "Frequency") 
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

<img src="ps7-resample-nonlinear_files/figure-markdown_github/1.6-1.png" style="display: block; margin: auto;" />

Although the average MSE of 10-fold cross-validation for 100 times is a little bit larger than that for 1-time, the plot above suggests that 10-fold cross-validation approach is much more steady than the validation set approach in the section 1.3, since in cross-validation approach, the much narrower distributrion of MSE (distribution range:397 to 400)is found than that of validation set approach (disribution range: 330-450). Therefore, the 10-fold validation approach is less vulnerable to the process of data splitting than validation set approach.

7)Compare the estimated parameters and standard errors from the original model in step 1 (the model estimated using all of the available data) to parameters and standard errors estimated using the bootstrap (\(n = 1000\))

By using the bootstrap approach (1000 times), the co-effecient of the model is listed as below.

``` r
set.seed(1234)
# Boot-strap 
biden_boot <- biden %>%
  modelr::bootstrap(1000) %>%
  mutate(model = map(strap, ~lm(biden ~ age + female + educ + dem + rep, data =.)),
  coef = map(model, tidy))

biden_boot %>%
  unnest(coef) %>%
  group_by(term) %>%
  summarize(est.boot = mean(estimate),
            se.boot = sd(estimate, na.rm = TRUE))
```

    ## # A tibble: 6 × 3
    ##          term     est.boot    se.boot
    ##         <chr>        <dbl>      <dbl>
    ## 1 (Intercept)  58.96180746 2.94989029
    ## 2         age   0.04756082 0.02846997
    ## 3         dem  15.42942074 1.11104505
    ## 4        educ  -0.35119332 0.19201866
    ## 5      female   4.07552938 0.94880851
    ## 6         rep -15.88710208 1.43153427

Compared to the co-effecients of the model in 1.1 (model based on all the observations):

``` r
coef(summary(biden_all))
```

    ##                 Estimate Std. Error    t value     Pr(>|t|)
    ## (Intercept)  58.81125899  3.1244366  18.822996 2.694143e-72
    ## age           0.04825892  0.0282474   1.708438 8.772744e-02
    ## female        4.10323009  0.9482286   4.327258 1.592601e-05
    ## educ         -0.34533479  0.1947796  -1.772952 7.640571e-02
    ## dem          15.42425563  1.0680327  14.441745 8.144928e-45
    ## rep         -15.84950614  1.3113624 -12.086290 2.157309e-32

the standard errors for `age`, `female`, `educ`, and `rep` in the model built by the bootstrap approach are slightly larger than those in the model 1.1. This is because bootstrap approach does not rely on distributional assumptions, and thus can give more robust estimations.

Part 2: College (bivariate) \[3 points\]
========================================

Instructional expenditure per student as predictor:
---------------------------------------------------

I first choose the expenditure as the independnet variable and plot the relation between Out-of-state tuition with the expenditure of student below. It is obvious the relationship between them is not linear. Following Tukey and Mosteller's “bulging rule”, I use the Log(X) for power transformation. \[Outstate = \beta_0 + \beta_{1}log(Expend) \]

Below, I show the regression curve (red) and the residual plots.

``` r
# plot of the expend 
ggplot(college, aes(x=Expend, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  labs(title = "Scatter plot of Out-of-state tuition on Instructional expenditure per student",
        x = "Instructional expenditure per student",
        y = "Out-of-state tuition")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.1-1.png" style="display: block; margin: auto;" />

``` r
# set up the model with log(X).
log_exp <- lm(Outstate ~ log(Expend), data = college)

grid1 <-college %>%
  add_predictions(log_exp) %>%
  add_residuals(log_exp)

ggplot(college, aes(x=Expend, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_line(aes(y=pred), data = grid1, color = 'red', size = 1) +
  labs(title = "Regression of Out-of-state tuition on Instructional expenditure per student",
        x = "Instructional expenditure per student",
        y = "Out-of-state tuition")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.1-2.png" style="display: block; margin: auto;" />

``` r
ggplot(grid1, aes(x = pred, y = resid)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_hline(yintercept = 0, color = 'blue', size = 1, linetype = 'dashed') +
  labs(title = "Predicted Value and Residuals of linear regression (Outstate vs. log(Expend))",
        x = "Predicted Out-of-state tuition",
        y = "Residuals")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.1-3.png" style="display: block; margin: auto;" />

To validate this model, 10-fold validation for log(x) transformation is conducted.

``` r
set.seed(1234)
ex10_data <- crossv_kfold(college, k = 10)
ex_error_fold10 <- vector("numeric", 5)
terms <- 1:5

for(i in terms){
  ex10_models <- map(ex10_data$train, ~ lm(Outstate ~ poly(Expend, i), data = .))
  ex10_mse <- map2_dbl(ex10_models, ex10_data$test, calc_mse)
  ex_error_fold10[[i]] <- mean(ex10_mse)
}

exlog_10fold <- crossv_kfold(college, k = 10)
exlog_10models <- map(exlog_10fold$train, ~ lm(Outstate ~ log(Expend), data = .))

exlog_10mses <- map2_dbl(exlog_10models, exlog_10fold$test, calc_mse)
mse_exlog10 <- mean(exlog_10mses, na.rm = TRUE)

data_frame(terms = terms,
           fold10 = ex_error_fold10) %>%
  ggplot(aes(x=terms, y=fold10)) +
  geom_line() +
  theme_bw()+
  geom_hline(aes(yintercept = mse_exlog10, color = 'MSE for log transformation'), linetype = 'dashed') + 
  scale_colour_manual("", values = c("MSE for log transformation"="orange")) +
  labs(title = "MSE estimates",
       x = "Degree of Polynomial",
       y = "Mean Squared Error")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.1-2-1.png" style="display: block; margin: auto;" />

Form the above graph, the 10-fold MSE is actually lower for a third degree polynomial of Expend than it is for the log(Expend). However, sine MSE decreases only by -0.05, I decided to use this Y~log(X) model.

``` r
pander(summary(log_exp))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>log(Expend)</strong></td>
<td align="center">7482</td>
<td align="center">229.9</td>
<td align="center">32.54</td>
<td align="center">4.059e-147</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">-57502</td>
<td align="center">2090</td>
<td align="center">-27.51</td>
<td align="center">8.347e-117</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: Outstate ~ log(Expend)</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">777</td>
<td align="center">2617</td>
<td align="center">0.5774</td>
<td align="center">0.5769</td>
</tr>
</tbody>
</table>

The parameter and co-effecient of this model is listed in the table above. There is a statistically significant (p-value&lt;0.001), strong and positive relation between expenditure and tuition. The interpretation of the co-effecient on log(Expend):one percent increase in instructional expenditure per student is associated with a $74.82 increase in Out-of-state tuition.

Graduation rate as predictor:
-----------------------------

Then I choose the Graduation rate as the independnet variable and make the scatter plot between Out-of-state tuition with it as below. It appears there is a linear relationship between graduation rate and the Out-of-state tuition. However, based on the residual plot below, it seems there is correlation between the residuals and the predicted values.

``` r
# scatter plot
ggplot(college, aes(x=Grad.Rate, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  labs(title = "Scatter plot of Out-of-state tuition on Graduation Rate",
        x = "Graduation Rate",
        y = "Out-of-state tuition")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.2.1-1.png" style="display: block; margin: auto;" />

``` r
# set up the model with Grad.Rate.
g_rate <- lm(Outstate ~ Grad.Rate, data = college)

grid2 <-college %>%
  add_predictions(g_rate) %>%
  add_residuals(g_rate)

ggplot(college, aes(x=Grad.Rate, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_line(aes(y=pred), data = grid2, color = 'red', size = 1) +
  labs(title = "Regression of Out-of-state tuition on Graduation Rate",
        x = "Graduation Rate",
        y = "Out-of-state tuition")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.2.1-2.png" style="display: block; margin: auto;" />

``` r
ggplot(grid2, aes(x = pred, y = resid)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_hline(yintercept = 0, color = 'blue', size = 1, linetype = 'dashed') +
  labs(title = "Predicted Value and Residuals of linear regression (Outstate vs. Graduation Rate)",
        x = "Predicted Out of State Tuition",
        y = "Residuals")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.2.1-3.png" style="display: block; margin: auto;" />

``` r
pander(summary(g_rate))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>Grad.Rate</strong></td>
<td align="center">133.8</td>
<td align="center">6.905</td>
<td align="center">19.38</td>
<td align="center">1.629e-68</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">1682</td>
<td align="center">467.3</td>
<td align="center">3.599</td>
<td align="center">0.0003393</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: Outstate ~ Grad.Rate</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">777</td>
<td align="center">3304</td>
<td align="center">0.3264</td>
<td align="center">0.3255</td>
</tr>
</tbody>
</table>

In order to validate this model, 10-fold validation for such simple linear regression is conducted.

``` r
set.seed(1234)
ex10_data <- crossv_kfold(college, k = 10)
ex_error_fold10 <- vector("numeric", 5)
terms <- 1:5

for(i in terms){
  ex10_models <- map(ex10_data$train, ~ lm(Outstate ~ poly(Grad.Rate, i), data = .))
  ex10_mse <- map2_dbl(ex10_models, ex10_data$test, calc_mse)
  ex_error_fold10[[i]] <- mean(ex10_mse)
}

exlog_10fold <- crossv_kfold(college, k = 10)
exlog_10models <- map(exlog_10fold$train, ~ lm(Outstate ~ Grad.Rate, data = .))

exlog_10mses <- map2_dbl(exlog_10models, exlog_10fold$test, calc_mse)
mse_exlog10 <- mean(exlog_10mses, na.rm = TRUE)

data_frame(terms = terms,
           fold10 = ex_error_fold10) %>%
  ggplot(aes(x=terms, y=fold10)) +
  geom_line() +
  theme_bw()+
  geom_hline(aes(yintercept = mse_exlog10, color = 'MSE for 10-fold cross validation'), linetype = 'dashed') + 
  scale_colour_manual("", values = c("MSE for identity transformation"="orange")) +
  labs(title = "MSE estimates",
       x = "Degree of Polynomial",
       y = "Mean Squared Error")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.2.2-1.png" style="display: block; margin: auto;" />

Looking at this graph of MSE, we see that a 4th degree polynomialprovides the lowest MSE under 10-fold cross validation. Thus, I create a 4th degree polynomial linear model and report the result as below: by removing some data (graduation rate larger than 100), the curve of the new model fit the data very well, (the R square value increase from 0.3264 to 0.349).

``` r
college2<-college %>%
 filter(Grad.Rate <= 100)

grate4 <- lm(Outstate ~ poly(Grad.Rate, 4), data = college2)

grid3 <-college2 %>%
  add_predictions(grate4) %>%
  add_residuals(grate4)

ggplot(college2, aes(x=Grad.Rate, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_line(aes(y=pred), data = grid3, color = 'red', size = 1) +
  labs(title = "Regression of Out-of-state tuition on Graduation Rate",
        x = "Graduation Rate",
        y = "Out-of-state tuition")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.2.3-1.png" style="display: block; margin: auto;" />

``` r
ggplot(grid3, aes(x = pred, y = resid)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_hline(yintercept = 0, color = 'blue', size = 1, linetype = 'dashed') +
  labs(title = "Predicted Value and Residuals of polynominal regression (Outstate vs. Graduation Rate)",
        x = "Predicted Out of State Tuition",
        y = "Residuals")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.2.3-2.png" style="display: block; margin: auto;" />

``` r
pander(summary(grate4))
```

<table style="width:97%;">
<colgroup>
<col width="36%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>poly(Grad.Rate, 4)1</strong></td>
<td align="center">64530</td>
<td align="center">3252</td>
<td align="center">19.85</td>
<td align="center">4.003e-71</td>
</tr>
<tr class="even">
<td align="center"><strong>poly(Grad.Rate, 4)2</strong></td>
<td align="center">8080</td>
<td align="center">3252</td>
<td align="center">2.485</td>
<td align="center">0.01317</td>
</tr>
<tr class="odd">
<td align="center"><strong>poly(Grad.Rate, 4)3</strong></td>
<td align="center">-12375</td>
<td align="center">3252</td>
<td align="center">-3.806</td>
<td align="center">0.0001525</td>
</tr>
<tr class="even">
<td align="center"><strong>poly(Grad.Rate, 4)4</strong></td>
<td align="center">-4924</td>
<td align="center">3252</td>
<td align="center">-1.514</td>
<td align="center">0.1303</td>
</tr>
<tr class="odd">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">10442</td>
<td align="center">116.7</td>
<td align="center">89.46</td>
<td align="center">0</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: Outstate ~ poly(Grad.Rate, 4)</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">776</td>
<td align="center">3252</td>
<td align="center">0.3509</td>
<td align="center">0.3475</td>
</tr>
</tbody>
</table>

In the above summary table for the co-effecients and parameters for the polynominal model, we can find there are statistically significant association between graduation rates with the Out-of-state tuition.

Room and board costs as predictor:
----------------------------------

Finally, I choose the Room and board costs as the independnet variable and make the scatter plot between Out-of-state tuition with it as below. There is a linear relationship between graduation rate and the Out-of-state tuition. However, based on the residual plot below, the residuals appear to be correlated with the predicted values.

``` r
# scatter plot
ggplot(college, aes(x=Room.Board, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  labs(title = "Scatter plot of Out-of-state tuition on Room and Board Cost",
        x = "Room and Board Cost",
        y = "Out-of-state tuition")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.3.1-1.png" style="display: block; margin: auto;" />

``` r
# set up the model with Grad.Rate.
rb <- lm(Outstate ~ Room.Board, data = college)

grid4 <-college %>%
  add_predictions(rb) %>%
  add_residuals(rb)

ggplot(college, aes(x=Room.Board, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_line(aes(y=pred), data = grid4, color = 'red', size = 1) +
  labs(title = "Regression of Out-of-state tuition on Room and Board Cost",
        x = "Room and Board Cost",
        y = "Out-of-state tuition")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.3.1-2.png" style="display: block; margin: auto;" />

``` r
ggplot(grid4, aes(x = pred, y = resid)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_hline(yintercept = 0, color = 'blue', size = 1, linetype = 'dashed') +
  labs(title = "Predicted Value and Residuals of linear regression (Outstate vs. Room and Board Cost)",
        x = "Predicted Out of State Tuition",
        y = "Residuals")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.3.1-3.png" style="display: block; margin: auto;" />

``` r
pander(summary (rb))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>Room.Board</strong></td>
<td align="center">2.4</td>
<td align="center">0.09965</td>
<td align="center">24.08</td>
<td align="center">4.135e-96</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">-17.45</td>
<td align="center">447.8</td>
<td align="center">-0.03896</td>
<td align="center">0.9689</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: Outstate ~ Room.Board</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">777</td>
<td align="center">3044</td>
<td align="center">0.4281</td>
<td align="center">0.4273</td>
</tr>
</tbody>
</table>

To validate this model, 10-fold validation for simple linear regression is conducted.

``` r
set.seed(1234)
ex10_data <- crossv_kfold(college, k = 10)
ex_error_fold10 <- vector("numeric", 5)
terms <- 1:5

for(i in terms){
  ex10_models <- map(ex10_data$train, ~ lm(Outstate ~ poly(Room.Board, i), data = .))
  ex10_mse <- map2_dbl(ex10_models, ex10_data$test, calc_mse)
  ex_error_fold10[[i]] <- mean(ex10_mse)
}

exlog_10fold <- crossv_kfold(college, k = 10)
exlog_10models <- map(exlog_10fold$train, ~ lm(Outstate ~ Room.Board, data = .))

exlog_10mses <- map2_dbl(exlog_10models, exlog_10fold$test, calc_mse)
mse_exlog10 <- mean(exlog_10mses, na.rm = TRUE)

data_frame(terms = terms,
           fold10 = ex_error_fold10) %>%
  ggplot(aes(x=terms, y=fold10)) +
  geom_line() +
  theme_bw()+
  geom_hline(aes(yintercept = mse_exlog10, color = 'MSE for 10-fold cross validation'), linetype = 'dashed') + 
  scale_colour_manual("", values = c("MSE for identity transformation"="orange")) +
  labs(title = "MSE estimates",
       x = "Degree of Polynomial",
       y = "Mean Squared Error")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.3.2-1.png" style="display: block; margin: auto;" />

Looking at this graph of MSE, we see that the lowest MSE could be observed at the degree of polynomial = 2. Therefore, I made a polinominal model.

``` r
rb2 <- lm(Outstate ~ poly(Room.Board, 2), data = college)

grid5 <-college %>%
  add_predictions(rb2) %>%
  add_residuals(rb2)

ggplot(college, aes(x=Room.Board, y=Outstate)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_line(aes(y=pred), data = grid5, color = 'red', size = 1) +
  labs(title = "Regression of Out-of-state tuition on Graduation Rate",
        x = "Graduation Rate",
        y = "Room and Board Cost")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.3.3-1.png" style="display: block; margin: auto;" />

``` r
ggplot(grid5, aes(x = pred, y = resid)) +
  geom_point(alpha=0.2) +
  theme_bw()+
  geom_hline(yintercept = 0, color = 'blue', size = 1, linetype = 'dashed') +
  labs(title = "Predicted Value and Residuals of polynominal regression (Outstate vs. Room and Board Cost)",
        x = "Predicted Out of State Tuition",
        y = "Residuals")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/2.3.3-2.png" style="display: block; margin: auto;" />

``` r
pander(summary(rb2))
```

<table style="width:99%;">
<colgroup>
<col width="37%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>poly(Room.Board, 2)1</strong></td>
<td align="center">73321</td>
<td align="center">3037</td>
<td align="center">24.14</td>
<td align="center">2.031e-96</td>
</tr>
<tr class="even">
<td align="center"><strong>poly(Room.Board, 2)2</strong></td>
<td align="center">-6535</td>
<td align="center">3037</td>
<td align="center">-2.151</td>
<td align="center">0.03175</td>
</tr>
<tr class="odd">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">10441</td>
<td align="center">109</td>
<td align="center">95.82</td>
<td align="center">0</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: Outstate ~ poly(Room.Board, 2)</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">777</td>
<td align="center">3037</td>
<td align="center">0.4315</td>
<td align="center">0.43</td>
</tr>
</tbody>
</table>

In the above summary table for the co-effecients and parameters for the polynominal model, we can find there are statistically significant association between board and room cost with the Out-of-state tuition. In addition, compared with the simple linear model, the R square value of this new model is enhanced from 0.4281 to 0.4315.

To sum up, the three predictors I chose, instructional expenditure per student,graduation rate, and room and board costs, all have statistically significant relationship with out-of-state tuition. Through 10-fold cross-validation approach, I confirm three bivariate models by using tuition as the dependent variable. The relation between it with graduation rate and room/board costs can be explained in two polinominal models. Because interpretation is more tractable for a regression on log(Expend), I choose the model of Tuition~log(Expenditure) rather than polynominal model.

Part 3: College (GAM) \[3 points\]
==================================

1)Split the data into a training set and a test set:

``` r
set.seed(1234)
split <- resample_partition(college, c(test = 0.3, train = 0.7))
```

2)Estimate an OLS model on the training data:

``` r
college_train <- lm(Outstate ~ Private + Room.Board + PhD + perc.alumni + Expend + Grad.Rate, 
                       data = split$train)

pander(summary(college_train))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>PrivateYes</strong></td>
<td align="center">2575</td>
<td align="center">253.9</td>
<td align="center">10.14</td>
<td align="center">3.101e-22</td>
</tr>
<tr class="even">
<td align="center"><strong>Room.Board</strong></td>
<td align="center">0.9927</td>
<td align="center">0.1028</td>
<td align="center">9.661</td>
<td align="center">1.826e-20</td>
</tr>
<tr class="odd">
<td align="center"><strong>PhD</strong></td>
<td align="center">36.53</td>
<td align="center">6.801</td>
<td align="center">5.371</td>
<td align="center">1.166e-07</td>
</tr>
<tr class="even">
<td align="center"><strong>perc.alumni</strong></td>
<td align="center">53.39</td>
<td align="center">9.048</td>
<td align="center">5.9</td>
<td align="center">6.435e-09</td>
</tr>
<tr class="odd">
<td align="center"><strong>Expend</strong></td>
<td align="center">0.2067</td>
<td align="center">0.0207</td>
<td align="center">9.987</td>
<td align="center">1.175e-21</td>
</tr>
<tr class="even">
<td align="center"><strong>Grad.Rate</strong></td>
<td align="center">30.73</td>
<td align="center">6.696</td>
<td align="center">4.589</td>
<td align="center">5.551e-06</td>
</tr>
<tr class="odd">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">-3596</td>
<td align="center">538.4</td>
<td align="center">-6.678</td>
<td align="center">6.043e-11</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: Outstate ~ Private + Room.Board + PhD + perc.alumni + Expend + Grad.Rate</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">544</td>
<td align="center">2090</td>
<td align="center">0.7263</td>
<td align="center">0.7232</td>
</tr>
</tbody>
</table>

As the table shown above, this model's R-square is 0.7263, implying it could explain about 72.63% of the variance in the training data. Based on the p-value reported in the table above, all the six independent variabls are statistically significant. I will interpret the co-effecients on each independent variable as below: a)PrivateYes: Holding the other variables constant, private university will have averagely 2575 dollars higher in its tuition than other non-private university. b)Roo,.Board: Holding the other variables constant, with room-board costs increasing by 1 dollar, the out-of-state tuition would increase 0.9927 dollar. c)PhD percentage: Holding the other variables constant, as the portion of facaulty with Ph.D. increase by 1 percent,averagely the tuition would get higher by 36.53 dollars. d)percentage of alumni: Holding the other variables constant, as the portion of alumni who donates increase by 1 percent, the tuition would be 53.39 dollars higher. e)Expend: Holding the other variables constant, as the expenditure per student increase by 1 dollar, the tuition will increase by 0.2067 dollars, on average. f)graduation rate: Holding the other variables constant, the tuition would be 30.73 dollars more if the graduation rate increases by 1 unit.

In the following graph, I make the residual plot for this linear model. Except the range with larger predicted values,it appears that the predicted value is not in correlation with the residuals.

``` r
college[unlist(split$train["idx"], use.names = FALSE),] %>%
  add_predictions(college_train) %>%
  add_residuals(college_train) %>%
  ggplot(aes(pred, resid)) +
  geom_point()+
  theme_bw()+
  labs(title = "Predicted Value and Residuals of linear regression",
        x = "Predicted Value",
        y = "Residuals")
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/3.2-2-1.png" style="display: block; margin: auto;" />

3)GAM model: I will use a GAM model that regresses Outstate on the binary predictor Private, a 2nd degree polynomial of Room.Board (from in part 2), local regressions for PhD and perc.alumni, the log(Expend) (from part 2), and a fourth degree polynomial for Grad.Rate (from part 2).

``` r
gam <- gam(Outstate ~ Private + poly(Room.Board, 2) + lo(PhD) + lo(perc.alumni) + log(Expend) + poly(Grad.Rate, 4),data = split$train)
pander(summary(gam))
```

-   **call**: `gam(formula = Outstate ~ Private + poly(Room.Board, 2) + lo(PhD) +      lo(perc.alumni) + log(Expend) + poly(Grad.Rate, 4), data = split$train)`
-   **terms**: Outstate ~ Private + poly(Room.Board, 2) + lo(PhD) + lo(perc.alumni) + log(Expend) + poly(Grad.Rate, 4)
-   **anova**:

    <table style="width:75%;">
    <caption>Anova for Nonparametric Effects</caption>
    <colgroup>
    <col width="36%" />
    <col width="13%" />
    <col width="12%" />
    <col width="12%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center"> </th>
    <th align="center">Npar Df</th>
    <th align="center">Npar F</th>
    <th align="center">Pr(F)</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center"><strong>(Intercept)</strong></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    </tr>
    <tr class="even">
    <td align="center"><strong>Private</strong></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    </tr>
    <tr class="odd">
    <td align="center"><strong>poly(Room.Board, 2)</strong></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    </tr>
    <tr class="even">
    <td align="center"><strong>lo(PhD)</strong></td>
    <td align="center">2.7</td>
    <td align="center">3.839</td>
    <td align="center">0.01249</td>
    </tr>
    <tr class="odd">
    <td align="center"><strong>lo(perc.alumni)</strong></td>
    <td align="center">2.4</td>
    <td align="center">0.9547</td>
    <td align="center">0.3998</td>
    </tr>
    <tr class="even">
    <td align="center"><strong>log(Expend)</strong></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    </tr>
    <tr class="odd">
    <td align="center"><strong>poly(Grad.Rate, 4)</strong></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    </tr>
    </tbody>
    </table>

-   **parametric.anova**:

    <table style="width:100%;">
    <caption>Anova for Parametric Effects</caption>
    <colgroup>
    <col width="36%" />
    <col width="8%" />
    <col width="13%" />
    <col width="13%" />
    <col width="13%" />
    <col width="13%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center"> </th>
    <th align="center">Df</th>
    <th align="center">Sum Sq</th>
    <th align="center">Mean Sq</th>
    <th align="center">F value</th>
    <th align="center">Pr(&gt;F)</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center"><strong>Private</strong></td>
    <td align="center">1</td>
    <td align="center">2.322e+09</td>
    <td align="center">2.322e+09</td>
    <td align="center">613.9</td>
    <td align="center">1.768e-90</td>
    </tr>
    <tr class="even">
    <td align="center"><strong>poly(Room.Board, 2)</strong></td>
    <td align="center">2</td>
    <td align="center">1.967e+09</td>
    <td align="center">983748292</td>
    <td align="center">260.1</td>
    <td align="center">2.409e-79</td>
    </tr>
    <tr class="odd">
    <td align="center"><strong>lo(PhD)</strong></td>
    <td align="center">1</td>
    <td align="center">833263189</td>
    <td align="center">833263189</td>
    <td align="center">220.3</td>
    <td align="center">6.638e-42</td>
    </tr>
    <tr class="even">
    <td align="center"><strong>lo(perc.alumni)</strong></td>
    <td align="center">1</td>
    <td align="center">422126959</td>
    <td align="center">422126959</td>
    <td align="center">111.6</td>
    <td align="center">8.503e-24</td>
    </tr>
    <tr class="odd">
    <td align="center"><strong>log(Expend)</strong></td>
    <td align="center">1</td>
    <td align="center">650773989</td>
    <td align="center">650773989</td>
    <td align="center">172.1</td>
    <td align="center">3.178e-34</td>
    </tr>
    <tr class="even">
    <td align="center"><strong>poly(Grad.Rate, 4)</strong></td>
    <td align="center">4</td>
    <td align="center">119747173</td>
    <td align="center">29936793</td>
    <td align="center">7.915</td>
    <td align="center">3.353e-06</td>
    </tr>
    <tr class="odd">
    <td align="center"><strong>Residuals</strong></td>
    <td align="center">527.9</td>
    <td align="center">1.996e+09</td>
    <td align="center">3782085</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    </tr>
    </tbody>
    </table>

-   **dispersion**:

    <table style="width:14%;">
    <colgroup>
    <col width="13%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">gaussian</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">3782085</td>
    </tr>
    </tbody>
    </table>

-   **df**: *16.13* and *527.9*
-   **deviance.resid**:

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="6%" />
    <col width="7%" />
    <col width="7%" />
    <col width="6%" />
    <col width="6%" />
    <col width="7%" />
    <col width="6%" />
    <col width="6%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">2</th>
    <th align="center">3</th>
    <th align="center">4</th>
    <th align="center">5</th>
    <th align="center">6</th>
    <th align="center">9</th>
    <th align="center">10</th>
    <th align="center">11</th>
    <th align="center">12</th>
    <th align="center">14</th>
    <th align="center">16</th>
    <th align="center">21</th>
    <th align="center">26</th>
    <th align="center">27</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-289.4</td>
    <td align="center">1371</td>
    <td align="center">-2771</td>
    <td align="center">-1708</td>
    <td align="center">4270</td>
    <td align="center">2500</td>
    <td align="center">937.4</td>
    <td align="center">2883</td>
    <td align="center">3793</td>
    <td align="center">561.9</td>
    <td align="center">-2223</td>
    <td align="center">-255.6</td>
    <td align="center">633.3</td>
    <td align="center">-1287</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="6%" />
    <col width="6%" />
    <col width="9%" />
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="9%" />
    <col width="6%" />
    <col width="6%" />
    <col width="6%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">28</th>
    <th align="center">29</th>
    <th align="center">31</th>
    <th align="center">34</th>
    <th align="center">36</th>
    <th align="center">39</th>
    <th align="center">40</th>
    <th align="center">41</th>
    <th align="center">42</th>
    <th align="center">44</th>
    <th align="center">45</th>
    <th align="center">46</th>
    <th align="center">47</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-1655</td>
    <td align="center">1670</td>
    <td align="center">1748</td>
    <td align="center">-381.3</td>
    <td align="center">-1783</td>
    <td align="center">-451.2</td>
    <td align="center">-4310</td>
    <td align="center">159.6</td>
    <td align="center">-892.1</td>
    <td align="center">-958.3</td>
    <td align="center">3530</td>
    <td align="center">1679</td>
    <td align="center">1268</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="6%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="6%" />
    <col width="8%" />
    <col width="6%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">48</th>
    <th align="center">50</th>
    <th align="center">53</th>
    <th align="center">54</th>
    <th align="center">56</th>
    <th align="center">57</th>
    <th align="center">58</th>
    <th align="center">60</th>
    <th align="center">61</th>
    <th align="center">63</th>
    <th align="center">66</th>
    <th align="center">68</th>
    <th align="center">70</th>
    <th align="center">72</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">8383</td>
    <td align="center">-3264</td>
    <td align="center">-1314</td>
    <td align="center">-2717</td>
    <td align="center">-3786</td>
    <td align="center">1006</td>
    <td align="center">-484.2</td>
    <td align="center">2852</td>
    <td align="center">684.8</td>
    <td align="center">631.9</td>
    <td align="center">183.4</td>
    <td align="center">93.14</td>
    <td align="center">-7378</td>
    <td align="center">-212</td>
    </tr>
    </tbody>
    </table>

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="6%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="6%" />
    <col width="6%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">74</th>
    <th align="center">76</th>
    <th align="center">77</th>
    <th align="center">79</th>
    <th align="center">80</th>
    <th align="center">81</th>
    <th align="center">82</th>
    <th align="center">84</th>
    <th align="center">86</th>
    <th align="center">87</th>
    <th align="center">89</th>
    <th align="center">90</th>
    <th align="center">92</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">4442</td>
    <td align="center">-1358</td>
    <td align="center">-2189</td>
    <td align="center">-971.1</td>
    <td align="center">-109.3</td>
    <td align="center">-1139</td>
    <td align="center">918.6</td>
    <td align="center">-1452</td>
    <td align="center">353.4</td>
    <td align="center">2585</td>
    <td align="center">1322</td>
    <td align="center">-220.5</td>
    <td align="center">93.61</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="6%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">96</th>
    <th align="center">100</th>
    <th align="center">102</th>
    <th align="center">105</th>
    <th align="center">109</th>
    <th align="center">111</th>
    <th align="center">113</th>
    <th align="center">116</th>
    <th align="center">117</th>
    <th align="center">120</th>
    <th align="center">121</th>
    <th align="center">122</th>
    <th align="center">123</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">1794</td>
    <td align="center">-2376</td>
    <td align="center">290.3</td>
    <td align="center">2484</td>
    <td align="center">-1290</td>
    <td align="center">-2690</td>
    <td align="center">1741</td>
    <td align="center">3593</td>
    <td align="center">300.7</td>
    <td align="center">3184</td>
    <td align="center">2025</td>
    <td align="center">-1937</td>
    <td align="center">2771</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">124</th>
    <th align="center">125</th>
    <th align="center">126</th>
    <th align="center">128</th>
    <th align="center">129</th>
    <th align="center">131</th>
    <th align="center">132</th>
    <th align="center">134</th>
    <th align="center">135</th>
    <th align="center">136</th>
    <th align="center">137</th>
    <th align="center">138</th>
    <th align="center">139</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2872</td>
    <td align="center">-1582</td>
    <td align="center">1145</td>
    <td align="center">-1986</td>
    <td align="center">1807</td>
    <td align="center">939.7</td>
    <td align="center">1343</td>
    <td align="center">-509.9</td>
    <td align="center">358.6</td>
    <td align="center">-2272</td>
    <td align="center">1058</td>
    <td align="center">1121</td>
    <td align="center">837.3</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="9%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">140</th>
    <th align="center">141</th>
    <th align="center">142</th>
    <th align="center">144</th>
    <th align="center">146</th>
    <th align="center">147</th>
    <th align="center">148</th>
    <th align="center">149</th>
    <th align="center">150</th>
    <th align="center">151</th>
    <th align="center">154</th>
    <th align="center">156</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">1781</td>
    <td align="center">1703</td>
    <td align="center">515.6</td>
    <td align="center">-711.8</td>
    <td align="center">1933</td>
    <td align="center">-954</td>
    <td align="center">1120</td>
    <td align="center">-630.2</td>
    <td align="center">2340</td>
    <td align="center">363.4</td>
    <td align="center">-168.5</td>
    <td align="center">-908.5</td>
    </tr>
    </tbody>
    </table>

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">157</th>
    <th align="center">158</th>
    <th align="center">160</th>
    <th align="center">161</th>
    <th align="center">162</th>
    <th align="center">163</th>
    <th align="center">166</th>
    <th align="center">167</th>
    <th align="center">169</th>
    <th align="center">170</th>
    <th align="center">171</th>
    <th align="center">172</th>
    <th align="center">173</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-469.9</td>
    <td align="center">1579</td>
    <td align="center">505.9</td>
    <td align="center">1934</td>
    <td align="center">997.3</td>
    <td align="center">2134</td>
    <td align="center">1091</td>
    <td align="center">-2074</td>
    <td align="center">-270.9</td>
    <td align="center">763.1</td>
    <td align="center">-250.6</td>
    <td align="center">506</td>
    <td align="center">2635</td>
    </tr>
    </tbody>
    </table>

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="9%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">174</th>
    <th align="center">176</th>
    <th align="center">177</th>
    <th align="center">178</th>
    <th align="center">181</th>
    <th align="center">182</th>
    <th align="center">183</th>
    <th align="center">185</th>
    <th align="center">187</th>
    <th align="center">188</th>
    <th align="center">190</th>
    <th align="center">191</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-2580</td>
    <td align="center">188</td>
    <td align="center">-230.5</td>
    <td align="center">-697.3</td>
    <td align="center">-168.7</td>
    <td align="center">563.8</td>
    <td align="center">-2329</td>
    <td align="center">2275</td>
    <td align="center">4046</td>
    <td align="center">-352.1</td>
    <td align="center">-2510</td>
    <td align="center">-4424</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">192</th>
    <th align="center">194</th>
    <th align="center">195</th>
    <th align="center">196</th>
    <th align="center">197</th>
    <th align="center">198</th>
    <th align="center">199</th>
    <th align="center">200</th>
    <th align="center">201</th>
    <th align="center">202</th>
    <th align="center">203</th>
    <th align="center">204</th>
    <th align="center">205</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-1052</td>
    <td align="center">-580.4</td>
    <td align="center">1035</td>
    <td align="center">-2397</td>
    <td align="center">102.8</td>
    <td align="center">2114</td>
    <td align="center">970.8</td>
    <td align="center">-868.4</td>
    <td align="center">3588</td>
    <td align="center">230.9</td>
    <td align="center">-1330</td>
    <td align="center">-847.5</td>
    <td align="center">188.8</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">206</th>
    <th align="center">210</th>
    <th align="center">211</th>
    <th align="center">213</th>
    <th align="center">214</th>
    <th align="center">215</th>
    <th align="center">216</th>
    <th align="center">217</th>
    <th align="center">218</th>
    <th align="center">219</th>
    <th align="center">220</th>
    <th align="center">221</th>
    <th align="center">223</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-234</td>
    <td align="center">-1305</td>
    <td align="center">4396</td>
    <td align="center">1018</td>
    <td align="center">-521</td>
    <td align="center">836.5</td>
    <td align="center">614.3</td>
    <td align="center">-1307</td>
    <td align="center">3355</td>
    <td align="center">3346</td>
    <td align="center">2236</td>
    <td align="center">-1694</td>
    <td align="center">-4681</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">226</th>
    <th align="center">227</th>
    <th align="center">228</th>
    <th align="center">229</th>
    <th align="center">230</th>
    <th align="center">231</th>
    <th align="center">232</th>
    <th align="center">234</th>
    <th align="center">235</th>
    <th align="center">237</th>
    <th align="center">239</th>
    <th align="center">240</th>
    <th align="center">243</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">5113</td>
    <td align="center">-124.3</td>
    <td align="center">973.1</td>
    <td align="center">984.3</td>
    <td align="center">-2500</td>
    <td align="center">-264.8</td>
    <td align="center">-1635</td>
    <td align="center">1254</td>
    <td align="center">2708</td>
    <td align="center">261.2</td>
    <td align="center">-2484</td>
    <td align="center">1214</td>
    <td align="center">1741</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">244</th>
    <th align="center">245</th>
    <th align="center">246</th>
    <th align="center">247</th>
    <th align="center">248</th>
    <th align="center">249</th>
    <th align="center">250</th>
    <th align="center">251</th>
    <th align="center">252</th>
    <th align="center">253</th>
    <th align="center">254</th>
    <th align="center">255</th>
    <th align="center">256</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">847.8</td>
    <td align="center">-1963</td>
    <td align="center">-1236</td>
    <td align="center">-2801</td>
    <td align="center">-1154</td>
    <td align="center">-4324</td>
    <td align="center">4366</td>
    <td align="center">-2389</td>
    <td align="center">-1275</td>
    <td align="center">1138</td>
    <td align="center">-1508</td>
    <td align="center">-2769</td>
    <td align="center">613.6</td>
    </tr>
    </tbody>
    </table>

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="9%" />
    <col width="9%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">259</th>
    <th align="center">260</th>
    <th align="center">261</th>
    <th align="center">263</th>
    <th align="center">264</th>
    <th align="center">266</th>
    <th align="center">267</th>
    <th align="center">269</th>
    <th align="center">270</th>
    <th align="center">271</th>
    <th align="center">272</th>
    <th align="center">273</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-1740</td>
    <td align="center">-659.5</td>
    <td align="center">226.3</td>
    <td align="center">-243.2</td>
    <td align="center">-522.8</td>
    <td align="center">-635.3</td>
    <td align="center">-1328</td>
    <td align="center">1823</td>
    <td align="center">2123</td>
    <td align="center">1986</td>
    <td align="center">-1196</td>
    <td align="center">-1599</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">274</th>
    <th align="center">276</th>
    <th align="center">277</th>
    <th align="center">278</th>
    <th align="center">283</th>
    <th align="center">284</th>
    <th align="center">285</th>
    <th align="center">286</th>
    <th align="center">287</th>
    <th align="center">289</th>
    <th align="center">290</th>
    <th align="center">291</th>
    <th align="center">293</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">1156</td>
    <td align="center">1228</td>
    <td align="center">-1130</td>
    <td align="center">-448.6</td>
    <td align="center">-3893</td>
    <td align="center">-1489</td>
    <td align="center">-3282</td>
    <td align="center">499.5</td>
    <td align="center">-1474</td>
    <td align="center">1022</td>
    <td align="center">70.06</td>
    <td align="center">1177</td>
    <td align="center">4404</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">295</th>
    <th align="center">296</th>
    <th align="center">297</th>
    <th align="center">299</th>
    <th align="center">302</th>
    <th align="center">303</th>
    <th align="center">304</th>
    <th align="center">305</th>
    <th align="center">308</th>
    <th align="center">309</th>
    <th align="center">310</th>
    <th align="center">311</th>
    <th align="center">312</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-1273</td>
    <td align="center">-1371</td>
    <td align="center">1872</td>
    <td align="center">-251.9</td>
    <td align="center">2800</td>
    <td align="center">2131</td>
    <td align="center">277.4</td>
    <td align="center">-1967</td>
    <td align="center">-1749</td>
    <td align="center">1242</td>
    <td align="center">2277</td>
    <td align="center">-1075</td>
    <td align="center">563.8</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">313</th>
    <th align="center">314</th>
    <th align="center">315</th>
    <th align="center">316</th>
    <th align="center">317</th>
    <th align="center">320</th>
    <th align="center">321</th>
    <th align="center">322</th>
    <th align="center">324</th>
    <th align="center">326</th>
    <th align="center">329</th>
    <th align="center">330</th>
    <th align="center">331</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-402.2</td>
    <td align="center">1829</td>
    <td align="center">531.8</td>
    <td align="center">-1315</td>
    <td align="center">-2389</td>
    <td align="center">-6250</td>
    <td align="center">982.1</td>
    <td align="center">1245</td>
    <td align="center">-2886</td>
    <td align="center">962.8</td>
    <td align="center">-583.9</td>
    <td align="center">-2303</td>
    <td align="center">1836</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">332</th>
    <th align="center">333</th>
    <th align="center">336</th>
    <th align="center">337</th>
    <th align="center">338</th>
    <th align="center">339</th>
    <th align="center">340</th>
    <th align="center">341</th>
    <th align="center">343</th>
    <th align="center">344</th>
    <th align="center">346</th>
    <th align="center">348</th>
    <th align="center">349</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2647</td>
    <td align="center">617</td>
    <td align="center">-1344</td>
    <td align="center">2355</td>
    <td align="center">706.5</td>
    <td align="center">-2410</td>
    <td align="center">-110.4</td>
    <td align="center">-533</td>
    <td align="center">1359</td>
    <td align="center">-1833</td>
    <td align="center">-3140</td>
    <td align="center">-2055</td>
    <td align="center">-2392</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="9%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">351</th>
    <th align="center">352</th>
    <th align="center">353</th>
    <th align="center">354</th>
    <th align="center">355</th>
    <th align="center">356</th>
    <th align="center">357</th>
    <th align="center">358</th>
    <th align="center">360</th>
    <th align="center">361</th>
    <th align="center">362</th>
    <th align="center">363</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">254.6</td>
    <td align="center">110.3</td>
    <td align="center">549.6</td>
    <td align="center">-197.4</td>
    <td align="center">536.5</td>
    <td align="center">-397.8</td>
    <td align="center">-1035</td>
    <td align="center">-209.7</td>
    <td align="center">-272.1</td>
    <td align="center">-1005</td>
    <td align="center">-3530</td>
    <td align="center">-486.4</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">364</th>
    <th align="center">365</th>
    <th align="center">366</th>
    <th align="center">367</th>
    <th align="center">369</th>
    <th align="center">370</th>
    <th align="center">371</th>
    <th align="center">373</th>
    <th align="center">374</th>
    <th align="center">376</th>
    <th align="center">377</th>
    <th align="center">379</th>
    <th align="center">380</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">780.5</td>
    <td align="center">-1763</td>
    <td align="center">-332</td>
    <td align="center">1311</td>
    <td align="center">-1433</td>
    <td align="center">-271.4</td>
    <td align="center">-937.5</td>
    <td align="center">-1205</td>
    <td align="center">1849</td>
    <td align="center">3934</td>
    <td align="center">420.9</td>
    <td align="center">557.5</td>
    <td align="center">285.8</td>
    </tr>
    </tbody>
    </table>

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="8%" />
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="9%" />
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">381</th>
    <th align="center">382</th>
    <th align="center">386</th>
    <th align="center">387</th>
    <th align="center">388</th>
    <th align="center">389</th>
    <th align="center">390</th>
    <th align="center">391</th>
    <th align="center">392</th>
    <th align="center">393</th>
    <th align="center">394</th>
    <th align="center">395</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">1335</td>
    <td align="center">-1129</td>
    <td align="center">-264.3</td>
    <td align="center">2177</td>
    <td align="center">-4555</td>
    <td align="center">758.8</td>
    <td align="center">-3657</td>
    <td align="center">2111</td>
    <td align="center">-487.6</td>
    <td align="center">-535.8</td>
    <td align="center">1081</td>
    <td align="center">698.4</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="9%" />
    <col width="9%" />
    <col width="7%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">396</th>
    <th align="center">398</th>
    <th align="center">399</th>
    <th align="center">400</th>
    <th align="center">401</th>
    <th align="center">403</th>
    <th align="center">404</th>
    <th align="center">405</th>
    <th align="center">407</th>
    <th align="center">408</th>
    <th align="center">409</th>
    <th align="center">410</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-602.4</td>
    <td align="center">-142.3</td>
    <td align="center">1144</td>
    <td align="center">-172.5</td>
    <td align="center">3557</td>
    <td align="center">1842</td>
    <td align="center">-292.5</td>
    <td align="center">-823.6</td>
    <td align="center">-2216</td>
    <td align="center">809.9</td>
    <td align="center">1756</td>
    <td align="center">-297.4</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">411</th>
    <th align="center">412</th>
    <th align="center">413</th>
    <th align="center">415</th>
    <th align="center">417</th>
    <th align="center">418</th>
    <th align="center">420</th>
    <th align="center">421</th>
    <th align="center">422</th>
    <th align="center">425</th>
    <th align="center">426</th>
    <th align="center">427</th>
    <th align="center">428</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-1347</td>
    <td align="center">2867</td>
    <td align="center">-2958</td>
    <td align="center">-2107</td>
    <td align="center">140.5</td>
    <td align="center">-2013</td>
    <td align="center">1299</td>
    <td align="center">2330</td>
    <td align="center">-1307</td>
    <td align="center">-1648</td>
    <td align="center">2419</td>
    <td align="center">1458</td>
    <td align="center">1936</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">429</th>
    <th align="center">430</th>
    <th align="center">431</th>
    <th align="center">432</th>
    <th align="center">434</th>
    <th align="center">436</th>
    <th align="center">438</th>
    <th align="center">439</th>
    <th align="center">441</th>
    <th align="center">442</th>
    <th align="center">443</th>
    <th align="center">445</th>
    <th align="center">446</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">3281</td>
    <td align="center">970.5</td>
    <td align="center">1002</td>
    <td align="center">3721</td>
    <td align="center">2003</td>
    <td align="center">-493.2</td>
    <td align="center">1025</td>
    <td align="center">-818.3</td>
    <td align="center">-1023</td>
    <td align="center">901.1</td>
    <td align="center">1751</td>
    <td align="center">1746</td>
    <td align="center">2035</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">447</th>
    <th align="center">451</th>
    <th align="center">452</th>
    <th align="center">453</th>
    <th align="center">454</th>
    <th align="center">455</th>
    <th align="center">456</th>
    <th align="center">459</th>
    <th align="center">460</th>
    <th align="center">461</th>
    <th align="center">462</th>
    <th align="center">463</th>
    <th align="center">470</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2693</td>
    <td align="center">1127</td>
    <td align="center">-3589</td>
    <td align="center">-1251</td>
    <td align="center">2676</td>
    <td align="center">398.7</td>
    <td align="center">-462.5</td>
    <td align="center">-103.5</td>
    <td align="center">350.8</td>
    <td align="center">1466</td>
    <td align="center">765.5</td>
    <td align="center">-739.9</td>
    <td align="center">4547</td>
    </tr>
    </tbody>
    </table>

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">472</th>
    <th align="center">473</th>
    <th align="center">474</th>
    <th align="center">475</th>
    <th align="center">476</th>
    <th align="center">478</th>
    <th align="center">479</th>
    <th align="center">480</th>
    <th align="center">482</th>
    <th align="center">483</th>
    <th align="center">488</th>
    <th align="center">490</th>
    <th align="center">492</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2019</td>
    <td align="center">63.89</td>
    <td align="center">-293.6</td>
    <td align="center">1437</td>
    <td align="center">-774</td>
    <td align="center">-956.9</td>
    <td align="center">-144.4</td>
    <td align="center">1679</td>
    <td align="center">-1656</td>
    <td align="center">-2699</td>
    <td align="center">732.7</td>
    <td align="center">597.9</td>
    <td align="center">20.15</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="9%" />
    <col width="9%" />
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">493</th>
    <th align="center">494</th>
    <th align="center">495</th>
    <th align="center">496</th>
    <th align="center">497</th>
    <th align="center">499</th>
    <th align="center">500</th>
    <th align="center">501</th>
    <th align="center">502</th>
    <th align="center">503</th>
    <th align="center">504</th>
    <th align="center">505</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">219.6</td>
    <td align="center">1344</td>
    <td align="center">1365</td>
    <td align="center">-264.8</td>
    <td align="center">-317.7</td>
    <td align="center">-241.6</td>
    <td align="center">2237</td>
    <td align="center">-1783</td>
    <td align="center">-1047</td>
    <td align="center">1106</td>
    <td align="center">-1651</td>
    <td align="center">-480</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">506</th>
    <th align="center">507</th>
    <th align="center">508</th>
    <th align="center">509</th>
    <th align="center">510</th>
    <th align="center">512</th>
    <th align="center">513</th>
    <th align="center">514</th>
    <th align="center">515</th>
    <th align="center">516</th>
    <th align="center">519</th>
    <th align="center">520</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-277.6</td>
    <td align="center">38.38</td>
    <td align="center">-2699</td>
    <td align="center">-1354</td>
    <td align="center">-2129</td>
    <td align="center">-429.1</td>
    <td align="center">3375</td>
    <td align="center">-1423</td>
    <td align="center">-2296</td>
    <td align="center">-447.9</td>
    <td align="center">-1020</td>
    <td align="center">-770.2</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">521</th>
    <th align="center">522</th>
    <th align="center">524</th>
    <th align="center">526</th>
    <th align="center">527</th>
    <th align="center">528</th>
    <th align="center">529</th>
    <th align="center">532</th>
    <th align="center">533</th>
    <th align="center">534</th>
    <th align="center">536</th>
    <th align="center">537</th>
    <th align="center">538</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">589.9</td>
    <td align="center">-1634</td>
    <td align="center">-777.6</td>
    <td align="center">13.83</td>
    <td align="center">1156</td>
    <td align="center">2967</td>
    <td align="center">836.2</td>
    <td align="center">737</td>
    <td align="center">637.5</td>
    <td align="center">-627.5</td>
    <td align="center">680.2</td>
    <td align="center">1115</td>
    <td align="center">-1281</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">540</th>
    <th align="center">541</th>
    <th align="center">543</th>
    <th align="center">547</th>
    <th align="center">548</th>
    <th align="center">549</th>
    <th align="center">550</th>
    <th align="center">551</th>
    <th align="center">552</th>
    <th align="center">553</th>
    <th align="center">554</th>
    <th align="center">556</th>
    <th align="center">557</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-1213</td>
    <td align="center">-1784</td>
    <td align="center">-5232</td>
    <td align="center">-1786</td>
    <td align="center">2607</td>
    <td align="center">2027</td>
    <td align="center">-490.6</td>
    <td align="center">-3403</td>
    <td align="center">-1182</td>
    <td align="center">-150.2</td>
    <td align="center">-3627</td>
    <td align="center">2313</td>
    <td align="center">-456.5</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="9%" />
    <col width="9%" />
    <col width="9%" />
    <col width="9%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">558</th>
    <th align="center">559</th>
    <th align="center">560</th>
    <th align="center">562</th>
    <th align="center">563</th>
    <th align="center">564</th>
    <th align="center">565</th>
    <th align="center">566</th>
    <th align="center">567</th>
    <th align="center">569</th>
    <th align="center">570</th>
    <th align="center">571</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">1845</td>
    <td align="center">-3314</td>
    <td align="center">-736</td>
    <td align="center">-2990</td>
    <td align="center">-3527</td>
    <td align="center">-3772</td>
    <td align="center">-560.8</td>
    <td align="center">-445.9</td>
    <td align="center">-61.54</td>
    <td align="center">-913.4</td>
    <td align="center">-1400</td>
    <td align="center">-290.7</td>
    </tr>
    </tbody>
    </table>

    <table>
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">572</th>
    <th align="center">573</th>
    <th align="center">574</th>
    <th align="center">575</th>
    <th align="center">576</th>
    <th align="center">577</th>
    <th align="center">578</th>
    <th align="center">579</th>
    <th align="center">581</th>
    <th align="center">584</th>
    <th align="center">585</th>
    <th align="center">587</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-831.1</td>
    <td align="center">-508.3</td>
    <td align="center">-1122</td>
    <td align="center">2379</td>
    <td align="center">-2226</td>
    <td align="center">940.6</td>
    <td align="center">-402.7</td>
    <td align="center">-627.9</td>
    <td align="center">-567.8</td>
    <td align="center">-2067</td>
    <td align="center">-588.4</td>
    <td align="center">-2292</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">588</th>
    <th align="center">589</th>
    <th align="center">590</th>
    <th align="center">591</th>
    <th align="center">592</th>
    <th align="center">593</th>
    <th align="center">596</th>
    <th align="center">597</th>
    <th align="center">598</th>
    <th align="center">599</th>
    <th align="center">600</th>
    <th align="center">601</th>
    <th align="center">602</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-986.6</td>
    <td align="center">173.1</td>
    <td align="center">-226</td>
    <td align="center">-1942</td>
    <td align="center">-4245</td>
    <td align="center">-738.9</td>
    <td align="center">-1413</td>
    <td align="center">-3000</td>
    <td align="center">2945</td>
    <td align="center">208.8</td>
    <td align="center">-3205</td>
    <td align="center">208</td>
    <td align="center">1381</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">603</th>
    <th align="center">607</th>
    <th align="center">609</th>
    <th align="center">610</th>
    <th align="center">611</th>
    <th align="center">612</th>
    <th align="center">613</th>
    <th align="center">614</th>
    <th align="center">615</th>
    <th align="center">616</th>
    <th align="center">617</th>
    <th align="center">619</th>
    <th align="center">621</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2203</td>
    <td align="center">-212.9</td>
    <td align="center">886.3</td>
    <td align="center">-1336</td>
    <td align="center">-1191</td>
    <td align="center">968.6</td>
    <td align="center">-2148</td>
    <td align="center">-888.4</td>
    <td align="center">-1974</td>
    <td align="center">2274</td>
    <td align="center">604.3</td>
    <td align="center">900.6</td>
    <td align="center">-2597</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">622</th>
    <th align="center">623</th>
    <th align="center">626</th>
    <th align="center">627</th>
    <th align="center">628</th>
    <th align="center">630</th>
    <th align="center">632</th>
    <th align="center">633</th>
    <th align="center">634</th>
    <th align="center">635</th>
    <th align="center">636</th>
    <th align="center">637</th>
    <th align="center">638</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2333</td>
    <td align="center">-3696</td>
    <td align="center">1486</td>
    <td align="center">-1414</td>
    <td align="center">2297</td>
    <td align="center">256</td>
    <td align="center">1249</td>
    <td align="center">1273</td>
    <td align="center">-892.7</td>
    <td align="center">-950.1</td>
    <td align="center">-974.9</td>
    <td align="center">1416</td>
    <td align="center">2876</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">639</th>
    <th align="center">641</th>
    <th align="center">642</th>
    <th align="center">643</th>
    <th align="center">645</th>
    <th align="center">646</th>
    <th align="center">647</th>
    <th align="center">649</th>
    <th align="center">651</th>
    <th align="center">652</th>
    <th align="center">653</th>
    <th align="center">654</th>
    <th align="center">656</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2850</td>
    <td align="center">-2037</td>
    <td align="center">-2042</td>
    <td align="center">467.2</td>
    <td align="center">957.7</td>
    <td align="center">-366.3</td>
    <td align="center">-213.9</td>
    <td align="center">64.76</td>
    <td align="center">110.8</td>
    <td align="center">-4058</td>
    <td align="center">1663</td>
    <td align="center">1887</td>
    <td align="center">-2035</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">660</th>
    <th align="center">661</th>
    <th align="center">662</th>
    <th align="center">663</th>
    <th align="center">665</th>
    <th align="center">667</th>
    <th align="center">669</th>
    <th align="center">670</th>
    <th align="center">671</th>
    <th align="center">672</th>
    <th align="center">674</th>
    <th align="center">676</th>
    <th align="center">677</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-1011</td>
    <td align="center">1564</td>
    <td align="center">-2741</td>
    <td align="center">3384</td>
    <td align="center">-135.7</td>
    <td align="center">4240</td>
    <td align="center">2445</td>
    <td align="center">-448</td>
    <td align="center">212.7</td>
    <td align="center">635.5</td>
    <td align="center">-2690</td>
    <td align="center">-5.137</td>
    <td align="center">-1353</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">678</th>
    <th align="center">679</th>
    <th align="center">680</th>
    <th align="center">681</th>
    <th align="center">682</th>
    <th align="center">683</th>
    <th align="center">687</th>
    <th align="center">690</th>
    <th align="center">691</th>
    <th align="center">693</th>
    <th align="center">694</th>
    <th align="center">695</th>
    <th align="center">696</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">2030</td>
    <td align="center">2702</td>
    <td align="center">1320</td>
    <td align="center">-416.7</td>
    <td align="center">346.4</td>
    <td align="center">-1156</td>
    <td align="center">-1874</td>
    <td align="center">-1139</td>
    <td align="center">306.5</td>
    <td align="center">4248</td>
    <td align="center">670.7</td>
    <td align="center">-3197</td>
    <td align="center">-1640</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="9%" />
    <col width="9%" />
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">697</th>
    <th align="center">699</th>
    <th align="center">700</th>
    <th align="center">701</th>
    <th align="center">702</th>
    <th align="center">703</th>
    <th align="center">705</th>
    <th align="center">706</th>
    <th align="center">707</th>
    <th align="center">708</th>
    <th align="center">711</th>
    <th align="center">713</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">835.5</td>
    <td align="center">1070</td>
    <td align="center">2705</td>
    <td align="center">-1641</td>
    <td align="center">2934</td>
    <td align="center">-1595</td>
    <td align="center">436</td>
    <td align="center">-509.5</td>
    <td align="center">-217.7</td>
    <td align="center">-19.58</td>
    <td align="center">1210</td>
    <td align="center">-198.5</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">714</th>
    <th align="center">715</th>
    <th align="center">716</th>
    <th align="center">718</th>
    <th align="center">719</th>
    <th align="center">721</th>
    <th align="center">722</th>
    <th align="center">723</th>
    <th align="center">727</th>
    <th align="center">728</th>
    <th align="center">729</th>
    <th align="center">731</th>
    <th align="center">732</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">1590</td>
    <td align="center">172.9</td>
    <td align="center">97.97</td>
    <td align="center">-1767</td>
    <td align="center">-2144</td>
    <td align="center">-5344</td>
    <td align="center">-1050</td>
    <td align="center">362.4</td>
    <td align="center">2061</td>
    <td align="center">-1664</td>
    <td align="center">-1912</td>
    <td align="center">-100.4</td>
    <td align="center">-1211</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="8%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">735</th>
    <th align="center">736</th>
    <th align="center">737</th>
    <th align="center">739</th>
    <th align="center">740</th>
    <th align="center">741</th>
    <th align="center">742</th>
    <th align="center">743</th>
    <th align="center">746</th>
    <th align="center">748</th>
    <th align="center">749</th>
    <th align="center">750</th>
    <th align="center">751</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-441.7</td>
    <td align="center">-3531</td>
    <td align="center">-263.3</td>
    <td align="center">885.1</td>
    <td align="center">368.7</td>
    <td align="center">3642</td>
    <td align="center">1939</td>
    <td align="center">1475</td>
    <td align="center">1641</td>
    <td align="center">14.03</td>
    <td align="center">834.1</td>
    <td align="center">828.5</td>
    <td align="center">-697.8</td>
    </tr>
    </tbody>
    </table>

    <table style="width:100%;">
    <caption>Table continues below</caption>
    <colgroup>
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="8%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    <col width="7%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">752</th>
    <th align="center">753</th>
    <th align="center">755</th>
    <th align="center">756</th>
    <th align="center">757</th>
    <th align="center">758</th>
    <th align="center">760</th>
    <th align="center">761</th>
    <th align="center">762</th>
    <th align="center">764</th>
    <th align="center">765</th>
    <th align="center">766</th>
    <th align="center">767</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">3882</td>
    <td align="center">-2306</td>
    <td align="center">-331.3</td>
    <td align="center">1761</td>
    <td align="center">2952</td>
    <td align="center">837.5</td>
    <td align="center">-726.3</td>
    <td align="center">1492</td>
    <td align="center">1126</td>
    <td align="center">160.1</td>
    <td align="center">-1348</td>
    <td align="center">-1526</td>
    <td align="center">-368.5</td>
    </tr>
    </tbody>
    </table>

    <table style="width:60%;">
    <colgroup>
    <col width="9%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    <col width="8%" />
    </colgroup>
    <thead>
    <tr class="header">
    <th align="center">769</th>
    <th align="center">770</th>
    <th align="center">772</th>
    <th align="center">773</th>
    <th align="center">775</th>
    <th align="center">776</th>
    <th align="center">777</th>
    </tr>
    </thead>
    <tbody>
    <tr class="odd">
    <td align="center">-779.7</td>
    <td align="center">3034</td>
    <td align="center">1451</td>
    <td align="center">2447</td>
    <td align="center">-2861</td>
    <td align="center">-1166</td>
    <td align="center">-3606</td>
    </tr>
    </tbody>
    </table>

-   **deviance**: *1.996e+09*
-   **null.deviance**: *8.568e+09*
-   **aic**: *9801*
-   **iter**: *2*
-   **na.action**:

<!-- end of list -->
As the simple linear model, all the independent variables in this model have statistically significant association with the Out-of-state tuition. While all of them have (at the 0 level) F-values for parametric effects, only `Ph.D.` has statistically significant (at the \(\alpha = .05\) level) nonparametric F-value, indicating that there is likely to be a nonparametric effect on `Outstate`. In the next step, I present the relation between the depedent variable with each of six independent variables as below:

``` r
clg_gam_terms <- preplot(gam, se = TRUE, rug = FALSE)

# PhD
data_frame(x = clg_gam_terms$`lo(PhD)`$x,
           y = clg_gam_terms$`lo(PhD)`$y,
           se.fit = clg_gam_terms$`lo(PhD)`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state Tuition",
       subtitle = "Local Regression",
       x = "PHD",
       y = expression(f[3](PhD)))
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/3.3-2-1.png" style="display: block; margin: auto;" />

``` r
# perc.alumni
data_frame(x = clg_gam_terms$`lo(perc.alumni)`$x,
           y = clg_gam_terms$`lo(perc.alumni)`$y,
           se.fit = clg_gam_terms$`lo(perc.alumni)`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state Tuition",
       subtitle = "Local Regression",
       x = "perc.alumni",
       y = expression(f[4](perc.alumni)))
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/3.3-2-2.png" style="display: block; margin: auto;" />

``` r
# Expend
data_frame(x = clg_gam_terms$`log(Expend)`$x,
           y = clg_gam_terms$`log(Expend)`$y,
           se.fit = clg_gam_terms$`log(Expend)`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state Tuition",
       subtitle = "Log Transformation",
       x = "Expend",
       y = expression(f[5](expend)))
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/3.3-2-3.png" style="display: block; margin: auto;" />

``` r
# Grad.Rate
data_frame(x = clg_gam_terms$`poly(Grad.Rate, 4)`$x,
           y = clg_gam_terms$`poly(Grad.Rate, 4)`$y,
           se.fit = clg_gam_terms$`poly(Grad.Rate, 4)`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state Tuition",
       subtitle = "4th Degree Polynominal",
       x = "Grad.Rate",
       y = expression(f[6](Grad.Rate)))
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/3.3-2-4.png" style="display: block; margin: auto;" />

``` r
# Private
data_frame(x = clg_gam_terms$Private$x,
           y = clg_gam_terms$Private$y,
           se.fit = clg_gam_terms$Private$se.y) %>%
  unique %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y, ymin = y_low, ymax = y_high)) +
  geom_errorbar() +
  geom_point() +
  labs(title = "GAM of Out-of-state Tuition",
       x = "Is Private School or Not",
       y = expression(f[1](private)))
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/3.3-2-5.png" style="display: block; margin: auto;" />

``` r
# Room.Board
data_frame(x = clg_gam_terms$`poly(Room.Board, 2)`$x,
           y = clg_gam_terms$`poly(Room.Board, 2)`$y,
           se.fit = clg_gam_terms$`poly(Room.Board, 2)`$se.y) %>%
  mutate(y_low = y - 1.96 * se.fit,
         y_high = y + 1.96 * se.fit) %>%
  ggplot(aes(x, y)) +
  geom_line() +
  geom_line(aes(y = y_low), linetype = 2) +
  geom_line(aes(y = y_high), linetype = 2) +
  labs(title = "GAM of Out-of-state Tuition",
       subtitle = "Linear Regression",
       x = "Room.Board",
       y = expression(f[2](Room.Board))) 
```

<img src="ps7-resample-nonlinear_files/figure-markdown_github/3.3-2-6.png" style="display: block; margin: auto;" />

According to the plots above, it appears that all these six variables have have substantial and significant relationships with out-of-state tuition. a)`PhD`:The wide confidence intervals for values of PhD below 30% indicate that the effect of the portion of facaulty with PhD is not so strong in that range. However, the confidence interval decreases in size as PhD grows larger, eventually showing us that as PhD increases, Out-of-state tuition does as well. b)`perc.alumni`: It appears that the effect of the portion of alumni who donate on the out-of-state tuition will be weaker when it is larger than 40% or smaller than 15%.
c)`Expend`:The plot shows as the expenditure increases, the positive effect of it on the out-of state tuition will be weaker, since the cofeident intervals become more broad at the higher value of `expend`. d)`Grad.Rate`:From the lower values of graduation rate, the effect is hard to determine. As it increases past these values, however, it starts to increase out-of-state tuition until the higher levels of graduation rate in which the confidence interval becomes wider again. e)`Private`: From the plot, we see being private or not has very strong relation with the amount of tuitions.Non-private universities are more likely to have lower tuition than the private universities. f)Room.Board: There is strong and positive relation between the room and board costs with the tuitions. In addition, the strength of this effect of board costs on the tution only persists in a middle range from 4000 to 6000 dollars.

4)Use the test set to evaluate the model fit of the estimated OLS and GAM models, and explain the results obtained.

``` r
mse_1 <- calc_mse(college_train, split$test)
mse_2 <- calc_mse(gam, split$test)
```

The MSE for the OLS is 3652035, larger than that of the GAM (3595260), indicating that the GAM model more accurately fit the test data.

5)For which variables, if any, is there evidence of a non-linear relationship with the response?

I test three variables in the model, Ph.D., Expend, and Grad.Rate to see whether they are really in non-linear relationship with the dependent variable. By

``` r
#PhD
gam_PhD_rm<-gam(Outstate ~ Private + poly(Room.Board, 2) + lo(perc.alumni) + log(Expend) + poly(Grad.Rate, 4), data = split$train)

gam_PhD_lm<-gam(Outstate ~ Private + poly(Room.Board, 2) + PhD+ lo(perc.alumni) + log(Expend) + poly(Grad.Rate, 4), data = split$train)

pander(anova(gam_PhD_rm, gam_PhD_lm, gam, test = "F"))
```

<table style="width:78%;">
<caption>Analysis of Deviance Table</caption>
<colgroup>
<col width="16%" />
<col width="18%" />
<col width="8%" />
<col width="15%" />
<col width="8%" />
<col width="11%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Resid. Df</th>
<th align="center">Resid. Dev</th>
<th align="center">Df</th>
<th align="center">Deviance</th>
<th align="center">F</th>
<th align="center">Pr(&gt;F)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">531.6</td>
<td align="center">2.056e+09</td>
<td align="center">NA</td>
<td align="center">NA</td>
<td align="center">NA</td>
<td align="center">NA</td>
</tr>
<tr class="even">
<td align="center">530.6</td>
<td align="center">2.035e+09</td>
<td align="center">1</td>
<td align="center">20619839</td>
<td align="center">5.452</td>
<td align="center">0.01992</td>
</tr>
<tr class="odd">
<td align="center">527.9</td>
<td align="center">1.996e+09</td>
<td align="center">2.693</td>
<td align="center">38835973</td>
<td align="center">3.813</td>
<td align="center">0.0129</td>
</tr>
</tbody>
</table>

Since the p-value for both linear model and the non-linear model in terms of `PhD` are not statistically significant, larger than 0.01. It is hard to determine whether the PhD has a linear relationship with the dependent variable or not.

``` r
#expend

gam_exp_rm <- gam(Outstate ~ Private + poly(Room.Board, 2) + lo(PhD) + lo(perc.alumni) +  poly(Grad.Rate, 4),data = split$train)

gam_exp_lm <- gam(Outstate ~ Private + poly(Room.Board, 2) + lo(PhD) + lo(perc.alumni) +  poly(Grad.Rate, 4) + Expend,data = split$train)

pander(anova(gam_exp_rm, gam_exp_lm, gam, test = "F"))
```

<table style="width:76%;">
<caption>Analysis of Deviance Table In the examination of <code>Expend</code>, I found the p-value for the linear model for it is very small, indicating that it is more appopriate to adopt the linear relationship between expenditure with the out-of-state tuition.</caption>
<colgroup>
<col width="16%" />
<col width="18%" />
<col width="6%" />
<col width="15%" />
<col width="6%" />
<col width="12%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Resid. Df</th>
<th align="center">Resid. Dev</th>
<th align="center">Df</th>
<th align="center">Deviance</th>
<th align="center">F</th>
<th align="center">Pr(&gt;F)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">528.9</td>
<td align="center">2.539e+09</td>
<td align="center">NA</td>
<td align="center">NA</td>
<td align="center">NA</td>
<td align="center">NA</td>
</tr>
<tr class="even">
<td align="center">527.9</td>
<td align="center">2.216e+09</td>
<td align="center">1</td>
<td align="center">322802761</td>
<td align="center">76.9</td>
<td align="center">2.479e-17</td>
</tr>
<tr class="odd">
<td align="center">527.9</td>
<td align="center">1.996e+09</td>
<td align="center">0</td>
<td align="center">219445176</td>
<td align="center">NA</td>
<td align="center">NA</td>
</tr>
</tbody>
</table>

``` r
#Grad.Rate
gam_gr_rm <- gam(Outstate ~ Private + poly(Room.Board, 2) + lo(PhD) + lo(perc.alumni) + log(Expend) ,data = split$train)
gam_gr_lm <- gam(Outstate ~ Private + poly(Room.Board, 2) + lo(PhD) + lo(perc.alumni) + log(Expend) + Grad.Rate,data = split$train)

pander(anova(gam_PhD_rm, gam_PhD_lm, gam, test = "F"))
```

<table style="width:78%;">
<caption>Analysis of Deviance Table</caption>
<colgroup>
<col width="16%" />
<col width="18%" />
<col width="8%" />
<col width="15%" />
<col width="8%" />
<col width="11%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Resid. Df</th>
<th align="center">Resid. Dev</th>
<th align="center">Df</th>
<th align="center">Deviance</th>
<th align="center">F</th>
<th align="center">Pr(&gt;F)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">531.6</td>
<td align="center">2.056e+09</td>
<td align="center">NA</td>
<td align="center">NA</td>
<td align="center">NA</td>
<td align="center">NA</td>
</tr>
<tr class="even">
<td align="center">530.6</td>
<td align="center">2.035e+09</td>
<td align="center">1</td>
<td align="center">20619839</td>
<td align="center">5.452</td>
<td align="center">0.01992</td>
</tr>
<tr class="odd">
<td align="center">527.9</td>
<td align="center">1.996e+09</td>
<td align="center">2.693</td>
<td align="center">38835973</td>
<td align="center">3.813</td>
<td align="center">0.0129</td>
</tr>
</tbody>
</table>

The result for `Grad.Rate` is similar to that in `PhD`, the p-value of linear or non-linear relationship is not statistically significant and close to each other, so there is no strong evidence to refute or confirm the GAM model in the section 3.2, by using this approach.
