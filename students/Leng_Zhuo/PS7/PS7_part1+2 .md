
---
title: "Problem set #7: resampling and nonlinearity"
---
author: Zhuo Leng
-----

-------

Part 1: Sexy Joe Biden (redux) [4 points]
----------

1.Estimate the training MSE of the model using the traditional approach.
Fit the linear regression model using the entire dataset and calculate the mean squared error for the training set.


```R
library(tidyverse)
library(modelr)
library(broom)
# install.packages('rcfss')
# library(rcfss)

set.seed(1234)

options(digits = 3)

theme_set(theme_minimal())

```


```R
biden_data <- read.csv(file = "biden.csv", header = T)
attach(biden_data)

biden_data <- biden_data %>%
  tbl_df()
#biden_data

lm.fit1 <- lm(biden ~ age + female + educ + dem + rep, data = biden_data )
summary(lm.fit1)
```


    
    Call:
    lm(formula = biden ~ age + female + educ + dem + rep, data = biden_data)
    
    Residuals:
       Min     1Q Median     3Q    Max 
    -75.55 -11.29   1.02  12.78  53.98 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  58.8113     3.1244   18.82  < 2e-16 ***
    age           0.0483     0.0282    1.71    0.088 .  
    female        4.1032     0.9482    4.33  1.6e-05 ***
    educ         -0.3453     0.1948   -1.77    0.076 .  
    dem          15.4243     1.0680   14.44  < 2e-16 ***
    rep         -15.8495     1.3114  -12.09  < 2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 19.9 on 1801 degrees of freedom
    Multiple R-squared:  0.282,	Adjusted R-squared:  0.28 
    F-statistic:  141 on 5 and 1801 DF,  p-value: <2e-16



From the table above, we could know the estimate coefficients and std.error for lm.fit1.


```R
##mse function
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}
mse(lm.fit1, biden_data)    # linear model lm.fit1
```


395.270169278648


The mean square error for training set is 395.270169278648

2.Estimate the test MSE of the model using the validation set approach.


```R
### 2. Estimate the test MSE of the model using the validation set approach.
set.seed(1234)
biden_split <- resample_partition(biden_data, c(test = 0.3, train = 0.7))

lm.fit2 <- lm(biden ~ age + female + educ + dem + rep, data = biden_split$train)
summary(lm.fit2)
```


    
    Call:
    lm(formula = biden ~ age + female + educ + dem + rep, data = biden_split$train)
    
    Residuals:
       Min     1Q Median     3Q    Max 
     -75.8  -10.7    0.9   12.9   53.7 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  57.3374     3.6977   15.51  < 2e-16 ***
    age           0.0373     0.0336    1.11  0.26770    
    female        4.1721     1.1267    3.70  0.00022 ***
    educ         -0.2602     0.2322   -1.12  0.26275    
    dem          16.3277     1.2766   12.79  < 2e-16 ***
    rep         -14.6070     1.5580   -9.38  < 2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 19.9 on 1259 degrees of freedom
    Multiple R-squared:  0.279,	Adjusted R-squared:  0.276 
    F-statistic: 97.3 on 5 and 1259 DF,  p-value: <2e-16




```R
mse(lm.fit2, biden_split$test)    # linear model lm.fit2
```


399.83030289036


The test MSE is 399.83030289036. Compare to training set mse in step1 395.270169278648, the mse increase. This indicate that the model fits on the training set data perhaps not that good fit on test set of data.

3.Repeat the validation set approach 100 times, using 100 different splits of the observations into a training set and a validation set. Comment on the results obtained.


```R
##3.repeat validation set approach 100 times

mse_variable <- function(biden_data){
  biden_split <- resample_partition(biden_data, c(test = 0.3, train = 0.7))
  mse = mse(lm.fit2, biden_split$test)
  return(mse)
}

rerun(100, mse_variable(biden_data))


```


<ol>
	<li>394.443745339653</li>
	<li>390.581107880246</li>
	<li>379.341348981344</li>
	<li>394.62200793705</li>
	<li>387.131373140576</li>
	<li>389.93267381006</li>
	<li>387.319811470941</li>
	<li>367.662766361275</li>
	<li>363.988045016102</li>
	<li>415.613295663415</li>
	<li>383.483803469048</li>
	<li>385.812486471785</li>
	<li>357.688133039022</li>
	<li>409.791468606991</li>
	<li>412.805713980367</li>
	<li>374.042989688246</li>
	<li>391.44068757044</li>
	<li>397.282040525935</li>
	<li>384.837557749433</li>
	<li>393.237272880405</li>
	<li>340.869550297965</li>
	<li>369.19870072522</li>
	<li>336.493936198818</li>
	<li>424.706794579425</li>
	<li>390.105985665742</li>
	<li>383.242611759005</li>
	<li>420.517786192237</li>
	<li>380.566670022169</li>
	<li>390.497135172626</li>
	<li>388.327062420989</li>
	<li>465.779423677344</li>
	<li>418.386023105483</li>
	<li>373.376196355354</li>
	<li>439.962493492067</li>
	<li>372.978483673395</li>
	<li>360.77615088328</li>
	<li>379.04024065373</li>
	<li>415.630614706321</li>
	<li>355.981547220753</li>
	<li>374.080050483771</li>
	<li>373.909935199897</li>
	<li>410.821378364811</li>
	<li>384.899967645163</li>
	<li>396.522772109136</li>
	<li>437.245331810871</li>
	<li>384.734116706184</li>
	<li>395.854855933275</li>
	<li>407.578855370415</li>
	<li>417.114781966538</li>
	<li>417.92187037182</li>
	<li>415.274078690785</li>
	<li>387.476044280655</li>
	<li>374.367869098387</li>
	<li>402.155897025855</li>
	<li>422.442229378488</li>
	<li>378.970319427434</li>
	<li>399.036821665611</li>
	<li>417.1233257845</li>
	<li>403.803718688398</li>
	<li>363.236045036919</li>
	<li>424.624862915409</li>
	<li>418.653053206655</li>
	<li>405.390443461575</li>
	<li>380.204761061138</li>
	<li>403.410978652975</li>
	<li>405.168798251651</li>
	<li>394.025440270553</li>
	<li>412.315731846946</li>
	<li>415.651124755696</li>
	<li>406.730046700352</li>
	<li>407.38796886919</li>
	<li>390.558222443786</li>
	<li>383.441428275943</li>
	<li>386.088108650444</li>
	<li>453.700031501077</li>
	<li>422.236118070872</li>
	<li>405.951696558653</li>
	<li>421.613609392902</li>
	<li>417.915204431769</li>
	<li>407.278458524363</li>
	<li>399.40195333815</li>
	<li>411.249336284109</li>
	<li>390.835068306475</li>
	<li>404.560485094451</li>
	<li>396.871927931107</li>
	<li>375.634825926762</li>
	<li>381.602565736412</li>
	<li>412.795684530756</li>
	<li>442.682385238185</li>
	<li>417.054725225204</li>
	<li>423.635467991072</li>
	<li>376.999693553599</li>
	<li>385.68437226993</li>
	<li>398.264127271859</li>
	<li>407.59677633345</li>
	<li>400.616265640762</li>
	<li>392.900426229056</li>
	<li>378.383306592325</li>
	<li>385.136782298026</li>
	<li>399.60314911872</li>
</ol>



MSE mostly fall in 360~460. This method is more accurate then only doing one time validation. We could calculate the mean mse of 100 times as the result. This could limit bias.

4.Estimate the test MSE of the model using the leave-one-out cross-validation (LOOCV) approach. Comment on the results obtained.


```R
## 4. leave-one-out cross-validation (LOOCV) 
loocv_data <- crossv_kfold(biden_data, k = nrow(biden_data))

loocv_models <- map(loocv_data$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
loocv_mse <- map2_dbl(loocv_models, loocv_data$test, mse)
mean(loocv_mse)
```


397.955504554888


The test MSE of the model using the leave-one-out cross validation approach is 397.955504554888. The speed of LOOCV is faster than run 100 times validation approach. And it's MSE is very stable.

5.Estimate the test MSE of the model using the $10$-fold cross-validation approach. Comment on the results obtained.


```R
##$10$-fold cross-validation approach

set.seed(1234)
cv10_data <- crossv_kfold(biden_data, k = 10)

cv10_models <- map(cv10_data$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
cv10_mse <- map2_dbl(cv10_models, cv10_data$test, mse)
mean(cv10_mse)

```


397.883723312889


The result of 10-fold cross validation is 396.928796280458, which is similar to the result got by LOOCV. However, the speed of 10-fold cross validation is much faster than LOOCV. This approach is more efficient when calculate mse.


```R
##$10$-fold cross-validation approach

cv10_data2 <- crossv_kfold(biden_data, k = 10)

cv_error_fold10_2 <- vector("numeric", 100)
terms <- 1:100

for(i in terms){
  cv10_models2 <- map(cv10_data2$train, ~ lm(biden ~ age + female + educ + dem + rep, data = .))
  cv10_mse2 <- map2_dbl(cv10_models2, cv10_data2$test, mse)
  cv_error_fold10_2[[i]] <- mean(cv10_mse2)
}

mean(cv_error_fold10_2)


```


398.532258630613


The mean MSE of 100 times is 398.532258630613, the value is very similar to the LOOCV approach.

7.Compare the estimated parameters and standard errors from the original model in step 1 (the model estimated using all of the available data) to parameters and standard errors estimated using the bootstrap ($n = 1000$).



```R
###7.bootstrap

# traditional parameter estimates and standard errors
tidy(lm.fit1)

### bootstrapped estimates of the parameter estimates and standard errors
biden_boot <- biden_data %>%
  modelr::bootstrap(1000) %>%
  mutate(model = map(strap, ~ lm(biden ~ age + female + educ + dem + rep, data = .)),
         coef = map(model, tidy))

biden_boot %>%
  unnest(coef) %>%
  group_by(term) %>%
  summarize(est.boot = mean(estimate),
            se.boot = sd(estimate, na.rm = TRUE))
```


<table>
<thead><tr><th scope=col>term</th><th scope=col>estimate</th><th scope=col>std.error</th><th scope=col>statistic</th><th scope=col>p.value</th></tr></thead>
<tbody>
	<tr><td>(Intercept)</td><td> 58.8113   </td><td>3.1244     </td><td> 18.82     </td><td>2.69e-72   </td></tr>
	<tr><td>age        </td><td>  0.0483   </td><td>0.0282     </td><td>  1.71     </td><td>8.77e-02   </td></tr>
	<tr><td>female     </td><td>  4.1032   </td><td>0.9482     </td><td>  4.33     </td><td>1.59e-05   </td></tr>
	<tr><td>educ       </td><td> -0.3453   </td><td>0.1948     </td><td> -1.77     </td><td>7.64e-02   </td></tr>
	<tr><td>dem        </td><td> 15.4243   </td><td>1.0680     </td><td> 14.44     </td><td>8.14e-45   </td></tr>
	<tr><td>rep        </td><td>-15.8495   </td><td>1.3114     </td><td>-12.09     </td><td>2.16e-32   </td></tr>
</tbody>
</table>




<table>
<thead><tr><th scope=col>term</th><th scope=col>est.boot</th><th scope=col>se.boot</th></tr></thead>
<tbody>
	<tr><td>(Intercept)</td><td> 58.9618   </td><td>2.9499     </td></tr>
	<tr><td>age        </td><td>  0.0476   </td><td>0.0285     </td></tr>
	<tr><td>dem        </td><td> 15.4294   </td><td>1.1110     </td></tr>
	<tr><td>educ       </td><td> -0.3512   </td><td>0.1920     </td></tr>
	<tr><td>female     </td><td>  4.0755   </td><td>0.9488     </td></tr>
	<tr><td>rep        </td><td>-15.8871   </td><td>1.4315     </td></tr>
</tbody>
</table>



The difference betweent the estimates from bootstrap is not that much to the original model in step 1. The std errors of bootstrap for age, demale, rep, dem variables are slightly highter than that in original model. The std error of bootstrp of educ is a little bit lower than that in original model. This is because bootstrap doesn't rely on distributional assumptions.

Part 2: College (bivariate) [3 points]
-----------
------------
first predictor: Expend
--------------



```R
college_data <- read.csv(file = "College.csv", header = T)
head(college_data)
attach(college_data)
```


<table>
<thead><tr><th scope=col>Private</th><th scope=col>Apps</th><th scope=col>Accept</th><th scope=col>Enroll</th><th scope=col>Top10perc</th><th scope=col>Top25perc</th><th scope=col>F.Undergrad</th><th scope=col>P.Undergrad</th><th scope=col>Outstate</th><th scope=col>Room.Board</th><th scope=col>Books</th><th scope=col>Personal</th><th scope=col>PhD</th><th scope=col>Terminal</th><th scope=col>S.F.Ratio</th><th scope=col>perc.alumni</th><th scope=col>Expend</th><th scope=col>Grad.Rate</th></tr></thead>
<tbody>
	<tr><td>Yes  </td><td>1660 </td><td>1232 </td><td>721  </td><td>23   </td><td>52   </td><td>2885 </td><td> 537 </td><td> 7440</td><td>3300 </td><td>450  </td><td>2200 </td><td>70   </td><td>78   </td><td>18.1 </td><td>12   </td><td> 7041</td><td>60   </td></tr>
	<tr><td>Yes  </td><td>2186 </td><td>1924 </td><td>512  </td><td>16   </td><td>29   </td><td>2683 </td><td>1227 </td><td>12280</td><td>6450 </td><td>750  </td><td>1500 </td><td>29   </td><td>30   </td><td>12.2 </td><td>16   </td><td>10527</td><td>56   </td></tr>
	<tr><td>Yes  </td><td>1428 </td><td>1097 </td><td>336  </td><td>22   </td><td>50   </td><td>1036 </td><td>  99 </td><td>11250</td><td>3750 </td><td>400  </td><td>1165 </td><td>53   </td><td>66   </td><td>12.9 </td><td>30   </td><td> 8735</td><td>54   </td></tr>
	<tr><td>Yes  </td><td> 417 </td><td> 349 </td><td>137  </td><td>60   </td><td>89   </td><td> 510 </td><td>  63 </td><td>12960</td><td>5450 </td><td>450  </td><td> 875 </td><td>92   </td><td>97   </td><td> 7.7 </td><td>37   </td><td>19016</td><td>59   </td></tr>
	<tr><td>Yes  </td><td> 193 </td><td> 146 </td><td> 55  </td><td>16   </td><td>44   </td><td> 249 </td><td> 869 </td><td> 7560</td><td>4120 </td><td>800  </td><td>1500 </td><td>76   </td><td>72   </td><td>11.9 </td><td> 2   </td><td>10922</td><td>15   </td></tr>
	<tr><td>Yes  </td><td> 587 </td><td> 479 </td><td>158  </td><td>38   </td><td>62   </td><td> 678 </td><td>  41 </td><td>13500</td><td>3335 </td><td>500  </td><td> 675 </td><td>67   </td><td>73   </td><td> 9.4 </td><td>11   </td><td> 9727</td><td>55   </td></tr>
</tbody>
</table>



For the no transformation model, I choose Expend value as predictor.


```R
# ###first model: No transformation
model1 <- glm(Outstate ~ Expend, data = college_data)
summary(model1)

grid1 <- college_data %>%
  add_predictions(model1) %>%
  add_residuals(model1) 


##plot
ggplot(grid1, aes(Expend)) +
  geom_point(aes(y = Outstate)) +
  geom_smooth(method = "lm", aes(y = pred), color = 'red') +
  labs(title = "Linear model for a linear relationship", y = "Predicted Out-of-state tuition")

ggplot(grid1, aes(x = pred)) +
  geom_point(aes(y = resid)) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Homoscedastic variance of error terms",
        x = "Predicted Out-of-state tuition",
        y = "Residuals")
```


    
    Call:
    glm(formula = Outstate ~ Expend, data = college_data)
    
    Deviance Residuals: 
       Min      1Q  Median      3Q     Max  
    -15781   -2089      58    2011    7785  
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept) 5.43e+03   2.25e+02    24.2   <2e-16 ***
    Expend      5.18e-01   2.05e-02    25.3   <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for gaussian family taken to be 8870412)
    
        Null deviance: 1.2559e+10  on 776  degrees of freedom
    Residual deviance: 6.8746e+09  on 775  degrees of freedom
    AIC: 14640
    
    Number of Fisher Scoring iterations: 2








![png](output_28_3.png)



![png](output_28_4.png)


In order to find the statistics relationship between Expand(Instructional expenditure per student) and Outstate tuition, I first fit a simple no transformation linear model 
$$Outstate = \beta_{0} + \beta_{1} * Expand +\epsilon$$
From the summary table, we could know the coefficient of Expand is 5.18e-01, which means with one unit increase in Expand, the Outstate tuition will increase 5.18e-01. The p-valeu < 0.05, and this could testify the statistics significant of Expand predictor. The residuals seem to be randomly located around 0. However, from the first plot, we could see that the data still not that fit the model.



```R
##$10$-fold cross-validation approach

cv10_college_data <- crossv_kfold(college_data, k = 10)
cv10_college_model <- map(cv10_college_data$train, ~ lm(Outstate ~ Expend, data = .))
cv10_college_mse <- map2_dbl(cv10_college_model, cv10_college_data$test, mse)
mean(cv10_college_mse)

```


9289484.55337569


Then after using 10-fold cross-validation methods, we get the mse = 9289484.55337569.

From the plot, and Bulging Rule, now we try to do monotonic transformation--log(x) transformation.


```R
##Monotonic transformation--log(x) transformation
model2 <- glm(Outstate ~ log(Expend), data = college_data)
summary(model2)

grid2 <- college_data %>%
  add_predictions(model2, var = "pred2") %>%
  add_residuals(model2)


#plot
ggplot(grid2, aes(x = Expend)) +
  geom_point(aes(y = Outstate)) +
  geom_smooth(method = "lm", aes(y = pred2), color = 'red') +
  labs(title = "Log Linear model for a linear relationship",
        x = "log(Expend)",
        y = "Predicted Out-of-state tuition")

ggplot(grid2, aes(x = pred2)) +
  geom_point(aes(y = resid)) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Homoscedastic variance of error terms",
        x = "Predicted Out-of-state tuition",
        y = "Residuals")


```


    
    Call:
    glm(formula = Outstate ~ log(Expend), data = college_data)
    
    Deviance Residuals: 
       Min      1Q  Median      3Q     Max  
    -10651   -1572     100    1806    6604  
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)   -57502       2090   -27.5   <2e-16 ***
    log(Expend)     7482        230    32.5   <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for gaussian family taken to be 6847844)
    
        Null deviance: 1.2559e+10  on 776  degrees of freedom
    Residual deviance: 5.3071e+09  on 775  degrees of freedom
    AIC: 14439
    
    Number of Fisher Scoring iterations: 2








![png](output_32_3.png)



![png](output_32_4.png)


In order to find the statistics relationship between Expand(Instructional expenditure per student) and Outstate tuition, I first fit a simple no transformation linear model 
$$Outstate = \beta_{0} + \beta_{1} * log(Expand) +\epsilon$$
From the summary table, we could know the coefficient of Expand is 7482, which means with one unit increase in log(Expand), the Outstate tuition will increase 7482. The p-valeu < 0.05, and this could testify the statistics significant of Expand predictor. Also, the residual seem to be randomly located.


```R
##$10$-fold cross-validation approach

cv10_college_model2 <- map(cv10_college_data$train, ~ lm(Outstate ~ log(Expend), data = .))
cv10_college_mse2 <- map2_dbl(cv10_college_model2, cv10_college_data$test, mse)
log_mse <- mean(cv10_college_mse2)
log_mse
```


6908134.32487044


The 10-fold cross-validation methods generate mse = 6890220.52725771, which decrease a lot compare with mse of the model1 = 9289484.55337569. So this model is better then the last one. Then we want to know if polynomial transformation better by test the 10 -fold CV MSE.


```R
cv10_college_data <- crossv_kfold(college_data, k = 10)
cv10_error_fold10 <- vector("numeric", 5)
terms <- 1:5

for(i in terms){
  cv10_college_model2 <- map(cv10_college_data$train, ~ lm(Outstate ~ poly(Expend, i), data = .))
  cv10_college_mse2 <- map2_dbl(cv10_college_model2, cv10_college_data$test, mse)
  cv10_error_fold10[[i]] <- mean(cv10_college_mse2)
}

data_frame(terms = terms,
           fold10 = cv10_error_fold10) %>%
  ggplot(aes(terms, fold10)) +
    geom_line() +
    labs(title = "MSE estimates for 10-fold cross validation",
         x = "Degree of Polynomial",
         y = "Mean Squared Error") +
    geom_hline(aes(color = 'Log(Expend) MSE', yintercept = log_mse)) + 
    theme(legend.position = 'bottom')

```




![png](output_36_1.png)


The 10-fold cross-validation between different degrees of polynomials and log transformation model shows that 3 degree of polynomial can actually generate lower mse. So we now try polynomials transformation for 3 degree.


```R
##Polynomial regressions

model3 <- glm(Outstate ~ I(Expend^1) + I(Expend^2) + I(Expend^3), data = college_data)
tidy(model3)

grid3 <- college_data %>%
  add_predictions(model3, var = "pred3")


##plot
ggplot(grid3, aes(x = Expend)) +
  geom_point(data = college_data, aes(y = Outstate, alpha = .05)) +
  geom_smooth(method = "lm", aes(y = pred3), color = 'red') +
  labs(title = "Polynomial regression for a linear relationship with df3")

```


<table>
<thead><tr><th scope=col>term</th><th scope=col>estimate</th><th scope=col>std.error</th><th scope=col>statistic</th><th scope=col>p.value</th></tr></thead>
<tbody>
	<tr><td>(Intercept)</td><td>-1.24e+03  </td><td>5.60e+02   </td><td>-2.22      </td><td>2.70e-02   </td></tr>
	<tr><td>I(Expend^1)</td><td> 1.70e+00  </td><td>1.09e-01   </td><td>15.57      </td><td>9.57e-48   </td></tr>
	<tr><td>I(Expend^2)</td><td>-4.63e-05  </td><td>5.76e-06   </td><td>-8.04      </td><td>3.30e-15   </td></tr>
	<tr><td>I(Expend^3)</td><td> 3.95e-10  </td><td>8.04e-11   </td><td> 4.91      </td><td>1.08e-06   </td></tr>
</tbody>
</table>






![png](output_38_2.png)



```R
##$10$-fold cross-validation approach

cv10_college_model3 <- map(cv10_college_data$train, ~ glm(Outstate ~ I(Expend^1) + I(Expend^2) + I(Expend^3), data = .))
cv10_college_mse3 <- map2_dbl(cv10_college_model3, cv10_college_data$test, mse)
mean(cv10_college_mse3)

```


6501560.58202698


The 10-fold cross-validation methods generate mse = 6501560.58202698, which decrease a lot compare with mse of the model1 = 6890220.52725771. So this model is better. This model is $$Outstate = \beta_{0} + \beta_{1} * Expend + beta_{2} * (Expend)^{2} + beta_{3} * (Expend)^{3} + \epsilon$$

Second predictor: Graduation rate
--------


```R
# ###first model: No transformation
model1_gr <- glm(Outstate ~ Grad.Rate, data = college_data)
summary(model1_gr)

grid1_gr <- college_data %>%
  add_predictions(model1_gr) %>%
  add_residuals(model1_gr) 


##plot
ggplot(grid1_gr, aes(x = Grad.Rate)) +
  geom_point(aes(y = Outstate)) +
  geom_smooth(method = "lm", aes(y = pred), color = 'red') +
  labs(title = "Linear model for a linear relationship", y = "Predicted Out-of-state tuition")

ggplot(grid1_gr, aes(x = pred)) +
  geom_point(aes(y = resid)) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Homoscedastic variance of error terms",
        x = "Predicted Out-of-state tuition",
        y = "Residuals")
```


    
    Call:
    glm(formula = Outstate ~ Grad.Rate, data = college_data)
    
    Deviance Residuals: 
       Min      1Q  Median      3Q     Max  
    -11222   -2252    -161    2157   12659  
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)   1681.9      467.3     3.6  0.00034 ***
    Grad.Rate      133.8        6.9    19.4  < 2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for gaussian family taken to be 10916506)
    
        Null deviance: 1.2559e+10  on 776  degrees of freedom
    Residual deviance: 8.4603e+09  on 775  degrees of freedom
    AIC: 14801
    
    Number of Fisher Scoring iterations: 2








![png](output_42_3.png)



![png](output_42_4.png)


In order to find the statistics relationship between grad.rate and Outstate tuition, I first fit a simple no transformation linear model 
$$Outstate = \beta_{0} + \beta_{1} * Grad.Rate +\epsilon$$
From the summary table, we could know the coefficient of Grade.Rate is 133.8, which means with one unit increase in Grade.Rate, the Outstate tuition will increase 133.8. The p-valeu < 0.05, and this could testify the statistics significant of Expand predictor. The residuals seem to be randomly located around 0. Also, from the first plot, we could see that the data fit the model well. To see if we could improve the model, I try to use cross validation to do piecewise polynomial.


```R
##$10$-fold cross-validation approach

cv10_college_model2 <- map(cv10_college_data$train, ~ glm(Outstate ~ Grad.Rate, data = .))
cv10_college_mse2 <- map2_dbl(cv10_college_model2, cv10_college_data$test, mse)
log_mse <- mean(cv10_college_mse2)
log_mse

cv10_error_fold10 <- vector("numeric", 5)
terms <- 1:5

for(i in terms){
  cv10_college_model2 <- map(cv10_college_data$train, ~ lm(Outstate ~ poly(Grad.Rate, i), data = .))
  cv10_college_mse2 <- map2_dbl(cv10_college_model2, cv10_college_data$test, mse)
  cv10_error_fold10[[i]] <- mean(cv10_college_mse2)
}

data_frame(terms = terms,
           fold10 = cv10_error_fold10) %>%
  ggplot(aes(terms, fold10)) +
    geom_line() +
    geom_hline(aes(color = 'MSE', yintercept = log_mse)) + 
    labs(title = "MSE estimates for 10-fold cross validation",
         x = "Degree of Polynomial",
         y = "Mean Squared Error")

```


10955042.0502037





![png](output_44_2.png)


So from the mse for 10-fold cv method of the non transformation model is 10955042.0502037. After see the graph above, we could know that 4-degree polynomial can provides minimum MSE.


```R
##Polynomial regressions

model3 <- glm(Outstate ~ I(Grad.Rate^1) + I(Grad.Rate^2) + I(Grad.Rate^3) + I(Grad.Rate^4), data = college_data)
tidy(model3)

grid3 <- college_data %>%
  add_predictions(model3, var = "pred3")


##plot
ggplot(grid3, aes(x = Grad.Rate)) +
  geom_point(data = college_data, aes(y = Outstate, alpha = .05)) +
  geom_smooth(method = "lm", aes(y = pred3), color = 'red') +
  labs(title = "Polynomial regression for a linear relationship with df4")

## mse
cv10_error_fold10[4]
```


<table>
<thead><tr><th scope=col>term</th><th scope=col>estimate</th><th scope=col>std.error</th><th scope=col>statistic</th><th scope=col>p.value</th></tr></thead>
<tbody>
	<tr><td>(Intercept)   </td><td> 7.94e+03     </td><td>4.21e+03      </td><td> 1.8861       </td><td>0.0597        </td></tr>
	<tr><td>I(Grad.Rate^1)</td><td>-5.00e+00     </td><td>3.17e+02      </td><td>-0.0158       </td><td>0.9874        </td></tr>
	<tr><td>I(Grad.Rate^2)</td><td>-3.84e+00     </td><td>8.45e+00      </td><td>-0.4550       </td><td>0.6492        </td></tr>
	<tr><td>I(Grad.Rate^3)</td><td> 1.12e-01     </td><td>9.45e-02      </td><td> 1.1808       </td><td>0.2381        </td></tr>
	<tr><td>I(Grad.Rate^4)</td><td>-6.63e-04     </td><td>3.77e-04      </td><td>-1.7587       </td><td>0.0790        </td></tr>
</tbody>
</table>






10577173.0711689



![png](output_46_3.png)


The new model is $$Outstate = \beta_{0} + \beta_{1} * Grad.Rate + beta_{2} * (Grad.Rate)^{2} + beta_{3} * (Grad.Rate)^{3} + beta_{4} * (Grad.Rate)^{4} + \epsilon$$
The mse of this model is 10577173.0711689 and this model seems better for Grad.Rate.

third predictor: S.F.Ratio
------


```R
# ###first model: No transformation
model1_sf <- glm(Outstate ~ S.F.Ratio, data = college_data)
summary(model1_sf)

grid1_sf <- college_data %>%
  add_predictions(model1_sf) %>%
  add_residuals(model1_sf) 


##plot
ggplot(grid1_sf, aes(x = S.F.Ratio)) +
  geom_point(aes(y = Outstate)) +
  geom_smooth(method = "lm", aes(y = pred), color = 'red') +
  labs(title = "Linear model for a linear relationship", y = "Predicted Out-of-state tuition")

ggplot(grid1_gr, aes(x = pred)) +
  geom_point(aes(y = resid)) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = "Homoscedastic variance of error terms",
        x = "Predicted Out-of-state tuition",
        y = "Residuals")
```


    
    Call:
    glm(formula = Outstate ~ S.F.Ratio, data = college_data)
    
    Deviance Residuals: 
       Min      1Q  Median      3Q     Max  
    -10168   -2637    -252    2268   13267  
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept)  18385.6      444.5    41.4   <2e-16 ***
    S.F.Ratio     -563.9       30.4   -18.6   <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    (Dispersion parameter for gaussian family taken to be 11217046)
    
        Null deviance: 1.2559e+10  on 776  degrees of freedom
    Residual deviance: 8.6932e+09  on 775  degrees of freedom
    AIC: 14822
    
    Number of Fisher Scoring iterations: 2








![png](output_49_3.png)



![png](output_49_4.png)


In order to find the statistics relationship between S.F.Ratio and Outstate tuition, I first fit a simple no transformation linear model 
$$Outstate = \beta_{0} + \beta_{1} * S.F.Ratio +\epsilon$$
From the summary table, we could know the coefficient of S.F.Ratio is -563.9 , which means with one unit increase in S.F.Ratio, the Outstate tuition will decrease 563.9. The p-valeu < 0.05, and this could testify the statistics significant of Expand predictor. The residuals seem to be randomly located around 0. Also, from the first plot, we could see that the data fit the model well. To see if we could improve the model, I try to use cross validation to do piecewise polynomial.


```R
##$10$-fold cross-validation approach

cv10_college_model2 <- map(cv10_college_data$train, ~ glm(Outstate ~ S.F.Ratio, data = .))
cv10_college_mse2 <- map2_dbl(cv10_college_model2, cv10_college_data$test, mse)
log_mse <- mean(cv10_college_mse2)
log_mse

cv10_error_fold10 <- vector("numeric", 5)
terms <- 1:5

for(i in terms){
  cv10_college_model2 <- map(cv10_college_data$train, ~ lm(Outstate ~ poly(S.F.Ratio, i), data = .))
  cv10_college_mse2 <- map2_dbl(cv10_college_model2, cv10_college_data$test, mse)
  cv10_error_fold10[[i]] <- mean(cv10_college_mse2)
}

data_frame(terms = terms,
           fold10 = cv10_error_fold10) %>%
  ggplot(aes(terms, fold10)) +
    geom_line() +
    geom_hline(aes(color = 'MSE', yintercept = log_mse)) + 
    labs(title = "MSE estimates for 10-fold cross validation",
         x = "Degree of Polynomial",
         y = "Mean Squared Error")
```


11270332.6439016





![png](output_51_2.png)


We could know that the mse for the no transformation model is 11270332.6439016. Although, from piecewise polynomial graph, we could know that when df = 4, the mse could be minimize. However,the change is not that significant. So we could keep to the no transformation model in order to prevent overfitting problem.

Part 3: College (GAM) [3 points]
-----------


```R
set.seed(1234)
```

1.Split the data into a training set and a test set.
-------


```R
require(mgcv)
library(mgcv)
library(splines)
install.packages('gam')
library(gam)
GAM_college_data = read.csv(file = "College.csv", header = T)
```

I don't know why I can't install the gam packages, so then I use R markdown to write part3. 
--------


```R


```

    Warning message in install.packages(update[instlib == l, "Package"], l, contriburl = contriburl, :
    “installation of package ‘nlme’ had non-zero exit status”Warning message in install.packages(update[instlib == l, "Package"], l, contriburl = contriburl, :
    “installation of package ‘openssl’ had non-zero exit status”Warning message in install.packages(update[instlib == l, "Package"], l, contriburl = contriburl, :
    “installation of package ‘pbdZMQ’ had non-zero exit status”Updating HTML index of packages in '.Library'
    Making 'packages.html' ... done



```R
install.packages('gam')
library(gam)
```

    Warning message in install.packages("gam"):
    “installation of package ‘gam’ had non-zero exit status”Updating HTML index of packages in '.Library'
    Making 'packages.html' ... done



    Error in library(gam): there is no package called ‘gam’
    Traceback:


    1. library(gam)

    2. stop(txt, domain = NA)



```R

```
