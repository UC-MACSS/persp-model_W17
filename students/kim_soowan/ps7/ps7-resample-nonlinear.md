# Problem set #7: resampling and nonlinearity
Soo Wan Kim  
February 25, 2017  



# Part 1: Sexy Joe Biden (redux) [4 points]

####Question  1. 
**Estimate the training MSE of the model using the traditional approach. Fit the linear regression model using the entire dataset and calculate the mean squared error for the training set.**


```r
#function to calculate MSE
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}

#estimate model
glm_trad <- glm(biden ~ age + female + educ + dem + rep, data = biden)
pander(tidy(glm_trad))
```


--------------------------------------------------------
   term      estimate   std.error   statistic   p.value 
----------- ---------- ----------- ----------- ---------
(Intercept)   58.81       3.124       18.82    2.694e-72

    age      0.04826     0.02825      1.708     0.08773 

  female      4.103      0.9482       4.327    1.593e-05

   educ      -0.3453     0.1948      -1.773     0.07641 

    dem       15.42       1.068       14.44    8.145e-45

    rep       -15.85      1.311      -12.09    2.157e-32
--------------------------------------------------------

```r
entire_mse <- mse(glm_trad, biden) #calculate MSE
```

The MSE for the model trained on the entire dataset is 395.2701693.

  2. **Estimate the test MSE of the model using the validation set approach. How does this value compare to the training MSE from step 1?**


```r
biden_split <- resample_partition(biden, c(test = 0.3, train = 0.7)) #split data into 70/30 training/test set
biden_train <- biden_split$train %>% 
  tbl_df()
biden_test <- biden_split$test %>% 
  tbl_df()

biden_train_lm <- glm(biden ~ age + female + educ + dem + rep, data = biden_train) #estimate model on training set
pander(tidy(biden_train_lm))
```


--------------------------------------------------------
   term      estimate   std.error   statistic   p.value 
----------- ---------- ----------- ----------- ---------
(Intercept)   57.34       3.698       15.51    9.235e-50

    age      0.03728     0.03362      1.109     0.2677  

  female      4.172       1.127       3.703    0.0002223

   educ      -0.2602     0.2322       -1.12     0.2628  

    dem       16.33       1.277       12.79    2.657e-35

    rep       -14.61      1.558      -9.375    3.086e-20
--------------------------------------------------------

```r
validation_mse <- mse(biden_train_lm, biden_test) #calculate MSE using test set
```

The MSE calculated using the validation approach is 399.8303029. This is slightly higher than the MSE calculated using the traditional approach (395.2701693). The model fitted using the validation approach only used 70% of the observations, so it is somewhat worse at predicting the results observed in the dataset than the model fitted on all the data.

  3. **Repeat the validation set approach 100 times, using 100 different splits of the observations into a training set and a validation set. Comment on the results obtained.**
  

```r
mse_list <- vector(, 100) #set up empty vector of 100 items

#function to calculate the validation set MSE a certain number of times
validation_mse <- function(data, model, reps) {
  count <- 0
  while (count < reps) {
    split <- resample_partition(biden, c(test = 0.3, train = 0.7)) #split data into 70/30 training/test set
    train <- tbl_df(split$train) 
    test <- dplyr::tbl_df(split$test)
    train_lm <- glm(model, data = data) #estimate model
    validation_mse <- mse(train_lm, test) #calculate MSE
    mse_list[count + 1] <- validation_mse #append MSE values into vector
    count <- count + 1
  }
  return(mse_list)
}

#vector of results from repeating validation approach 100 times
mse_list <- validation_mse(biden, biden ~ age + female + educ + dem + rep, 100)

summary(mse_list)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   338.0   385.7   396.8   398.1   412.7   450.2
```

```r
mse_sd <- sd(mse_list)
```

The MSEs vary quite a bit, ranging from 338 to 450. The standard deviation of MSE estimates is 20.6102169. This shows that the results are highly dependent on which observations are picked for the training set. The average of these values is very close to the MSE obtained using the entire dataset.

  4. **Estimate the test MSE of the model using the leave-one-out cross-validation (LOOCV) approach. Comment on the results obtained.**


```r
loocv_data <- crossv_kfold(biden, k = nrow(biden)) #divide data into k folds where k = number of observations
loocv_models <- map(loocv_data$train, ~ glm(biden ~ age + female + educ + dem + rep, data = .)) #estimate model
loocv_mse_map <- map2_dbl(loocv_models, loocv_data$test, mse) #calculate MSEs
loocv_mse <- mean(loocv_mse_map, na.rm = TRUE) #get mean of MSEs
```

The MSE calculated using the LOOCV approach is 397.9555046. This is similar to the MSE obtained using the entire dataset and the validation approach, and particularly to the mean of the MSEs obtained by repeating the validation approach 100 times. 

  5. **Estimate the test MSE of the model using the 10-fold cross-validation approach. Comment on the results obtained.**


```r
biden_cv10 <- crossv_kfold(biden, k = 10) %>% #divide data set into 10 folds
  mutate(model = map(train, ~ lm(biden ~ age + female + educ + dem + rep, data = .)), #estimate model
         mse = map2_dbl(model, test, mse)) #calculate MSEs
cv10_mse <- mean(biden_cv10$mse, na.rm = TRUE) #get mean MSE
```

The MSE calculated using the 10-fold cross-validation approach is 398.0728532. Again, this is similar to the other MSE estimates. It is especially very close to the LOOCV MSE and the mean of the MSEs from repeating the validation set approach 100 times.

  6. **Repeat the 10-fold cross-validation approach 100 times, using 100 different splits of the observations into $10$-folds. Comment on the results obtained.**
  

```r
cv10_mse_list <- vector(, 100) #create vector of length 100

#function
cv10_mse <- function(data, model, reps) {
  count <- 0
  while (count < reps) {
    folded <- crossv_kfold(data, k = 10) #divide data set into 10 folds
    folded$mod <- map(folded$train, ~glm(model, data = .)) #estimate model
    folded$mse <- map2_dbl(folded$mod, folded$test, mse) #calculate MSEs
    cv10_mse <- mean(folded$mse, na.rm = TRUE) #get mean of MSEs
    cv10_mse_list[count + 1] <- cv10_mse #store in vector
    count <- count + 1
  }
  return(cv10_mse_list)
}

cv10_mse_list <- cv10_mse(biden, biden ~ age + female + educ + dem + rep, 100)
summary(cv10_mse_list)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   396.3   397.6   397.9   398.0   398.3   400.3
```

```r
mse_sd <- sd(cv10_mse_list)
```

This time, the estimates vary very little, with a standard deviation of only 0.5394957. This shows that the 10-fold cross validation method produces estimates much more reliable and unbiased than that of the validation set approach. However, the mean of the estimates using this method is almost identical to the mean of the MSEs from repeating the validation set approach 100 times. 

  7. **Compare the estimated parameters and standard errors from the original model in step 1 (the model estimated using all of the available data) to parameters and standard errors estimated using the bootstrap ($n = 1000$).**

**Original model estimates**


```r
pander(tidy(glm_trad))
```


--------------------------------------------------------
   term      estimate   std.error   statistic   p.value 
----------- ---------- ----------- ----------- ---------
(Intercept)   58.81       3.124       18.82    2.694e-72

    age      0.04826     0.02825      1.708     0.08773 

  female      4.103      0.9482       4.327    1.593e-05

   educ      -0.3453     0.1948      -1.773     0.07641 

    dem       15.42       1.068       14.44    8.145e-45

    rep       -15.85      1.311      -12.09    2.157e-32
--------------------------------------------------------

**Bootstrap estimates**


```r
#estimate model using bootstrap, display in tidy format
boot <- biden %>%
  modelr::bootstrap(1000) %>%
  mutate(model = map(strap, ~ glm(biden ~ age + female + educ + dem + rep, data = .)), 
         coef = map(model, tidy))

boot_est <- boot %>%
  unnest(coef) %>%
  group_by(term) %>%
  summarize(est.boot = mean(estimate),
            se.boot = sd(estimate, na.rm = TRUE))
boot_est
```

```
## # A tibble: 6 × 3
##          term     est.boot    se.boot
##         <chr>        <dbl>      <dbl>
## 1 (Intercept)  58.91337251 2.97814255
## 2         age   0.04770968 0.02883481
## 3         dem  15.43020645 1.10724812
## 4        educ  -0.34950530 0.19214401
## 5      female   4.08800549 0.94879605
## 6         rep -15.87431840 1.44433208
```

The estimates for parameters and standard errors are very nearly the same across the two approaches.
