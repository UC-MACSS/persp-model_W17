Problem set \#7: resampling and nonlinearity
================
Zhuo Leng

-   [Part 3: GAM](#part-3-gam)
-   [Estimate an OLS model](#estimate-an-ols-model)
-   [3.GAM model](#gam-model)
-   [4. Testing Model](#testing-model)
-   [5. Non-linear Relationship with the response](#non-linear-relationship-with-the-response)

Part 3: GAM
-----------

Estimate an OLS model
---------------------

``` r
attach(GAM_college_data)
##1.split the data

part3_split <- resample_partition(GAM_college_data, c(test = 0.3, train = 0.7))

part3.model1 <- lm(Outstate ~  Private + Room.Board + PhD + perc.alumni + Expend + Grad.Rate, data = part3_split$train)
summary(part3.model1)
```

    ## 
    ## Call:
    ## lm(formula = Outstate ~ Private + Room.Board + PhD + perc.alumni + 
    ##     Expend + Grad.Rate, data = part3_split$train)
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

As the summary table above, we could see that the OLS model coefficients are as above. From their p-value, we could see all the six predictors are statistics significant. The Adjusted R-squared: 0.738, which is close to 1, means the model fit the data well.

3.GAM model
-----------

``` r
part3.model2 <- gam(Outstate ~  Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + poly(Expend, 4) + poly(Grad.Rate, 3), data = part3_split$train)

summary(part3.model2)
```

    ## 
    ## Call: gam(formula = Outstate ~ Private + lo(Room.Board) + lo(PhD) + 
    ##     lo(perc.alumni) + poly(Expend, 4) + poly(Grad.Rate, 3), data = part3_split$train)
    ## Deviance Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -6976.03 -1081.40   -23.99  1255.16  7168.81 
    ## 
    ## (Dispersion Parameter for gaussian family taken to be 3565862)
    ## 
    ##     Null Deviance: 8567705097 on 543 degrees of freedom
    ## Residual Deviance: 1868126040 on 523.8918 degrees of freedom
    ## AIC: 9772.815 
    ## 
    ## Number of Local Scoring Iterations: 2 
    ## 
    ## Anova for Parametric Effects
    ##                        Df     Sum Sq    Mean Sq  F value    Pr(>F)    
    ## Private              1.00 2333041989 2333041989 654.2715 < 2.2e-16 ***
    ## lo(Room.Board)       1.00 2023176008 2023176008 567.3736 < 2.2e-16 ***
    ## lo(PhD)              1.00  823687764  823687764 230.9926 < 2.2e-16 ***
    ## lo(perc.alumni)      1.00  414259055  414259055 116.1736 < 2.2e-16 ***
    ## poly(Expend, 4)      4.00  856229523  214057381  60.0296 < 2.2e-16 ***
    ## poly(Grad.Rate, 3)   3.00  103825427   34608476   9.7055 3.047e-06 ***
    ## Residuals          523.89 1868126040    3565862                       
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Anova for Nonparametric Effects
    ##                    Npar Df Npar F   Pr(F)  
    ## (Intercept)                                
    ## Private                                    
    ## lo(Room.Board)         3.0 2.5559 0.05498 .
    ## lo(PhD)                2.7 2.3855 0.07515 .
    ## lo(perc.alumni)        2.4 1.2001 0.30635  
    ## poly(Expend, 4)                            
    ## poly(Grad.Rate, 3)                         
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

For my gam model, I use linear regression on private and 4 degree polynomial on Grad.Rate, 3 degree polynomial on Expend. For other variables, I use local regression. From the summary table, we could see that the almost all variables are statistically significant.

``` r
# clg_gam_terms <- preplot(part3_model2, se = TRUE, rug = FALSE)
# 
# # Private
# data_frame(x = clg_gam_terms$Private$x,
#            y = clg_gam_terms$Private$y,
#            se.fit = part3.model2.terms$Private$se.y) %>%
#   unique %>%
#   mutate(y_low = y - 1.96 * se.fit,
#          y_high = y + 1.96 * se.fit) %>%
#   ggplot(aes(x, y, ymin = y_low, ymax = y_high)) +
#   geom_line() +
#   labs(title = "GAM of Out-of-state Tuition",
#        x = "Private",
#        y = expression(f[5](private)))
# ```
# 
# 
# ```{r  , include=TRUE}
# # Room.Board
# data_frame(x = part3.model2.terms$`lo(Room.Board)`$x,
#            y = part3.model2.terms$`lo(Room.Board)`$y,
#            se.fit = part3.model2.terms$lo(Room.Board)$se.y) %>%
#   mutate(y_low = y - 1.96 * se.fit,
#          y_high = y + 1.96 * se.fit) %>%
#   ggplot(aes(x, y)) +
#   geom_line() +
#   geom_line(aes(y = y_low), linetype = 2) +
#   geom_line(aes(y = y_high), linetype = 2) +
#   labs(title = "GAM of Out-of-state Tuition",
#        x = "Room.Board",
#        y = expression(f[6](Room.Board)))   
# ```
# 
# 
# 
# ```{r  , include=TRUE}
# 
# # PhD
# data_frame(x = clg_gam_terms$`lo(PhD)`$x,
#            y = clg_gam_terms$`lo(PhD)`$y,
#            se.fit = clg_gam_terms$`lo(PhD)`$se.y) %>%
#   mutate(y_low = y - 1.96 * se.fit,
#          y_high = y + 1.96 * se.fit) %>%
#   ggplot(aes(x, y)) +
#   geom_line() +
#   geom_line(aes(y = y_low), linetype = 2) +
#   geom_line(aes(y = y_high), linetype = 2) +
#   labs(title = "GAM of Out-of-state Tuition",
#        x = "PHD",
#        y = expression(f[1](PhD)))
# ```
# 
# 
# 
# ```{r  , include=TRUE}
# # perc.alumni
# data_frame(x = clg_gam_terms$`lo(perc.alumni)`$x,
#            y = clg_gam_terms$`lo(perc.alumni)`$y,
#            se.fit = clg_gam_terms$`lo(perc.alumni)`$se.y) %>%
#   mutate(y_low = y - 1.96 * se.fit,
#          y_high = y + 1.96 * se.fit) %>%
#   ggplot(aes(x, y)) +
#   geom_line() +
#   geom_line(aes(y = y_low), linetype = 2) +
#   geom_line(aes(y = y_high), linetype = 2) +
#   labs(title = "GAM of Out-of-state Tuition",
#        x = "perc.alumni",
#        y = expression(f[2](perc.alumni)))
# ```
# 
# 
# 
# ```{r  , include=TRUE}
# # Expend
# data_frame(x = clg_gam_terms$`bs(Expend, degree = 4)`$x,
#            y = clg_gam_terms$`bs(Expend, degree = 4)`$y,
#            se.fit = clg_gam_terms$`bs(Expend, degree = 4)`$se.y) %>%
#   mutate(y_low = y - 1.96 * se.fit,
#          y_high = y + 1.96 * se.fit) %>%
#   ggplot(aes(x, y)) +
#   geom_line() +
#   geom_line(aes(y = y_low), linetype = 2) +
#   geom_line(aes(y = y_high), linetype = 2) +
#   labs(title = "GAM of Out-of-state Tuition",
#        x = "Expend",
#        y = expression(f[3](expend)))
# ```
# 
# 
# ```{r  , include=TRUE}
# # Grad.Rate
# data_frame(x = clg_gam_terms$`bs(Grad.Rate, degree = 3)`$x,
#            y = clg_gam_terms$`bs(Grad.Rate, degree = 3)`$y,
#            se.fit = clg_gam_terms$`bs(Grad.Rate, degree = 3)`$se.y) %>%
#   mutate(y_low = y - 1.96 * se.fit,
#          y_high = y + 1.96 * se.fit) %>%
#   ggplot(aes(x, y)) +
#   geom_line() +
#   geom_line(aes(y = y_low), linetype = 2) +
#   geom_line(aes(y = y_high), linetype = 2) +
#   labs(title = "GAM of Out-of-state Tuition",
#        x = "Grad.Rate",
#        y = expression(f[4](Grad.Rate)))
```

4. Testing Model
----------------

``` r
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}
mse_model1 <- mse(part3.model1, part3_split$test)
mse_model2 <- mse(part3.model2, part3_split$test)
mse_model1
```

    ## [1] 3595260

``` r
mse_model2
```

    ## [1] 3617093

The mse from OLS is 3787199, and the mse from gam is 3750909. So the mse from gam is smaller, we could know that the gam modal fit the data better. However, the difference is not very big, so perhaps the model has overfitting problem.

5. Non-linear Relationship with the response
--------------------------------------------

``` r
# Room.Board
gam_no_rb <- gam(Outstate ~  Private + lo(PhD) + lo(perc.alumni) + poly(Expend, 4) + poly(Grad.Rate, 3), data = part3_split$train)

gam_lin_rb <- gam(Outstate ~ Private + Room.Board + lo(PhD) + lo(perc.alumni) + poly(Expend, 4) + poly(Grad.Rate,3), data = part3_split$train)

# PhD
gam_no_phd <- gam(Outstate ~ Private + lo(Room.Board) + lo(perc.alumni) + poly(Expend,4) + poly(Grad.Rate, 3), data = part3_split$train)

gam_lin_phd <- gam(Outstate ~ Private + lo(Room.Board) + PhD + lo(perc.alumni) + poly(Expend,4) +  poly(Grad.Rate,3), data = part3_split$train)

## perc.alumni
gam_no_pa <- gam(Outstate ~ Private + lo(Room.Board) + lo(PhD) + poly(Expend, 4) + poly(Grad.Rate, 3), data = part3_split$train)

gam_lin_pa <- gam(Outstate ~ Private + lo(Room.Board) + lo(PhD) + perc.alumni + poly(Expend, 4) + poly(Grad.Rate, 3), data = part3_split$train)

# Expend
gam_no_expend<- gam(Outstate ~Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + poly(Grad.Rate, 3), data = part3_split$train)

gam_lin_expend <- gam(Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + Expend + poly(Grad.Rate,3), data = part3_split$train)


##Grad.Rate
gam_no_gr <- gam(Outstate ~ Private + lo(PhD) + lo(Room.Board) + lo(perc.alumni) + poly(Expend, 4) , data = part3_split$train)

gam_lin_gr <- gam(Outstate ~ Private + lo(Room.Board) + PhD + lo(perc.alumni) + poly(Expend, 4) + Grad.Rate, data = part3_split$train)
```

We need to know the new model that omits predictor and the GAM model that have the linear predictor. We will not include the Private here because it's binary. Then we conduct ANOVA test among the three model: 1.full model 2. linear predictor model 3. omits predictor model

``` r
anova(gam_no_rb, gam_lin_rb, part3.model2)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: Outstate ~ Private + lo(PhD) + lo(perc.alumni) + poly(Expend, 
    ##     4) + poly(Grad.Rate, 3)
    ## Model 2: Outstate ~ Private + Room.Board + lo(PhD) + lo(perc.alumni) + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ## Model 3: Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ##   Resid. Df Resid. Dev     Df  Deviance  Pr(>Chi)    
    ## 1    527.87 2101046439                               
    ## 2    526.87 1895085298 1.0000 205961141 2.963e-14 ***
    ## 3    523.89 1868126040 2.9792  26959258   0.05514 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
anova(gam_no_phd, gam_lin_phd, part3.model2)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: Outstate ~ Private + lo(Room.Board) + lo(perc.alumni) + poly(Expend, 
    ##     4) + poly(Grad.Rate, 3)
    ## Model 2: Outstate ~ Private + lo(Room.Board) + PhD + lo(perc.alumni) + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ## Model 3: Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ##   Resid. Df Resid. Dev     Df Deviance Pr(>Chi)  
    ## 1    527.58 1906239295                           
    ## 2    526.58 1890581737 1.0000 15657558  0.03613 *
    ## 3    523.89 1868126040 2.6927 22455697  0.07834 .
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
anova(gam_no_pa, gam_lin_pa, part3.model2)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: Outstate ~ Private + lo(Room.Board) + lo(PhD) + poly(Expend, 
    ##     4) + poly(Grad.Rate, 3)
    ## Model 2: Outstate ~ Private + lo(Room.Board) + lo(PhD) + perc.alumni + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ## Model 3: Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ##   Resid. Df Resid. Dev     Df Deviance  Pr(>Chi)    
    ## 1    527.33 1948119869                              
    ## 2    526.33 1878966270 1.0000 69153599 1.064e-05 ***
    ## 3    523.89 1868126040 2.4363 10840229    0.2898    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
anova(gam_no_expend, gam_lin_expend, part3.model2)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + 
    ##     poly(Grad.Rate, 3)
    ## Model 2: Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + 
    ##     Expend + poly(Grad.Rate, 3)
    ## Model 3: Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ##   Resid. Df Resid. Dev Df  Deviance  Pr(>Chi)    
    ## 1    527.89 2530158316                           
    ## 2    526.89 2198491434  1 331666882 < 2.2e-16 ***
    ## 3    523.89 1868126040  3 330365394 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
anova(gam_no_gr, gam_lin_gr, part3.model2)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: Outstate ~ Private + lo(PhD) + lo(Room.Board) + lo(perc.alumni) + 
    ##     poly(Expend, 4)
    ## Model 2: Outstate ~ Private + lo(Room.Board) + PhD + lo(perc.alumni) + 
    ##     poly(Expend, 4) + Grad.Rate
    ## Model 3: Outstate ~ Private + lo(Room.Board) + lo(PhD) + lo(perc.alumni) + 
    ##     poly(Expend, 4) + poly(Grad.Rate, 3)
    ##   Resid. Df Resid. Dev      Df Deviance Pr(>Chi)  
    ## 1    526.89 1966154024                            
    ## 2    528.58 1907523660 -1.6927 58630364           
    ## 3    523.89 1868126040  4.6927 39397620  0.04173 *
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

After the anova test, we could conclude that the Phd,and Grad.Rate are statistically significant in 2th model, so they have linear relationship. Phd and perc.alumni and Expend have non-linear relationshionship.
