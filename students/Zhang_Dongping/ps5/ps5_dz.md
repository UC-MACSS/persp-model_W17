MACS 30100: Problem Set 5
================
Dongping Zhang
2/10/2017

-   [I. Describe the data:](#i.-describe-the-data)
-   [II. Simple Linear Regression Model](#ii.-simple-linear-regression-model)
-   [III. Multiple Linear Regression (Part I)](#iii.-multiple-linear-regression-part-i)
-   [IV. Multiple Linear Regression (Part II)](#iv.-multiple-linear-regression-part-ii)
-   [V. Interactive Linear Regression](#v.-interactive-linear-regression)

I. Describe the data:
=====================

Plot a histogram of biden with a binwidth of 1. Make sure to give the graph a title and proper x and y-axis labels. In a few sentences, describe any interesting features of the graph.

-   Load the `biden.csv` dataset

``` r
biden <- read.csv('biden.csv')
```

-   Plot a histogram of biden with a binwidth of `1`.

``` r
ggplot(biden, aes(x = biden)) + geom_histogram(aes(y = ..density..), binwidth = 1) + 
  ggtitle("Histogram of Biden Plot") +
  labs(x = "Thermometer Scale (in degree)", y = "Percent of observations in bin") +
  scale_x_continuous(breaks = seq(0, 100, by = 10)) +
  theme(plot.title = element_text(hjust = 0.5))
```

![](ps5_dz_files/figure-markdown_github/plot%20biden-1.png)

According to the histogram above, the sample population in general has favorable and warm feelings toward Biden, because most of the ratings are falling between 50 degrees and 100 degrees. The tallest bin is at 50 degrees, meaning about 20% of the sample population is indifferent toward Biden. One feature of the dataset that makes it interesting is that most of the survey respondents would select multiples of 5 when responding to this question.

II. Simple Linear Regression Model
==================================

-   Run a simple linear regression of `biden` on `age`

``` r
simple_lm = lm(biden~age, data = biden)
summary(simple_lm)
```

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

**1. Is there a relationship between the predictor and the response?** <br /> In order to determine whether there is a relationship between the predictor and the response, I would refer to the p-value of the `age` variable. Becasue the p-value for the `age` variable is 0.0563, it implies that the probability of observing *β*<sub>1</sub> value equals to 0.06241, or larger, is 0.0563. This probability is greater using a 5% significance level, so we fail to reject *H*<sub>0</sub> and thus can claim that the effect of the `age` variable is indeed 0.

**2. How strong is the relationship between the predictor and the response?** <br /> The relationship between `age` and `biden` is that 1 unit increase of `age` would lead to an increase of response variable `biden` by 0.06241 degree on average. Thus, we can conclude that the relationship between the predictor and the response is not strong at all.

**3. Is the relationship between the predictor and the response positive or negative?** <br /> The relationship between the predictor and the response, or the coefficient of the `age` variable, is 0.06241, which has a positive effect, meaning an increase in the predictor would likely to cause an increase in the response.

**4. Report the *R*<sup>2</sup> of the model. What percentage of the variation in biden does age alone explain? Is this a good or bad model?**

``` r
summary(simple_lm)$r.squared
```

    ## [1] 0.002017624

<br /> The *R*<sup>2</sup> of the model is 0.002018, meaning that 0.2018% of variability in `biden` can be explained by using `age`. This low *R*<sup>2</sup> statistic indicates that this regression model did not explain much of the variability in the response because the model might be wrong, or the inherent error *σ*<sup>2</sup> is high, or both.

**5. What is the predicted biden associated with an age of 45? What are the associated 95% confidence intervals?** <br /> The predicted `biden` associated with an age of 45 is 62.00581, and the associated 95% confidence interval is (60.91248, 63.09872).

``` r
(pred_ci45 <- augment(simple_lm, 
                      newdata = data_frame(age = c(45))) %>%
   mutate(ymin = .fitted - .se.fit * 1.96,
          ymax = .fitted + .se.fit * 1.96))
```

    ##   age .fitted   .se.fit     ymin     ymax
    ## 1  45 62.0056 0.5577123 60.91248 63.09872

**6. Plot the response and predictor. Draw the least squares regression line**

``` r
# create a dataframe
biden_grid <-  biden %>%
  data_grid(age) %>%
  add_predictions(simple_lm)

# plotting
ggplot(biden, aes(age)) +
  geom_point(aes(y = biden)) +
  geom_line(aes(y = pred), data = biden_grid, color = "red", size = 1) + 
  ggtitle("Least Square Regression Line: Biden on Age") +
  labs(x = "Age", y = "Thermometer Scale (in degree)") +
  scale_x_continuous(breaks = seq(0, 100, by = 10)) +
  scale_y_continuous(breaks = seq(0, 100, by = 10)) + 
  theme(plot.title = element_text(hjust = 0.5)) 
```

![](ps5_dz_files/figure-markdown_github/least%20square%20regression%20line-1.png)

III. Multiple Linear Regression (Part I)
========================================

-   Run a multiple linear regression of `biden` on `age`, `female`, and `educ`

``` r
multiple_lm = lm(biden~age + female + educ, data = biden)
summary(multiple_lm)
```

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

**1. Is there a statistically significant relationship between the predictors and response?** <br /> According to the p-values of each regressors, `female` and `educ` are both statistically significant meaning there are linear relationships between those two regressors with the response because their p-vales, 1.86 × 10<sup>−8</sup> and 7.94 × 10<sup>−5</sup>, are both lower than the typically used 5% significance level. Meanwhile, `age` is still not statistically significant becasue its p-value, 0.198, is greater than the 5% significance level, thus we would fail to reject *H*<sub>0</sub> of *β*<sub>*A**g**e*</sub> = 0.

**2. What does the parameter for female suggest?** <br /> Because `female` is a dummy variable, its coefficient represents the average differences between females and males, ceteris paribus. According to the model above, the parameter for `female` suggests a female respondent would likely give a response 6.19607 degrees higher than a scores given by a male respondent on average, ceteris paribus.

**3. Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does `age`, `gender`, and `education` explain? Is this a better or worse model than the age-only model?**

``` r
(r2_comparison <- c("simple"   = summary(simple_lm)$r.squared,
                    "multiple" = summary(multiple_lm)$r.squared))
```

    ##      simple    multiple 
    ## 0.002017624 0.027227269

<br /> The *R*<sup>2</sup> of this multiple regression model is 0.02722727. The proportion of variability in `biden` that can be explained by those three variables is about 2.72%. Comparing the *R*<sup>2</sup> statistic of the age-only model with the multiple regression model, this later is certainly a better model.

**4.Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type. Is there a problem with this model? If so, what?** <br /> According to the residual vs. fitted values plot shown below, it is easily observable that the residuals are not centered at 0 and are differing by the parties the respondents affiliated to. The current model tends to overestimate the responses by Republicans while underestimate the responses by Democrats. This suggests we might want to add other variales to differentiate respondents by parties so as to make better predictions.

``` r
biden_stats <- augment(multiple_lm, biden) %>%
  mutate(rep = rep * 2, 
         party_id = factor(dem + rep)) %>%
  mutate(Party = factor(party_id, labels = c("Others", "Democrats", "Republicans"))) %>%
  mutate(Party = factor(Party, levels = rev(levels(Party))))

ggplot(biden_stats, aes(x = .fitted, y = .resid)) +
  geom_point(aes(color = Party), size = 0.3) + 
  geom_smooth(aes(color = Party), method = 'loess', size = 1) +
  labs(x = "Fitted Values",
       y = "Residuals",
       title = "Fitted Value vs. Residual Plot \n (Multiple Regression Model 1)") + 
    theme(plot.title = element_text(hjust = 0.5)) 
```

![](ps5_dz_files/figure-markdown_github/pred%20vals%20vs.%20residuals-1.png)

IV. Multiple Linear Regression (Part II)
========================================

-   Run a multiple linear regression of `biden` on `age`, `female`, `educ`, `dem`, and `rep`

``` r
multiple_lm2 = lm(biden~age + female + educ + dem + rep, data = biden)
summary(multiple_lm2)
```

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

**5. Did the relationship between gender and Biden warmth change?** <br /> The relationship between `female` and `biden` has changed and it has decreased from 6.19607 to 4.10323 but maintaining a positive effect.

**6. Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does `age`, `gender`, `education`, and `party` identification explain? Is this a better or worse model than the age + gender + education model?**

``` r
(r2_comparison2 <- c("simple"   = summary(simple_lm)$r.squared,
                     "multiple" = summary(multiple_lm)$r.squared,
                     "multiple2" = summary(multiple_lm2)$r.squared))
```

    ##      simple    multiple   multiple2 
    ## 0.002017624 0.027227269 0.281539147

<br /> The *R*<sup>2</sup> of this version multiple regression model is 0.281539147. So now, the proportion of variability in `biden` that can be explained by current selected regressors is about 28.15%. Comparing the *R*<sup>2</sup> statistic with the previous multiple regression model, the current model is definitely a lot and the prediction would be more reliable.

**7. Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type. By adding variables for party ID to the regression model, did we fix the previous problem?** <br /> According to the residual vs. fitted values plot shown below, the patterns of residuals across all parties are centering around 0, so the current multiple regression model has fixed, or at leaset has improved, the previous problems by adding dummies of parties. Nevertheless, we are still able to see that comparing with "Others", "Republicans" and "Democrates" still has fluctuating residuals, and it implies that there might be some other variables out there not included but can potentially improve the current model.

``` r
biden_stats <- augment(multiple_lm2, biden) %>%
  mutate(rep = rep * 2, 
         party_id = factor(dem + rep)) %>%
  mutate(Party = factor(party_id, labels = c("Others", "Democrats", "Republicans"))) %>%
  mutate(Party = factor(Party, levels = rev(levels(Party))))

ggplot(biden_stats, aes(x = .fitted, y = .resid)) +
  geom_point(aes(color = Party), size = 0.3) + 
  geom_smooth(aes(color = Party), method = 'loess', size = 1) +
  labs(x = "Fitted Values",
       y = "Residuals",
       title = "Fitted Value vs. Residual Plot \n (Multiple Regression Model 2)") + 
    theme(plot.title = element_text(hjust = 0.5)) 
```

![](ps5_dz_files/figure-markdown_github/pred%20vals%20vs.%20residuals%202-1.png)

V. Interactive Linear Regression
================================

-   Run a multiple linear regression of `biden` on `female`, `dem`, and `female` × `dem`

``` r
dem_rep = biden[biden$dem == 1 | biden$rep == 1, ]
interactive_lm = lm(biden~female*dem, data = dem_rep)
summary(interactive_lm)
```

    ## 
    ## Call:
    ## lm(formula = biden ~ female * dem, data = dem_rep)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -75.519 -13.070   4.223  11.930  55.618 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   39.382      1.455  27.060  < 2e-16 ***
    ## female         6.395      2.018   3.169  0.00157 ** 
    ## dem           33.688      1.835  18.360  < 2e-16 ***
    ## female:dem    -3.946      2.472  -1.597  0.11065    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 19.42 on 1147 degrees of freedom
    ## Multiple R-squared:  0.3756, Adjusted R-squared:  0.374 
    ## F-statistic:   230 on 3 and 1147 DF,  p-value: < 2.2e-16

**1. Estimate predicted Biden warmth feeling thermometer ratings and 95% confidence intervals for female Democrats, female Republicans, male Democrats, and male Republicans. Does the relationship between party ID and Biden warmth differ for males/females? Does the relationship between gender and Biden warmth differ for Democrats/Republicans?**

``` r
(pred_ci_party <- augment(interactive_lm, 
                          newdata = data.frame(female = c(1, 0, 1, 0), dem = c(1, 1, 0, 0))) %>%
   mutate(ymin = .fitted - .se.fit * 1.96,
          ymax = .fitted + .se.fit * 1.96))
```

    ##   female dem  .fitted   .se.fit     ymin     ymax
    ## 1      1   1 75.51883 0.8881114 73.77813 77.25953
    ## 2      0   1 73.06954 1.1173209 70.87959 75.25949
    ## 3      1   0 45.77720 1.3976638 43.03778 48.51662
    ## 4      0   0 39.38202 1.4553632 36.52951 42.23453

<br /> The relationship between party ID and Biden warmth differ for males/females. Female Democrats would return a response of 29.74 units more than female Republicans on average while male democrats would return a response of 33.68752 units more than male republicans on average. At the same time, there are also differences between gender and Biden warmth for Democrats/Republicans. Female Democratics would return a response of 2.44929 degrees higher than male Democratics on average while female Republicans would return a response of 6.39518 degrees higher than male Republicans on average. In conclusion, we could claim using these statistics that Democrats favor Biden regardless of gender, but females, regardless of political party, favor Biden more than males.
