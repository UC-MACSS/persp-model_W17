PS6
================
Cheng Yee Lim
17th February 2017

Part 1: Modeling Voter Turnout
------------------------------

### Describe the data (1 point)

#### Plot a histogram of voter turnout. Make sure to give the graph a title and proper *x* and *y*-axis labels. What is the unconditional probability of a given individual turning out to vote?

``` r
health %>% 
  filter(!is.na(vote96)) %>%
  ggplot() + 
  geom_bar(aes(x = factor(vote96)), fill = "deepskyblue1") + 
  labs(x = "Voter Turnout", 
       y = "Count") + 
  scale_x_discrete(breaks = c(0,1), 
                   labels = c("Did not vote", "Voted"))
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-1-1.png) Generate a scatterplot of the relationship between mental health and observed voter turnout and overlay a linear smoothing line. What information does this tell us? What is problematic about this linear smoothing line?

``` r
health %>%
  filter(!is.na(mhealth_sum) & !is.na(vote96)) %>%
  ggplot(aes(x = mhealth_sum, y = vote96)) + 
  geom_point() + 
  geom_smooth(method = "lm", color = "deepskyblue1") + 
  labs(x = "Mental Health Index", 
       y = "Voter Turnout")
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-2-1.png)

### Basic model (3 points)

Estimate a logistic regression model of the relationship between mental health and voter turnout.

*v**o**t**e*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*h**e**a**l**t**h*<sub>*i*</sub>

``` r
m_voter <- glm(vote96 ~ mhealth_sum, data = health, family = binomial)
summary(m_voter)
```

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum, family = binomial, data = health)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.6834  -1.2977   0.7452   0.8428   1.6911  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  1.13921    0.08444  13.491  < 2e-16 ***
    ## mhealth_sum -0.14348    0.01969  -7.289 3.13e-13 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1672.1  on 1321  degrees of freedom
    ## Residual deviance: 1616.7  on 1320  degrees of freedom
    ##   (1510 observations deleted due to missingness)
    ## AIC: 1620.7
    ## 
    ## Number of Fisher Scoring iterations: 4

#### Is the relationship between mental health and voter turnout statistically significant?

The relationship between mental health and voter turnout has a p-value of 3.13 \* 10<sup>−13</sup> is statistically significant at 1% significance level.

#### Interpret the estimated parameter for mental health in terms of log-odds.

For every one-unit increase in an individual's mental health (where 0 is an individual with no depressed feelings, and 9 is an individual with the most severe depressed mood), we expect the log-odds of voting to decrease by 0.143.

#### Generate a graph of the relationship between mental health and the log-odds of voter turnout.

``` r
prob2odds <- function(x){
  x / (1 - x)
}

prob2logodds <- function(x){
  log(prob2odds(x))
}

health <- health %>%
  filter(!is.na(mhealth_sum) & mhealth_sum > 0) %>%
  add_predictions(m_voter) %>% 
  mutate(logodds = prob2logodds(pred), 
         odds = prob2odds(pred)) 

health %>% 
  ggplot(aes(x = mhealth_sum, y = logodds)) + 
  geom_line() + 
  labs(x = "Mental Health Index", 
       y = "Log-odds") 
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-4-1.png)

#### Interpret the estimated parameter for mental health in terms of odds. Generate a graph of the relationship between mental health and the odds of voter turnout.

The relationship between mental health and the odds of turning up to vote is 0.892.

``` r
exp(-0.114)
```

    ## [1] 0.892258

``` r
health %>% 
  ggplot(aes(x = mhealth_sum, y = odds)) + 
  geom_line() + 
  labs(x = "Mental Health Index", 
       y = "Odds") 
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-5-1.png)

#### Interpret the estimated parameter for mental health in terms of probabilities. Generate a graph of the relationship between mental health and the probability of voter turnout. What is the first difference for an increase in the mental health index from 1 to 2? What about for 5 to 6?

The expected change in probability given an unit increase in the mental health index is -0.0273.
The first difference for an increase in the mental health index from 1 to 2 is -0.0292.
The first difference for an increase in the mental health index from 5 to 6 is -0.0348.

``` r
health %>% 
  ggplot(aes(x = mhealth_sum, y = pred)) + 
  geom_line() + 
  labs(x = "Mental Health Index", 
       y = "Predicted Probabilities") 
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
probability <- function(x, y){ 
  exp(1.13921 - 0.14348*y)/(1 + exp(1.13921 - 0.14348*y)) - exp(1.13921 - 0.14348*x)/(1+exp(1.13921 - 0.14348*x))
}
cat("First Difference from an increase in mental health index = ", probability(1,2))
```

    ## First Difference from an increase in mental health index =  -0.0291793

``` r
cat("First Difference from an increase in mental health index = ", probability(5,6))
```

    ## First Difference from an increase in mental health index =  -0.03477953

#### Estimate the accuracy rate, proportional reduction in error (PRE), and the AUC for this model. Do you consider it to be a good model?

It is a decent model, it results in an improvement of

``` r
logit2prob <- function(x){
  exp(x) / (1 + exp(x))
} #logit2prob function

getmode <- function(v) {
   uniqv <- unique(v)
   uniqv[which.max(tabulate(match(v, uniqv)))]
} # determining best model 

mh_accuracy <- health %>%
  filter(!is.na(mhealth_sum & !is.na(vote96))) %>%
  add_predictions(m_voter) %>%
  mutate(pred2 = logit2prob(pred),
         pred2 = as.numeric(pred > .5))

mean(mh_accuracy$vote96 == mh_accuracy$pred2, na.rm = TRUE) #accuracy rate
```

    ## [1] 0.6680203

``` r
# function to calculate PRE for a logistic regression model
PRE <- function(model){
  # get the actual values for y from the data
  y <- model$y
  
  # get the predicted values for y from the model
  y.hat <- round(model$fitted.values)
  
  # calculate the errors for the null model and your model
  E1 <- sum(y != median(y))
  E2 <- sum(y != y.hat)
  
  # calculate the proportional reduction in error
  PRE <- (E1 - E2) / E1
  return(PRE)
}

PRE(m_voter)
```

    ## [1] 0.01616628

``` r
confusionMatrix(mh_accuracy$pred2, mh_accuracy$vote96)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 149 129
    ##          1 198 509
    ##                                           
    ##                Accuracy : 0.668           
    ##                  95% CI : (0.6376, 0.6974)
    ##     No Information Rate : 0.6477          
    ##     P-Value [Acc > NIR] : 0.0962707       
    ##                                           
    ##                   Kappa : 0.238           
    ##  Mcnemar's Test P-Value : 0.0001696       
    ##                                           
    ##             Sensitivity : 0.4294          
    ##             Specificity : 0.7978          
    ##          Pos Pred Value : 0.5360          
    ##          Neg Pred Value : 0.7199          
    ##              Prevalence : 0.3523          
    ##          Detection Rate : 0.1513          
    ##    Detection Prevalence : 0.2822          
    ##       Balanced Accuracy : 0.6136          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

### Multiple variable model (3 points)

Our GLM of voter turnout consists of three components:

Firstly, we assume our outcome variable, voter turnout, is drawn from the binomial distribution with probability *π*, given the values of the predicator variables in the model. *π* is the probability that, for any observation *i*, *Y* will take on the particular value *Y*<sub>*i*</sub>.
In our model, *Y*<sub>*i*</sub> takes on the expected value of 1 with probability *π* and 0 with probability 1 − *π*, so *π*<sub>*i*</sub> is the conditional probability of sampling a 1 in this group.

Pr(*Y*<sub>*i*</sub> = *y*<sub>*i*</sub>|*π*) = *π*<sub>*i*</sub><sup>*y*<sub>*i*</sub></sup> (1 − *π*<sub>*i*</sub>)<sup>1 − *y*<sub>*i*</sub></sup>

``` r
health %>% 
  filter(!is.na(vote96)) %>%
  ggplot() + 
  geom_histogram(aes(x = vote96, 
                     y = ..density..), 
                 fill = "deepskyblue1", 
                 na.rm = TRUE) + 
  labs(x = "Voter Turnout") + 
  scale_x_continuous(breaks = c(0,1), 
                   labels = c("Did not vote", "Voted")) #fix density
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-8-1.png)

Secondly, since the probability of voter turnout may systematically vary to given known predictors, we incorporate that into the model with a linear predictor. The linear predictor is:
*g*(*π*<sub>*i*</sub>)≡*ρ*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*h**e**a**l**t**h*<sub>*i*</sub> + *β*<sub>2</sub>*a**g**e*<sub>*i*</sub> + *β*<sub>3</sub>*e**d**u**c*<sub>*i*</sub> + *β*<sub>4</sub>*b**l**a**c**k*<sub>*i*</sub> + *β*<sub>5</sub>*f**e**m**a**l**e*<sub>*i*</sub> + *β*<sub>6</sub>*m**a**r**r**i**e**d*<sub>*i*</sub> + *β*<sub>7</sub>*i**n**c**o**m**e*<sub>*i*</sub> + *ϵ*<sub>*i*</sub>

Thirdly, we use a logit link function to constrain the linear predictor to the \[0,1\] range. A link function, *g*(*π*<sub>*i*</sub>) = *e*<sup>*ρ*<sub>*i*</sub></sup> / (1 + *e*<sup>*ρ*<sub>*i*</sub></sup>), transforms the expectation of the vector turnout to the linear predictor.

``` r
voter <- glm(vote96 ~ mhealth_sum + age + educ + black + female + married + inc10, data = health, family = binomial)

summary(voter)
```

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum + age + educ + black + female + 
    ##     married + inc10, family = binomial, data = health)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.4657  -1.0406   0.5544   0.8893   2.0390  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -3.918136   0.559168  -7.007 2.43e-12 ***
    ## mhealth_sum -0.103186   0.027633  -3.734 0.000188 ***
    ## age          0.040388   0.005378   7.510 5.90e-14 ***
    ## educ         0.211969   0.032250   6.573 4.94e-11 ***
    ## black        0.140554   0.220730   0.637 0.524276    
    ## female       0.053360   0.157010   0.340 0.733971    
    ## married      0.484264   0.173003   2.799 0.005124 ** 
    ## inc10        0.043238   0.029856   1.448 0.147550    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1146.29  on 884  degrees of freedom
    ## Residual deviance:  977.89  on 877  degrees of freedom
    ##   (178 observations deleted due to missingness)
    ## AIC: 993.89
    ## 
    ## Number of Fisher Scoring iterations: 4

Interpret the results in paragraph format. This should include a discussion of your results as if you were reviewing them with fellow computational social scientists. Discuss the results using any or all of log-odds, odds, predicted probabilities, and first differences - choose what makes sense to you and provides the most value to the reader. Use graphs and tables as necessary to support your conclusions.

In this multivariate logistic regression model, the response variable is the binary voter turnout variable where 1 means the respondent voted and 0 means the respondent did not vote. The predictors include the mental health index, age, education, race (Black or not), gender (female or not), marital status (married or not), and family income (in $10,000s). The regression results indicate that four of the coefficients are statistically significant; these coefficients are, respectively, -0.089102 for the mental health index, 0.042534 for age, 0.228686 for education and 0.069614 for income. These coefficients are given in terms of log-odds.

In terms of odds, hold other variables constant, a unit increase in the mental health index leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 0.9147523. Likewise, holding other variables constant, one year increase in age leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 1.043452. Again, holding other variables constant, one year increase in the number of years of formal education leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 1.256947. Finally, holding other variables constant, a unit increase in income leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 1.072094. In terms of predicted probabilities, these values correspond to, respectively, a multiplicative factor of 0.4777392 for each unit increase in the mental health index holding other variables constant, 0.510632 for age, 0.5569236 for educaiton, and 0.5173964 for income.

Part 2: Modelling TV Consumption
--------------------------------

``` r
gss <- read.csv("./data/gss2006.csv")
```

### Describe the data

`health` contains a subset of the 2006 General Social Survey of 4510 American individuals in 2006. The response variable `tvhours` contains a count of TV hours watched per day for each individual in 2006.

``` r
gss %>% 
  filter(!is.na(tvhours)) %>%
  ggplot() + 
  geom_histogram(aes(x = tvhours), fill = "deepskyblue1", binwidth=2) + 
  labs(x = "Number of hours of TV watched per day", 
       y = "Count", 
       title = "Histogram of TV hours watched per day") 
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-11-1.png)

### Multivariate Poisson Regression Model of Hours of TV Watched

Firstly, we assume our outcome variable, number of hours of TV watched, is drawn from a poisson distribution, where Pr(*T**V**h**o**u**r**s*<sub>*i*</sub> = *k*|*μ*) = (*μ*<sup>*k*</sup>*e*<sup>−*μ*</sup>) / *k*! .

In our model of predicting number of hours of TV watched per day, we include the following additional variables:
1. `age` - Age (in years)
2. `childs` - Number of children
3. `educ` - Highest year of formal schooling completed
4. `female` - 1 if female, 0 if male
5. `hrsrelax` - Hours per day respondent has to relax
6. `black` - 1 if respondent is black, 0 otherwise
7. `social_connect` - Ordinal scale of social connectedness, with values low-moderate-high (0-1-2)
8. `voted04` - 1 if respondent voted in the 2004 presidential election, 0 otherwise
9. `xmovie` - 1 if respondent saw an X-rated movie in the last year, 0 otherwise

Thus, the linear predictor of our model is:

*η*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*a**g**e* + *β*<sub>2</sub>*c**h**i**l**d**r**e**n* + *β*<sub>3</sub>*e**d**u**c* + *β*<sub>4</sub>*f**e**m**a**l**e* + *β*<sub>5</sub>*h**r**s**r**e**l**a**x* + *β*<sub>6</sub>*b**l**a**c**k*
+*β*<sub>7</sub>*s**o**c**i**a**l*<sub>*c*</sub>*o**n**n**e**c**t* + *β*<sub>8</sub>*v**o**t**e**d*04 + *β*<sub>9</sub>*x**m**o**v**i**e* + *ϵ*<sub>*i*</sub>

The link function for the poisson distribution is:
*μ*<sub>*i*</sub> = log(*η*<sub>*i*</sub>)

The results of the estimated multivariate poisson model is as follows:

``` r
tv_hours <- glm(tvhours ~ age + childs + educ + female + hrsrelax + black + social_connect + voted04 + xmovie, data = gss, family = poisson)
summary(tv_hours)
```

    ## 
    ## Call:
    ## glm(formula = tvhours ~ age + childs + educ + female + hrsrelax + 
    ##     black + social_connect + voted04 + xmovie, family = poisson, 
    ##     data = gss)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.9656  -0.7125  -0.0833   0.4486   5.3006  
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     1.001794   0.197884   5.063 4.14e-07 ***
    ## age             0.001022   0.002668   0.383   0.7016    
    ## childs         -0.005058   0.022149  -0.228   0.8194    
    ## educ           -0.027138   0.011410  -2.378   0.0174 *  
    ## female          0.012445   0.060887   0.204   0.8380    
    ## hrsrelax        0.046949   0.009637   4.872 1.11e-06 ***
    ## black           0.434400   0.069766   6.227 4.77e-10 ***
    ## social_connect  0.031811   0.037713   0.844   0.3989    
    ## voted04        -0.124845   0.070469  -1.772   0.0765 .  
    ## xmovie          0.095053   0.068068   1.396   0.1626    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 578.52  on 487  degrees of freedom
    ## Residual deviance: 480.02  on 478  degrees of freedom
    ##   (4022 observations deleted due to missingness)
    ## AIC: 1749.4
    ## 
    ## Number of Fisher Scoring iterations: 5

From the regression results, we can identify that only `educ`, `hrsrelax`, and `black` are statistically significant at a 5% significance level. Since we used a multivariate poisson regression, the effects of variables will be calculated as exponential of the estimated parameters.

``` r
exp(coef(tv_hours))
```

    ##    (Intercept)            age         childs           educ         female 
    ##      2.7231633      1.0010229      0.9949548      0.9732266      1.0125227 
    ##       hrsrelax          black social_connect        voted04         xmovie 
    ##      1.0480684      1.5440365      1.0323226      0.8826336      1.0997168

#### Education and Ethnicity

For a one year increase in maximum years of schooling, the number of TV hours watched per day is expected to decrease by 0.973, given the rest of the variables in the model are held constant. The decline in hours of TV watched per day as maximum education years increase can also be shown in by the overall decreasing trend in the line graphs below. This trend is also valid across black and non-black ethnicities

Furthermore, a black individual is, on average, expected to watch 1.54 more hours of TV per day than a non-black individual, given that the rest of the variables in the model are held constant. The differential between hours of TV watched by an average black and non-black individual can be visualized with the disparity between the blue and red lines.

``` r
gss %>%
  na.omit() %>%
  add_predictions(tv_hours) %>%
  group_by(educ, black) %>% 
  summarize(pred = mean(exp(pred))) %>%
  ggplot(aes(x = educ, y = pred)) +
  geom_line(aes(color = factor(black))) +
  labs(y = "Predicted Number of Hours of TV watched per day", 
       x = "Maximum Years of Education") + 
  scale_color_discrete(name = "Race", 
                     breaks = c(0,1), 
                     labels = c("Non-black", "Black"))
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-14-1.png)

#### Taste and Preferences

Unsurprisingly, according to the regression results, individuals who devote more time to relaxation per day are more likely to watch more hours of television. For an hour increase in hours the respondent has to relax, the number of TV hours watched per day is expected to increase by 1.048, given the rest of the variables in the model are held constant.

``` r
gss %>%
  na.omit() %>%
  add_predictions(tv_hours) %>%
  group_by(hrsrelax) %>% 
  summarize(pred = mean(exp(pred))) %>%
  ggplot(aes(x = hrsrelax, y = pred)) +
  geom_line() +
  labs(y = "Predicted Number of Hours of TV watched per day", 
       x = "Hours per day Respondent has to Relax")
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-15-1.png)

#### Over or under-dispersion

The dispersion parameter is 1.116. While the dispersion parameter is larger than 1, it is rather close to 1 so over-dispersion does not appear to be a significant problem.

``` r
tv_disp <- glm(tvhours ~ age + childs + educ + female + hrsrelax + black + social_connect + voted04 + xmovie, data = gss, family = "quasipoisson")
summary(tv_disp)
```

    ## 
    ## Call:
    ## glm(formula = tvhours ~ age + childs + educ + female + hrsrelax + 
    ##     black + social_connect + voted04 + xmovie, family = "quasipoisson", 
    ##     data = gss)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.9656  -0.7125  -0.0833   0.4486   5.3006  
    ## 
    ## Coefficients:
    ##                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     1.001794   0.206888   4.842 1.74e-06 ***
    ## age             0.001022   0.002789   0.366   0.7142    
    ## childs         -0.005058   0.023157  -0.218   0.8272    
    ## educ           -0.027138   0.011929  -2.275   0.0234 *  
    ## female          0.012445   0.063657   0.196   0.8451    
    ## hrsrelax        0.046949   0.010075   4.660 4.11e-06 ***
    ## black           0.434400   0.072940   5.956 5.03e-09 ***
    ## social_connect  0.031811   0.039429   0.807   0.4202    
    ## voted04        -0.124845   0.073675  -1.695   0.0908 .  
    ## xmovie          0.095053   0.071165   1.336   0.1823    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for quasipoisson family taken to be 1.093068)
    ## 
    ##     Null deviance: 578.52  on 487  degrees of freedom
    ## Residual deviance: 480.02  on 478  degrees of freedom
    ##   (4022 observations deleted due to missingness)
    ## AIC: NA
    ## 
    ## Number of Fisher Scoring iterations: 5
