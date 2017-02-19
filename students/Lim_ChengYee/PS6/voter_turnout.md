PS6
================
Cheng Yee Lim
17th February 2017

Describe the data (1 point)
---------------------------

1.  Plot a histogram of voter turnout. Make sure to give the graph a title and proper *x* and *y*-axis labels. What is the unconditional probability of a given individual turning out to vote?

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

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-1-1.png) 1. Generate a scatterplot of the relationship between mental health and observed voter turnout and overlay a linear smoothing line. What information does this tell us? What is problematic about this linear smoothing line?

``` r
health %>%
  filter(!is.na(mhealth_sum) & !is.na(vote96)) %>%
  ggplot(aes(x = mhealth_sum, y = vote96)) + 
  geom_point() + 
  geom_smooth(model = "lm", color = "deepskyblue1") + 
  labs(x = "Mental Health Index", 
       y = "Voter Turnout")
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-2-1.png)

Basic model (3 points)
----------------------

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

### Is the relationship between mental health and voter turnout statistically significant?

The relationship between mental health and voter turnout has a p-value of 3.13 \* 10<sup>−13</sup> is statistically significant at 1% significance level.

### Interpret the estimated parameter for mental health in terms of log-odds.

For every one-unit increase in an individual's mental health (where 0 is an individual with no depressed feelings, and 9 is an individual with the most severe depressed mood), we expect the log-odds of voting to decrease by 0.143.

### Generate a graph of the relationship between mental health and the log-odds of voter turnout.

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

### Interpret the estimated parameter for mental health in terms of odds. Generate a graph of the relationship between mental health and the odds of voter turnout.

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

### Interpret the estimated parameter for mental health in terms of probabilities. Generate a graph of the relationship between mental health and the probability of voter turnout. What is the first difference for an increase in the mental health index from 1 to 2? What about for 5 to 6?

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

### Estimate the accuracy rate, proportional reduction in error (PRE), and the AUC for this model. Do you consider it to be a good model?

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

Multiple variable model (3 points)
----------------------------------

Our GLM of voter turnout consists of three components:

Firstly, we assume our outcome variable, voter turnout, is drawn from the binomial distribution with probability *π*, given the values of the predicator variables in the model. *π* is the probability that, for any observation *i*, *Y* will take on the particular value *Y*<sub>*i*</sub>.
In our model, *Y*<sub>*i*</sub> takes on the expected value of 1 with probability *π* and 0 with probability 1 − *π*, so *π*<sub>*i*</sub> is the conditional probability of sampling a 1 in this group.

``` r
health %>% 
  filter(!is.na(vote96)) %>%
  ggplot() + 
  geom_histogram(aes(x = vote96, y = ..density..), fill = "deepskyblue1", na.rm = TRUE) + 
  labs(x = "Voter Turnout") + 
  scale_x_continuous(breaks = c(0,1), 
                   labels = c("Did not vote", "Voted")) #fix density
```

![](voter_turnout_files/figure-markdown_github/unnamed-chunk-8-1.png)

Secondly, since the probability of voter turnout may systematically vary to given known predictors, we incorporate that into the model with a linear predictor.

*π*<sub>*i*</sub> = *ρ*<sub>*i*</sub>

Thus, the linear predictor is:

*g*(*π*<sub>*i*</sub>)≡*ρ*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*h**e**a**l**t**h*<sub>*i*</sub> + *β*<sub>2</sub>*a**g**e*<sub>*i*</sub> + *β*<sub>3</sub>*e**d**u**c*<sub>*i*</sub> + *β*<sub>4</sub>*b**l**a**c**k*<sub>*i*</sub> + *β*<sub>5</sub>*f**e**m**a**l**e*<sub>*i*</sub> + *β*<sub>6</sub>*m**a**r**r**i**e**d*<sub>*i*</sub> + *β*<sub>7</sub>*i**n**c**o**m**e*<sub>*i*</sub> + *ϵ*<sub>*i*</sub>

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

Thirdly, we use a logit link function to constrain the linear predictor to the \[0,1\] range. A link function
$$g(\\pi\_i) = \\frac{e^{\\rho\_i}}{1+e^{\\rho\_i}}$$
 which transforms the expectation of the vector turnout to the linear predictor.