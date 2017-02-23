MACS 30100 PS6
================
Erin M. Ochoa
2017 February 20

-   [Part 1: Modeling voter turnout](#part-1-modeling-voter-turnout)
    -   [Describing the data](#describing-the-data)
    -   [Estimating a basic logistic model](#estimating-a-basic-logistic-model)
    -   [Multiple variable model](#multiple-variable-model)
-   [Part 2: Modeling television consumption](#part-2-modeling-television-consumption)
    -   [Estimate a regression model](#estimate-a-regression-model)

Part 1: Modeling voter turnout
==============================

We begin by reading in the data. Because it does not make sense to consider respondents for whom either voting behavior or depression index score is missing, we subset the data to include only those respondents with valid responses in both variables. We also add a factor-type variable to describe voting; this will decrease the time necessary to construct plots.

``` r
df = read.csv('data/mental_health.csv')
df = df[(!is.na(df$vote96) & !is.na(df$mhealth_sum)), ]
df$Turnout = factor(df$vote96, labels = c("Did not vote", "Voted"))
```

Describing the data
-------------------

We plot a histogram of voter turnout:

![](ps6-emo_files/figure-markdown_github/vote_histogram-1.png)

The unconditional probability of a given respondent voting in the election is 0.672466. The data are distributed bimodally, with about twice as many respondents voting as not voting.

We generate a scatterplot of voter turnout versus depression index score, with points colored by whether the respondent voted. Because mental health scores are integers ranging from \[0,16\] and turnout is categorical, there can be a maximum of 34 points on the plot. This would not be terribly informative, so we jitter the points, increase their transparency, and add a horizontal line between the distributions of voters and non-voters; however, we must remember that the jittered position is not the true position: we must imagine the same number of points, with all the blue ones at 1 and all the red ones at 0. (That is: any within-group variability in the y direction is false.) These additions are somewhat helpful, but because voter turnout is dichotomous, it is not well suited to a scatterplot. (We will address this soon.)

![](ps6-emo_files/figure-markdown_github/scatterplot_turnout_vs_mentalhealth-1.png)

The regression line shows that respondents with higher depression scores trend toward not voting. We note again, however, that because voter turnout is dichotomous—a respondent either votes (1) or doesn't (0), with no possile outcomes in between—the regression line is misleading. It suggests, for example, that potential respondents with scores so high that they are off the index could have a negative chance of voting, which makes no sense; similarly, respondents with scores well below zero could have greater than a 1.0 chance of voting. Additionally, because the depression index score ranges from \[0,16\], it does not have support over the entire domain of real numbers; the regression line, however, suggests that such scores are possible and points to probabilites for them—and some of those probabilites fall outside of the realm of possible probabilities (which range from \[0,1\]). These problems imply that linear regression is the wrong type of analysis for the type of data with which we are dealing.

We now return to the matter of visualizing the distribution of depression scores by voter turnout. Because the outcome is dichotomous and the predictor is continuous but over a short interval, the scatterplot does a poor job of clearly showing the correlation between depression score and turnout. We therefore turn to a density plot:

![](ps6-emo_files/figure-markdown_github/density_plot-1.png)

Now we can clearly see that voters and non-voters had very different distributions of depression scores: most voters scored between \[0,5\] on the depression index, but only about half of non-voters did. While there were voters and non-voters over nearly the entire range of possible depression scores, non-voters tended to have higher scores (but the only respondents to score 16 on the depression index were actually voters).

Estimating a basic logistic model
---------------------------------

First, we define the functions necessary in this section:

``` r
PRE = function(model){
  y = model$y
  y.hat = round(model$fitted.values)
  
  E1 = sum(y != median(y))
  E2 = sum(y != y.hat)
  
  PRE = (E1 - E2) / E1
  return(PRE)
}

logit2prob = function(x){
  exp(x) / (1 + exp(x))
}

prob2odds = function(x){
  x / (1 - x)
}

prob2logodds = function(x){
  log(prob2odds(x))
}

calcodds = function(x){
  exp(int + coeff * x)
}

oddsratio = function(x,y){
  exp(coeff * (x - y))
}

calcprob = function(x){
  exp(int + coeff * x) / (1 + exp(int + coeff * x))
}

firstdifference = function(x,y){
  calcprob(y) - calcprob(x)
}

threshold_compare = function(thresh, dataframe, model){
  pred <- dataframe %>%
          add_predictions(model) %>%
          mutate(pred = logit2prob(pred),
          pred = as.numeric(pred > thresh))
  
  cm = confusionMatrix(pred$pred, pred$vote96, dnn = c("Prediction", "Actual"), positive = '1')

  data_frame(threshold = thresh,
             sensitivity = cm$byClass[["Sensitivity"]],
             specificity = cm$byClass[["Specificity"]],
             accuracy = cm$overall[["Accuracy"]])
}
```

We estimate a logistic regression model of the relationship between mental health and voter turnout:

``` r
logit_voted_depression = glm(vote96 ~ mhealth_sum, family = binomial, data=df)
summary(logit_voted_depression)
```

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum, family = binomial, data = df)
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
    ## AIC: 1620.7
    ## 
    ## Number of Fisher Scoring iterations: 4

We generate the additional dataframes and variables necessary to continue:

``` r
int = tidy(logit_voted_depression)[1,2]
coeff = tidy(logit_voted_depression)[2,2]

voted_depression_pred = df %>%
                        add_predictions(logit_voted_depression) %>%
                        mutate(prob = logit2prob(pred)) %>%
                        mutate(odds = prob2odds(prob))

voted_depression_accuracy = df %>%
                            add_predictions(logit_voted_depression) %>%
                            mutate(pred = logit2prob(pred),
                            pred = as.numeric(pred > .5))
```

We find a statistically significant relationship at the p &lt; .001 level between depression score and voting behavior. The relationship is negative and the coefficient is -0.1434752. Because the model is not linear, we cannot simply say that a change in depression index score results in a corresponding change in voter turnout. Instead, we must interpret the coefficient in terms of log-odds, odds, and probability. We will interpret the coefficient thus in the following responses.

Log-odds: For every one-unit increase in depression score, we expect the log-odds of voting to decrease by -0.1434752. Note that this is the mhealth\_sum coefficient from the logistic model summarized above. This is because the logistic model describes the linear change in log-odds as a function of the intercept (1.1392097) and of depression index score (as multiplied by the corresponding coefficient, -0.1434752).

We graph the relationship between mental health and the log-odds of voter turnout:

![](ps6-emo_files/figure-markdown_github/log_odds_plot-1.png)

Odds: The coefficient for depression index score cannot be interpreted in terms of odds without being evaluated at a certain depression score. This is because the relationship between depression score and odds is logistic, not linear: ![](./eq1.png)

For example, for a respondent with a depression index score of 12 would have odds of voting equal to 0.5585044. This means the respondent is 0.5585044 times more likely to vote than not vote; because this is less than one, such a respondent would be unlikely to vote. In contrast, a respondent with a depression index score of 3 would be 2.0315196 times more likely to vote than not vote. A respondent with a depression score of 8 would be approximately just as likely to vote as not vote because the odds for that score equal 0.9914448.

We graph the relationship between depression index score and the odds of voting:

![](ps6-emo_files/figure-markdown_github/voted_depression_odds_plot-1.png)

Probability: The relationship between depression index score and voting is not linear; like with odds, we must use a specific depression index score in order to calculate the probability of such a respondent voting. For example, a respondent with a depression index score of 3 would have a probability of voting equal to 0.6701324 and a respondent who scored 12 would have a probability of 0.3583592. As we noted earlier, a respondent with a score of 8 would be about equally likely to vote as not vote, with a probability of 0.497852.

The first difference for an increase in the depression index score from 1 to 2 is -0.0291782; for 5 to 6, it is -0.0347782. This means that as depression index score increases from 1 to 2, the probability of voting decreases by 0.0291782; for an increase from 5 to 6, the probability of voting decreases by 0.0347782

We plot the probabilty of voting against depression score, including the actual responses as points (this time, without jitter):

![](ps6-emo_files/figure-markdown_github/logit_voted_depression_prob_plot-1.png)

We define the variables necessary to answer the next question:

``` r
ar = mean(voted_depression_accuracy$vote96 == voted_depression_accuracy$pred, na.rm = TRUE)

uc = median(df$vote96)

cm.5_voted_depression <- confusionMatrix(voted_depression_accuracy$pred, voted_depression_accuracy$vote96,
                         dnn = c("Prediction", "Actual"), positive = '1')

cm.5_table = cm.5_voted_depression$table

actlpos = cm.5_table[1,2] + cm.5_table[2,2]
predposcrrct = cm.5_table[2,2]

actlneg = cm.5_table[1,1] + cm.5_table[2,1]
prednegcrrct = cm.5_table[1,1]

tpr.notes =  predposcrrct / actlpos
tnr.notes =  prednegcrrct / actlneg

tpr.cm.5 = sum(cm.5_voted_depression$byClass[1])
tnr.cm.5 = sum(cm.5_voted_depression$byClass[2])

threshold_x = seq(0, 1, by = .001) %>%
              map_df(threshold_compare, df, logit_voted_depression)

auc_x_voted_depression <- auc(voted_depression_pred$vote96, voted_depression_pred$prob)
```

Using a threshold value of .5, we estimate the accuracy rate of the logistic model at 0.677761.

We find that the useless classifier for this data predicts that all voters will vote; because the voter variable is dichotomous, we find this by simply taking the median of the distribution: 1.

With the useless classifier (which predicts all respondents will vote), we find that the proportional reduction in error is 0.0161663. This means that the model based only on depression index scores provides an improvement in the proportional reduction in error of 1.6166282% over the useless-classifier model.

The AUC for this model is 0.6243087.

This model's performance is poor, but we temper that by considering that it uses only one predictor. We expect to improve the model by including additional predictors.

For good measure we plot the accuracy, sensitivity, and specificy rates for thresholds between 0 and 1:

![](ps6-emo_files/figure-markdown_github/ar_vs_threshold_plot-1.png)

This plot suggests that using a threshold of approximately .7 would strike a good balance between sensitivity and specificity, but at the slight expense of accuracy. Interestingly, the accuracy curve is roughly flat below thresholds below approximately .67 and then drops steeply before leveling out at a threshold of approximately .75. The accuracy curve shows that regardless of the threshold, this model can never achieve accuracy much greater than 0.677761.

We also plot the ROC curve:

![](ps6-emo_files/figure-markdown_github/roc_plot-1.png)

We can see that the ROC curve falls above the graph's diagonal, which means that the model performs better than simply guessing, but not by much. The area under the curve is 0.6243087, implying that model performance is poor. Given the moderate AUC (0.6243087) but very low PRE (0.0161663), we conclude that this model is weak.

Multiple variable model
-----------------------

Using depression index score, age, and income, we now estimate a multivariate logistic regression. We have chosen to use the depression index score because it alone yielded somewhat reasonable results, so we suppose that it will form a good basis for a more robust model. Age is important because it is common knowledge that voting participation increases with age; the same can be said for income. With these three variables, we hope to achieve higher rates of sensitivity and sprecificy as well as a higher AUC compared to the bivariate model.

Here we define the three components of the GLM for the model to be constructed:

—Probability distribution (random component): Because we are using logistic regression, we assume that the outcome (y: voted or did not vote) is drawn from the Bernoulli distribution, with probability *π*:

![](./eq2.png)

—Linear predictor: The linear predictor is the following multivariate linear model:

![](./eq3.png)

—Link function: The link function is the logit function:

![](./eq4.png)

We estimate the model:

``` r
logit_voted_mv = glm(vote96 ~ mhealth_sum + age + inc10, family = binomial, data=df)
summary(logit_voted_mv)
```

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum + age + inc10, family = binomial, 
    ##     data = df)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.2470  -1.1170   0.6101   0.8653   1.7640  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -0.953205   0.238966  -3.989 6.64e-05 ***
    ## mhealth_sum -0.107597   0.022582  -4.765 1.89e-06 ***
    ## age          0.032162   0.004355   7.384 1.53e-13 ***
    ## inc10        0.143886   0.023571   6.104 1.03e-09 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1472.2  on 1167  degrees of freedom
    ## Residual deviance: 1314.5  on 1164  degrees of freedom
    ##   (154 observations deleted due to missingness)
    ## AIC: 1322.5
    ## 
    ## Number of Fisher Scoring iterations: 4

We generate the additional dataframes and variables necessary to answer the question:

``` r
b0_mv = tidy(logit_voted_mv)[1,2]
b1_mv = tidy(logit_voted_mv)[2,2]
b2_mv = tidy(logit_voted_mv)[3,2]
b3_mv = tidy(logit_voted_mv)[4,2]

voted_mv_pred = df[(!is.na(df$age) & !is.na(df$inc10)), ] %>%
                data_grid(mhealth_sum, age = c(25,65), inc10 = c(5,10,15)) %>%
                add_predictions(logit_voted_mv) %>%
                mutate(prob = logit2prob(pred)) %>%
                mutate(odds = prob2odds(prob)) %>%
                mutate(age = factor(age, levels = c(25, 65), labels = c("Younger", "Older")),
                       inc10 = factor(inc10, levels = c(5, 10, 15),
                                       labels = c("Low income", "Medium income", "High income")))# %>%
                #mutate(age_inc = interaction(factor(age), factor(inc10)))
  
voted_mv_accuracy <- df[(!is.na(df$age) & !is.na(df$inc10)), ] %>%
                     add_predictions(logit_voted_mv) %>%
                     mutate(pred = logit2prob(pred),
                     pred = as.numeric(pred > .5))
```

We also define some helpful functions:

``` r
calcodds_mv = function(mhealth,age,income10K){
  exp(b0_mv + b1_mv * mhealth + b2_mv * age + b3_mv * income10K)
  }

oddsratio_mv = function(x,y,n){
  if (n == 1){
    exp(b1_mv * (x - y))
  }
  else if (n == 2){
    exp(b2_mv * (x - y))
  }
  else if (n == 3){
    exp(b3_mv * (x - y))
  }
}

calcprob_mv = function(mhealth,age,income10K){
  power = b0_mv + b1_mv * mhealth + b2_mv * age + b3_mv * income10K
  exp(power) / (1 + exp(power))
}


firstdifference_mv = function(x,y,n){
  mh_med = median(df$mhealth_sum, na.rm = TRUE)
  age_med = median(df$age, na.rm = TRUE)
  inc_med = median(df$inc10, na.rm = TRUE)
  if(n == 3){
      calcprob_mv(mh_med,age_med,y) - calcprob_mv(mh_med,age_med,x)
  }
  else if (n == 1){
      calcprob_mv(y,age_med,inc_med) - calcprob_mv(x,age_med,inc_med)
  }
  else if(n == 2)
      calcprob_mv(mh_med,y,inc_med) - calcprob_mv(mh_med,x,inc_med)
  }
```

![](ps6-emo_files/figure-markdown_github/mv_log_odds_plot-1.png)

We can see that the log-odds lines are parallel to each other. Log-odds of voting compared to not voting decrease linearly as depression index scores increase; it is important to note, however, that the predicted log-odds of voting for older respondents will always be higher than those for younger respondents with similar income and depression index score. Additionally, respondents with higher incomes will always have higher log-odds of voting than similar respondents (in terms of age and depression index score) with lower incomes. For example, a 40-year-old respondent with a depression score of 8 and an income of $40,000 would have log-odds of voting of 0.0480321, while a 40-year-old respondent with a depression score of 8 and an income of $20,000 would have log-odds of voting of -0.23974.

![](ps6-emo_files/figure-markdown_github/voted_mv_odds_plot-1.png)

This plot is much more informative than the last one. Here, we can clearly see the logistic relationships by group between depression index score and odds of voting. Note that for younger respondents with low incomes, odds are always low and though they are negatively impacted by increasing depression index scores, the difference between the odds at a score of zero and at a score of 16 are minor (from approximately .4 to 2). Compare this with the odds curve for older high-income respondents: those with low depression index scores have very high odds of voting—approximately 27 for a depression index score of zero—but for those with a scores of 16, the odds decrease to approximately 5. Because the odds start off so much higher for this group, the odds decrease much more swiftly than those of respondents in other groups. Even for a depression index score of 16, though, older high-income respondents maintain higher odds of voting than respondents in any other category. In fact, an older high-income respondent with a depression index score of 16 has only slightly lower odds of voting than an older medium-income respondent with a score of 8. This difference is remarkable and indicates that age and income together have a substantive impact on the odds of voting.

The odds ratio for a respondent with depression score of 0 compared to 16 is 5.5931589.

![](ps6-emo_files/figure-markdown_github/logit_mv_prob_plot-1.png)

Of the three plots thus far, we find this one to paint the clearest picture of what is predicted to happen to voting behavior by group as depression index scores increase. Of all the curves, the one for older high-income respondents appears the most stable over the domain of depression index: as scores increase from 0 to 16, the probability of voting decreases by only approximately .13. Compare this to younger low-income voters, whose probabilty of voting decreases by .4 as scores increase from 0 to 16.

We can calculate the first difference for changes in one category by imputing the median values for the other two categories. For example, the first difference for a change in age from 30 to 55 is an increase of 0.1767575 in the probability of voting.

``` r
ar_mv = mean(voted_mv_accuracy$vote96 == voted_mv_accuracy$pred, na.rm = TRUE)

uc_mv = median(voted_mv_accuracy$vote96)

cm.5_mv = confusionMatrix(voted_mv_accuracy$pred, voted_mv_accuracy$vote96,
                         dnn = c("Prediction", "Actual"), positive = '1')

cm.5_mv_table = cm.5_mv$table

actlpos_mv = cm.5_mv_table[1,2] + cm.5_mv_table[2,2]
predposcrrct_mv = cm.5_mv_table[2,2]

actlneg_mv = cm.5_mv_table[1,1] + cm.5_mv_table[2,1]
prednegcrrct_mv = cm.5_mv_table[1,1]

tpr.notes_mv =  predposcrrct_mv / actlpos_mv
tnr.notes_mv =  prednegcrrct_mv / actlneg_mv

tpr.cm.5_mv = sum(cm.5_mv$byClass[1])
tnr.cm.5_mv = sum(cm.5_mv$byClass[2])

threshold_x_mv = seq(0, 1, by = .001) %>%
                 map_df(threshold_compare, df, logit_voted_mv)


auc_x_voted_mv = auc(voted_mv_accuracy$vote96, voted_mv_accuracy$pred)
```

The multivariate model has a proportional error reduction of `PRE(logit_voted_mv) * 100`% over the useless-classifier model. This is a greater reduction than we saw with the bivariate model.

![](ps6-emo_files/figure-markdown_github/amv_r_vs_threshold_plot-1.png)

By plotting the accuracy, sensitivity, and specificity rates, we see that while the curves are smoother than they were for the model based solely on depression index, the multivariate model does not improve upon the bivariate model.

![](ps6-emo_files/figure-markdown_github/mv_roc_plot-1.png)

This ROC curve does not appear to improve upon the earlier one. In fact, with AUC of 0.6029709 (compared to 0.6243087), it offers slightly worse performance. This is despite the fact that depression index score (with a coefficient of -0.1075965), age (with a coefficient of 0.0321616), and income (with a coefficient of 0.143886) are all statistically significant at the p&lt;.001 level.

Given the uninspiring performance of the multivariate model, we conclude that together, depression index score, age, and income significantly predict voter turnout, but not substantively so.

Part 2: Modeling television consumption
=======================================

We begin by reading in the data. Having chosen the variables of interest, we subset the dataframe such that all cases contain valid responses for all such variables. We also convert social\_connect to a factor variable and label the levels.

``` r
df2 = read.csv('data/gss2006.csv')
df2 = df2[(!is.na(df2$tvhours) & !is.na(df2$hrsrelax) & !is.na(df2$social_connect)), ]
df2$social_connect = factor(df2$social_connect, labels = c("Low", "Medium", "High"))
```

We have chosen to investigate the effect of relaxation hours and social connectedness on television consumption. We expect that relaxation hours will have a positive effect on television consumption, while social connectedness will have a negative effect.

Estimate a regression model
---------------------------

Using the other variables in the dataset, derive and estimate a multiple variable Poisson regression model of hours of TV watched.

1.  We define the three components of the GLM for the model of interest:

—Probability distribution (random component): Because we are creating a model of the number of hours of television that respondents watch, we use a Poisson distribution:

![](./eq5.png)

—Linear predictor: The linear preditor is the following log-linear regression model:

![](./eq6.png)

—Link function: The link function for the Poisson distribution is the log function:

![](./eq7.png)

We estimate a Poisson model to explain television consumption with leisure time and social connectedness:

``` r
poisson_tv <- glm(tvhours ~ hrsrelax + social_connect, family = "poisson", data = df2)
summary(poisson_tv)
```

    ## 
    ## Call:
    ## glm(formula = tvhours ~ hrsrelax + social_connect, family = "poisson", 
    ##     data = df2)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.8981  -0.9040  -0.2332   0.4077   6.4822  
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)          0.702974   0.038316  18.347  < 2e-16 ***
    ## hrsrelax             0.041523   0.006137   6.766 1.32e-11 ***
    ## social_connectMedium 0.067613   0.044401   1.523    0.128    
    ## social_connectHigh   0.066088   0.048398   1.366    0.172    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 1346.9  on 1119  degrees of freedom
    ## Residual deviance: 1301.0  on 1116  degrees of freedom
    ## AIC: 4178.4
    ## 
    ## Number of Fisher Scoring iterations: 5

We find that hours of relaxation is statistically significant at the p&lt;.001 level; it has a coefficient of 0.0415226, which means that an increase of one hour in relaxation time for a low-connected respondent will result in a 0.0415226-fold increase in the mean number of television hours consumed. This effect is statistically significant, but does not seem substantively significant.

Contrary to our earlier expectation, social connectedness is not statistically significant.

![](ps6-emo_files/figure-markdown_github/poisson_log_count_plot-1.png)

The predicted log-count of television hours increases linearly with hours of relaxation. Additionally, respondents with medium and high social connectedness have the same predicted log-count of television consumption, which is in all cases higher than the predictions for low-connected respondents.

![](ps6-emo_files/figure-markdown_github/poisson_tv_predicted_count_plot-1.png)

We can see that the predicted count of television hours increases with hours of relaxation. Additionally, and somewhat counter to what we expected, respondents with medium and high social connectedness have higher television consumption than respondents with low connectedness. This suggests that television watching occurs more frequently as a social event than it does as an individual activity.

In order to test for under- or over-dispersion, we estimate a quasipoisson model with the same variables:

``` r
quasipoisson_tv <- glm(tvhours ~ hrsrelax + social_connect, family = "quasipoisson", data = df2)
summary(quasipoisson_tv)
```

    ## 
    ## Call:
    ## glm(formula = tvhours ~ hrsrelax + social_connect, family = "quasipoisson", 
    ##     data = df2)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.8981  -0.9040  -0.2332   0.4077   6.4822  
    ## 
    ## Coefficients:
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)          0.702974   0.044485  15.803  < 2e-16 ***
    ## hrsrelax             0.041523   0.007125   5.828 7.34e-09 ***
    ## social_connectMedium 0.067613   0.051549   1.312     0.19    
    ## social_connectHigh   0.066088   0.056189   1.176     0.24    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for quasipoisson family taken to be 1.347905)
    ## 
    ##     Null deviance: 1346.9  on 1119  degrees of freedom
    ## Residual deviance: 1301.0  on 1116  degrees of freedom
    ## AIC: NA
    ## 
    ## Number of Fisher Scoring iterations: 5

The dispersion parameter for the quasipoisson distriubution is 1.347905; this indicates that the model is over-dispersed (the true variance of the distribution is greater than its mean) and that therefore the Poisson distribution is not the most appropriate random component for the model.
