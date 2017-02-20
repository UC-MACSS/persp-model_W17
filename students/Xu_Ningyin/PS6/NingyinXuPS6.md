Xu\_Ningyin\_PS6: Generalized Linear Models
================
Ningyin Xu
2/16/2017

-   [Part 1: Modeling voter turnout](#part-1-modeling-voter-turnout)
    -   [Problem 1. Describe the data](#problem-1.-describe-the-data)
    -   [Problem 2. Basic Model](#problem-2.-basic-model)
    -   [Problem 3. Multiple Variable Model](#problem-3.-multiple-variable-model)
-   [Part 2: Modeling TV Consumption](#part-2-modeling-tv-consumption)
    -   [Estimate a regression model](#estimate-a-regression-model)

Part 1: Modeling voter turnout
==============================

Problem 1. Describe the data
----------------------------

![](NingyinXuPS6_files/figure-markdown_github/hist-1.png)

1). The unconditional probability of a given individual turning out to vote is 62.96%.

![](NingyinXuPS6_files/figure-markdown_github/linear-1.png)

2). One can tell from the graph that there seem to be a negative correlation between individual's mental health and her/his voting decision, revealing depression decreases people's desire to participate in politics on some level.

However, this linear line has problems: the range of reponse is also problematic. The linear line shows the range of "voting decision" is `(0.25, 0.75)`, while response only have 2 values: 0, 1. This line doesn't explain the relationship between these two variables very well.

Problem 2. Basic Model
----------------------

    ## 
    ## =============================================
    ##                       Dependent variable:    
    ##                   ---------------------------
    ##                             vote96           
    ## ---------------------------------------------
    ## mhealth_sum                -0.143***         
    ##                             (0.020)          
    ##                                              
    ## Constant                   1.139***          
    ##                             (0.084)          
    ##                                              
    ## ---------------------------------------------
    ## Observations                 1,322           
    ## Log Likelihood             -808.360          
    ## Akaike Inf. Crit.          1,620.720         
    ## =============================================
    ## Note:             *p<0.1; **p<0.05; ***p<0.01

1). The relationship between mental health and voter turnour is statistically but not substantively significant. 1&gt; the p-value of mhealth\_sum is very small, indicating the probablity to reject the null hypothesis is greather than 99%. This is statistical significance. 2&gt; The size of estimate parameter of mental health is -0.1435. It means with one unit increase in mental health, there would be 0.1435 decrease in the log-odds of voting decesion, so 1.1543 decrease in odds of voting. This effect size is relatively small, so the relationship is not substantively significant.

![](NingyinXuPS6_files/figure-markdown_github/logit2-1.png)

2). With one unit increase in mental health (more depressed), there would be 0.1435 decrease in the log-odds of voting decesion.

![](NingyinXuPS6_files/figure-markdown_github/logit3-1.png)

3). With one unit increase in mental health (more depressed), there would be 1.1543 decrease in odds of voting.

![](NingyinXuPS6_files/figure-markdown_github/logit4-1.png)

4). With one unit increase in mental health (more depressed), there would be 0.5358 decrease in odds of voting. The first difference for an increase in the mental health index from 1 to 2 is -0.0292, for 5 to 6 is -0.0348. So the probability of voting would decrease by 2.92% when the individual's mental health increase from 1 to 2, the probability of voting would decrease by 3.48% when the individual's mental health increase from 5 to 6.

5). The accuracy rate, proportional reduction in error (PRE), and the AUC for this model are 67.78%, 1.62%, and 0.5401 respectively. This model is thus not so good. The accuracy rate seems okay but without baseline it doesn't say too much about the model's performance. The PRE says the statistical model reduces only little prediction error. The AUC is close to 0.5, which is the auc under random condition.

Problem 3. Multiple Variable Model
----------------------------------

1). For the model I chose, the Probability distribution is the Bernoulli distribution: \[
Pr({Y_i} = {y_i}|\pi ) = {\pi_{i}}^{y_i}{(1 - {\pi_i})}^{(1-{y_i})}
\]

The linear predictor is: \[
{\eta_i} = {\beta_0} + {\beta_1}{MentalHealth_i} + {\beta_2}{Age_i} + {\beta_3}{Education_i}
\]

The link function is: \[{\pi_i} = \frac{e^{\eta_i}}{1 + e^{\eta_i}}\]

    ## 
    ## =============================================
    ##                       Dependent variable:    
    ##                   ---------------------------
    ##                             vote96           
    ## ---------------------------------------------
    ## mhealth_sum                -0.099***         
    ##                             (0.021)          
    ##                                              
    ## age                        0.045***          
    ##                             (0.004)          
    ##                                              
    ## educ                       0.260***          
    ##                             (0.027)          
    ##                                              
    ## Constant                   -4.375***         
    ##                             (0.463)          
    ##                                              
    ## ---------------------------------------------
    ## Observations                 1,317           
    ## Log Likelihood             -710.201          
    ## Akaike Inf. Crit.          1,428.403         
    ## =============================================
    ## Note:             *p<0.1; **p<0.05; ***p<0.01

3). From the results of the model, one can see the model performs well: 1&gt;. The chosen variables all have significant relationship with the response. The three predictors we chose are all statistically significant. 2&gt;. The model's fits the reality relatively well. The accuracy rate, proportional reduction in error (PRE), and the AUC for this model are 72.29%, 15.12%, and 0.6379 respectively. Comparing to the signle-variable model, the accuracy rate has been improved, there's more reduction in error, and the AUC is enlarged. One can also use p-value 0, which is way less than 0.05, showing that this model as a whole fits significantly better than an empty model.

Based on the fitness of this model, one could interpret the relationship between response and the three predictors using this model. The size of estimate parameter of mental health is -0.0985. It means with one unit increase in mental health, there would be 0.0985 decrease in the log-odds of voting decesion, so 1.1036 decrease in odds of voting.

The size of estimate parameter of age is 0.0449. It means with one unit increase in age, there would be -0.0449 decrease in the log-odds of voting decesion, so 0.9561 decrease in odds of voting.

The size of estimate parameter of education level is 0.2605. It means with one unit increase in education level, there would be -0.2605 decrease in the log-odds of voting decesion, so 0.7707 decrease in odds of voting.

First difference could be demonstrated from mental health, age, and education level. To compare with the basic model before, I'm only gonna use mental health here, controlling age and education level as 22 and 15. The first difference for an increase in the mental health index from 1 to 2 is NA, for 5 to 6 is NA. So the probability of voting would decrease by NA% when the individual's mental health increase from 1 to 2, the probability of voting would decrease by NA% when the individual's mental health increase from 5 to 6.

Part 2: Modeling TV Consumption
===============================

Estimate a regression model
---------------------------

    ## 
    ## =============================================
    ##                       Dependent variable:    
    ##                   ---------------------------
    ##                             tvhours          
    ## ---------------------------------------------
    ## educ                       -0.039***         
    ##                             (0.011)          
    ##                                              
    ## grass                       -0.108*          
    ##                             (0.062)          
    ##                                              
    ## hrsrelax                   0.047***          
    ##                             (0.009)          
    ##                                              
    ## black                      0.451***          
    ##                             (0.072)          
    ##                                              
    ## Constant                   1.225***          
    ##                             (0.169)          
    ##                                              
    ## ---------------------------------------------
    ## Observations                  441            
    ## Log Likelihood             -781.721          
    ## Akaike Inf. Crit.          1,573.442         
    ## =============================================
    ## Note:             *p<0.1; **p<0.05; ***p<0.01

1). For the model I chose, the Probability distribution is: \[Pr(tvhours = tvhours_{i}|\mu)=\frac{\mu^{k}e^tvhours_{i}}{tvhours_{i}!}\]

The linear predictor is: \[tvhours_{i} = \beta_{0} + \beta_{1}educ_{i} + \beta_{2}grass_{i} + \beta_{3}hrsrelax_{i} + \beta_{4}black_{i}\]

The link function is: \[\mu_{i} = ln(\eta_{i})\]

\[
\begin{equation}
test=\frac{1}{2}
\end{equation}
\]
