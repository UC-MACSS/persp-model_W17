Kang\_persp-model\_PS6
================
BobaeKang
February 20, 2017

-   [Part 1: Modeling voter turnout](#part-1-modeling-voter-turnout)
    -   [Describe the data (1 point)](#describe-the-data-1-point)
    -   [Basic model (3 points)](#basic-model-3-points)
    -   [Multiple variable model (3 points)](#multiple-variable-model-3-points)
-   [Part 2: Modeling tv consumption](#part-2-modeling-tv-consumption)
    -   [Estimate a regression model (3 points)](#estimate-a-regression-model-3-points)

Part 1: Modeling voter turnout
==============================

Describe the data (1 point)
---------------------------

Here are two histograms of the voter turnout data, one with missing values and the other witout:
![](Kang_persp-model_PS6_files/figure-markdown_github/Part%201%20histogram%201-1.png) ![](Kang_persp-model_PS6_files/figure-markdown_github/Part%201%20histogram%202-1.png)

The unconditional probabilities for each voter turnout category are as follows:

    ## [1] "Unconditional probability with missing values"

<table style="width:50%;">
<colgroup>
<col width="12%" />
<col width="37%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">vote96</th>
<th align="center">probability (with missing values)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">0</td>
<td align="center">0.29307910</td>
</tr>
<tr class="even">
<td align="center">1</td>
<td align="center">0.62959040</td>
</tr>
<tr class="odd">
<td align="center">NA</td>
<td align="center">0.07733051</td>
</tr>
</tbody>
</table>

    ## [1] "Unconditional probability without missing values"

<table style="width:54%;">
<colgroup>
<col width="12%" />
<col width="41%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">vote96</th>
<th align="center">probability (without missing values)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">0</td>
<td align="center">0.3176426</td>
</tr>
<tr class="even">
<td align="center">1</td>
<td align="center">0.6823574</td>
</tr>
</tbody>
</table>

The following scattorplot with a overlaying linear smoothing line indicates a negative correlation between the voter turnout and the mental health index. The points are jittered to better ilustrate how many observations exist for each mental health index score. Without jittering, multiple observations for the given mental health index appear on the plot as if there is only a single observation.

The linear model suffers from a potential problem of higher-than-one or lower-then-zero probability, because it assumes that the range of the response variable is all real numbers.

![](Kang_persp-model_PS6_files/figure-markdown_github/Part%201%20scatterplot-1.png)

Basic model (3 points)
----------------------

The relationship between mental health and voter turnout is statistically significant, as the low p-value for the coefficient (3.13e-13) suggests. Is this substantively significant? The coefficient, -0.14348, indicates the log-odds of a unit increase in the value of mental health state. Exponentiating the coefficient givese the odds of voting for a unit increase in the value of mental health state, 0.8663381. That is, the odds that voter turnout is 1 change by a multiplicative factor of 0.8663381. In the same mannter, the probability that the voter turnout = 1 change by a multiplicative factor of 0.8663381/(1 + 0.8663381) = 0.4641914, and this seems substantively significant.

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum, family = binomial(), data = data1)
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

Here is a line plot for log-odds:

![](Kang_persp-model_PS6_files/figure-markdown_github/Part%201%20basic%20model%20log-odds%20plot-1.png)

And a line plot for odds:

![](Kang_persp-model_PS6_files/figure-markdown_github/Part%201%20basic%20model%20odds%20plot-1.png)

    ## mapping: y = vote96 
    ## geom_point: na.rm = FALSE
    ## stat_identity: na.rm = FALSE
    ## position_identity

Finally, a line plot for probability with the jittered scattorplot for voter turnout:

![](Kang_persp-model_PS6_files/figure-markdown_github/Part%201%20basic%20model%20probabilities%20plot-1.png)

The first difference in probability for an increase in the mental health index from 1 to 2, and for 5 to 6.

    ## The difference in probability for an increase in the mental health index from 1 to 2 is 0.02917824

    ## The difference in probability for an increase in the mental health index from 5 to 6 is 0.03477821

As for the current model, the accuracy rate is 0.677761, the proportional reduction in error is 0.01616628, and the area under the curve (AUC) is 0.6243. In other words, this is not a good model. Its accuracy rate is little better than the baseline rate (only 1.6% reduction in error) and its performance is hardly superior to the random guess, which would have the AUC of 0.5.

    ## Accuracy rate: 0.677761

    ## Proportional reduction in error: 0.01616628

    ## Area under the curve: 0.6243

Multiple variable model (3 points)
----------------------------------

The three components of the logistic regression:
\* Probability distribution (random component): the Bernoulli distribution, Pr(*Y*<sub>*i*</sub> = *y*<sub>*i*</sub>|*π*) = *π*<sub>*i*</sub><sup>*y*<sub>*i*</sub></sup> (1 − *π*<sub>*i*</sub>)<sup>1 − *y*<sub>*i*</sub></sup>
\* Linear predictor: *η*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1, *i*</sub> + *β*<sub>2</sub>*X*<sub>2, *i*</sub> + *β*<sub>3</sub>*X*<sub>3, *i*</sub> + *β*<sub>4</sub>*X*<sub>4, *i*</sub> + *β*<sub>5</sub>*X*<sub>5, *i*</sub> + *β*<sub>6</sub>*X*<sub>6, *i*</sub> + *β*<sub>7</sub>*X*<sub>7, *i*</sub>,
where *X*<sub>1</sub> is `mhealth_sum`, *X*<sub>2</sub> is `age`, *X*<sub>3</sub> is `educ`, *X*<sub>4</sub> is `blakc`, *X*<sub>5</sub> is `female`, *X*<sub>6</sub> is `married`, and *X*<sub>7</sub> is `inc10`.
\* Link function: the logit function, *π*<sub>*i*</sub> = *e*<sup>*η*<sub>*i*</sub></sup> / (1 + *e*<sup>*η*<sub>*i*</sub></sup>)

Estimate the model and report your results.

    ## 
    ## Call:
    ## glm(formula = vote96 ~ ., family = binomial(), data = data1)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.4843  -1.0258   0.5182   0.8428   2.0758  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -4.304103   0.508103  -8.471  < 2e-16 ***
    ## mhealth_sum -0.089102   0.023642  -3.769 0.000164 ***
    ## age          0.042534   0.004814   8.835  < 2e-16 ***
    ## educ         0.228686   0.029532   7.744 9.65e-15 ***
    ## black        0.272984   0.202585   1.347 0.177820    
    ## female      -0.016969   0.139972  -0.121 0.903507    
    ## married      0.296915   0.153164   1.939 0.052557 .  
    ## inc10        0.069614   0.026532   2.624 0.008697 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1468.3  on 1164  degrees of freedom
    ## Residual deviance: 1241.8  on 1157  degrees of freedom
    ##   (1667 observations deleted due to missingness)
    ## AIC: 1257.8
    ## 
    ## Number of Fisher Scoring iterations: 4

In this multivariate logistric regression model, the response variable is the binary voter turnout varaible where 1 means the respondent voted and 0 means the respondent did not vote. The predictors include the mental health index, age, education, race (Black or not), gender (female or not), marital status (married or not), and family income (in $10,000s). The regression results indicate that four of the coefficients are statistically significant; these coefficients are, respectively, -0.089102 for the mental health index, 0.042534 for age, 0.228686 for education and 0.069614 for income. These coefficients are given in terms of log-odds.

In terms of odds, hold other variables constant, a unit increase in the mental health index leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 0.9147523. Likewise, holding other variables constant, one year increase in age leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 1.043452. Again, holding other variables constant, one year increase in the number of years of formal education leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 1.256947. Finally, holding other variables constant, a unit increase in income leads to an average change in the odds of voter turnout = 1 by a multiplicative factor of 1.072094. In terms of predicted probabilities, these values correspond to, respectively, a multiplicative factor of 0.4777392 for each unit increase in the mental health index holding other variables constant, 0.510632 for age, 0.5569236 for educaiton, and 0.5173964 for income.

The accuracy rate, proportional reduction in error (PRE) and area under the curve (AUC) ofthe current model indicate that the model is better than the "simple" logistic regression model. Nonetheless, even with more predictors, the current logistic regression model shows a rather poor performance. The accuracy, PRE and AUC scores of the current model are as follows:

    ## Accuracy rate: 0.7236052

    ## Proportional reduction in error: 0.1481481

    ## Area Under the Curve: 0.759624

We can also compare the current model with the previous model using the same two cases of first difference in the mental health index, 1 to 2 and 5 to 6. To hold other variables constant, I will use the case of a 30-old-year black female single with 16 years of education and income of $50,000. In the previous model, they were 0.02917824 for 1 to 2 and 0.03477821 for 5 to 6. For the multivariate model:

    ## The difference in predicted probability for an increase in the mental health index from 1 to 2 is 0.0141544

    ## The difference in predicted probability for an increase in the mental health index from 5 to 6 is 0.01717635

Finally, the following plot illustrate the difference between the respondents with college education and the others. While the higher mental health index is associated with less probability of voting for both groups, the effect of higher education is remarkable.

    ## `geom_smooth()` using method = 'loess'

![](Kang_persp-model_PS6_files/figure-markdown_github/Part%201%20multivariate%20model%20plot-1.png)

Part 2: Modeling tv consumption
===============================

Estimate a regression model (3 points)
--------------------------------------

The three components of the Poisson regression:
\* Probability distribution (random component): the Poisson distribution, Pr(*Y*<sub>*i*</sub> = *y*<sub>*i*</sub>|*μ*) = *μ*<sup>*y*<sub>*i*</sub></sup> *e*<sup>−*μ*</sup> / *y*<sub>*i*</sub>!
\* Linear predictor: *η*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1, *i*</sub> + *β*<sub>2</sub>*X*<sub>2, *i*</sub> + *β*<sub>3</sub>*X*<sub>3, *i*</sub> + *β*<sub>4</sub>*X*<sub>4, *i*</sub> + *β*<sub>5</sub>*X*<sub>5, *i*</sub> + *β*<sub>6</sub>*X*<sub>6, *i*</sub> + *β*<sub>7</sub>*X*<sub>7, *i*</sub>,
where *X*<sub>1</sub> is `age`, *X*<sub>2</sub> is `childs`, *X*<sub>3</sub> is `educ`, *X*<sub>4</sub> is `female`, *X*<sub>5</sub> is `grass`, *X*<sub>6</sub> is `hrsrelax`, and *X*<sub>7</sub> is `black`.
\* Link function: the log function, *μ*<sub>*i*</sub> = ln(*η*<sub>*i*</sub>)

Estimate the model and report your results:

    ## 
    ## Call:
    ## glm(formula = tvhours ~ . - social_connect - xmovie - zodiac - 
    ##     voted04 - dem - rep - ind, family = poisson(), data = data2)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.0603  -0.7566  -0.1010   0.4653   5.3834  
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  1.2175645  0.1969556   6.182 6.33e-10 ***
    ## age          0.0002017  0.0027479   0.073 0.941496    
    ## childs      -0.0042927  0.0237624  -0.181 0.856643    
    ## educ        -0.0393047  0.0112682  -3.488 0.000486 ***
    ## female       0.0184765  0.0637082   0.290 0.771802    
    ## grass       -0.1090472  0.0631473  -1.727 0.084191 .  
    ## hrsrelax     0.0471133  0.0097755   4.820 1.44e-06 ***
    ## black        0.4495892  0.0742844   6.052 1.43e-09 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 527.72  on 440  degrees of freedom
    ## Residual deviance: 442.37  on 433  degrees of freedom
    ##   (4069 observations deleted due to missingness)
    ## AIC: 1579.3
    ## 
    ## Number of Fisher Scoring iterations: 5

In this Poisson regression model, the response variable is the number of hours for watching TV per day. The predictors I chose include the following: age, number of children, education, gender (1 if female), opinion on legalizing marijuana, hours to relex, and race (1 if black). The regression result illustrates that the coefficients for only three predictors are statistically significant. These coefficients are, respectively -0.0380001 for education, 0.0457914 for hours to relax and 0.4363657 for race. Each of these coefficients indicates the extent of a change in the log-count of the respondent's Tv-watching hours, to which a unit increase in the given predictor will lead on average when other variables are held constant.

These cofficients also mean the following: a unit increase in edcuation is associated with a 0.9627128-fold change in the mean number of the hours of watching TV per day. Also, each additioanl hour of relaxing is associated with a 1.046856-fold change in the mean number of the hours of watching TV per day. Finally, being black is associated with a 1.547074-fold change in the mean number of the hours of watching TV per day.

Finally, the following plot shows the effect of three statistically significant predictors on the hours of watching TV. The plot illustrates that while leisure and racial factors are positively correlated with the hours of watching TV, education is negatively correlated with the hours of watching TV.

    ## `geom_smooth()` using method = 'loess'

![](Kang_persp-model_PS6_files/figure-markdown_github/Part%202%20multivariate%20model%20plot-1.png)
