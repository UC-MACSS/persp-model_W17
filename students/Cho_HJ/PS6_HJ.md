Perspectives on Computational Modeling PS6
================
HyungJin Cho
February 20, 2017

Part 1: Modeling voter turnout
==============================

I.Describe the data
-------------------

#### I.1.(a)Plot a histogram of voter turnout. Make sure to give the graph a title and proper *x* and *y*-axis labels.

![](PS6_HJ_files/figure-markdown_github/I.1.(a)Histogram-1.png)

#### I.1.(b)What is the unconditional probability of a given individual turning out to vote?

|  vote96|  unconditional probability of a given individual turning out to vote|
|-------:|--------------------------------------------------------------------:|
|       0|                                                            0.3176426|
|       1|                                                            0.6823574|

#### I.2.(a)Generate a scatterplot of the relationship between mental health and observed voter turnout and overlay a linear smoothing line.

![](PS6_HJ_files/figure-markdown_github/I.2.(a)Scatterplot%20&%20Linear%20Smoothing%20Line-1.png)

#### I.2.(b)What information does this tell us? What is problematic about this linear smoothing line?

The scatterplot and the linear smoothing line indicates that lower value of the mental health index is associated with voting, which means an individual with lower depressed mood tends to vote. The problem of this linear smoothing line is the vote status is treated as continuous variable instead of categorical variable.

II.Basic model
--------------

#### II&gt;1.Is the relationship between mental health and voter turnout statistically and/or substantively significant?

    ## 
    ## Call:
    ## glm(formula = vote96 ~ mhealth_sum, family = binomial(), data = DATA_1)
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

The relationship between mental health and voter turnout is statistically significant (p-value=3.133882910^{-13} &lt; .001). The relationship between mental health and voter turnout appears substantively significant. The coefficient (-0.1434752) suggests the log-odds and the exponentiated coefficient (0.8663423) indicates the odds along with the probability (0.4641926).

#### II.2.Interpret the estimated parameter for mental health in terms of log-odds. Generate a graph of the relationship between mental health and the log-odds of voter turnout.

![](PS6_HJ_files/figure-markdown_github/II.2.Log-odds%20Plot-1.png)

The estimated parameter for mental health in terms of log-odds is -0.1434752. The change in log-odds by a unit increase in mental health is -0.1434752. The graph shows linear relationship.

#### II.3.Interpret the estimated parameter for mental health in terms of odds. Generate a graph of the relationship between mental health and the odds of voter turnout.

![](PS6_HJ_files/figure-markdown_github/II.3.Odds%20Plot-1.png)

The estimated parameter for mental health in terms of odds is 0.8663423. The change in odds by a unit increase in mental health is 0.8663423.

#### II.4.Interpret the estimated parameter for mental health in terms of probabilities. Generate a graph of the relationship between mental health and the probability of voter turnout. What is the first difference for an increase in the mental health index from 1 to 2? What about for 5 to 6?

![](PS6_HJ_files/figure-markdown_github/II.4.Probability%20Plot-1.png)

    ## $title
    ## [1] "Probability of Voter Turout for Mental Health"
    ## 
    ## $x
    ## [1] "Mental Health"
    ## 
    ## $y
    ## [1] "Probability of Voter Turnout"
    ## 
    ## attr(,"class")
    ## [1] "labels"

The estimated parameter for mental health in terms of probabilities is 0.4641926. The difference for an increase in the mental health index from 1 to 2 is 0.0291782. The difference for an increase in the mental health index from 5 to 6 is 0.0347782.

#### II.5.Estimate the accuracy rate, proportional reduction in error (PRE), and the AUC for this model. Do you consider it to be a good model?

    ## [1] 0.677761

The accuracy rate is 0.677761, the proportional reduction in error (PRE) is 0.0161663, and the area under the curve (AUC) is 0.6243087. This is not a good model because the accuracy rate only better than the baseline rate by 0.0161663 proportional reduction in error and the area under the curve is only larger than the random guess by 0.1243087.

III.Multiple variable model
---------------------------

#### III.1.Write out the three components of the GLM for your specific model of interest. This includes the Probability distribution (random component), Linear predictor, Link function.

The random component of the probability distribution: The Bernoulli distribution.
$$Pr(\\sum\_{i=1}^{n}vote96\_i = k|p) = \\binom{n}{k}p^k(1-p)^{n-k}$$
 Linear predictor:
*v**o**t**e*96<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1, *i*</sub> + *β*<sub>2</sub>*X*<sub>2, *i*</sub> + *β*<sub>3</sub>*X*<sub>3, *i*</sub> + *β*<sub>4</sub>*X*<sub>4, *i*</sub> + *β*<sub>5</sub>*X*<sub>5, *i*</sub> + *β*<sub>6</sub>*X*<sub>6, *i*</sub> + *β*<sub>7</sub>*X*<sub>7, *i*</sub>
 *X*<sub>1</sub>=mhealth\_sum, *X*<sub>2</sub>=age, *X*<sub>3</sub>=educ, *X*<sub>4</sub>=blakc, *X*<sub>5</sub>=female, *X*<sub>6</sub>=married, *X*<sub>7</sub>=inc10.
Link function: Logit function.
$$g(vote96\_i) = \\frac{e^{vote96\_i}}{1 + e^{vote96\_i}}$$

#### III.2.Estimate the model and report your results.

    ## Parsed with column specification:
    ## cols(
    ##   vote96 = col_double(),
    ##   mhealth_sum = col_double(),
    ##   age = col_double(),
    ##   educ = col_double(),
    ##   black = col_double(),
    ##   female = col_double(),
    ##   married = col_double(),
    ##   inc10 = col_double()
    ## )

    ## 
    ## Call:
    ## glm(formula = vote96 ~ ., family = binomial(), data = DATA_1)
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

The estimatation of this multivariate logistric regression model shows the results that 4 factors are statistically significant: mental health index, age, education, and income.

#### III.3.Interpret the results in paragraph format. This should include a discussion of your results as if you were reviewing them with fellow computational social scientists. Discuss the results using any or all of log-odds, odds, predicted probabilities, and first differences - choose what makes sense to you and provides the most value to the reader. Use graphs and tables as necessary to support your conclusions.

In terms of log-odds, coeeficients of mental health index (-0.089102), age (0.042534), education (0.228686), and income (0.069614) are found to be statistically significant. In respect to odds, multiplicative factor of health index (0.9147523), age (1.043452), education (1.256947), and income (1.072094) are found. In regard to probabilities, predicted probabilities are associated with health index (0.4777392), age (0.510632), education (0.5569236), and income (0.5173964)

Part 2: Modeling tv consumption
===============================

IIII.Estimate a regression model
--------------------------------

#### IIII.1.Write out the three components of the GLM for your specific model of interest. This includes the Probability distribution (random component), Linear predictor, Link function.

The random component of the probability distribution: Poisson distribution.
$$Pr(tvhours = k|\\lambda) = \\frac{\\lambda^{k}e^{-\\lambda}}{k!}$$
 Linear predictor:
*t**v**h**o**u**r**s*<sub>*i*</sub> = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1, *i*</sub> + *β*<sub>2</sub>*X*<sub>2, *i*</sub> + *β*<sub>3</sub>*X*<sub>3, *i*</sub> + *β*<sub>4</sub>*X*<sub>4, *i*</sub> + *β*<sub>5</sub>*X*<sub>5, *i*</sub> + *β*<sub>6</sub>*X*<sub>6, *i*</sub> + *β*<sub>7</sub>*X*<sub>7, *i*</sub>
+*β*<sub>8</sub>*X*<sub>8, *i*</sub> + *β*<sub>9</sub>*X*<sub>9, *i*</sub> + *β*<sub>10</sub>*X*<sub>10, *i*</sub> + *β*<sub>11</sub>*X*<sub>11, *i*</sub> + *β*<sub>12</sub>*X*<sub>12, *i*</sub> + *β*<sub>13</sub>*X*<sub>13, *i*</sub> + *β*<sub>14</sub>*X*<sub>14, *i*</sub>
 *η*<sub>*i*</sub>=tvhours\_{i}, *X*<sub>1</sub>=age, *X*<sub>2</sub>=childs, *X*<sub>3</sub>=educ, *X*<sub>4</sub>=female, *X*<sub>5</sub>=grass, *X*<sub>6</sub>=hrsrelax, and *X*<sub>7</sub>=black, *X*<sub>8</sub>=social\_connect, *X*<sub>9</sub>=voted04, *X*<sub>10</sub>=xmovie, *X*<sub>11</sub>=zodiac, *X*<sub>12</sub>=dem, *X*<sub>13</sub>=rep, *X*<sub>14</sub>=ind.
Link function: Log function.
*g*(*t**v**h**o**u**r**s*<sub>*i*</sub>)=log(*t**v**h**o**u**r**s*<sub>*i*</sub>)

#### IIII.2.Estimate the model and report your results.

    ## 
    ## Call:
    ## glm(formula = tvhours ~ ., family = poisson(), data = DATA_2)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -3.1120  -0.6741  -0.1144   0.4224   4.9257  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                     Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)        1.0795865  0.2419794   4.461 8.14e-06 ***
    ## age                0.0016522  0.0028397   0.582   0.5607    
    ## childs            -0.0003896  0.0238729  -0.016   0.9870    
    ## educ              -0.0292174  0.0126351  -2.312   0.0208 *  
    ## female             0.0457000  0.0652987   0.700   0.4840    
    ## grass             -0.1002726  0.0686146  -1.461   0.1439    
    ## hrsrelax           0.0468472  0.0102790   4.558 5.18e-06 ***
    ## black              0.4657924  0.0841629   5.534 3.12e-08 ***
    ## social_connect     0.0437349  0.0407999   1.072   0.2837    
    ## voted04           -0.0994787  0.0785680  -1.266   0.2055    
    ## xmovie             0.0708408  0.0773420   0.916   0.3597    
    ## zodiacAries       -0.1011364  0.1508248  -0.671   0.5025    
    ## zodiacCancer       0.0267776  0.1451557   0.184   0.8536    
    ## zodiacCapricorn   -0.2155760  0.1657034  -1.301   0.1933    
    ## zodiacGemini       0.0285895  0.1481143   0.193   0.8469    
    ## zodiacLeo         -0.1515676  0.1553215  -0.976   0.3291    
    ## zodiacLibra       -0.0392537  0.1379102  -0.285   0.7759    
    ## zodiacNaN         -0.2985240  0.2126126  -1.404   0.1603    
    ## zodiacPisces      -0.1446731  0.1649895  -0.877   0.3806    
    ## zodiacSagittarius -0.2177846  0.1577638  -1.380   0.1674    
    ## zodiacScorpio      0.0225911  0.1538460   0.147   0.8833    
    ## zodiacTaurus      -0.1273891  0.1644799  -0.774   0.4386    
    ## zodiacVirgo       -0.1240442  0.1564495  -0.793   0.4279    
    ## dem                0.0103276  0.0917055   0.113   0.9103    
    ## rep                0.0148615  0.0927662   0.160   0.8727    
    ## ind                       NA         NA      NA       NA    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 527.72  on 440  degrees of freedom
    ## Residual deviance: 429.42  on 416  degrees of freedom
    ##   (4069 observations deleted due to missingness)
    ## AIC: 1600.4
    ## 
    ## Number of Fisher Scoring iterations: 5

The estimatation of this model shows the results that 3 factors are statistically significant: education, relax hours, and race.

#### IIII.3.Interpret the results in paragraph format. This should include a discussion of your results as if you were reviewing them with fellow computational social scientists. Discuss the results using any or all of log-counts, predicted event counts, and first differences - choose what makes sense to you and provides the most value to the reader. Is the model over or under-dispersed? Use graphs and tables as necessary to support your conclusions.

In terms of log-odds, coeeficients of education (-0.0292174), relax hours (0.0468472), and race (0.4657924) are found to be statistically significant. In respect to odds, multiplicative factor of education (0.9712053), relax hours (1.047962), and race (1.593276) are found. In regard to probabilities, predicted probabilities are associated with education (0.4926962), relax hours (0.5117097), and race (0.6143874)
