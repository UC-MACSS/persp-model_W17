Xu\_Ningyin\_PS5: Linear Regression
================
Ningyin Xu
2/8/2017

-   [Problem 1. Describe the data](#problem-1.-describe-the-data)
-   [Problem 2. Simple linear regression](#problem-2.-simple-linear-regression)

Problem 1. Describe the data
----------------------------

![](test_files/figure-markdown_github/unnamed-chunk-1-1.png)

From the histogram shown above, one can tell most people have high scores (greater than 50) in feeling thermometer for Joe Biden, which means they like him. However, the highest frequency count among all the bins appears in the score of 50, saying the group of people who have indifferent attitude about him is the largest one.

Problem 2. Simple linear regression
-----------------------------------

The summary of simple linear regression is shown as below.

    ## 
    ## Call:
    ## lm(formula = biden ~ age, data = bidendata)
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

To make it clearer, the estimates of parameters and their standard errors, R-squared, and adjusted R-squared are the following:

    ##          term    estimate  std.error statistic       p.value
    ## 1 (Intercept) 59.19736008 1.64791889 35.922496 1.145056e-213
    ## 2         age  0.06240535 0.03266815  1.910281  5.625534e-02

    ## [1] 0.002017624

    ## [1] 0.001464725

1&2). One could say there is a relationship between the predictor and the response, since the p-value of age in the summary shows that there are more than 90% chance of rejecting the null hypothesis, but statistically speaking the relationship is not very strong/significant, which requires greater than 95%.

3). The positive sign of the estimate of age shows that it has a positive effect on the response.

4). The R-squared is about 0.002018, and the adjusted R-squared is about 0.001465. This means only about 0.2% of variation is explained by age, implying that the model is not fitting the actual data well.

![](test_files/figure-markdown_github/simple_lm_pred-1.png)
