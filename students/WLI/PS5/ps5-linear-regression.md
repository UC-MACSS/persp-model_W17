Problem set \#5: linear regression
================
Weijia Li
**Due Monday February 13th at 11:30am**

-   [Describe the data (1 point)](#describe-the-data-1-point)
-   [Simple linear regression (2 points)](#simple-linear-regression-2-points)
-   [Multiple linear regression (2 points)](#multiple-linear-regression-2-points)
-   [Multiple linear regression model (with even more variables!) (3 points)](#multiple-linear-regression-model-with-even-more-variables-3-points)
-   [Interactive linear regression model (2 points)](#interactive-linear-regression-model-2-points)

`biden.csv` contains a selection of variables from the [2008 American National Election Studies survey](http://www.electionstudies.org/) that allow you to test competing factors that may influence attitudes towards Joe Biden. The variables are coded as follows:

-   `biden` - feeling thermometer ranging from 0-100[1]
-   `female` - 1 if respondent is female, 0 if respondent is male
-   `age` - age of respondent in years
-   `dem` - 1 if respondent is a Democrat, 0 otherwise
-   `rep` - 1 if respondent is a Republican, 0 otherwise
-   `educ` - number of years of formal education completed by respondent
    -   `17` - 17+ years (aka first year of graduate school and up)

Describe the data (1 point)
===========================

Plot a histogram of `biden` with a binwidth of `1`. Make sure to give the graph a title and proper *x* and *y*-axis labels. In a few sentences, describe any interesting features of the graph.

``` r
ggplot(biden, mapping = aes(x = biden)) +
  geom_histogram(binwidth = 1, alpha=0.7) +
  labs(title = "Distribution of feeling thermometer of Biden",
       x = "Feeling thermometer",
       y = "Frequency count of individuals")
```

![](ps5-linear-regression_files/figure-markdown_github/unnamed-chunk-1-1.png)

The histogram looks interesting that not all numbers between 0 and 100 are recorded in the data, in fact, there are only a small set of numbers are recorded as feeling thermometer of Biden. This is probably because people will generally give a multiple of 5 when they are asked to score something.

Simple linear regression (2 points)
===================================

Estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub>

where *Y* is the Joe Biden feeling thermometer and *X*<sub>1</sub> is age. Report the parameters and standard errors.

``` r
biden_mod <- lm(biden ~ age, data = biden)
summary(biden_mod)
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

``` r
grid <- biden %>%
  data_grid(age) %>%
  add_predictions(biden_mod)

tidy(biden_mod)
```

    ##          term    estimate  std.error statistic       p.value
    ## 1 (Intercept) 59.19736008 1.64791889 35.922496 1.145056e-213
    ## 2         age  0.06240535 0.03266815  1.910281  5.625534e-02

*β*<sub>0</sub> is 59.19736 and *β*<sub>1</sub> is 0.06241. The intercept means the estimated feeling warmth is ~59.2 when *X*<sub>1</sub> is 0. The standard error for *β*<sub>0</sub> is 1.64792 and is 0.03267 for *β*<sub>1</sub>.

1.  Is there a relationship between the predictor and the response?

p-value is 0.05626 so there is a relationship between the predictor and the response at significant level 0.01.

1.  How strong is the relationship between the predictor and the response?

Estimated *β* is 0.06241, that means 20 years age difference only makes 1.2 point change.

1.  Is the relationship between the predictor and the response positive or negative?

0.0624 is positive, so the relationship is positive.

1.  Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does `age` alone explain? Is this a good or bad model?

The *R*<sup>2</sup> is 0.002018, that is, the age variation only explains 0.2% of the variation in feeling thermometer. This is a bad model that is not explainary.

1.  What is the predicted `biden` associated with an `age` of 45? What are the associated 95% confidence intervals?

``` r
(pred_ci <- augment(biden_mod, newdata = data_frame(age = c(45))) %>%
  mutate(ymin = .fitted - .se.fit * 1.96,
         ymax = .fitted + .se.fit * 1.96))
```

    ##   age .fitted   .se.fit     ymin     ymax
    ## 1  45 62.0056 0.5577123 60.91248 63.09872

The predicted feeling for Biden at age 45 is 62.0056, and the associated 95% confidence intervals are (60.91248, 63.09872).

1.  Plot the response and predictor. Draw the least squares regression line.

``` r
ggplot(biden, aes(age)) +
  geom_point(aes(y = biden)) +
  geom_line(aes(y = pred), data = grid, color = "red", size = 1) +
  labs(title = 'Plot of Joe Biden Feeling thermometer with Least Squares Regression Line',
       y = 'Feeling thermometer')
```

![](ps5-linear-regression_files/figure-markdown_github/unnamed-chunk-4-1.png)

Multiple linear regression (2 points)
=====================================

It is unlikely `age` alone shapes attitudes towards Joe Biden. Estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>3</sub>

where *Y* is the Joe Biden feeling thermometer, *X*<sub>1</sub> is age, *X*<sub>2</sub> is gender, and *X*<sub>3</sub> is education. Report the parameters and standard errors.

``` r
biden_mult <- lm(biden ~ age + female + educ, data = biden)
tidy(biden_mult)
```

    ##          term    estimate  std.error statistic      p.value
    ## 1 (Intercept) 68.62101396 3.59600465 19.082571 4.337464e-74
    ## 2         age  0.04187919 0.03248579  1.289154 1.975099e-01
    ## 3      female  6.19606946 1.09669702  5.649755 1.863612e-08
    ## 4        educ -0.88871263 0.22469183 -3.955251 7.941295e-05

1.  Is there a statistically significant relationship between the predictors and response?

There is a statistically significant relationship between the predictors and response. The predictors 'female' and 'educ' both have very small p-values (1.863612e-08 and 7.941295e-05 respectively) yet 'age' is not significant anymore with p-value 0.1975 &gt; 0.1.

1.  What does the parameter for `female` suggest?

The coefficient for 'female' is 6.196, that means there will be on average ~6.2 points higher if the respondent is female(with age and education to be constants).

1.  Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does age, gender, and education explain? Is this a better or worse model than the age-only model?

``` r
summary(biden_mult)$r.squared
```

    ## [1] 0.02722727

*R*<sup>2</sup> is 0.027, thus age, gender and education only explains 2.7% of variation in 'biden'.

1.  Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type. Is there a problem with this model? If so, what?

``` r
biden %>%
  select(age, educ, female, biden, dem, rep) %>%
  add_predictions(biden_mult) %>%
  add_residuals(biden_mult) %>%
  {.} -> grid

griddem = filter(grid, dem == 1)
gridrep = filter(grid, rep == 1)
gridother = filter(grid, dem == 0 & rep == 0)

dem_resid_lm = lm(resid ~ pred, data = griddem)
rep_resid_lm = lm(resid ~ pred, data = gridrep)
other_resid_lm = lm(resid ~ pred, data = gridother)

griddem <- griddem %>%
  add_predictions(dem_resid_lm, var = 'pred1')
gridrep <- gridrep %>%
  add_predictions(rep_resid_lm, var = 'pred1')
gridother <- gridother %>%
  add_predictions(other_resid_lm, var = 'pred1')

ggplot(grid, aes(pred, resid)) +
  geom_point() +
  geom_line(aes(y = pred1 , color = 'Dem'), data = griddem, size = 1) +
  geom_line(aes(y = pred1, color = 'Rep'), data = gridrep, size = 1) +
  geom_line(aes(y = pred1, color = 'Ind'), data = gridother, size = 1) +
  scale_colour_manual("", values = c("Dem"="blue","Rep"="red", "Ind"="green")) +
  labs(title = "Predicted Value and Residuals of multiple variables regression",
        x = "Predicted Biden Warmth Score",
        y = "Residuals")
```

![](ps5-linear-regression_files/figure-markdown_github/unnamed-chunk-7-1.png)

There is indeed a problem. While we can see political preferences has distinct effects on residue values, the model did not seperate the data by the party effects. Thus, we need to include party identification into our model.

Multiple linear regression model (with even more variables!) (3 points)
=======================================================================

Estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>3</sub> + *β*<sub>4</sub>*X*<sub>4</sub> + *β*<sub>5</sub>*X*<sub>5</sub>

where *Y* is the Joe Biden feeling thermometer, *X*<sub>1</sub> is age, *X*<sub>2</sub> is gender, *X*<sub>3</sub> is education, *X*<sub>4</sub> is Democrat, and *X*<sub>5</sub> is Republican.[2] Report the parameters and standard errors.

``` r
biden_mm = lm(biden ~ age + female + educ + dem + rep, data = biden)
tidy(biden_mm)
```

    ##          term     estimate std.error  statistic      p.value
    ## 1 (Intercept)  58.81125899 3.1244366  18.822996 2.694143e-72
    ## 2         age   0.04825892 0.0282474   1.708438 8.772744e-02
    ## 3      female   4.10323009 0.9482286   4.327258 1.592601e-05
    ## 4        educ  -0.34533479 0.1947796  -1.772952 7.640571e-02
    ## 5         dem  15.42425563 1.0680327  14.441745 8.144928e-45
    ## 6         rep -15.84950614 1.3113624 -12.086290 2.157309e-32

1.  Did the relationship between gender and Biden warmth change?

It indeed changed. The relationship between gender and Biden warmth was 1.863612e-08 but now is 1.592601e-05 when 'dem' and 'rep' are considered.

1.  Report the *R*<sup>2</sup> of the model. What percentage of the variation in `biden` does age, gender, education, and party identification explain? Is this a better or worse model than the age + gender + education model?

``` r
summary(biden_mm)$r.squared
```

    ## [1] 0.2815391

The *R*<sup>2</sup> of the model is 0.2815391, i.e. 28.15% of the variation in 'biden' was explained by age, gender, education and party identification. Though this number is not good enough, but is still significantly higher than the previous models.

1.  Generate a plot comparing the predicted values and residuals, drawing separate smooth fit lines for each party ID type. By adding variables for party ID to the regression model, did we fix the previous problem?

``` r
biden %>%
  select(age, educ, female, biden, dem, rep) %>%
  add_predictions(biden_mm) %>%
  add_residuals(biden_mm) %>%
  {.} -> grid

griddem = filter(grid, dem == 1)
gridrep = filter(grid, rep == 1)
gridind = filter(grid, dem == 0 & rep == 0)

dem_resid_lm = lm(resid ~ pred, data = griddem)
rep_resid_lm = lm(resid ~ pred, data = gridrep)
ind_resid_lm = lm(resid ~ pred, data = gridind)

griddem %>%
  add_predictions(dem_resid_lm, var='pred1') %>%
  {.} -> griddem

gridrep %>%
  add_predictions(rep_resid_lm, var='pred1') %>%
  {.} -> gridrep

gridind %>%
  add_predictions(ind_resid_lm, var='pred1') %>%
  {.} -> gridind

ggplot(grid, aes(pred, resid)) +
  labs(title = 'Residuals vs. Predicted values of Warmth Score',
       y = 'Residual value',
       x = 'Predicted warmth score') +
  geom_point() +
  geom_line(aes(y = pred1, color = "DEM"), data = griddem, size = 1) +
  geom_line(aes(y = pred1, color = "REP"), data = gridrep, size = 1) +
  geom_line(aes(y = pred1, color = "IND"), data = gridind, size = 1) +
  scale_color_manual('', values = c("DEM" = "blue", "REP" = "red", "IND" = "green"))
```

![](ps5-linear-regression_files/figure-markdown_github/unnamed-chunk-10-1.png)

We did solve the previous problem. Now all three lines have slope and intercepts approximately 0 and such similar pattern suggests

Before the, the smooth fit regression lines for each of the three possible party affiliations was distinct, with slightly differing slopes and very different (visually) intercepts. Now, however, after we have included party affiliation into our model, we see that all three smooth fit lines for Democrats, Republicans, and Independents have a slope of approximately 0 as well as a 0 intercept. They are all quite similar now, suggesting that in our current model that party affiliation, or lack thereof, has no effect on our residuals.

Note however, that there is still a distinct pattern to the residuals. While this is not ideal, it is because age, and years of education (educ) are only measured as integers in our data, and since we did not include them as factors (which would be cumbersome, to be honest) in our model, they are treated as if they exist in the real numbers.

Interactive linear regression model (2 points)
==============================================

Let's explore this relationship between gender and Biden warmth more closely. Perhaps the effect of gender on Biden warmth differs between partisan affiliation. That is, not only do we need to account for the effect of party ID in our linear regression model, but that gender has a different effect for Democrats and Republicans. Democrats are already predisposed to favor Joe Biden and have warm thoughts about him, whereas Republicans are predisposed to dislike him. But because Biden is so charming, he can woo female Republicans better than male Republicans. This suggests an **interactive** relationship between gender and party ID.

Filter your dataset to remove any independent respondents (keeping only those who identify as Democrats or Republicans), and estimate the following linear regression:

*Y* = *β*<sub>0</sub> + *β*<sub>1</sub>*X*<sub>1</sub> + *β*<sub>2</sub>*X*<sub>2</sub> + *β*<sub>3</sub>*X*<sub>1</sub>*X*<sub>2</sub>

where *Y* is the Joe Biden feeling thermometer, *X*<sub>1</sub> is gender, and *X*<sub>2</sub> is Democrat. Report the parameters and standard errors.

``` r
biden_int <- biden %>% 
  filter(dem == 1 | rep == 1)  %>%
  lm(biden ~ female * dem, data = .)
tidy(biden_int)
```

    ##          term  estimate std.error statistic       p.value
    ## 1 (Intercept) 39.382022  1.455363 27.059928 4.045546e-125
    ## 2      female  6.395180  2.017807  3.169371  1.568102e-03
    ## 3         dem 33.687514  1.834799 18.360328  3.295008e-66
    ## 4  female:dem -3.945888  2.471577 -1.596506  1.106513e-01

1.  Estimate predicted Biden warmth feeling thermometer ratings and 95% confidence intervals for female Democrats, female Republicans, male Democrats, and male Republicans. Does the relationship between party ID and Biden warmth differ for males/females? Does the relationship between gender and Biden warmth differ for Democrats/Republicans?

``` r
(pred_ci <- biden_int$model %>%
    data_grid(female, dem) %>%
    augment(biden_int, newdata = .) %>%
    mutate(ymin = .fitted - .se.fit * 1.96,
         ymax = .fitted + .se.fit * 1.96) %>%
    rename(c('female' = 'gender', 'dem' = 'party', '.fitted' = 'feeling warmth', 'ymin' = 'lb_confident interval', 'ymax' = 'ub_confident interval')) %>%
    mutate(gender = ifelse(gender == 0, "Male", "Female"),
         party = ifelse(party == 0, "Republican", "Democrat")))
```

    ##   gender      party feeling warmth   .se.fit lb_confident interval
    ## 1   Male Republican       39.38202 1.4553632              36.52951
    ## 2   Male   Democrat       73.06954 1.1173209              70.87959
    ## 3 Female Republican       45.77720 1.3976638              43.03778
    ## 4 Female   Democrat       75.51883 0.8881114              73.77813
    ##   ub_confident interval
    ## 1              42.23453
    ## 2              75.25949
    ## 3              48.51662
    ## 4              77.25953

Comparing feeling warmth between genders inside parties, warmth for male is generally lower than that of for female (39.38202 vs 45.77720 for Republicans and 73.06954 vs 75.51883 for Democrats) but significant. On the other hand, when comparing feelings for Biden between parties for each gender, there appear a large differnce in the numbers with 39.38202 vs 73.06954 for male and 45.77720 vs 75.51883 for female. Similar patterns can be found in 95% confidnet intervals as well.

[1] Feeling thermometers are a common metric in survey research used to gauge attitudes or feelings of warmth towards individuals and institutions. They range from 0-100, with 0 indicating extreme coldness and 100 indicating extreme warmth.

[2] Independents must be left out to serve as the baseline category, otherwise we would encounter perfect multicollinearity.
