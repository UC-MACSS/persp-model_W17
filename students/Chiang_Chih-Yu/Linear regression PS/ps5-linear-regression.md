PS\#5: linear regression
================
Chih-Yu Chiang
UCID 12145146
Feb. 11, 2017

Describe the data
-----------------

``` r
ggplot(data = df, aes(x = biden)) +
  geom_histogram(binwidth = 1) +
  scale_x_continuous(breaks = round(seq(min(df$biden), max(df$biden), by = 5), 1)) +
  labs(title = "Histogram of Biden's feeling thermometer",
       x = "Feeling thermometer",
       y = "Number of people")
```

![](ps5-linear-regression_files/figure-markdown_github/Describe%20the%20data-1.png)

As expected, The feeling of respondents toward Biden skews to the left. That is, it seems generally more respondents feel positive than negative toward Biden. In addition, while almost all answers fall on the multiple of 5, the feeling thermometer is probabliy evaluated by a 0-100 scale with 5-unit brackets. Strangely, there are a few answers do not fall on the multiple of 5, one at 8 and two near to 90. I'd suspect the three odd answers are probably miscoded and should be further examined regarding their effectiveness.

Simple linear regression
------------------------

``` r
#----Simple linear regression
#--Model
lm_1 <- lm(biden ~ age, data = df)
pander(summary(lm_1))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>age</strong></td>
<td align="center">0.06241</td>
<td align="center">0.03267</td>
<td align="center">1.91</td>
<td align="center">0.05626</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">59.2</td>
<td align="center">1.648</td>
<td align="center">35.92</td>
<td align="center">1.145e-213</td>
</tr>
</tbody>
</table>

<table style="width:86%;">
<caption>Fitting linear model: biden ~ age</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="12%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">1807</td>
<td align="center">23.44</td>
<td align="center">0.002018</td>
<td align="center">0.001465</td>
</tr>
</tbody>
</table>

``` r
#--Prediction
#Use augment to generate predictions
#Calculate 95% confidence intervals
pred_data <- augment(lm_1, newdata = data_frame(age = 45)) %>%
  mutate(ymin = .fitted - .se.fit * 1.96,
         ymax = .fitted + .se.fit * 1.96)
pander(pred_data)
```

<table style="width:56%;">
<colgroup>
<col width="8%" />
<col width="13%" />
<col width="13%" />
<col width="9%" />
<col width="9%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">age</th>
<th align="center">.fitted</th>
<th align="center">.se.fit</th>
<th align="center">ymin</th>
<th align="center">ymax</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">45</td>
<td align="center">62.01</td>
<td align="center">0.5577</td>
<td align="center">60.91</td>
<td align="center">63.1</td>
</tr>
</tbody>
</table>

``` r
#--Plot
#Create prediction values
grid <- df %>% 
  data_grid(age) %>% 
  add_predictions(lm_1)

#Plot
ggplot(df, aes(x = age)) +
  geom_point(aes(y = biden)) +
  geom_line(aes(y = pred), data = grid, color = "red", size = 1) +
  labs(title = "Age and Biden's feeling thermometer",
       x = "Age",
       y = "Feeling thermometer")
```

![](ps5-linear-regression_files/figure-markdown_github/Simple%20linear%20regression-1.png)

The parameters are 59.19736, for the intercept, and 0.06241, for the predictor, `age`; the standard errors are 1.64792, for the intercept, and 0.03267, for `age`.

1.  As the p-value of this model is 0.05626, smaller than 0.1, the predictor, `age`, has a weak relationship with the response, `biden`, at a 90% condifence level.

2.  The relationship is, though statistically significant at a 90% condifence level, not strong. First, With p-value 0.05626, there are more than 5% chance we observe the dataset in hand while the null hypothesis holds, the predictor has no relationship with the response. Second, the estimate (parameter) of `age` is fairly small (0.06241), meaning, even a 10-years difference in `age` can only influence approximately 0.6 unit (out of 100) of the `biden`, a tiny difference.

3.  With the estimate (parameter) of the predictor, `age`, is positive (0.06241), the relationship between it and the response, `biden`, is positive. That means, when `age` increases, `biden` increases as well.

4.  The *R*<sup>2</sup> of this model is 0.002018. That indicates this model can only explain 0.2% of the variation in `biden`. While the percentage is pretty low, the model is not a good one in explaning the response, `biden`.

5.  The predicted `biden` associated with `age` at 45 is 62.0056. Its associated 95% confidence interval is (60.91248, 63.09872).

6.  The line has a slightly positive slope, which indicates a potential positive relationship between `age` and `biden`. However, there's no clear pattern as observing the dots themselves; that means the relationship between `age` and `biden` is relatively weak, conforming to the statistics conclusions.

Multiple linear regression
--------------------------

``` r
#----Multiple linear regression
#--Model
lm_2 <- lm(biden ~ age + female + educ, data = df)
pander(summary(lm_2))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>age</strong></td>
<td align="center">0.04188</td>
<td align="center">0.03249</td>
<td align="center">1.289</td>
<td align="center">0.1975</td>
</tr>
<tr class="even">
<td align="center"><strong>female</strong></td>
<td align="center">6.196</td>
<td align="center">1.097</td>
<td align="center">5.65</td>
<td align="center">1.864e-08</td>
</tr>
<tr class="odd">
<td align="center"><strong>educ</strong></td>
<td align="center">-0.8887</td>
<td align="center">0.2247</td>
<td align="center">-3.955</td>
<td align="center">7.941e-05</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">68.62</td>
<td align="center">3.596</td>
<td align="center">19.08</td>
<td align="center">4.337e-74</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: biden ~ age + female + educ</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">1807</td>
<td align="center">23.16</td>
<td align="center">0.02723</td>
<td align="center">0.02561</td>
</tr>
</tbody>
</table>

``` r
#--Plot
#Create prediction and residual values
df_lm2 <- df %>% 
  add_predictions(lm_2) %>%
  add_residuals(lm_2)

ggplot(df_lm2, aes(x = pred, y = resid)) + geom_point() +
  stat_smooth(data = filter(df_lm2, dem == 1), mapping = aes(colour = "Democrat"), size = 1) +
  stat_smooth(data = filter(df_lm2, rep == 1), mapping = aes(colour = "Republican"), size = 1) +
  stat_smooth(data = filter(df_lm2, rep == 0, dem == 0), mapping = aes(colour = "Independent"), size = 1) +
  labs(title = "Predicted Biden's feeling thermometer and model residual",
       x = "Predicted Biden's feeling thermometer",
       y = "Model residual") +
  scale_color_manual(name = "Party", values = c("blue", "purple", "red"))
```

    ## `geom_smooth()` using method = 'loess'
    ## `geom_smooth()` using method = 'loess'
    ## `geom_smooth()` using method = 'loess'

![](ps5-linear-regression_files/figure-markdown_github/Multiple%20linear%20regression-1.png)

The parameters are 68.62101, for the intercept, 0.04188, for `age`, 6.19607, for `female`, and -0.88871, for `educ`; the standard errors are 3.59600, for the intercept, 0.03249, for `age`, 1.09670, for `female`, and 0.22469, for `educ`.

1.  There is a statistically significant relationship between the predictors and response. at a 99.9% condifence level, as the model p-value is 8.876e-11, very close to 0.

2.  The parameter (estimate) of `female` is 6.19607, which suggests a positive and strong (relative to `age`) relationship between `female` and `biden`. That indicates, when controlling `age` and `edu`, female respondents have an average 6.19607 higher score in `biden` than the male respondents.

3.  The *R*<sup>2</sup> of this model is 0.02723. That indicates the predictors in this model can only explain 2.7% of the variation in `biden`. While the percentage is pretty low, the model is not a good one in explaning the response, `biden`. As the percentage (2.7%) is higher than of the age-only model (0.2%), this model is superior to the age-only model regarding its explanation power. This model also has a residual median closer to 0 and 1Q and 3Q more equal-distance from the median, which indicates the residuals disperse more noramally. These characteristics show that the residuals behave more like the real, natural error and support the notion this is a better model than the last one.

4.  One potential problem of this model is that it does not include the party factor, which obviously has a relationship with `biden`. We can observe from the smoothing lines that there's a structural difference between Democrat and Republican respondents regarding the residuals. The smooth line for Democrat is constantly above 0, while the Republican one is under 0. This indicates that Democrats generally give a higher `biden` score than Republicans.

Multiple linear regression model
--------------------------------

``` r
#----Multiple linear regression model
#--Model
lm_3 <- lm(biden ~ age + female + educ + dem + rep, data = df)
pander(summary(lm_3))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>age</strong></td>
<td align="center">0.04826</td>
<td align="center">0.02825</td>
<td align="center">1.708</td>
<td align="center">0.08773</td>
</tr>
<tr class="even">
<td align="center"><strong>female</strong></td>
<td align="center">4.103</td>
<td align="center">0.9482</td>
<td align="center">4.327</td>
<td align="center">1.593e-05</td>
</tr>
<tr class="odd">
<td align="center"><strong>educ</strong></td>
<td align="center">-0.3453</td>
<td align="center">0.1948</td>
<td align="center">-1.773</td>
<td align="center">0.07641</td>
</tr>
<tr class="even">
<td align="center"><strong>dem</strong></td>
<td align="center">15.42</td>
<td align="center">1.068</td>
<td align="center">14.44</td>
<td align="center">8.145e-45</td>
</tr>
<tr class="odd">
<td align="center"><strong>rep</strong></td>
<td align="center">-15.85</td>
<td align="center">1.311</td>
<td align="center">-12.09</td>
<td align="center">2.157e-32</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">58.81</td>
<td align="center">3.124</td>
<td align="center">18.82</td>
<td align="center">2.694e-72</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: biden ~ age + female + educ + dem + rep</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">1807</td>
<td align="center">19.91</td>
<td align="center">0.2815</td>
<td align="center">0.2795</td>
</tr>
</tbody>
</table>

``` r
#--Plot
#Create prediction and residual values
df_lm3 <- df %>% 
  add_predictions(lm_3) %>%
  add_residuals(lm_3)

ggplot(df_lm3, aes(x = pred, y = resid)) +
  geom_point() +
  geom_smooth(data = filter(df_lm3, dem == 1), aes(color = "Democrat"), size = 1) +
  geom_smooth(data = filter(df_lm3, rep == 1), aes(color = "Republican"), size = 1) +
  stat_smooth(data = filter(df_lm3, rep == 0, dem == 0), mapping = aes(colour = "Independent"), size = 1) +
  labs(title = "Predicted Biden's feeling thermometer and model residual",
       x = "Predicted Biden's feeling thermometer",
       y = "Model residual") +
  scale_color_manual(name = "Party", values = c("blue", "purple", "red"))
```

    ## `geom_smooth()` using method = 'loess'
    ## `geom_smooth()` using method = 'loess'
    ## `geom_smooth()` using method = 'loess'

![](ps5-linear-regression_files/figure-markdown_github/Multiple%20linear%20regression%20model-1.png)

The parameters are 58.81126, for the intercept, 0.04826, for `age`, 4.10323, for `female`, -0.34533, for `educ`, 15.42426, for `dem`, and -15.84951, for `rep`; the standard errors are 3.12444, for the intercept, 0.02825, for `age`, 0.94823, for `female`, 0.19478, for `educ`, 1.06803, for `dem`, and 1.31136, for `rep`.

1.  Yes. The relationship between gender and Biden warmth is reletively weaker in this model. The parameter (estimate) of `female` is smaller (4.10323 compared to 6.19607), which means a weaker positive correlation between gender and `biden`; the p-value is larger (1.59e-05 compared to 1.86e-08), though still small enough to claim the statistical significance, indicates a higher probability that the null hypotesis, gender has no relationship with `biden`, holds.

2.  The *R*<sup>2</sup> of this model is 0.2815. That indicates the predictors in this model explain 28.2% of the variation in `biden`, much can be attributed to the party tags (both estimates are significant and large, with 15.42426 and -15.84951 respectively). As the percentage (28.2%) is a lot higher than of the age + gender + education model (2.7%), this model is superior to the last model regarding its explanation power. This model also has residual 1Q and 3Q more equal-distance from the median, which indicates the residuals disperse more noramally. This characteristics show that the residuals behave more like the real, natural error and support the notion this is a better model than the last one.

3.  The problem observed in the last model has been fixed, as we can see the smooth lines of Democrat, Independent, and Republican are at a similar level close to 0, and the residuals are generally normally distributed with mean close to 0.

Interactive linear regression model
-----------------------------------

``` r
#----Interactive linear regression model
#--Model
#Filter out neutral respondents
df_lm4 <- df %>%
  filter(dem == 1 | rep == 1)

#Train model
lm_4 <- lm(biden ~ female + dem + female * dem, data = df_lm4)
pander(summary(lm_4))
```

<table style="width:86%;">
<colgroup>
<col width="25%" />
<col width="15%" />
<col width="18%" />
<col width="13%" />
<col width="13%" />
</colgroup>
<thead>
<tr class="header">
<th align="center"> </th>
<th align="center">Estimate</th>
<th align="center">Std. Error</th>
<th align="center">t value</th>
<th align="center">Pr(&gt;|t|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>female</strong></td>
<td align="center">6.395</td>
<td align="center">2.018</td>
<td align="center">3.169</td>
<td align="center">0.001568</td>
</tr>
<tr class="even">
<td align="center"><strong>dem</strong></td>
<td align="center">33.69</td>
<td align="center">1.835</td>
<td align="center">18.36</td>
<td align="center">3.295e-66</td>
</tr>
<tr class="odd">
<td align="center"><strong>female:dem</strong></td>
<td align="center">-3.946</td>
<td align="center">2.472</td>
<td align="center">-1.597</td>
<td align="center">0.1107</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">39.38</td>
<td align="center">1.455</td>
<td align="center">27.06</td>
<td align="center">4.046e-125</td>
</tr>
</tbody>
</table>

<table style="width:85%;">
<caption>Fitting linear model: biden ~ female + dem + female * dem</caption>
<colgroup>
<col width="20%" />
<col width="30%" />
<col width="11%" />
<col width="22%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">Observations</th>
<th align="center">Residual Std. Error</th>
<th align="center"><span class="math inline"><em>R</em><sup>2</sup></span></th>
<th align="center">Adjusted <span class="math inline"><em>R</em><sup>2</sup></span></th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">1151</td>
<td align="center">19.42</td>
<td align="center">0.3756</td>
<td align="center">0.374</td>
</tr>
</tbody>
</table>

``` r
#--Prediction
#Create data for prediction
pred_data <- data_frame(female = c(0, 0, 1, 1), dem = c(0, 1, 0, 1))

#Use augment to generate predictions
pred_aug <- augment(lm_4, newdata = pred_data)

#Calculate 95% confidence intervals
pred_ci <- mutate(pred_aug,
                  ymin = .fitted - .se.fit * 1.96,
                  ymax = .fitted + .se.fit * 1.96)
pander(pred_ci)
```

<table style="width:68%;">
<colgroup>
<col width="12%" />
<col width="8%" />
<col width="13%" />
<col width="13%" />
<col width="9%" />
<col width="9%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">female</th>
<th align="center">dem</th>
<th align="center">.fitted</th>
<th align="center">.se.fit</th>
<th align="center">ymin</th>
<th align="center">ymax</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">0</td>
<td align="center">0</td>
<td align="center">39.38</td>
<td align="center">1.455</td>
<td align="center">36.53</td>
<td align="center">42.23</td>
</tr>
<tr class="even">
<td align="center">0</td>
<td align="center">1</td>
<td align="center">73.07</td>
<td align="center">1.117</td>
<td align="center">70.88</td>
<td align="center">75.26</td>
</tr>
<tr class="odd">
<td align="center">1</td>
<td align="center">0</td>
<td align="center">45.78</td>
<td align="center">1.398</td>
<td align="center">43.04</td>
<td align="center">48.52</td>
</tr>
<tr class="even">
<td align="center">1</td>
<td align="center">1</td>
<td align="center">75.52</td>
<td align="center">0.8881</td>
<td align="center">73.78</td>
<td align="center">77.26</td>
</tr>
</tbody>
</table>

The parameters are 39.382, for the intercept, 6.395, for `female`, 33.688, for `dem`, and -3.946, for the interactive term of `female` and `dem`; the standard errors are 1.455, for the intercept, 2.018, for `female`, 1.835, for `dem`. and 2.472, for the interactive term of `female` and `dem`.

-   Generally, the gender factor has a negative effect on the party ID factor's relationship with `biden` and vice versa, while we can observe the negative parameter (-3.946) of their interactive term; however, with a 0.11065 p-value, this effect is not statistically significant even at a 90% confidence level. That is, we can not conclude that the interactive relationship exists with the current sample and confidence level.

-   Examining the confidence interval, we can see some trace indicating the potentially different relationships between party ID and `biden` regarding different genders. Generally, the party ID's relationship with `biden` is weaker for females than males, with all differences between party IDs in predicted value (33.68752 for male and 29.74163 for female), CI upper bound (33.02496 for male and 28.74291 for female), and CI lower bound (35.35008 for male and 30.74035 for female) are smaller for females than males.

-   From another angle, the CIs illustrate the potentially different relationship between `biden` and gender factor, regarding different party IDs.The relationship between `biden` and gender factor is potentially stronger for Republicans than Democrats. For one, all differences of this relationship are larger as for Republican than for Democrats: difference in predicted value (33.68752 for Republican and 29.74163 for Democrats), CI upper bound (33.68752 for Republican and 29.74163 for Democrats), and CI lower bound (33.68752 for Republican and 29.74163 for Democrats). For another, the Democrats' CIs of different genders overlaps each other ((70.87959, 75.25949) for male Democrats and (73.77813, 77.25953) for female Democrats), implicating that we are not able to conclude that there's a statistically significant difference between the female Democrats and male Democrats' relationships with `biden` at a 95% confidence level. Compared to Democrats', the Republican's CIs of different genders do not overlap each other ((36.52951, 42.23453) for male Republicans and (43.03778, 48.51662) for female Republicans) and indicates a statistically significant difference between the female Republicans and male Republicans' relationships with `biden` at a 95% confidence level. The change between these two relationships implies the stronger relationship between gender and `biden` for Republicans than Democrats.
