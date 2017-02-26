Problem set \#6: GLM
================
Tong Ju/ ID: 12146007
**2017/2/19**

-   [1) Describe the data](#describe-the-data)
-   [2) Basic Model](#basic-model)
-   [3) Multiple Variable Model](#multiple-variable-model)
-   [4) Modeling TV Consumption](#modeling-tv-consumption)

1) Describe the data
====================

``` r
# generate the new counted dataframe, remove the missing value and calculate the percentage of voting and no voting.
m<-mhealth %>%
  count(vote96) %>%
  na.omit %>%
  mutate (vote = ifelse (vote96 == 0, "No","Yes"), percent = 100*n / sum(n)) 


# barchart for tournout   
ggplot(data = m, aes(x=vote, y=n, fill=vote)) +
  geom_bar(stat='identity') + 
  theme_bw()+
  labs(title = 'Voter Turnout in 1996', x = 'Vote Status', y = 'Number of people') +
  theme(plot.title = element_text(hjust = 0.5))
```

![](ps6_files/figure-markdown_github/Histogram-1.png)

``` r
# Remove all the missing data in vote96 and mhealth_sum 
df = mhealth[(!is.na(mhealth$vote96) & !is.na(mhealth$mhealth_sum)), ]
# Plot the linear regression fitting line
ggplot(df, aes(mhealth_sum, vote96)) +
  geom_point(alpha=0.2) +
  geom_smooth(method = "lm") + 
  theme_bw()+
  scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1)) + 
  labs(title = "Voting in 1996 vs. Mental Health Index \n(with Smoothing Linear Line)",
       y = "Voting: Yes (1) vs. No (0)",
       x = "Mental Health Index (no depression = 0, extreme depression = 9 )")+
  theme(plot.title = element_text(hjust = 0.5))
```

![](ps6_files/figure-markdown_github/Histogram-2.png)

``` r
# make the table for question 1
pander(m)
```

<table style="width:42%;">
<colgroup>
<col width="12%" />
<col width="6%" />
<col width="9%" />
<col width="12%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">vote96</th>
<th align="center">n</th>
<th align="center">vote</th>
<th align="center">percent</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">0</td>
<td align="center">830</td>
<td align="center">No</td>
<td align="center">31.76426</td>
</tr>
<tr class="even">
<td align="center">1</td>
<td align="center">1783</td>
<td align="center">Yes</td>
<td align="center">68.23574</td>
</tr>
</tbody>
</table>

1)As the table shown above, the unconditional probability of a given individual turning out to vote is 68.24%. 2)The scatterplot with the linear smoothing line indicates, in general, people who are more depressive (with higher value of mental health index) will be less likely to vote. Therefore, there is certain association between mental health with the action of voting based on this plotting. However, this graph is very misleading when considering the relationship between voting status (binary variable) and the mental health index (discrete variable): 1) most importantly,the voter turnout follows the binorminal distribution, whose feature cannot be captured by the linear relationship. Since voter turnout is dichotomous, with no possile outcomes in between, the regression line is misleading. 2) Even though we consider the predicted value by the regression line between 0 and 1 as the predicted probability, there are still other problems: because the regression line is linear and continuous, it extends infinitely in both directions of the mental health index. But as we know, the predictions based on the mental health value, lower than 0 or higher than 9, do not make any sense. 3) According to this linear regression, there is no upper and lower bound for the predicted voting probability, the prediction may exceed 1.0 or may be lower than 0, which do not comply with the definition that the probability is between 0 and 1.

2) Basic Model
==============

``` r
# basic glm model
basic <- glm(vote96 ~ mhealth_sum, data = df, family = binomial)
# make the table
pander(summary(basic))
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
<th align="center">z value</th>
<th align="center">Pr(&gt;|z|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>mhealth_sum</strong></td>
<td align="center">-0.1435</td>
<td align="center">0.01969</td>
<td align="center">-7.289</td>
<td align="center">3.134e-13</td>
</tr>
<tr class="even">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">1.139</td>
<td align="center">0.08444</td>
<td align="center">13.49</td>
<td align="center">1.759e-41</td>
</tr>
</tbody>
</table>

(Dispersion parameter for binomial family taken to be 1 )

<table style="width:61%;">
<colgroup>
<col width="26%" />
<col width="34%" />
</colgroup>
<tbody>
<tr class="odd">
<td align="center">Null deviance:</td>
<td align="left">1672 on 1321 degrees of freedom</td>
</tr>
<tr class="even">
<td align="center">Residual deviance:</td>
<td align="left">1617 on 1320 degrees of freedom</td>
</tr>
</tbody>
</table>

``` r
# defibe the fundamental functions
logit2prob <- function(x){
  exp(x) / (1 + exp(x))
}

prob2odds <- function(x){
  x / (1 - x)
}

prob2logodds <- function(x){
  log(prob2odds(x))
}



basic_pred<-df %>%
  dplyr::select(vote96, mhealth_sum) %>%
  add_predictions(basic, var = 'logit') %>%
  mutate(prob = logit2prob(logit)) %>%
  mutate(odds = prob2odds(prob))
```

1)We find a statistically significant relationship at the p &lt; .0005 level between depression score and voting behavior. The relationship is negative and the coefficient is -0.1435. Therefore, we can conclude that there is a statistically significant. The co-effecient on `mhealth_sum` is -0.1435, indicating that one-unit increase in depression score could result in the decrease of log-odds of voting by 0.1435. Thus, the association between these two variable is moderately substantive and negative.

``` r
# log odds plotting
ggplot(basic_pred, aes(mhealth_sum)) + 
  theme_bw()+
  geom_line(aes(y = logit), color = "red") + 
  labs(title = "Log Odds of Voting in 1996 vs. Mental Health Status", 
       x = "Mental Health Index",
       y = "Log odds of Voting")+
   theme(plot.title = element_text(hjust = 0.5))
```

![](ps6_files/figure-markdown_github/basic%20model%202-1.png)

2)The Log-odds graph is shown as above. The interpretation for the estimated parameter for mental health in terms of log-odds: as each one unit increase in mental health score, we expect the log-odds of voting status to decrease by 0.1435.

``` r
# odds plotting
ggplot(basic_pred, aes(mhealth_sum)) + 
  theme_bw()+
  geom_line(aes(y = odds), color = "red") + 
  labs(title = "Odds of Voting in 1996 vs. Mental Health Status", 
       x = "Mental Health Index",
       y = "Log odds of Voting")+
   theme(plot.title = element_text(hjust = 0.5))
```

![](ps6_files/figure-markdown_github/basic%20model%203-1.png)

3)Interpretation of the estimated parameter for mental health in terms of odds: as each one unit increase in mental health score, we expect to see 86.63% (note: it is calculated from exp(-0.1435) ) decrease of the odds of voting.

``` r
# probability plotting
ggplot(basic_pred, aes(mhealth_sum)) + 
  theme_bw()+
  geom_line(aes(y = prob), color = "red") + 
  labs(title = "Probability of Voting in 1996 vs. Mental Health Status", 
       x = "Mental Health Index",
       y = "Log odds of Voting")+
   theme(plot.title = element_text(hjust = 0.5))
```

![](ps6_files/figure-markdown_github/basic%20model%204-1.png)

``` r
# Add the predicted value to 0 to 9
score <- tibble(mhealth_sum = 0:9)  %>%
  add_predictions(basic, var = 'logit') %>%
  mutate(prob = logit2prob(logit))

pander(score, caption ="Predicted Prob. for Mental Health Index")
```

<table style="width:50%;">
<caption>Predicted Prob. for Mental Health Index</caption>
<colgroup>
<col width="19%" />
<col width="18%" />
<col width="12%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">mhealth_sum</th>
<th align="center">logit</th>
<th align="center">prob</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">0</td>
<td align="center">1.139209676</td>
<td align="center">0.7575345</td>
</tr>
<tr class="even">
<td align="center">1</td>
<td align="center">0.995734471</td>
<td align="center">0.7302191</td>
</tr>
<tr class="odd">
<td align="center">2</td>
<td align="center">0.852259266</td>
<td align="center">0.7010409</td>
</tr>
<tr class="even">
<td align="center">3</td>
<td align="center">0.708784061</td>
<td align="center">0.6701324</td>
</tr>
<tr class="odd">
<td align="center">4</td>
<td align="center">0.565308857</td>
<td align="center">0.6376800</td>
</tr>
<tr class="even">
<td align="center">5</td>
<td align="center">0.421833652</td>
<td align="center">0.6039219</td>
</tr>
<tr class="odd">
<td align="center">6</td>
<td align="center">0.278358447</td>
<td align="center">0.5691437</td>
</tr>
<tr class="even">
<td align="center">7</td>
<td align="center">0.134883242</td>
<td align="center">0.5336698</td>
</tr>
<tr class="odd">
<td align="center">8</td>
<td align="center">-0.008591963</td>
<td align="center">0.4978520</td>
</tr>
<tr class="even">
<td align="center">9</td>
<td align="center">-0.152067168</td>
<td align="center">0.4620563</td>
</tr>
</tbody>
</table>

4)Based on the table above, we can caluculate out the predicted probability difference the mental health index from 1 to 2 is 0.0273154; the difference between the index from 5 to 6 is 0.0347782. In order to interpret parameter in terms of probabilities, we need to specify our initial value for the mental health socre, since the relationship between the probabality of voting and score is not linear. For example, to compare the first difference in the probablity of voting for people with score 1 and those with socre 2, we can say, depression index score increases from 1 to 2, the probability of voting decreases by factor of 0.0273154.

``` r
pred<-basic_pred %>%
  mutate (pred_vote= ifelse(prob > .5, 1, 0))

acc_rate <- mean(pred$vote96 == pred$pred_vote)
e2 <- 1 - acc_rate
e1 <- 1 - mean(pred$vote96 == 1)

pre <- (e1 - e2) / e1

auc_score = auc(pred$vote96, pred$pred_vote)
```

5)Given a threshhold of .5, the accuracy rate is 67.78% and the proportional reduction in error is 1.62%. The AUC is 0.5401 (the AUC score takes into account all possible threshhold values). First, accuracy rate tells us that there is 67.78% of the predictions based on mental health index were correct. But only based on this we cannot conclude this model is good or not. So from the AUC, we know area under the (ROC) curve (AUC) is only 0.5401 (AUC of random guess is 0.5). In addition, this model only reduce 1.62% of the prediction error. I argue this model is not so perfect, which is only a littl bit better than a random guess. We need to include other variable into the model to see whether we can improve it.

3) Multiple Variable Model
==========================

1)The random component of the probability distribution: vote96 is distributed as a binomial random variable. \(vote96_i\) (each row of our dataframe) follows a Bernoulli distribution and thus the sum of all individual \(vote96_{i}\ 's\) (i.e. the entire column vote96) is distributed as a binomial random variable.

![](data/3-1.png)

In my case, I only choose mental health index, age, education, gender as predictors. The linear predictor is: \[vote96_{i} = \beta_{0} + \beta_{1}ageh + \beta_{2}educ + \beta_{3}married + \beta_{4}inc10\]

Link function is:

![](data/3-2.png)

``` r
# model
logit_m = glm(vote96 ~ age + educ + married + inc10, family = binomial, data=df)

pander(summary(logit_m))
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
<th align="center">z value</th>
<th align="center">Pr(&gt;|z|)</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center"><strong>age</strong></td>
<td align="center">0.04388</td>
<td align="center">0.004741</td>
<td align="center">9.256</td>
<td align="center">2.125e-20</td>
</tr>
<tr class="even">
<td align="center"><strong>educ</strong></td>
<td align="center">0.2384</td>
<td align="center">0.02917</td>
<td align="center">8.171</td>
<td align="center">3.052e-16</td>
</tr>
<tr class="odd">
<td align="center"><strong>married</strong></td>
<td align="center">0.3279</td>
<td align="center">0.1516</td>
<td align="center">2.163</td>
<td align="center">0.03051</td>
</tr>
<tr class="even">
<td align="center"><strong>inc10</strong></td>
<td align="center">0.08077</td>
<td align="center">0.02615</td>
<td align="center">3.089</td>
<td align="center">0.002008</td>
</tr>
<tr class="odd">
<td align="center"><strong>(Intercept)</strong></td>
<td align="center">-4.783</td>
<td align="center">0.4761</td>
<td align="center">-10.04</td>
<td align="center">9.672e-24</td>
</tr>
</tbody>
</table>

(Dispersion parameter for binomial family taken to be 1 )

<table style="width:61%;">
<colgroup>
<col width="26%" />
<col width="34%" />
</colgroup>
<tbody>
<tr class="odd">
<td align="center">Null deviance:</td>
<td align="left">1468 on 1164 degrees of freedom</td>
</tr>
<tr class="even">
<td align="center">Residual deviance:</td>
<td align="left">1258 on 1160 degrees of freedom</td>
</tr>
</tbody>
</table>

2)To establish a model by using the varaibles rather than the mental health index: through examination of the models with variables provided forward and backward, age, education, married status ,and income are chosen as the predictors estimating the voter turnout, since these four show stronger relationships in the tested models with the voter turnout than others do. The model is estimated and reported as in the table above.

3)Based on the summarized table above, three variables in this model have statistically significant associations (p-value &lt; 0.005) with voting variable: p-value for age, education, and income are 2.125e-20 , 3.052e-16 , and 0.002008. For variable of married status has only moderately statistical significance with voting rurnout, p-value is around 0.03051. In addition, all four independent variables have positive relation with voter turnout, and the co-effecient on age, education, maritial status and income are 0.04388, 0.3279, 0.08077, separately.

Interpretation of co-effecients in terms of log odds, odds: As each one year increase in age, we expect the log-odds of voting status to increase by 0.04388;and expect to see see 104.49% increase in the odds of voting, when holding other variables constant. As each one year increase in education year, we expect the log-odds of voting status to increase by 0.2384; and expect to see see 126.93% increase in the odds of voting, when holding other variables constant. As each 10k increase in annual income, we expect the log-odds of voting status to increase by 0.08077; and expect to see see 108.41% increase in the odds of voting, when holding other variables constant. Married people has higher log odds of voting than unmarried ones by factor of 0.3279; or say, Married people has increased their odds of voting by 138.81% than those unmarried ones, when controlling all the other variables.

Holding the income and education year as their means, the probability of voting is estimated by age and maritial status. The result is plotted as below, which indicates that as the age increases, the probability of being a viter will increase, and the married people are more likely to vote than those who are unmarried. Taken together, people who are elder and married have higher possibility to vote than those younger and unmarried. For example, At age 20, the turnout rate is 5% higher for married people than unmarried people; at age 80, the difference shrinks to around 2.5%. This result is consistent with the literature's hypothesis.

``` r
# add the predicted values
pred_1<- df %>%
  data_grid(age, married, educ = mean(educ, na.rm = TRUE), inc10 = mean(inc10, na.rm = TRUE)) %>%
  add_predictions(logit_m, var = 'logit') %>%
  mutate(prob = logit2prob(logit)) %>%
  mutate(odds = prob2odds(prob))

ggplot (pred_1 , aes(age, prob, color = ifelse(married == 1, 'Married', 'Unmarried')))+
  theme_bw()+
  geom_line() +
  scale_color_discrete(name = "Marital status") +
  labs(title = "Probability of voter turnout",
       subtitle = "by age and marital status, controlling education and income",
       x = "Age",
       y = "Probability of voter turnout")
```

![](ps6_files/figure-markdown_github/multi%202-1.png)

Holding the age and income at their average values, the probability of voting is estimated by eudcation year and married status. The result is plotted as below, which indicates that as the education years increases, the probability of being a viter will increase, and the married people are more likely to vote than those who are unmarried. This result is also consistent with the literature's hypothesis.

``` r
# add the predicted values
pred_2<- df %>%
  data_grid(age = mean(age, na.rm = TRUE), married, educ, inc10 =mean(inc10, na.rm = TRUE)) %>%
  add_predictions(logit_m, var = 'logit') %>%
  mutate(prob = logit2prob(logit)) %>%
  mutate(odds = prob2odds(prob))

ggplot (pred_2 , aes(educ, prob, color = ifelse(married == 1, 'Married', 'Unmarried')))+
  theme_bw()+
  geom_line() +
  scale_color_discrete(name = "Marital status") +
  labs(title = "Probability of voter turnout",
       subtitle = "by education and marital status, controlling age and income",
       x = "Education years",
       y = "Probability of voter turnout")
```

![](ps6_files/figure-markdown_github/multi%203-1.png) These two plots have shown there are positive relationship between age or education year with the probability to vote. In addition, married people are more likely to vote than those unmarried.

4) Modeling TV Consumption
==========================

1)The random component of the probability distribution, tvhours is distributed as a poisson random variable (Y is the hours of TV watched a day; mu is the predictor):

![](data/4-1.png)

Linear predictor: The linear preditor is the following log-linear regression model

![](data/4-2.png)

Link function: The link function for the Poisson distribution is the log function:

![](data/4-3.png)
