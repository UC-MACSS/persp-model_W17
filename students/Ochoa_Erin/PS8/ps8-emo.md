MACS 30100 PS8
================
Erin M. Ochoa
2017 March 6

-   [Part 1: Joe Biden (redux times two)](#part-1-joe-biden-redux-times-two)
-   [Part 2: Modeling voter turnout](#part-2-modeling-voter-turnout)
-   [Part 3: OJ Simpson \[4 points\]](#part-3-oj-simpson-4-points)
-   [Submission instructions](#submission-instructions)
    -   [If you use R](#if-you-use-r)
    -   [If you use Python](#if-you-use-python)

``` r
mse = function(model, data) {
  x = modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}


# There seems to be a bug in the gbm function.
# Work-around method found here: http://www.samuelbosch.com/2015/09/workaround-ntrees-is-missing-in-r.html

predict.gbm = function (object, newdata, n.trees, type = "link", single.tree = FALSE, ...) {
  if (missing(n.trees)) {
    if (object$train.fraction < 1) {
      n.trees = gbm.perf(object, method = "test", plot.it = FALSE)
    }
    else if (!is.null(object$cv.error)) {
      n.trees = gbm.perf(object, method = "cv", plot.it = FALSE)
    }
    else {
      n.trees = length(object$train.error)
    }
    cat(paste("Using", n.trees, "trees...\n"))
    gbm::predict.gbm(object, newdata, n.trees, type, single.tree, ...)
  }
}
```

Part 1: Joe Biden (redux times two)
===================================

We read in the data:

``` r
df = read.csv('data/biden.csv')
```

Next, we split the dataset into training and validation sets in a ratio of 7:3, then estimate a regression tree:

``` r
set.seed(1234)

biden_split7030 = resample_partition(df, c(test = 0.3, train = 0.7))
biden_train70 = biden_split7030$train %>%
                tbl_df()
biden_test30 = biden_split7030$test %>%
               tbl_df()

# estimate model
biden_tree1 = tree(biden ~ female + age + educ + dem + rep, data = biden_train70)
mse_test30 = mse(biden_tree1,biden_test30)
```

We evaluate the model with the testing data and find that the mean squared error is 406.417. Next, we plot the tree:

![](ps8-emo_files/figure-markdown_github/plot_biden_tree1-1.png)

The model shows that identifying as a Democrat is the strongest predictor of feelings of warmth toward Vice President Biden, and identifying as a Republican is the second-strongest predictor. Together, these splits indicate that party affiliation (whether Democratic, Republican, or neither) is the most important factor when it comes to predicting an individual subject's feelings of warmth toward Mr. Biden.

We now fit another, more complex tree model:

``` r
biden_tree2 = tree(biden ~ female + age + educ + dem + rep, data = biden_train70,
                   control = tree.control(nobs = nrow(biden_train70), mindev = .0000001))
```

We prune the tree 49 different times, increasing the number of leaves from 2 to 50 and storing the MSE for each pruned tree:

``` r
rounds = 50

mse_list_biden_50 = vector("numeric", rounds - 1)
leaf_list_biden_50 = vector("numeric", rounds - 1)

set.seed(1234)

for(i in 2:rounds) {
    biden_mod = prune.tree(biden_tree2, best=i)

    mse_val = mse(biden_mod,biden_test30)
    mse_list_biden_50[[i-1]] = mse_val
    leaf_list_biden_50[[i-1]] = i
}

mse_df_biden_50 = as.data.frame(mse_list_biden_50)
mse_df_biden_50$branches = leaf_list_biden_50
```

We plot the MSE for each tree vs. the number of leaves:

![](ps8-emo_files/figure-markdown_github/plot%20biden_50_mse-1.png)

We can clearly see that the lowest MSE is 401.075 for a tree with 11 leaves. We plot that tree:

``` r
biden_pruned11 <- prune.tree(biden_tree2, best=11)
mse_test30_2 = mse(biden_pruned11,biden_test30)

plot(biden_pruned11, col='darkturquoise', lwd=2.5)
title("Regression Tree (Best 11) for Warmth Toward Joe Biden (2008)\n", sub = "Validation Dataset")
text(biden_pruned11, col='deeppink')
```

![](ps8-emo_files/figure-markdown_github/plotting_optimal_biden_tree-1.png)

The tree indicates that for Democrats, age is the next most important variable and education after that, but that gender is not important. For unaffiliated voters, gender is important; for women, education and age are both important, but not so for men. Among Republican voters, age is important within the whole group and education is important, but only for voters between ages 44 and 47; gender is not an important predictor of feelings of warmth toward Joe Biden among Republican voters.

Pruning the tree reduces the MSE from 406.417 to 401.075.

We use the bagging approach to analyze this data, computing 500 bootstrapped trees using the training data and testing the resulting model with the validation set:

``` r
set.seed(1234)

biden_bag_data_train = biden_train70 %>%
                       rename() %>%
                       mutate_each(funs(as.factor(.)), dem, rep) %>%
                       na.omit

biden_bag_data_test = biden_test30 %>%
                      rename() %>%
                      mutate_each(funs(as.factor(.)), dem, rep) %>%
                      na.omit

(bag_biden <- randomForest(biden ~ ., data = biden_bag_data_train, mtry = 5, ntree = 500, importance=TRUE))
```

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden_bag_data_train,      mtry = 5, ntree = 500, importance = TRUE) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##           Mean of squared residuals: 494
    ##                     % Var explained: 9.49

``` r
mse_bag_biden = mse(bag_biden, biden_bag_data_test)

bag_biden_importance = as.data.frame(importance(bag_biden))
```

Using the validation data, the model returns a test MSE of 484.358, which is considerably higher than the MSE found when pruning the tree earlier (401.075).

Next, we review variable importance measures:

![](ps8-emo_files/figure-markdown_github/biden_bag_importance-1.png)

The variable importance plot shows that age and Democrat are the two most important variables as these yield the greatest average decreases (of approximately 225,000 and 150,000, respectively) in node impurity across 500 bagged regression trees. Despite the higher test MSE, the bagged tree model is likely a better model than the pruned tree above because the bagged model uses bootstrapping to create 500 different training sets, whereas the pruned tree above uses only a single training set. The bagged model averages the variance across the bootstrapped trees, which, together, suggest that age and Democrat are the most important variables while gender is the least important. It is worth noting, however, that the bagged model only accounts for 9.49% of the variance in feelings of warmth toward Joe Biden.

Next, we estimate a random forest model with 500 trees:

``` r
set.seed(1234)

m = floor(sqrt(5))

(rf_biden = randomForest(biden ~ ., data = biden_bag_data_train, mtry = m, ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden_bag_data_train,      mtry = m, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##           Mean of squared residuals: 404
    ##                     % Var explained: 25.9

``` r
mse_rf_biden = mse(rf_biden, biden_bag_data_test)
```

The random forest model returns a test MSE of 403.306, which is much lower than the one returned by bagging (484.358). Furthermore, the random forest model explains a greater proportion of variance (25.9%) than the bagged model does (9.49%). Still, with the variane explained at only a quarter, this suggests that there are likely unobserved and unknown variables that have a notable effect on feelings of warmth for Joe Biden.

The notable decrease in MSE is attributable to the effect of limiting the variables available every split to only *m* (2) randomly-selected predictors. This means that the trees in the random forest model will be uncorrelated to each other, the variance for the final model will be lower, and the test MSE will be lower.

We plot the importance of the predictors:

![](ps8-emo_files/figure-markdown_github/plot_rf_importance-1.png)

The random forest model estimates that Democrat is the sinlge most important predictor of feelings toward Joe Biden and that Republican is next in line; these have mean increased node purity of approximately 110,000 and 70,000, respectively. As was the case with the bagging model, gender is the least important predictor.

Finally, we estimate three boosting models, each of different depths and with 10,000 trees:

``` r
set.seed(1234)
biden_models = list("boosting_depth1" = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train,
                                            n.trees = 10000, interaction.depth = 1),
                    "boosting_depth2" = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train,
                                            n.trees = 10000, interaction.depth = 2),
                    "boosting_depth4" = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train,
                                            n.trees = 10000, interaction.depth = 4))
```

    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...

For each depth, we find the optimal number of iterations:

``` r
set.seed(1234)
data_frame(depth = c(1, 2, 4),
           model = biden_models[c("boosting_depth1", "boosting_depth2", "boosting_depth4")],
           optimal = map_dbl(model, gbm.perf, plot.it = FALSE)) %>%
           select(-model) %>%
           knitr::kable(caption = "Optimal number of boosting iterations",
                        col.names = c("Depth", "Optimal number of iterations"))
```

    ## Using OOB method...

    ## Warning in .f(.x[[i]], ...): OOB generally underestimates the optimal
    ## number of iterations although predictive performance is reasonably
    ## competitive. Using cv.folds>0 when calling gbm usually results in improved
    ## predictive performance.

    ## Using OOB method...

    ## Warning in .f(.x[[i]], ...): OOB generally underestimates the optimal
    ## number of iterations although predictive performance is reasonably
    ## competitive. Using cv.folds>0 when calling gbm usually results in improved
    ## predictive performance.

    ## Using OOB method...

    ## Warning in .f(.x[[i]], ...): OOB generally underestimates the optimal
    ## number of iterations although predictive performance is reasonably
    ## competitive. Using cv.folds>0 when calling gbm usually results in improved
    ## predictive performance.

|  Depth|  Optimal number of iterations|
|------:|-----------------------------:|
|      1|                          3302|
|      2|                          2700|
|      4|                          2094|

Now we estimate the boosting models with the optimal number of treesh for each depth:

``` r
set.seed(1234)

biden_boost1 = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train, n.trees = 3302, interaction.depth = 1)
```

    ## Distribution not specified, assuming gaussian ...

``` r
biden_boost2 = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train, n.trees = 2700, interaction.depth = 2)
```

    ## Distribution not specified, assuming gaussian ...

``` r
biden_boost4 = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train, n.trees = 2094, interaction.depth = 4)
```

    ## Distribution not specified, assuming gaussian ...

``` r
mse_boost1_biden = mse(biden_boost1,biden_bag_data_test)
```

    ## Using 3302 trees...

``` r
mse_boost2_biden = mse(biden_boost2,biden_bag_data_test)
```

    ## Using 2700 trees...

``` r
mse_boost4_biden = mse(biden_boost4,biden_bag_data_test)
```

    ## Using 2094 trees...

The boosting model with a depth of 1 has a test MSE of 405.424; for the model with a depth of 2, it is 402.541 and for the model with a depth of 4 it is 404.544. This indicates that the boosting approach yields the lowest MSE for trees with a single split compared to those with two or four splits.

Next, we increase the value of the *λ* from the default of .001 to .1:

``` r
set.seed(1234)

boost1_biden_lambda = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train,
                          n.trees = 3302, interaction.depth = 1, shrinkage=0.1)
```

    ## Distribution not specified, assuming gaussian ...

``` r
boost2_biden_lambda = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train,
                          n.trees = 2700, interaction.depth = 2, shrinkage=0.1)
```

    ## Distribution not specified, assuming gaussian ...

``` r
boost4_biden_lambda = gbm(as.numeric(biden) - 1 ~ ., data = biden_bag_data_train,
                          n.trees = 2094, interaction.depth = 4, shrinkage=0.1)
```

    ## Distribution not specified, assuming gaussian ...

``` r
mse_boost1_biden_lambda = mse(boost1_biden_lambda,biden_bag_data_test)
```

    ## Using 3302 trees...

``` r
mse_boost2_biden_lambda = mse(boost2_biden_lambda,biden_bag_data_test)
```

    ## Using 2700 trees...

``` r
mse_boost4_biden_lambda = mse(boost4_biden_lambda,biden_bag_data_test)
```

    ## Using 2094 trees...

The test MSE for single-split trees has increased from 405.424 to 414.067; for trees with a depth of two, it has increased from 402.541 to 431.52 and for trees of depth 4, it has increased from 404.544 to 463.908. This suggests that increasing the step size leads to the model learning faster but not as well. Ideally, the next step would be to try different values of *λ* and determine which yields the lowest MSE for each depth.

Part 2: Modeling voter turnout
==============================

An important question in American politics is why do some people participate in the political process, while others do not? Participation has a direct impact on outcomes -- if you fail to participate in politics, the government and political officials are less likely to respond to your concerns. Typical explanations focus on a resource model of participation -- individuals with greater resources, such as time, money, and civic skills, are more likely to participate in politics. One area of importance is understanding voter turnout, or why people participate in elections. Using the resource model of participation as a guide, we can develop several expectations. First, women, who more frequently are the primary caregiver for children and earn a lower income, are less likely to participate in elections than men. Second, older Americans, who typically have more time and higher incomes available to participate in politics, should be more likely to participate in elections than younger Americans. Finally, individuals with more years of education, who are generally more interested in politics and understand the value and benefits of participating in politics, are more likely to participate in elections than individuals with fewer years of education.

While these explanations have been repeatedly tested by political scientists, an emerging theory assesses an individual's mental health and its effect on political participation.[1] Depression increases individuals' feelings of hopelessness and political efficacy, so depressed individuals will have less desire to participate in politics. More importantly to our resource model of participation, individuals with depression suffer physical ailments such as a lack of energy, headaches, and muscle soreness which drain an individual's energy and requires time and money to receive treatment. For these reasons, we should expect that individuals with depression are less likely to participate in election than those without symptoms of depression.

The 1998 General Social Survey included several questions about the respondent's mental health. `mental_health.csv` reports several important variables from this survey.

-   `vote96` - 1 if the respondent voted in the 1996 presidential election, 0 otherwise
-   `mhealth_sum` - index variable which assesses the respondent's mental health, ranging from 0 (an individual with no depressed mood) to 9 (an individual with the most severe depressed mood)[2]
-   `age` - age of the respondent
-   `educ` - Number of years of formal education completed by the respondent
-   `black` - 1 if the respondent is black, 0 otherwise
-   `female` - 1 if the respondent is female, 0 if male
-   `married` - 1 if the respondent is currently married, 0 otherwise
-   `inc10` - Family income, in $10,000s

1.  Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five tree-based models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)
2.  Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five SVM models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)

Part 3: OJ Simpson \[4 points\]
===============================

In October 1995 in what was termed "the trial of the century", O.J. Simpson was acquitted by a jury of murdering his wife Nicole Brown Simpson and her friend Ronald Goldman. The topic of Simpson's guilt or innocence divided Americans along racial lines, as Simpson is black and his wife was white. Especially in the aftermath of the 1992 Los Angeles riots sparked by the videotaped beating of Rodney King and subsequent acquittal of officers charged with police brutality, the Simpson trial illustrated many of the racial divides in the United States.

The CBS News/New York Times monthly poll conducted in September 1995 asked respondents several questions about the closing arguments and verdict in the case. All the relevant variables are contained in `simpson.csv`.

-   `guilt` - 1 if the respondent thinks OJ Simpson was "probably guilty", 0 if the respondent thinks OJ Simpson was "probably not guilty"
-   `dem` - Democrat
-   `rep` - Republican
-   `ind` - Independent
-   `age` - Age of respondent
-   `educ` - Highest education level of the respondent
-   `female` - Respondent is female
-   `black` - Respondent is black
-   `hispanic` - Respondent is hispanic
-   `income` - Self-reported income

1.  What is the relationship between race and belief of OJ Simpson's guilt? Develop a robust statistical learning model and use this model to explain the impact of an individual's race on their beliefs about OJ Simpson's guilt.
2.  How can you predict whether individuals believe OJ Simpson to be guilty of these murders? Develop a robust statistical learning model to predict whether individuals believe OJ Simpson to be either probably guilty or probably not guilty and demonstrate the effectiveness of this model using methods we have discussed in class.

You can make full use of any of the statistical learning techniques to complete this part of the assignment:

-   Linear regression
-   Logistic regression
-   Generalized linear models
-   Non-linear linear models
-   Tree-based models
-   Support vector machines
-   Resampling methods

Select methods that are appropriate for each question and **justify the use of these methods**.

Submission instructions
=======================

Assignment submission will work the same as earlier assignments. Submit your work as a pull request before the start of class on Monday. Store it in the same locations as you've been using. However the format of your submission should follow the procedures outlined below.

If you use R
------------

Submit your assignment as a single [R Markdown document](http://rmarkdown.rstudio.com/). R Markdown is similar to Juptyer Notebooks and compiles all your code, output, and written analysis in a single reproducible file.

If you use Python
-----------------

Either:

1.  Submit your assignment following the same procedures as required by Dr. Evans. Submit a Python script containing all your code, plus a $\\LaTeX$ generated PDF document with your results and substantive analysis.
2.  Submit your assignment as a single Jupyter Notebook with your code, output, and written analysis compiled there.

[1] [Ojeda, C. (2015). Depression and political participation. *Social Science Quarterly*, 96(5), 1226-1243.](http://onlinelibrary.wiley.com.proxy.uchicago.edu/doi/10.1111/ssqu.12173/abstract)

[2] The variable is an index which combines responses to four different questions: "In the past 30 days, how often did you feel: 1) so sad nothing could cheer you up, 2) hopeless, 3) that everything was an effort, and 4) worthless?" Valid responses are none of the time, a little of the time, some of the time, most of the time, and all of the time.
