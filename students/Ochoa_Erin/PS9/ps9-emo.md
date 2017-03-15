MACS 30100 PS9
================
Erin M. Ochoa
2017 March 15

-   [Setup](#setup)
-   [Attitudes towards feminists](#attitudes-towards-feminists)
    -   [Split the data into a training and test set (70/30%)](#split-the-data-into-a-training-and-test-set-7030)
    -   [Calculate the test MSE for KNN models with *K* = 5, 10, 15, …, 100, using whatever combination of variables you see fit. Which model produces the lowest test MSE?](#calculate-the-test-mse-for-knn-models-with-k-5-10-15-dots-100-using-whatever-combination-of-variables-you-see-fit.-which-model-produces-the-lowest-test-mse)
    -   [Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?](#compare-the-test-mse-for-the-best-knnwknn-models-to-the-test-mse-for-the-equivalent-linear-regression-decision-tree-boosting-and-random-forest-methods-using-the-same-combination-of-variables-as-before.-which-performs-the-best-why-do-you-think-this-method-performed-the-best-given-your-knowledge-of-how-it-works)
-   [Voter turnout and depression \[2 points\]](#voter-turnout-and-depression-2-points)
-   [Colleges](#colleges)
-   [Clustering states \[3 points\]](#clustering-states-3-points)

Setup
=====

``` r
knitr::opts_chunk$set(
    message = FALSE,
    warning = FALSE,
    cache = TRUE
)
library(ggdendro)
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(tidyverse)
```

    ## Loading tidyverse: ggplot2
    ## Loading tidyverse: tibble
    ## Loading tidyverse: tidyr
    ## Loading tidyverse: readr
    ## Loading tidyverse: purrr
    ## Loading tidyverse: dplyr

    ## Conflicts with tidy packages ----------------------------------------------

    ## combine(): dplyr, randomForest
    ## filter():  dplyr, stats
    ## lag():     dplyr, stats
    ## margin():  ggplot2, randomForest

``` r
library(forcats)
library(broom)
library(modelr)
```

    ## 
    ## Attaching package: 'modelr'

    ## The following object is masked from 'package:broom':
    ## 
    ##     bootstrap

``` r
library(stringr)
library(grid)
library(gridExtra)
```

    ## 
    ## Attaching package: 'gridExtra'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     combine

``` r
#library(rcfss)   #not available for this version of R
library(tree)
library(e1071)
library(gbm)
```

    ## Loading required package: survival

    ## Loading required package: lattice

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.1

``` r
library(pander)
library(knitr)
library(FNN)
library(kknn)
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(ISLR)

options(digits = 3)
set.seed(1234)
theme_set(theme_minimal())
```

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

Attitudes towards feminists
===========================

Split the data into a training and test set (70/30%)
----------------------------------------------------

We read in the data:

``` r
fem = read.csv('data/feminist.csv')
```

Next, we split the dataset into training and validation sets in a ratio of 7:3:

``` r
set.seed(1234)

fem_split7030 = resample_partition(fem, c(test = 0.3, train = 0.7))
fem_train70 = fem_split7030$train %>%
              tbl_df()
fem_test30 = fem_split7030$test %>%
             tbl_df()
```

Calculate the test MSE for KNN models with *K* = 5, 10, 15, …, 100, using whatever combination of variables you see fit. Which model produces the lowest test MSE?
------------------------------------------------------------------------------------------------------------------------------------------------------------------

Using female, education, income, Democrat, and Republican, we calculate the test MSE for KNN models with K ranging from \[5,100\] in 5-point increments:

``` r
fem_knn = data_frame(k = seq(5, 100, by = 5), 
                     knn = map(k, ~ knn.reg(select(fem_train70, -feminist, -age),
                                            y = fem_train70$feminist,
                                            test = select(fem_test30, -feminist, -age), k = .)), 
                     mse = map_dbl(knn, ~ mean((fem_test30$feminist - .$pred)^2))) 
```

The models with K=35, K=40, and K=50 produce the lowest MSE (440.547). Next, we plot the MSE for each model:

![](ps9-emo_files/figure-markdown_github/fem_knn_plot-1.png) \#\# Calculate the test MSE for weighted KNN models with K = 5, 10, 15, , 100 using the same combination of variables as before. Which model produces the lowest test MSE?

``` r
fem_w_knn = data_frame(k = seq(5, 100, by = 5), 
                       wknn = map(k, ~ kknn(feminist ~ female + educ + income + dem + rep,
                                            train = fem_train70, test = fem_test30, k = .)), 
                       mse_wknn = map_dbl(wknn, ~ mean((fem_test30$feminist - .$fitted.values)^2))) %>%
            left_join(fem_knn, by = "k") %>%
            mutate(fem_knn = mse)%>%
            select(k, fem_knn, mse_wknn) %>%
            gather(method, mse, -k) %>%
            mutate(method = str_replace(method, "mse_", ""))%>%
            mutate(Method = factor(method, levels = c("fem_knn","wknn"), labels = c("KNN","Weighted KNN")))
```

![](ps9-emo_files/figure-markdown_github/fem_w_knn_plot-1.png)

Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

``` r
lm_fem = lm(feminist ~ female + educ + income + dem + rep, data=fem_train70)
mse_lm_fem = mse(lm_fem, fem_test30)
```

``` r
fem_tree = tree(feminist ~ female + educ + income + dem + rep, data = fem_train70)
mse_fem_tree = mse(fem_tree,fem_test30)
```

``` r
set.seed(1234)
fem_boost_mdls = list("boosting_depth1" = gbm(as.numeric(feminist) - 1 ~ female + educ + income + dem + rep,
                                              data = fem_train70,
                                              n.trees = 10000, interaction.depth = 1),
                      "boosting_depth2" = gbm(as.numeric(feminist) - 1 ~ female + educ + income + dem + rep,
                                              data = fem_train70,
                                              n.trees = 10000, interaction.depth = 2),
                      "boosting_depth4" = gbm(as.numeric(feminist) - 1 ~ female + educ + income + dem + rep,
                                              data = fem_train70,
                                              n.trees = 10000, interaction.depth = 4))
```

    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...

``` r
set.seed(1234)
data_frame(depth = c(1, 2, 4),
           model = fem_boost_mdls[c("boosting_depth1", "boosting_depth2", "boosting_depth4")],
           optimal = map_dbl(model, gbm.perf, plot.it = FALSE)) %>%
           select(-model) %>%
           knitr::kable(caption = "Optimal number of boosting iterations",
                        col.names = c("Depth", "Optimal number of iterations"))
```

    ## Using OOB method...
    ## Using OOB method...
    ## Using OOB method...

|  Depth|  Optimal number of iterations|
|------:|-----------------------------:|
|      1|                          2558|
|      2|                          1850|
|      4|                          1400|

``` r
set.seed(1234)

fem_boost1 = gbm(as.numeric(feminist) - 1 ~ female + educ + income + dem + rep, data = fem_train70,
                 n.trees = 2558, interaction.depth = 1)
```

    ## Distribution not specified, assuming gaussian ...

``` r
fem_boost2 = gbm(as.numeric(feminist) - 1 ~ female + educ + income + dem + rep, data = fem_train70,
                 n.trees = 1850, interaction.depth = 2)
```

    ## Distribution not specified, assuming gaussian ...

``` r
fem_boost4 = gbm(as.numeric(feminist) - 1 ~ female + educ + income + dem + rep, data = fem_train70,
                 n.trees = 1400, interaction.depth = 4)
```

    ## Distribution not specified, assuming gaussian ...

``` r
mse_fem_boost1 = mse(fem_boost1,fem_test30)
```

    ## Using 2558 trees...

``` r
mse_fem_boost2 = mse(fem_boost2,fem_test30)
```

    ## Using 1850 trees...

``` r
mse_fem_boost4 = mse(fem_boost4,fem_test30)
```

    ## Using 1400 trees...

``` r
set.seed(1234)

m = floor(sqrt(5))

(fem_rf = randomForest(feminist ~ female + educ + income + dem + rep, data = fem_train70, mtry = m,
                        ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = feminist ~ female + educ + income + dem +      rep, data = fem_train70, mtry = m, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##           Mean of squared residuals: 434
    ##                     % Var explained: 6.22

``` r
mse_fem_rf = mse(fem_rf, fem_test30)
```

![](ps9-emo_files/figure-markdown_github/fem_compare_methods_plot-1.png)

Boosting with a depth of 4 produces the lowest MSE (431.643). This is because this boosted model relies on trees with four splits, representing the complex relationship between the predictor variables and the outcome.

Voter turnout and depression \[2 points\]
=========================================

The 1998 General Social Survey included several questions about the respondent's mental health. `mental_health.csv` reports several important variables from this survey.

-   `vote96` - 1 if the respondent voted in the 1996 presidential election, 0 otherwise
-   `mhealth_sum` - index variable which assesses the respondent's mental health, ranging from 0 (an individual with no depressed mood) to 9 (an individual with the most severe depressed mood)[1]
-   `age` - age of the respondent
-   `educ` - Number of years of formal education completed by the respondent
-   `black` - 1 if the respondent is black, 0 otherwise
-   `female` - 1 if the respondent is female, 0 if male
-   `married` - 1 if the respondent is currently married, 0 otherwise
-   `inc10` - Family income, in $10,000s

Estimate a series of models explaining/predicting voter turnout.

1.  Split the data into a training and test set (70/30).
2.  Calculate the test error rate for KNN models with *K* = 1, 2, …, 10, using whatever combination of variables you see fit. Which model produces the lowest test MSE?
3.  Calculate the test error rate for weighted KNN models with *K* = 1, 2, …, 10 using the same combination of variables as before. Which model produces the lowest test error rate?
4.  Compare the test error rate for the best KNN/wKNN model(s) to the test error rate for the equivalent logistic regression, decision tree, boosting, random forest, and SVM methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?

Colleges
========

The `College` dataset in the `ISLR` library contains statistics for a large number of U.S. colleges from the 1995 issue of U.S. News and World Report.

-   `Private` - A factor with levels `No` and `Yes` indicating private or public university.
-   `Apps` - Number of applications received.
-   `Accept` - Number of applications accepted.
-   `Enroll` - Number of new students enrolled.
-   `Top10perc` - Percent of new students from top 10% of H.S. class.
-   `Top25perc` - Percent of new students from top 25% of H.S. class.
-   `F.Undergrad` - Number of fulltime undergraduates.
-   `P.Undergrad` - Number of parttime undergraduates.
-   `Outstate` - Out-of-state tuition.
-   `Room.Board` - Room and board costs.
-   `Books` - Estimated book costs.
-   `Personal` - Estimated personal spending.
-   `PhD` - Percent of faculty with Ph.D.'s.
-   `Terminal` - Percent of faculty with terminal degrees.
-   `S.F.Ratio` - Student/faculty ratio.
-   `perc.alumni` - Percent of alumni who donate.
-   `Expend` - Instructional expenditure per student.
-   `Grad.Rate` - Graduation rate.

Perform PCA analysis on the college dataset and plot the first two principal components. Describe the results. What variables appear strongly correlated on the first principal component? What about the second principal component?

We assign the dataframe (imported via the ISLR library) to a new object and recode the Private variable to be numeric:

``` r
college = College
college$Private = as.numeric(college$Private)
```

Next, we perform principal component analysis and view the loadings for the first two components:

``` r
college_pca = prcomp(college, scale = TRUE)
college_pca_1_2 = as.data.frame(college_pca$rotation)[1:2]
college_pca_1_2
```

    ##                 PC1     PC2
    ## Private     -0.0890  0.3459
    ## Apps        -0.1996 -0.3436
    ## Accept      -0.1538 -0.3726
    ## Enroll      -0.1178 -0.3997
    ## Top10perc   -0.3603  0.0162
    ## Top25perc   -0.3448 -0.0177
    ## F.Undergrad -0.0941 -0.4107
    ## P.Undergrad  0.0175 -0.2931
    ## Outstate    -0.3277  0.1915
    ## Room.Board  -0.2665  0.0940
    ## Books       -0.0572 -0.0573
    ## Personal     0.0719 -0.1928
    ## PhD         -0.3033 -0.1162
    ## Terminal    -0.3039 -0.1042
    ## S.F.Ratio    0.2103 -0.2044
    ## perc.alumni -0.2367  0.1941
    ## Expend      -0.3330  0.0703
    ## Grad.Rate   -0.2731  0.1178

Because

For the first component, Top10perc and Top25perc load very negatively, suggesting that schools with high scores for PC1 will have low proportions of students in the top 10% or top 25% of their high-school classes. Similarly, Outstate, Expend, PhD, and Terminal also load very negatively, indicating that schools with high scores for PC1 will have high proportions of in-state students, low per-student expenditure, and low proportions of faculty with PhDs or terminal degrees.

For the second component, F.Undergrad loads highly negatively, suggesting that schools with high scores for PC2 will have low enrollment of full-time undergraduates. Private loads positively, suggesting that schools with high scores for this component will tend to be private. Enroll, Accept, and Apps load very negatively; schools with high scores for PC2 should enroll few new students, accept few applications, and receive few applications.

-   `Private` - A factor with levels `No` and `Yes` indicating private or public university.
-   `Apps` - Number of applications received.
-   `Accept` - Number of applications accepted.
-   `Enroll` - Number of new students enrolled.
-   `Top10perc` - Percent of new students from top 10% of H.S. class.
-   `Top25perc` - Percent of new students from top 25% of H.S. class.
-   `F.Undergrad` - Number of fulltime undergraduates.
-   `P.Undergrad` - Number of parttime undergraduates.
-   `Outstate` - Out-of-state tuition.
-   `Room.Board` - Room and board costs.
-   `Books` - Estimated book costs.
-   `Personal` - Estimated personal spending.
-   `PhD` - Percent of faculty with Ph.D.'s.
-   `Terminal` - Percent of faculty with terminal degrees.
-   `S.F.Ratio` - Student/faculty ratio.
-   `perc.alumni` - Percent of alumni who donate.
-   `Expend` - Instructional expenditure per student.
-   `Grad.Rate` - Graduation rate.

![](ps9-emo_files/figure-markdown_github/college_biplot-1.png)

Clustering states \[3 points\]
==============================

The `USArrests` dataset contains 50 observations (one for each state) from 1973 with variables on crime statistics:

-   `Murder` - Murder arrests (per 100,000)
-   `Assault` - Assault arrests (per 100,000)
-   `Rape` - Rape arrests (per 100,000)
-   `UrbanPop` - Percent urban population

1.  Perform PCA on the dataset and plot the observations on the first and second principal components.
2.  Perform *K*-means clustering with *K* = 2. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.
3.  Perform *K*-means clustering with *K* = 4. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.
4.  Perform *K*-means clustering with *K* = 3. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.
5.  Perform *K*-means clustering with *K* = 3 on the first two principal components score vectors, rather than the raw data. Describe your results and compare them to the clustering results with *K* = 3 based on the raw data.
6.  Using hierarchical clustering with complete linkage and Euclidean distance, cluster the states.
7.  Cut the dendrogram at a height that results in three distinct clusters. Which states belong to which clusters?
8.  Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation 1. What effect does scaling the variables have on the hierarchical clustering obtained? In your opinion, should the variables be scaled before the inter-observation dissimilarities are computed? Provide a justification for your answer.

[1] The variable is an index which combines responses to four different questions: "In the past 30 days, how often did you feel: 1) so sad nothing could cheer you up, 2) hopeless, 3) that everything was an effort, and 4) worthless?" Valid responses are none of the time, a little of the time, some of the time, most of the time, and all of the time.
