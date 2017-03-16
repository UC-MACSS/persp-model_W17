Problem set \#9: nonparametric methods and unsupervised learning
================
Soo Wan Kim
March 15, 2017

-   [Attitudes towards feminists \[3 points\]](#attitudes-towards-feminists-3-points)
-   [Voter turnout and depression \[2 points\]](#voter-turnout-and-depression-2-points)
-   [Colleges \[2 points\]](#colleges-2-points)
-   [Clustering states \[3 points\]](#clustering-states-3-points)

Attitudes towards feminists \[3 points\]
========================================

1.  Split the data into a training and test set (70/30%).

``` r
fem <- read.csv("data/feminist.csv")

fem_split <- resample_partition(fem, c(test = 0.3, train = 0.7)) #split data into 70/30 training/test set
fem_train <- fem_split$train %>% 
  tbl_df()
fem_test <- fem_split$test %>% 
  tbl_df()
```

1.  Calculate the test MSE for KNN models with *K* = 5, 10, 15, …, 100, using whatever combination of variables you see fit. Which model produces the lowest test MSE?
2.  Calculate the test MSE for weighted KNN models with *K* = 5, 10, 15, …, 100 using the same combination of variables as before. Which model produces the lowest test MSE?
3.  Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?

Voter turnout and depression \[2 points\]
=========================================

1.  Split the data into a training and test set (70/30).

``` r
mh <- read.csv("data/mental_health.csv")

mh_split <- resample_partition(mh, c(test = 0.3, train = 0.7)) #split data into 70/30 training/test set
mh_train <- mh_split$train %>% 
  tbl_df()
mh_test <- mh_split$test %>% 
  tbl_df()
```

1.  Calculate the test error rate for KNN models with *K* = 1, 2, …, 10, using whatever combination of variables you see fit. Which model produces the lowest test MSE?

2.  Calculate the test error rate for weighted KNN models with *K* = 1, 2, …, 10 using the same combination of variables as before. Which model produces the lowest test error rate?

3.  Compare the test error rate for the best KNN/wKNN model(s) to the test error rate for the equivalent logistic regression, decision tree, boosting, random forest, and SVM methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?

Colleges \[2 points\]
=====================

Perform PCA analysis on the college dataset and plot the first two principal components. Describe the results. What variables appear strongly correlated on the first principal component? What about the second principal component?

``` r
college <- read.csv("data/college.csv")
```

Clustering states \[3 points\]
==============================

1.  Perform PCA on the dataset and plot the observations on the first and second principal components.

2.  Perform *K*-means clustering with *K* = 2. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.

3.  Perform *K*-means clustering with *K* = 4. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.

4.  Perform *K*-means clustering with *K* = 3. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.

5.  Perform *K*-means clustering with *K* = 3 on the first two principal components score vectors, rather than the raw data. Describe your results and compare them to the clustering results with *K* = 3 based on the raw data.

6.  Using hierarchical clustering with complete linkage and Euclidean distance, cluster the states.

7.  Cut the dendrogram at a height that results in three distinct clusters. Which states belong to which clusters?

8.  Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation 1. What effect does scaling the variables have on the hierarchical clustering obtained? In your opinion, should the variables be scaled before the inter-observation dissimilarities are computed? Provide a justification for your answer.
