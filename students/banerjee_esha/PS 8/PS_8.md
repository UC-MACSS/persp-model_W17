PS 8: tree-based methods and support vector machines
================
Esha Banerjee
5 March 2017

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

    ## filter(): dplyr, stats
    ## lag():    dplyr, stats

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
library(tree)
library(randomForest)
```

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(stringr)
library(gridExtra)
```

    ## 
    ## Attaching package: 'gridExtra'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     combine

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
library(grid)
library(titanic)
#install.packages("pROC")
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(gbm)
```

    ## Loading required package: survival

    ## Loading required package: lattice

    ## Loading required package: splines

    ## Loading required package: parallel

    ## Loaded gbm 2.1.1

``` r
library(caret)
```

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:survival':
    ## 
    ##     cluster

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(ggdendro)
library(e1071)

theme_set(theme_minimal())
```

#### Part 1: Sexy Joe Biden (redux times two)

``` r
# read in data
biden <- read_csv("biden.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   biden = col_integer(),
    ##   female = col_integer(),
    ##   age = col_integer(),
    ##   educ = col_integer(),
    ##   dem = col_integer(),
    ##   rep = col_integer()
    ## )

Split the data into a training set (70%) and a validation set (30%). Be sure to set your seed prior to this part of your code to guarantee reproducibility of results.
======================================================================================================================================================================

``` r
# For reproducibility
set.seed(123) 
# Split data
biden_split <- resample_partition(biden, c(test = .3, train = .7))
```

Fit a decision tree to the training data, with biden as the response variable and the other variables as predictors. Plot the tree and interpret the results. What is the test MSE? Leave the control options for tree() at their default values
================================================================================================================================================================================================================================================

``` r
set.seed(123)
biden_tree <- tree(biden ~ ., data = biden_split$train)

# plot tree
plot(biden_tree, col='black', lwd=2.5)
title("Decision Tree for Biden Scores", sub = 'All predictors, Default Controls')
text(biden_tree, col='black')
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-4-1.png)

``` r
# function to get MSE
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}

mse_biden_1 <- mse(biden_tree,biden_split$test)
mse_biden_1
```

    ## [1] 423.709

We evaluate the model with the testing data and find that the mean squared error is 423.7090397.

The model shows that being a Democrat is the strongest predictor of feelings of warmth toward Joe Biden, being a Republican is the second-strongest predictor. These splits indicate that party affiliation is the most important factor in predicting an individual's feelings of warmth toward Joe Biden.

Now fit another tree to the training data with the following control options: tree(control = tree.control(nobs = \# number of rows in the training set,mindev = 0)). Use cross-validation to determine the optimal level of tree complexity, plot the optimal tree, and interpret the results. Does pruning the tree improve the test MSE?
==========================================================================================================================================================================================================================================================================================================================================

``` r
set.seed(123) # For reproducibility

biden_tree_2 <- tree(biden ~ ., data = biden_split$train,
     control = tree.control(nobs = nrow(biden_split$train),
                            mindev = 0))

rounds = 50

mse_list_biden_50 = vector("numeric", rounds - 1)
leaf_list_biden_50 = vector("numeric", rounds - 1)

for(i in 2:rounds) {
    biden_mod = prune.tree(biden_tree_2, best=i)

    mse_val = mse(biden_mod,biden_split$test)
    mse_list_biden_50[[i-1]] = mse_val
    leaf_list_biden_50[[i-1]] = i
}

mse_df_biden_50 = as.data.frame(mse_list_biden_50)
mse_df_biden_50$branches = leaf_list_biden_50

ggplot(mse_df_biden_50, aes(branches, mse_list_biden_50)) +
       geom_line(color='black',size=1) +
       labs(title = "Comparing Regression Trees for Warmth Toward Joe Biden",
       subtitle = "Using Validation Set",
       x = "Number of nodes",
       y = "Mean Squared Error") + 
       theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
             panel.border = element_rect(linetype = "solid", color = "grey70", fill=NA, size=1.1))
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-5-1.png)

``` r
mse_test <- mse(biden_tree_2,biden_split$test)
mse_test
```

    ## [1] 497.3772

Using cross validation, we find that the MSE is lowest for a tree with 6 nodes.

``` r
biden_pruned6 <- prune.tree(biden_tree_2, best=6)
mse_biden_pruned <- mse(biden_pruned6,biden_split$test)

plot(biden_pruned6, col='black', lwd=2.5)
title("Decision Tree for Biden Scores", sub = 'Only 6 nodes')
text(biden_pruned6, col='black')
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-6-1.png)

``` r
mse_biden_pruned
```

    ## [1] 422.8927

Pruning to 6 nodes reduces the MSE from 423.7090397 (which was obtained using all defaults) to 422.8926521.However 497.377166 which was obtained in this set after splitting and before pruning gave a high mse for the test data implying that it overfitted during training with the 70 % data. The tree indicates that for Democrats, age is the next most important variable. Among Republicans age is important followed by education but educationis a factor only for voters aged 43.5 years and above. Gender, strangely has no effect.

Use the bagging approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results.
==========================================================================================================================================

``` r
set.seed(123)

biden <- read.csv('biden.csv')
biden$Party[biden$dem == 1] = 'Democrat'
biden$Party[biden$dem == 0 & biden$rep == 0] = 'Independent'
biden$Party[biden$rep == 1] = 'Republican'

biden_split <- resample_partition(biden, c(test = 0.3, train = 0.7))

biden_bag_data_train <- biden_split$train %>%
                       tbl_df()%>%
                       select(-Party) %>%
                       mutate_each(funs(as.factor(.)), dem, rep) %>%
                       na.omit

biden_bag_data_test  <- biden_split$test %>%
  tbl_df()%>%
                      select(-Party) %>%
                      mutate_each(funs(as.factor(.)), dem, rep) %>%
                      na.omit

# estimate model
(bag_biden <- randomForest(biden ~ ., data = biden_bag_data_train, mtry = 5, ntree = 500, importance=TRUE))
```

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden_bag_data_train,      mtry = 5, ntree = 500, importance = TRUE) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##           Mean of squared residuals: 482.4251
    ##                     % Var explained: 10.76

``` r
# find MSE
mse_bag_biden <- mse(bag_biden, biden_bag_data_test)
mse_bag_biden
```

    ## [1] 501.2769

The MSE for the model with bagging is 501.2769109 , which is much higher than we had for the pruned tree with 422.8926521. The % variation explained is also very low, at 10.76%.

``` r
set.seed(123)
bag_biden_importance = as.data.frame(importance(bag_biden))

ggplot(bag_biden_importance, mapping=aes(x=rownames(bag_biden_importance), y=IncNodePurity)) +
       geom_bar(stat="identity", aes(fill=IncNodePurity)) +
       labs(title = "Average Increased Node Purity Across 500 Regression Trees",
       subtitle = "Predicted Warmth Toward Joe Biden",
       x = "Variable",
       y = "Mean Increased Node Purity") + 
       theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
             panel.border = element_rect(linetype = "solid", color = "grey70", fill=NA, size=1.1), legend.position = 'none') 
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-8-1.png) The variable importance plot shows that age and Democrat are the two most important variables as these yield the greatest average decrease in node impurity across 500 bagged regression trees. Despite the higher test MSE, the bagged tree model is likely a better model than the pruned tree above because the bagged model uses bootstrapping to create 500 different training sets, whereas the pruned tree above uses only a single training set.It can thus address variances based on the composition of the sets better. Here too, gender is the least important variable. The bagged model accounts only for 10.76% of the variance in feelings of warmth toward Joe Biden.

#### Use the random forest approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results. Describe the effect of *m*, the number of variables considered at each split, on the error rate obtained.

``` r
set.seed(123)

(biden_rf <- randomForest(biden ~ ., data = biden_bag_data_train,mtry =2,ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden_bag_data_train,      mtry = 2, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##           Mean of squared residuals: 395.5181
    ##                     % Var explained: 26.83

``` r
mse_rf_biden <- mse(biden_rf, biden_bag_data_test)
mse_rf_biden
```

    ## [1] 424.9684

The random forest model gives a test MSE of 424.9683515, which is much lower than the one returned by bagging 501.2769109. Random forest also explains variance (26.83%) in the data compared to the bagged model (10.76%). Still, with the % var explained is low, so that there are probably other unknown variables that effect feelings of warmth for Joe Biden.

The notable decrease in MSE is attributable to the effect of limiting the variables available every split to only randomly-selected predictors. This ensures that the trees in the random forest model are uncorrelated to each other, the variance in the final models is lower, and hence the test MSE is lower.

Plotting the importance of the predictors:

``` r
rf_biden_importance = as.data.frame(importance(biden_rf))

ggplot(rf_biden_importance, mapping=aes(x=rownames(rf_biden_importance), y=IncNodePurity)) +
       geom_bar(stat="identity", aes(fill=IncNodePurity)) +
       labs(title = "Average Increased Node Purity Across 500 Regression Trees",
       subtitle = "Predicted Warmth Toward Joe Biden",
       x = "Variable",
       y = "Mean Increased Node Purity") + 
       theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
             panel.border = element_rect(linetype = "solid", color = "grey70", fill=NA, size=1.1), legend.position = 'none') 
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-10-1.png) The random forest model estimates that Democrat is the sinlge most important predictor of feelings toward Joe Biden and that Republican is next in line. As was the case with the bagging model, gender is the least important predictor.

#### Use the boosting approach to analyze the data. What test MSE do you obtain? How does the value of the shrinkage parameter *λ* influence the test MSE?

We first run the boosting model using depths of 1,2 and 4 respoectively, to find the optimal number of iterations for lowest MSE.

``` r
set.seed(123)
biden_models <- list("boosting_depth1" = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 10000, interaction.depth = 1),
                       "boosting_depth2" = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 10000, interaction.depth = 2),
                       "boosting_depth4" = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 10000, interaction.depth = 4))
```

    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...

``` r
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
|      1|                          3468|
|      2|                          2665|
|      4|                          2160|

``` r
biden_boost_1 = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 3302, interaction.depth = 1)
```

    ## Distribution not specified, assuming gaussian ...

``` r
biden_boost_2 = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 2700, interaction.depth = 2)
```

    ## Distribution not specified, assuming gaussian ...

``` r
biden_boost_4 = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 2094, interaction.depth = 4)
```

    ## Distribution not specified, assuming gaussian ...

``` r
predict.gbm <- function (object, newdata, n.trees, type = "link", single.tree = FALSE, ...) {
  if (missing(n.trees)) {
    if (object$train.fraction < 1) {
      n.trees <- gbm.perf(object, method = "test", plot.it = FALSE)
    }
    else if (!is.null(object$cv.error)) {
      n.trees <- gbm.perf(object, method = "cv", plot.it = FALSE)
    }
    else {
      n.trees <- length(object$train.error)
    }
    cat(paste("Using", n.trees, "trees...\n"))
    gbm::predict.gbm(object, newdata, n.trees, type, single.tree, ...)
  }
}
mse_1 = mse(biden_boost_1,biden_bag_data_test)
```

    ## Using 3302 trees...

``` r
mse_1
```

    ## [1] 423.3929

``` r
mse_2 = mse(biden_boost_2,biden_bag_data_test)
```

    ## Using 2700 trees...

``` r
mse_2
```

    ## [1] 420.5549

``` r
mse_4 = mse(biden_boost_4,biden_bag_data_test)
```

    ## Using 2094 trees...

``` r
mse_4
```

    ## [1] 422.7429

The boosting model with a depth of 1 has a test MSE of 423.3929491; for the model with a depth of 2, it is 420.5549103 and for the model with a depth of 4 it is 422.742913. The boosting approach yields the lowest MSE for trees with two splits compared to those with one or four splits. These values are much better than those obtained by bagging and random forest models.

Next, we increase the value of the *λ* from the default of .001 to .1:

``` r
set.seed(123)

biden_boost_1 = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 3302, interaction.depth = 1,shrinkage=0.02)
```

    ## Distribution not specified, assuming gaussian ...

``` r
biden_boost_2 = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 2700, interaction.depth = 2,shrinkage=0.02)
```

    ## Distribution not specified, assuming gaussian ...

``` r
biden_boost_4 = gbm(as.numeric(biden) - 1 ~ .,
                                               data = biden_bag_data_train,
                                               n.trees = 2094, interaction.depth = 4,shrinkage=0.02)
```

    ## Distribution not specified, assuming gaussian ...

``` r
mse_1 = mse(biden_boost_1,biden_bag_data_test)
```

    ## Using 3302 trees...

``` r
mse_1
```

    ## [1] 440.6273

``` r
mse_2 = mse(biden_boost_2,biden_bag_data_test)
```

    ## Using 2700 trees...

``` r
mse_2
```

    ## [1] 446.1531

``` r
mse_4 = mse(biden_boost_4,biden_bag_data_test)
```

    ## Using 2094 trees...

``` r
mse_4
```

    ## [1] 464.9515

We notice that all the MSE values have increased.Shrinkage is used for reducing, or shrinking, the impact of each additional fitted base-learner (tree). It reduces the size of incremental steps and thus penalizes the importance of each consecutive iteration. So since we increased the step size, the negative impact of an erroneous boosting iteration could not be rectified and we end up with a high MSE.

Part 2: Modeling voter turnout
==============================

#### Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five tree-based models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)

``` r
mh <- read_csv("mental_health.csv") %>%
  na.omit()%>%
  mutate_each(funs(as.factor(.)), vote96, black, female, married) 
```

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

``` r
set.seed(123)
mh_split <- resample_partition(mh, p = c("test" = .3, "train" = .7))
```

``` r
mh_tree <- tree(vote96 ~ educ, data = as_tibble(mh_split$train))
mh_tree
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ## 1) root 816 1040.0 1 ( 0.3346 0.6654 )  
    ##   2) educ < 15.5 601  807.0 1 ( 0.3960 0.6040 )  
    ##     4) educ < 11.5 133  183.1 0 ( 0.5489 0.4511 ) *
    ##     5) educ > 11.5 468  607.5 1 ( 0.3526 0.6474 ) *
    ##   3) educ > 15.5 215  191.0 1 ( 0.1628 0.8372 ) *

``` r
plot(mh_tree)
text(mh_tree, pretty = 0)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-14-1.png)

``` r
fitted <- predict(mh_tree, as_tibble(mh_split$test), type = "class")
tree_err <- mean(as_tibble(mh_split$test)$vote96 != fitted)
tree_err
```

    ## [1] 0.3065903

``` r
roc_tree1 <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_tree1)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-14-2.png)

``` r
auc(roc_tree1)
```

    ## Area under the curve: 0.561

``` r
mh_tree <- tree(vote96 ~ educ + mhealth_sum, data = as_tibble(mh_split$train))
mh_tree
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ## 1) root 816 1040.0 1 ( 0.3346 0.6654 )  
    ##   2) mhealth_sum < 4.5 637  741.1 1 ( 0.2684 0.7316 )  
    ##     4) educ < 15.5 444  565.2 1 ( 0.3333 0.6667 ) *
    ##     5) educ > 15.5 193  141.0 1 ( 0.1192 0.8808 ) *
    ##   3) mhealth_sum > 4.5 179  244.6 0 ( 0.5698 0.4302 ) *

``` r
plot(mh_tree)
text(mh_tree, pretty = 0)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-15-1.png)

``` r
fitted <- predict(mh_tree, as_tibble(mh_split$test), type = "class")
tree_err <- mean(as_tibble(mh_split$test)$vote96 != fitted)
tree_err
```

    ## [1] 0.3037249

``` r
roc_tree2 <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_tree2)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-15-2.png)

``` r
auc(roc_tree2)
```

    ## Area under the curve: 0.5875

``` r
mh_tree <- tree(vote96 ~ educ + mhealth_sum + age, data = as_tibble(mh_split$train))
mh_tree
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 816 1040.00 1 ( 0.3346 0.6654 )  
    ##    2) age < 39.5 358  495.70 1 ( 0.4804 0.5196 )  
    ##      4) educ < 12.5 140  172.70 0 ( 0.6929 0.3071 )  
    ##        8) educ < 11.5 40   26.01 0 ( 0.9000 0.1000 ) *
    ##        9) educ > 11.5 100  133.70 0 ( 0.6100 0.3900 ) *
    ##      5) educ > 12.5 218  280.60 1 ( 0.3440 0.6560 )  
    ##       10) mhealth_sum < 4.5 172  201.70 1 ( 0.2733 0.7267 )  
    ##         20) educ < 15.5 94  124.10 1 ( 0.3723 0.6277 ) *
    ##         21) educ > 15.5 78   66.97 1 ( 0.1538 0.8462 ) *
    ##       11) mhealth_sum > 4.5 46   61.58 0 ( 0.6087 0.3913 ) *
    ##    3) age > 39.5 458  483.30 1 ( 0.2205 0.7795 )  
    ##      6) mhealth_sum < 4.5 378  353.20 1 ( 0.1772 0.8228 )  
    ##       12) educ < 9.5 34   46.66 1 ( 0.4412 0.5588 ) *
    ##       13) educ > 9.5 344  292.20 1 ( 0.1512 0.8488 )  
    ##         26) age < 48.5 122  133.80 1 ( 0.2377 0.7623 ) *
    ##         27) age > 48.5 222  147.80 1 ( 0.1036 0.8964 ) *
    ##      7) mhealth_sum > 4.5 80  109.10 1 ( 0.4250 0.5750 )  
    ##       14) mhealth_sum < 8.5 58   80.13 0 ( 0.5345 0.4655 ) *
    ##       15) mhealth_sum > 8.5 22   17.53 1 ( 0.1364 0.8636 ) *

``` r
plot(mh_tree)
text(mh_tree, pretty = 0)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-16-1.png)

``` r
fitted <- predict(mh_tree, as_tibble(mh_split$test), type = "class")
tree_err <- mean(as_tibble(mh_split$test)$vote96 != fitted)
tree_err
```

    ## [1] 0.3094556

``` r
roc_tree3 <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_tree3)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-16-2.png)

``` r
auc(roc_tree3)
```

    ## Area under the curve: 0.6214

``` r
mh_tree <- tree(vote96 ~ educ + mhealth_sum + age + inc10, data = as_tibble(mh_split$train))
mh_tree
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 816 1040.00 1 ( 0.3346 0.6654 )  
    ##    2) age < 39.5 358  495.70 1 ( 0.4804 0.5196 )  
    ##      4) educ < 12.5 140  172.70 0 ( 0.6929 0.3071 )  
    ##        8) educ < 11.5 40   26.01 0 ( 0.9000 0.1000 ) *
    ##        9) educ > 11.5 100  133.70 0 ( 0.6100 0.3900 ) *
    ##      5) educ > 12.5 218  280.60 1 ( 0.3440 0.6560 )  
    ##       10) mhealth_sum < 4.5 172  201.70 1 ( 0.2733 0.7267 )  
    ##         20) educ < 15.5 94  124.10 1 ( 0.3723 0.6277 ) *
    ##         21) educ > 15.5 78   66.97 1 ( 0.1538 0.8462 ) *
    ##       11) mhealth_sum > 4.5 46   61.58 0 ( 0.6087 0.3913 ) *
    ##    3) age > 39.5 458  483.30 1 ( 0.2205 0.7795 )  
    ##      6) inc10 < 2.40745 151  198.10 1 ( 0.3642 0.6358 ) *
    ##      7) inc10 > 2.40745 307  259.40 1 ( 0.1498 0.8502 )  
    ##       14) inc10 < 12.7897 267  245.40 1 ( 0.1723 0.8277 )  
    ##         28) age < 48.5 102  121.80 1 ( 0.2843 0.7157 ) *
    ##         29) age > 48.5 165  109.50 1 ( 0.1030 0.8970 ) *
    ##       15) inc10 > 12.7897 40    0.00 1 ( 0.0000 1.0000 ) *

``` r
plot(mh_tree)
text(mh_tree, pretty = 0)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-17-1.png)

``` r
fitted <- predict(mh_tree, as_tibble(mh_split$test), type = "class")
tree_err <- mean(as_tibble(mh_split$test)$vote96 != fitted)
tree_err
```

    ## [1] 0.2922636

``` r
roc_tree4 <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_tree4)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-17-2.png)

``` r
auc(roc_tree4)
```

    ## Area under the curve: 0.6065

``` r
mh_tree <- tree(vote96 ~ ., data = as_tibble(mh_split$train))
mh_tree
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 816 1040.00 1 ( 0.3346 0.6654 )  
    ##    2) age < 39.5 358  495.70 1 ( 0.4804 0.5196 )  
    ##      4) educ < 12.5 140  172.70 0 ( 0.6929 0.3071 )  
    ##        8) educ < 11.5 40   26.01 0 ( 0.9000 0.1000 ) *
    ##        9) educ > 11.5 100  133.70 0 ( 0.6100 0.3900 ) *
    ##      5) educ > 12.5 218  280.60 1 ( 0.3440 0.6560 )  
    ##       10) mhealth_sum < 4.5 172  201.70 1 ( 0.2733 0.7267 )  
    ##         20) educ < 15.5 94  124.10 1 ( 0.3723 0.6277 ) *
    ##         21) educ > 15.5 78   66.97 1 ( 0.1538 0.8462 ) *
    ##       11) mhealth_sum > 4.5 46   61.58 0 ( 0.6087 0.3913 ) *
    ##    3) age > 39.5 458  483.30 1 ( 0.2205 0.7795 )  
    ##      6) inc10 < 2.40745 151  198.10 1 ( 0.3642 0.6358 ) *
    ##      7) inc10 > 2.40745 307  259.40 1 ( 0.1498 0.8502 )  
    ##       14) inc10 < 12.7897 267  245.40 1 ( 0.1723 0.8277 )  
    ##         28) age < 48.5 102  121.80 1 ( 0.2843 0.7157 ) *
    ##         29) age > 48.5 165  109.50 1 ( 0.1030 0.8970 ) *
    ##       15) inc10 > 12.7897 40    0.00 1 ( 0.0000 1.0000 ) *

``` r
plot(mh_tree)
text(mh_tree, pretty = 0)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-18-1.png)

``` r
fitted <- predict(mh_tree, as_tibble(mh_split$test), type = "class")
tree_err <- mean(as_tibble(mh_split$test)$vote96 != fitted)
tree_err
```

    ## [1] 0.2922636

``` r
roc_tree5 <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_tree5)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-18-2.png)

``` r
auc(roc_tree5)
```

    ## Area under the curve: 0.6065

``` r
plot(roc_tree1, print.auc = TRUE, col = "blue", print.auc.x = .2)
plot(roc_tree2, print.auc = TRUE, col = "red", print.auc.x = .2, print.auc.y = .4, add = TRUE)
plot(roc_tree3, print.auc = TRUE, col = "orange", print.auc.x = .2, print.auc.y = .3, add = TRUE)
plot(roc_tree4, print.auc = TRUE, col = "green", print.auc.x = .2, print.auc.y = .2, add = TRUE)
plot(roc_tree5, print.auc = TRUE, col = "purple", print.auc.x = .2, print.auc.y = .1, add = TRUE)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-19-1.png) Chosing a combination of different variables: education for first model, education and mental health score for the second, education, age and mental health score in the third, education, age, income and mental health score in the fourth and all variables in the fifth. The area under the curve is highest at 0.621 for the model containing education, age and mental health score. While the error is slightly higher (0.30) than the last two (~ 0.29), it can be overlooked since the other variables are not better predictors.

Looking at the tree for the third model, age is the most important factor, followed by education and mental health score. We can interpret the tree for model 3 (shown below) using hypothetical observations. For younger people, education is the more dominant factor than mental health score when it comes to voting. People younger than 39.5 years are most likely to vote if they have a high education of more then 12.5 years. Older people are more likely to have voted, even if they have received only some education.

``` r
mh_tree <- tree(vote96 ~ educ + mhealth_sum + age, data = as_tibble(mh_split$train))
mh_tree
```

    ## node), split, n, deviance, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 816 1040.00 1 ( 0.3346 0.6654 )  
    ##    2) age < 39.5 358  495.70 1 ( 0.4804 0.5196 )  
    ##      4) educ < 12.5 140  172.70 0 ( 0.6929 0.3071 )  
    ##        8) educ < 11.5 40   26.01 0 ( 0.9000 0.1000 ) *
    ##        9) educ > 11.5 100  133.70 0 ( 0.6100 0.3900 ) *
    ##      5) educ > 12.5 218  280.60 1 ( 0.3440 0.6560 )  
    ##       10) mhealth_sum < 4.5 172  201.70 1 ( 0.2733 0.7267 )  
    ##         20) educ < 15.5 94  124.10 1 ( 0.3723 0.6277 ) *
    ##         21) educ > 15.5 78   66.97 1 ( 0.1538 0.8462 ) *
    ##       11) mhealth_sum > 4.5 46   61.58 0 ( 0.6087 0.3913 ) *
    ##    3) age > 39.5 458  483.30 1 ( 0.2205 0.7795 )  
    ##      6) mhealth_sum < 4.5 378  353.20 1 ( 0.1772 0.8228 )  
    ##       12) educ < 9.5 34   46.66 1 ( 0.4412 0.5588 ) *
    ##       13) educ > 9.5 344  292.20 1 ( 0.1512 0.8488 )  
    ##         26) age < 48.5 122  133.80 1 ( 0.2377 0.7623 ) *
    ##         27) age > 48.5 222  147.80 1 ( 0.1036 0.8964 ) *
    ##      7) mhealth_sum > 4.5 80  109.10 1 ( 0.4250 0.5750 )  
    ##       14) mhealth_sum < 8.5 58   80.13 0 ( 0.5345 0.4655 ) *
    ##       15) mhealth_sum > 8.5 22   17.53 1 ( 0.1364 0.8636 ) *

``` r
plot(mh_tree)
text(mh_tree, pretty = 0)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-20-1.png) \#\#\#\# Use cross-validation techniques and standard measures of model fit to compare and evaluate at least five SVM models of voter turnout. Select the best model and interpret the results using whatever methods you see fit.

``` r
mh_lin_tune <- tune(svm, vote96 ~ educ + age + mhealth_sum, data = as_tibble(mh_split$train),
                    kernel = "linear",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))

mh_lin <- mh_lin_tune$best.model
summary(mh_lin)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ educ + age + mhealth_sum, 
    ##     data = as_tibble(mh_split$train), ranges = list(cost = c(0.001, 
    ##         0.01, 0.1, 1, 5, 10, 100)), kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  10 
    ##       gamma:  0.3333333 
    ## 
    ## Number of Support Vectors:  521
    ## 
    ##  ( 261 260 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_lin, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes


roc_line <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
```

    ## Warning in roc.default(as_tibble(mh_split$test)$vote96, fitted
    ## $decision.values): Deprecated use a matrix as predictor. Unexpected results
    ## may be produced, please pass a numeric vector.

``` r
auc(roc_line)
```

    ## Area under the curve: 0.7468

``` r
plot(roc_line, main = "ROC of Voter Turnout - Linear Kernel, Partial Model")
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-21-1.png) Area under the curve: 0.7468

``` r
mh_lin_all <- tune(svm, vote96 ~ ., data = as_tibble(mh_split$train),
                    kernel = "linear",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))

mh_lall <- mh_lin_all$best.model
summary(mh_lall)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mh_split$train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  100 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  515
    ## 
    ##  ( 259 256 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_lall, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes


roc_line_all <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
```

    ## Warning in roc.default(as_tibble(mh_split$test)$vote96, fitted
    ## $decision.values): Deprecated use a matrix as predictor. Unexpected results
    ## may be produced, please pass a numeric vector.

``` r
auc(roc_line_all)
```

    ## Area under the curve: 0.7502

``` r
plot(roc_line_all, main = "ROC of Voter Turnout- Linear Kernel, Total Model")
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-22-1.png) Area under the curve: 0.7502

``` r
mh_poly_tune <- tune(svm, vote96 ~ age + educ + mhealth_sum, data = as_tibble(mh_split$train),
                    kernel = "polynomial",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))

mh_poly <- mh_poly_tune$best.model
summary(mh_poly)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ age + educ + mhealth_sum, 
    ##     data = as_tibble(mh_split$train), ranges = list(cost = c(0.001, 
    ##         0.01, 0.1, 1, 5, 10, 100)), kernel = "polynomial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  polynomial 
    ##        cost:  100 
    ##      degree:  3 
    ##       gamma:  0.3333333 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  508
    ## 
    ##  ( 256 252 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_poly, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_poly <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
```

    ## Warning in roc.default(as_tibble(mh_split$test)$vote96, fitted
    ## $decision.values): Deprecated use a matrix as predictor. Unexpected results
    ## may be produced, please pass a numeric vector.

``` r
auc(roc_poly)
```

    ## Area under the curve: 0.7411

``` r
plot(roc_poly, main = "ROC of Voter Turnout - Polynomial Kernel, Partial Model")
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-23-1.png)

Area under the curve: 0.7411

``` r
mh_poly_all <- tune(svm, vote96 ~ ., data = as_tibble(mh_split$train),
                    kernel = "polynomial",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))

mh_poly <- mh_poly_all$best.model
summary(mh_poly)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mh_split$train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "polynomial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  polynomial 
    ##        cost:  1 
    ##      degree:  3 
    ##       gamma:  0.125 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  518
    ## 
    ##  ( 268 250 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_poly, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_poly_all <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
```

    ## Warning in roc.default(as_tibble(mh_split$test)$vote96, fitted
    ## $decision.values): Deprecated use a matrix as predictor. Unexpected results
    ## may be produced, please pass a numeric vector.

``` r
auc(roc_poly_all)
```

    ## Area under the curve: 0.7416

``` r
plot(roc_poly_all, main = "ROC of Voter Turnout - Polynomial Kernel, Total Model")
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-24-1.png)

Area under the curve: 0.7395

``` r
mh_rad_tune <- tune(svm, vote96 ~ ., data = as_tibble(mh_split$train),
                    kernel = "radial",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))

mh_rad <- mh_rad_tune$best.model
summary(mh_rad)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mh_split$train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "radial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  1 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  515
    ## 
    ##  ( 271 244 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_rad, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_rad <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
```

    ## Warning in roc.default(as_tibble(mh_split$test)$vote96, fitted
    ## $decision.values): Deprecated use a matrix as predictor. Unexpected results
    ## may be produced, please pass a numeric vector.

``` r
auc(roc_rad)
```

    ## Area under the curve: 0.7466

``` r
plot(roc_rad, main= "ROC of Voter Turnout - Radial Kernel, Total Model")
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-25-1.png) Area under the curve: 0.7466

``` r
plot(roc_line, print.auc = TRUE, col = "blue", print.auc.x = .2)
plot(roc_line_all, print.auc = TRUE, col = "red", print.auc.x = .2, print.auc.y = .4, add = TRUE)
plot(roc_poly, print.auc = TRUE, col = "orange", print.auc.x = .2, print.auc.y = .3, add = TRUE)
plot(roc_poly_all, print.auc = TRUE, col = "green", print.auc.x = .2, print.auc.y = .2, add = TRUE)
plot(roc_rad, print.auc = TRUE, col = "purple", print.auc.x = .2, print.auc.y = .1, add = TRUE)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-26-1.png)

Looking at the area under the curve, the best model is the linear one with all the variables. Its error however is high, This model has a cost of 1, so the margins are narrow around the linear hyperplane. As we can see from the plot below, the error hovers around 0.32

``` r
plot(mh_lin_all)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-27-1.png) \# Part 3: OJ Simpson

#### What is the relationship between race and belief of OJ Simpson's guilt? Develop a robust statistical learning model and use this model to explain the impact of an individual's race on their beliefs about OJ Simpson's guilt.

``` r
oj = read.csv('simpson.csv')
oj <- oj %>%
  na.omit()
```

Starting off with linear regression just to get a basic idea, it is seen that the variable black is highly significant in predicting the belief in Simpson's guilt.Dealing with the non-binary variables: age, educationand income, age can be significant but its not substantive, education and income seem to be significant which is intuitive since education and income are often influenced by race.

``` r
# age, educ, income left
oj_binaries <- lm(guilt ~ dem + rep + black + hispanic + female, data = oj)
oj_age <- lm(guilt ~ age, data = oj)
oj_educ <- lm (guilt ~ educ, data = oj )
oj_income <- lm (guilt ~ income, data = oj )
summary (oj_age)
```

    ## 
    ## Call:
    ## lm(formula = guilt ~ age, data = oj)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -0.8814 -0.6235  0.2740  0.3376  0.4047 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 0.5316786  0.0335835  15.832  < 2e-16 ***
    ## age         0.0035330  0.0007077   4.992 6.71e-07 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.4597 on 1414 degrees of freedom
    ## Multiple R-squared:  0.01732,    Adjusted R-squared:  0.01663 
    ## F-statistic: 24.92 on 1 and 1414 DF,  p-value: 6.709e-07

``` r
summary (oj_income)
```

    ## 
    ## Call:
    ## lm(formula = guilt ~ income, data = oj)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -0.8688 -0.5569  0.2190  0.3503  0.4794 
    ## 
    ## Coefficients:
    ##                         Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)              0.64972    0.02417  26.883  < 2e-16 ***
    ## income$30,000-$50,000    0.03612    0.03227   1.119 0.263208    
    ## income$50,000-$75,000    0.13123    0.03961   3.313 0.000945 ***
    ## incomeOVER $75,000       0.21903    0.04332   5.056 4.83e-07 ***
    ## incomeREFUSED/NO ANSWER -0.12917    0.05845  -2.210 0.027276 *  
    ## incomeUNDER $15,000     -0.09283    0.04269  -2.175 0.029823 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.4547 on 1410 degrees of freedom
    ## Multiple R-squared:  0.04105,    Adjusted R-squared:  0.03765 
    ## F-statistic: 12.07 on 5 and 1410 DF,  p-value: 1.809e-11

``` r
summary (oj_educ)
```

    ## 
    ## Call:
    ## lm(formula = guilt ~ educ, data = oj)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -0.7964 -0.6409  0.2036  0.3591  0.4720 
    ## 
    ## Coefficients:
    ##                                     Estimate Std. Error t value Pr(>|t|)
    ## (Intercept)                          0.79638    0.02172  36.669  < 2e-16
    ## educHIGH SCHOOL GRAD                -0.15546    0.03012  -5.162 2.79e-07
    ## educNOT A HIGH SCHOOL GRAD          -0.26838    0.04626  -5.802 8.07e-09
    ## educREFUSED                          0.20362    0.26451   0.770    0.442
    ## educSOME COLLEGE(TRADE OR BUSINESS) -0.12608    0.03225  -3.910 9.67e-05
    ##                                        
    ## (Intercept)                         ***
    ## educHIGH SCHOOL GRAD                ***
    ## educNOT A HIGH SCHOOL GRAD          ***
    ## educREFUSED                            
    ## educSOME COLLEGE(TRADE OR BUSINESS) ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.4566 on 1411 degrees of freedom
    ## Multiple R-squared:  0.03243,    Adjusted R-squared:  0.02969 
    ## F-statistic: 11.82 on 4 and 1411 DF,  p-value: 1.888e-09

``` r
summary (oj_binaries)
```

    ## 
    ## Call:
    ## lm(formula = guilt ~ dem + rep + black + hispanic + female, data = oj)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -0.8864 -0.1365  0.1717  0.2150  0.9001 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  0.78495    0.03322  23.630  < 2e-16 ***
    ## dem          0.03660    0.03422   1.069  0.28503    
    ## rep          0.10147    0.03495   2.903  0.00375 ** 
    ## black       -0.62695    0.02726 -22.995  < 2e-16 ***
    ## hispanic    -0.05501    0.04062  -1.354  0.17590    
    ## female      -0.05812    0.02085  -2.787  0.00538 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.3852 on 1410 degrees of freedom
    ## Multiple R-squared:  0.312,  Adjusted R-squared:  0.3096 
    ## F-statistic: 127.9 on 5 and 1410 DF,  p-value: < 2.2e-16

``` r
tidy(oj_binaries)
```

    ##          term    estimate  std.error  statistic       p.value
    ## 1 (Intercept)  0.78495083 0.03321771  23.630490 2.815886e-104
    ## 2         dem  0.03659985 0.03422151   1.069498  2.850282e-01
    ## 3         rep  0.10146704 0.03494954   2.903244  3.750656e-03
    ## 4       black -0.62695090 0.02726455 -22.995096  1.262703e-99
    ## 5    hispanic -0.05500863 0.04062158  -1.354172  1.758984e-01
    ## 6      female -0.05811636 0.02084963  -2.787405  5.384502e-03

``` r
oj_binaries1 <- lm(guilt ~ black + hispanic, data = oj)
summary (oj_binaries1)
```

    ## 
    ## Call:
    ## lm(formula = guilt ~ black + hispanic, data = oj)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -0.8149 -0.1647  0.1851  0.1851  0.9002 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  0.81490    0.01180   69.05   <2e-16 ***
    ## black       -0.65021    0.02634  -24.68   <2e-16 ***
    ## hispanic    -0.06487    0.04079   -1.59    0.112    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.3876 on 1413 degrees of freedom
    ## Multiple R-squared:  0.3016, Adjusted R-squared:  0.3006 
    ## F-statistic: 305.1 on 2 and 1413 DF,  p-value: < 2.2e-16

``` r
ggplot(oj, aes(black, guilt)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "black",
       y = "Guilt")
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-29-1.png)

``` r
blah <- lm(guilt ~ age + income + educ + female, data = oj)
```

Plotting moddels considering the variables, we find it is safe to not consider the factors other than black and hispanic for estimation purposes.

``` r
oj %>%
  add_predictions(blah) %>%
  add_residuals(blah) %>%
  {.} -> grid
gridblack <- filter(grid, black == 1)
gridhispanic <- filter(grid, hispanic == 1)
gridother <- filter(grid, black == 0 & hispanic == 0)
ggplot(grid, aes(x = pred, y = resid)) +
  geom_point() +
  geom_smooth(method ="lm", aes(y = resid , color = 'Black'), data = gridblack, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = 'Hispanic'), data = gridhispanic, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = 'Other'), data = gridother, size = 1) +
  scale_colour_manual("", values = c("Black"="blue","Hispanic"="red", "Other"="green")) +
  labs(title = "Predicted Value and Residuals of model with age,income, gender, education",
        x = "Predicted Guilt",
        y = "Residuals") +
  theme_minimal()
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-30-1.png)

``` r
blah1 = lm(guilt ~ age + income + educ + female + black + hispanic, data = oj)
blah2 = lm(guilt ~ age + income + educ + female + dem + rep, data = oj)
blah3 = lm(guilt ~ age + income + educ + dem + rep + black + hispanic, data = oj)
oj %>%
  add_predictions(blah1) %>%
  add_residuals(blah1) %>%
  {.} -> grid
griddem <- filter(grid, dem == 1)
gridrep <- filter(grid, rep == 1)
gridother <- filter(grid, dem == 0 & rep == 0)
ggplot(grid, aes(x = pred, y = resid)) +
  geom_point() +
  geom_smooth(method ="lm", aes(y = resid , color = 'Dem'), data = griddem, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = 'Rep'), data = gridrep, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = 'Other'), data = gridother, size = 1) +
  scale_colour_manual("", values = c("Dem"="blue","Rep"="red", "Other"="green")) +
  labs(title = "Predicted Value and Residuals of model with age,income, gender, education, race",
        x = "Predicted Guilt",
        y = "Residuals") +
  theme_minimal()
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-30-2.png)

``` r
oj %>%
  add_predictions(blah2) %>%
  add_residuals(blah2) %>%
  {.} -> grid
gridblack <- filter(grid, black == 1)
gridhispanic <- filter(grid, hispanic == 1)
gridother <- filter(grid, black == 0 & hispanic == 0)
ggplot(grid, aes(x = pred, y = resid)) +
  geom_point() +
  geom_smooth(method ="lm", aes(y = resid , color = 'Black'), data = gridblack, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = 'Hispanic'), data = gridhispanic, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = 'Other'), data = gridother, size = 1) +
  scale_colour_manual("", values = c("Black"="blue","Hispanic"="red", "Other"="green")) +
  labs(title = "Predicted Value and Residuals of model with age,income, gender, education, party",
        x = "Predicted Guilt",
        y = "Residuals") +
  theme_minimal()
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-30-3.png)

``` r
oj %>%
  add_predictions(blah3) %>%
  add_residuals(blah3) %>%
  {.} -> grid
gridfemale <- filter(grid, female == 1)
gridmale <- filter(grid, female == 0)
ggplot(grid, aes(x = pred, y = resid)) +
  geom_point() +
  geom_smooth(method ="lm", aes(y = resid , color = 'F'), data = gridfemale, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = 'M'), data = gridmale, size = 1) +
  #geom_smooth(method ="lm", aes(y = resid, color = 'Other'), data = gridother, size = 1) +
  scale_colour_manual("", values = c("F"="blue","M"="red")) +
  labs(title = "Predicted Value and Residuals of model with age,income, party, education, race",
        x = "Predicted Guilt",
        y = "Residuals") +
  theme_minimal()
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-30-4.png)

``` r
blah4 = lm(guilt ~ educ + black + hispanic, data = oj)
oj %>%
  add_predictions(blah4) %>%
  add_residuals(blah4) %>%
  {.} -> grid
grid1 <- filter(grid, income == "UNDER $15,000")
grid2 <- filter(grid, income == "$15,000-$30,000")
grid3 <- filter(grid, income == "$30,000-$50,000")
grid4 <- filter(grid, income == "$50,000-$75,000")
grid5 <- filter(grid, income == "OVER $75,000")
ggplot(grid, aes(x = pred, y = resid)) +
  geom_point() +
  geom_smooth(method ="lm", aes(y = resid , color = "UNDER $15,000"), data = grid1, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = "$15,000-$30,000"), data = grid2, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = "$30,000-$50,000"), data = grid3, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = "$50,000-$75,000"), data = grid4, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = "OVER $75,000"), data = grid5, size = 1) +
  scale_colour_manual("", values = c("UNDER $15,000"="blue","$15,000-$30,000"="red", "$30,000-$50,000"="green", "$50,000-$75,000"="pink", "OVER $75,000" = "yellow" )) +
  labs(title = "Predicted Value and Residuals of model with education, race",
        x = "Predicted Guilt",
        y = "Residuals") +
  theme_minimal()
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-31-1.png)

``` r
blah5 = lm(guilt ~ income + black + hispanic, data = oj)
oj %>%
  add_predictions(blah5) %>%
  add_residuals(blah5) %>%
  {.} -> grid
grid1 <- filter(grid, educ == "NOT A HIGH SCHOOL GRAD")
grid2 <- filter(grid, educ == "HIGH SCHOOL GRAD")
grid3 <- filter(grid, educ == "SOME COLLEGE(TRADE OR BUSINESS)")
grid4 <- filter(grid, educ == "COLLEGE GRAD AND BEYOND")
#grid5 <- filter(grid, income == "OVER $75,000")
ggplot(grid, aes(x = pred, y = resid)) +
  geom_point() +
  geom_smooth(method ="lm", aes(y = resid , color = "NOT A HIGH SCHOOL GRAD"), data = grid1, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = "HIGH SCHOOL GRAD"), data = grid2, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = "SOME COLLEGE(TRADE OR BUSINESS)"), data = grid3, size = 1) +
  geom_smooth(method ="lm", aes(y = resid, color = "COLLEGE GRAD AND BEYOND"), data = grid4, size = 1) +
  #geom_smooth(method ="lm", aes(y = resid, color = "OVER $75,000"), data = grid5, size = 1) +
  scale_colour_manual("", values = c("NOT A HIGH SCHOOL GRAD"="blue","HIGH SCHOOL GRAD"="red", "SOME COLLEGE(TRADE OR BUSINESS)"="green", "COLLEGE GRAD AND BEYOND"="pink")) +
  labs(title = "Predicted Value and Residuals of model with income, race",
        x = "Predicted Guilt",
        y = "Residuals") +
  theme_minimal()
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-31-2.png) Concentrating on race:

``` r
oj$Opinion <- factor(oj$guilt, levels = c(0,1), labels = c("Probably not guilty", "Probably guilty"))
 
ggplot(oj, aes(x=black, fill=Opinion)) + geom_bar(position = "dodge") + 
       ylab("Frequency count of respondents") +
       xlab("Race") +
       ggtitle("Opinion of Simpson Guilt Based on Race") +
       theme(plot.title = element_text(hjust = 0.5),
       panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(),
       panel.border = element_rect(linetype = "solid", color = "grey70", fill=NA, size=1.1)) +
       scale_x_continuous(breaks = c(0,1), labels = c("Not Black", "Black"))
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-32-1.png) High proportion of non-black people believe OJ Simpson to be guilty. Considering a logistic regression, since we are using dichotomous variables to predict a dichotomous outcome (belief or no-belief in guilt), it seems to be the most logical choice.

``` r
logit_guilty1 <- glm(guilt ~ black, data = oj, family = binomial)
logit_guilty2 <- glm(guilt ~ hispanic, data = oj, family = binomial)
logit_guilty3 <- glm(guilt ~ black + hispanic, data = oj, family = binomial)
summary(logit_guilty1)
```

    ## 
    ## Call:
    ## glm(formula = guilt ~ black, family = binomial, data = oj)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.8233  -0.5926   0.6487   0.6487   1.9110  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  1.45176    0.07524   19.30   <2e-16 ***
    ## black       -3.10221    0.18271  -16.98   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1758.1  on 1415  degrees of freedom
    ## Residual deviance: 1352.2  on 1414  degrees of freedom
    ## AIC: 1356.2
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
summary(logit_guilty2)
```

    ## 
    ## Call:
    ## glm(formula = guilt ~ hispanic, family = binomial, data = oj)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.5319  -1.5319   0.8603   0.8603   0.9291  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  0.80328    0.05957  13.484   <2e-16 ***
    ## hispanic    -0.18650    0.22098  -0.844    0.399    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1758.1  on 1415  degrees of freedom
    ## Residual deviance: 1757.4  on 1414  degrees of freedom
    ## AIC: 1761.4
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
summary(logit_guilty3)
```

    ## 
    ## Call:
    ## glm(formula = guilt ~ black + hispanic, family = binomial, data = oj)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -1.8374  -0.5980   0.6394   0.6394   2.0758  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)  1.48367    0.07853  18.892   <2e-16 ***
    ## black       -3.11438    0.18316 -17.003   <2e-16 ***
    ## hispanic    -0.40056    0.25315  -1.582    0.114    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1758.1  on 1415  degrees of freedom
    ## Residual deviance: 1349.8  on 1413  degrees of freedom
    ## AIC: 1355.8
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
logit2prob <- function(x){
  exp(x) / (1 + exp(x))
}

prob2odds <- function(x){
  x / (1 - x)
}

prob2logodds <- function(x){
  log(prob2odds(x))
}

guilt_pred <- oj %>%
  add_predictions(logit_guilty1) %>%
  mutate(prob = logit2prob(pred)) %>%
  mutate(odds = prob2odds(prob)) %>%
  mutate(logodds = prob2logodds(prob))



ggplot(guilt_pred, aes(black)) +
  geom_point(aes(y = guilt)) +
  geom_line(aes(y = prob), color = "blue", size = 1) +
  labs(x = "Black",
       y = "Probability of guilt")
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-33-1.png)

``` r
#Logistic regression & chi-squared, phi-coeff


accuracy1 <- oj %>%
  add_predictions(logit_guilty1) %>%
  mutate(pred = logit2prob(pred),
         prob = pred,
         pred = as.numeric(pred > .5))
accuracy2 <- oj %>%
  add_predictions(logit_guilty2) %>%
  mutate(pred = logit2prob(pred),
         prob = pred,
         pred = as.numeric(pred > .5))
accuracy3 <- oj %>%
  add_predictions(logit_guilty3) %>%
  mutate(pred = logit2prob(pred),
         prob = pred,
         pred = as.numeric(pred > .5))

accuracy_rate1 <- 100*mean(accuracy1$guilt == accuracy1$pred, na.rm = TRUE)
accuracy_rate2 <- 100*mean(accuracy2$guilt == accuracy2$pred, na.rm = TRUE)
accuracy_rate3 <- 100*mean(accuracy3$guilt == accuracy3$pred, na.rm = TRUE)
accuracy_rate1
```

    ## [1] 81.5678

``` r
accuracy_rate2
```

    ## [1] 68.78531

``` r
accuracy_rate3
```

    ## [1] 81.5678

``` r
# function to calculate PRE for a logistic regression model
PRE <- function(model){
  # get the actual values for y from the data
  y <- model$y

  # get the predicted values for y from the model
  y.hat <- round(model$fitted.values)

  # calculate the errors for the null model and your model
  E1 <- sum(y != median(y))
  E2 <- sum(y != y.hat)

  # calculate the proportional reduction in error
  PRE <- 100*(E1 - E2) / E1
  return(PRE)
}

PRE(logit_guilty1)
```

    ## [1] 40.95023

``` r
PRE(logit_guilty2)
```

    ## [1] 0

``` r
PRE(logit_guilty3)
```

    ## [1] 40.95023

``` r
auc1 <- auc(accuracy1$guilt, accuracy1$prob)
auc1
```

    ## Area under the curve: 0.7313

``` r
auc2 <- auc(accuracy2$guilt, accuracy2$prob)
auc2
```

    ## Area under the curve: 0.5061

``` r
auc3 <- auc(accuracy3$guilt, accuracy3$prob)
auc3
```

    ## Area under the curve: 0.7397

The model with both black and hispanic as factors have the highest area under the curve but it is only marginally more than the one with only black in it. Also, there is no significant error reduction from black only to black + hispanic model. Beingblack reduces the log-odds of an individuals belief in OJ's guilt by -3.1022, i.e.,it lowers likelihood of believing in OJ's guilt.

Race is a highly dominant factor, so using a decision tree makes sense. It is also easy to interpret.

``` r
set.seed(123) # For reproducibility
oj = read.csv('simpson.csv')
oj = oj[(!is.na(oj$guilt)), ]
oj$Opinion = factor(oj$guilt, levels = c(0,1), labels = c("Innocent", "Guilty"))
oj_split7030 = resample_partition(oj, c(test = 0.3, train = 0.7))
oj_train70 = oj_split7030$train %>%
                tbl_df()
oj_test30 = oj_split7030$test %>%
               tbl_df()

oj_data_train = oj_train70 %>%
                select(-guilt) %>%
                mutate_each(funs(as.factor(.)), dem, rep) %>%
                na.omit

oj_data_test = oj_test30 %>%
               select(-guilt) %>%
               mutate_each(funs(as.factor(.)), dem, rep) %>%
               na.omit

# estimate model
oj_tree <- tree(Opinion ~ ., data = oj_data_train)

# plot tree
tree_data <- dendro_data(oj_tree)

ptree <- ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro()+
  labs(title = "Decision Tree for OJ's Guilt",
       subtitle = 'All predictors, Default Controls')
ptree
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-34-1.png) The decision tree clearly shows that being black is the single most important predictor of belief in Simpson's guilt. Using the Random forest approach with all variables,

``` r
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}

(rf_oj = randomForest(Opinion ~ ., data = oj_data_train, mtry = 3,ntree = 2000))
```

    ## 
    ## Call:
    ##  randomForest(formula = Opinion ~ ., data = oj_data_train, mtry = 3,      ntree = 2000) 
    ##                Type of random forest: classification
    ##                      Number of trees: 2000
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 19.56%
    ## Confusion matrix:
    ##          Innocent Guilty class.error
    ## Innocent      161    149   0.4806452
    ## Guilty         45    637   0.0659824

The error rate is only 19.1%which is pretty good. This is comparable to the values from using logistic regression too (Logistic regression makes most sense in this classification problem). Validating our belief in race to be a guiding factor:

``` r
rf_oj_importance = as.data.frame(importance(rf_oj))

ggplot(rf_oj_importance, mapping=aes(x=rownames(rf_oj_importance), y=MeanDecreaseGini)) +
       geom_bar(stat="identity", aes(fill=MeanDecreaseGini)) + 
       labs(title = "Mean Decrease in Gini Index Across 2000 Random Forest Regression Trees",
       subtitle = "Predicted Opinion of Simpson Guilt",
       x = "Variable",
       y = "Mean Decrease in Gini Index") + 
       theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 0.5),
             panel.border = element_rect(linetype = "solid", color = "grey70", fill=NA, size=1.1), legend.position = 'none') 
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-36-1.png) it yields the highest decrease in Gini Index.

Using linear SVM with all variables

``` r
set.seed(123)
simpson_lin_tune <- tune(svm, Opinion ~ ., data = na.omit(as_tibble(oj_data_train)),
                    kernel = "linear",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
```

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

    ## Warning in svm.default(x, y, scale = scale, ..., na.action = na.action):
    ## Variable(s) 'ind' constant. Cannot scale data.

``` r
summary(simpson_lin_tune)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost
    ##  0.01
    ## 
    ## - best performance: 0.1844343 
    ## 
    ## - Detailed performance results:
    ##    cost     error dispersion
    ## 1 1e-03 0.3125455 0.04260242
    ## 2 1e-02 0.1844343 0.03901566
    ## 3 1e-01 0.1844343 0.03901566
    ## 4 1e+00 0.1844343 0.03901566
    ## 5 5e+00 0.1844343 0.03901566
    ## 6 1e+01 0.1844343 0.03901566
    ## 7 1e+02 0.1844343 0.03901566

``` r
simpson_lin <- simpson_lin_tune$best.model
summary(simpson_lin)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = Opinion ~ ., data = na.omit(as_tibble(oj_data_train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  0.01 
    ##       gamma:  0.05882353 
    ## 
    ## Number of Support Vectors:  637
    ## 
    ##  ( 327 310 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  Innocent Guilty

``` r
#Best
simpson_lin <- simpson_lin_tune$best.model
summary(simpson_lin)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = Opinion ~ ., data = na.omit(as_tibble(oj_data_train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  0.01 
    ##       gamma:  0.05882353 
    ## 
    ## Number of Support Vectors:  637
    ## 
    ##  ( 327 310 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  Innocent Guilty

``` r
#ROC
fitted <- predict(simpson_lin, as_tibble(oj_data_test), decision.values = TRUE) %>%
  attributes

roc_line <- roc(as_tibble(oj_data_test)$Opinion, fitted$decision.values)
```

    ## Warning in roc.default(as_tibble(oj_data_test)$Opinion, fitted
    ## $decision.values): Deprecated use a matrix as predictor. Unexpected results
    ## may be produced, please pass a numeric vector.

``` r
plot(roc_line)
```

![](PS_8_files/figure-markdown_github/unnamed-chunk-37-1.png)

``` r
auc(roc_line)
```

    ## Area under the curve: 0.7841

Area under the curve: 0.7841. So, it doesnt better our accuracy from Random Forest. Cross validation using logistic regression with black being the only variable and random forest with all variables both seem to be good approaches, but I would choose random forest as the better one as it will lead to better results for different seeds. We know from previous work that the cross-validation (70:30) split becomes highly dictated by set composition.
