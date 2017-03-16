Problem set \#8: tree-based methods and support vector machines
================
MACS 30100 - Perspectives on Computational Modeling
**Due Monday March 6th at 11:30am**

-   [Part 1: Sexy Joe Biden (redux times two) \[3 points\]](#part-1-sexy-joe-biden-redux-times-two-3-points)
    -   [1. Data Partition](#data-partition)
    -   [2. Decision tree with default control options.](#decision-tree-with-default-control-options.)
    -   [3. Decision tree with specific contorl options.](#decision-tree-with-specific-contorl-options.)
    -   [4.Bagging approach](#bagging-approach)
    -   [5.Random Forest approach](#random-forest-approach)
    -   [6.Boosting approach](#boosting-approach)
-   [Part 2: Modeling voter turnout \[3 points\]](#part-2-modeling-voter-turnout-3-points)
    -   [1.Five tree-based models.](#five-tree-based-models.)
    -   [2. five SVM models.](#five-svm-models.)
-   [Part 3: OJ Simpson \[4 points\]](#part-3-oj-simpson-4-points)
    -   [1. Relationship between race and belief of OJ Simpson's guilt.](#relationship-between-race-and-belief-of-oj-simpsons-guilt.)
    -   [2. Predict belief of OJ Simpson's guilt.](#predict-belief-of-oj-simpsons-guilt.)

Part 1: Sexy Joe Biden (redux times two) \[3 points\]
=====================================================

-   `biden` - feeling thermometer ranging from 0-100[1]
-   `female` - 1 if respondent is female, 0 if respondent is male
-   `age` - age of respondent in years
-   `dem` - 1 if respondent is a Democrat, 0 otherwise
-   `rep` - 1 if respondent is a Republican, 0 otherwise
-   `educ` - number of years of formal education completed by respondent
    -   `17` - 17+ years (aka first year of graduate school and up)

#### 1. Data Partition

First, we split the data into a training set (70%) and a validation set (30%).

``` r
bid_split <- resample_partition(bid, p = c("test" = .3, "train" = .7))
```

#### 2. Decision tree with default control options.

Then we fit a decision tree to the training data with `biden` as the response variable and the other variables as predictors, leaving control as default.

``` r
# estimate model
set.seed(1234)
bid_tree <- tree(biden ~ ., data = as_tibble(bid_split$train))

# plot tree
tree_data <- dendro_data(bid_tree)
ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = 'Decision Tree with Default Control Options') +
  theme(plot.title = element_text(hjust = 0.5))
```

![](PS8_files/figure-markdown_github/biden%20tree-1.png)

``` r
# calculate test MSE
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}
mse(bid_tree, bid_split$test) 
```

    ## [1] 406.4167

As shown by the above decision tree, there are three terminal nodes (57.6, 43.23, 74.51), two internal nodes (`dem` &lt; 0.5 and `rep` &lt; 0.5), and three branches. This decision tree indicates that: If a person is a Democrat, then the model estimates his/her Biden score to be 74.51. If one is not a democrat, then we proceed down the left branch to the next internal node to check whether or not he is a Republican. If this person is a Republican, then the model estimates his/her Biden score to be 43.23. If this person is not a Republican, then the model estimates his/her Biden score to be 57.6. In other words, Democrats are estimated to give Biden the highest score, followed by the Independents, and the Republicans are estimated to give the lowest score.

The test MSE of this decision tree model is 406.4167.

#### 3. Decision tree with specific contorl options.

Now we fit another tree to the training data with the following `control` options:

``` r
set.seed(1234)

# fit the decision tree with specific controls
bid_tree2 <- tree(biden ~ ., data = as_tibble(bid_split$train), control = tree.control(nobs = nrow(bid_split$train),
                            mindev = 0))
summary(bid_tree2)
```

    ## 
    ## Regression tree:
    ## tree(formula = biden ~ ., data = as_tibble(bid_split$train), 
    ##     control = tree.control(nobs = nrow(bid_split$train), mindev = 0))
    ## Number of terminal nodes:  192 
    ## Residual mean deviance:  339.3 = 364000 / 1073 
    ## Distribution of residuals:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     -66     -10       1       0      11      47

``` r
test_mse <- mse(bid_tree2, bid_split$test) 

# generate 10-fold cross-validation trees to determine the optimal tree
bid_cv <- crossv_kfold(bid, k = 10) %>%
  mutate(tree = map(train, ~ tree(biden ~ ., data = .,
     control = tree.control(nobs = nrow(bid),
                            mindev = 0))))

# calculate each possible prune result for each fold
bid_cv <- expand.grid(bid_cv$.id, 2:25) %>% 
  as_tibble() %>%
  mutate(Var2 = as.numeric(Var2)) %>%
  rename(.id = Var1,
         k = Var2) %>%
  left_join(bid_cv) %>%
  mutate(prune = map2(tree, k, ~ prune.tree(.x, best = .y)),
         mse = map2_dbl(prune, test, mse))

bid_cv %>%
  select(k, mse) %>%
  group_by(k) %>%
  summarize(test_mse = mean(mse),
            sd = sd(mse, na.rm = TRUE)) %>%
  ggplot(aes(k, test_mse)) +
  geom_point() +
  geom_line() +
  labs(x = "Number of terminal nodes",
       y = "Test MSE")
```

![](PS8_files/figure-markdown_github/biden%20tree2-1.png)

``` r
test_mse
```

    ## [1] 481.4899

We use 10-fold cross-validation approach to determine the optimal tree. According to the above graph, it turns out that both 3 and 4 terminal nodes will give us the lowest test MSE. Further analysis shows that the test MSE associated with 3 nodes is 402.0678 whereas 4 nodes is 402.1037. Therefore we prune the decision tree to the optimal number of terminal nodes, which is 3.

``` r
set.seed(1234)
mse_3nodes = mean(bid_cv$mse[bid_cv$k==3])
mse_4nodes = mean(bid_cv$mse[bid_cv$k==4])

mod <- prune.tree(bid_tree2, best = 3)

# plot tree
tree_data2 <- dendro_data(mod)
ggplot(segment(tree_data2)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data2), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data2), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = 'Decision Tree with Optimal Terminal Nodes') +
  theme(plot.title = element_text(hjust = 0.5))
```

![](PS8_files/figure-markdown_github/bid%20tree2%20plot-1.png) The optimal tree is precisely the same as the decision tree with default control options. The models estimates that Democrats will give Biden the higest score of 74.51, Independents will give 57.6, and Republicans will give the lowest score of 43.23. While the test MSE of original decision tree is 481.4899366, the optimal test MSE with three terminal nodes is 402.0678. This is a substantial decrease. We thus concludes that pruning the tree indeed improves the test MSE.

#### 4.Bagging approach

Now we use bagging approach to analyze this data.

``` r
set.seed(1234)
(bid_bag <- randomForest(biden ~ ., data = bid,
                             mtry = 5, ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = bid, mtry = 5, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##           Mean of squared residuals: 492.7417
    ##                     % Var explained: 10.44

``` r
# variable importance measures
data_frame(var = rownames(importance(bid_bag)),
           RSSDecrease = importance(bid_bag)[,1]) %>%
  mutate(var = fct_reorder(var, RSSDecrease, fun = median)) %>%
  ggplot(aes(var, RSSDecrease)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting Biden Score",
       subtitle = "Bagging",
       x = NULL,
       y = "the total amount of reduction in the RSS")
```

![](PS8_files/figure-markdown_github/biden%20bagging-1.png) As shown by the above figure, for the Biden bagged model, `age` and `dem` are the most important predictors, as they contribute to the largest amounts of reduction in the RSS. In other words, the bagged model estimates that one's age and affiliation to Democratic Party would highly influence one's feeling thermometer score to Joe Biden. In contrast, years of education, gender, and being a republican or not are relatively unimportant to the model. However, the test MSE of the bagged model is 492.7417, a substantial increase from the MSE of the previous model. This suggests the bagged model is not good and may have an overfitting issue.

#### 5.Random Forest approach

Now we use the random forest approach to analyze this data.

``` r
set.seed(1234)
bid_rf_models <- list("rf_mtry1" = randomForest(biden ~ ., data = bid,
                            mtry=1, ntree = 500),
                       "rf_mtry2" = randomForest(biden ~ ., data = bid,
                            mtry=2, ntree = 500),
                       "rf_mtry3" = randomForest(biden ~ ., data = bid,
                            mtry=3, ntree = 500),
                       "rf_mtry4" = randomForest(biden ~ ., data = bid,
                            mtry=4, ntree = 500)
                       )
bid_rf_models 
```

    ## $rf_mtry1
    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = bid, mtry = 1, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 1
    ## 
    ##           Mean of squared residuals: 407.013
    ##                     % Var explained: 26.02
    ## 
    ## $rf_mtry2
    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = bid, mtry = 2, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##           Mean of squared residuals: 402.0724
    ##                     % Var explained: 26.92
    ## 
    ## $rf_mtry3
    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = bid, mtry = 3, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##           Mean of squared residuals: 429.6944
    ##                     % Var explained: 21.9
    ## 
    ## $rf_mtry4
    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = bid, mtry = 4, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 4
    ## 
    ##           Mean of squared residuals: 476.5571
    ##                     % Var explained: 13.38

``` r
# variable importance measures
data_frame(var = rownames(importance(bid_rf_models$rf_mtry2)),
           RSSDecrease = importance(bid_rf_models$rf_mtry2)[,1]) %>%
  mutate(var = fct_reorder(var, RSSDecrease, fun = median)) %>%
  ggplot(aes(var, RSSDecrease)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting Biden Score",
       subtitle = "Random Forest",
       x = NULL,
       y = "the total amount of reduction in the RSS")
```

![](PS8_files/figure-markdown_github/biden%20randomforest-1.png) We tested different number of variables *m* tried at each split for the random forest model. In the bagged model, we use all predictors in each split, but this results in correlation and similarity among the trees. By contrast, the random forest model intentionally ignores a random set of variables, and only considers a random sample of *m* predictors. This ensures that each tree is uncorrelated, and thus leads to a lower test MSE. After testing for different *m*, we found that in our case, the test MSE is lowest at *m*=2. The variance is slightly higher at 1, and the variance increases substantially as *m* increases after the threshold of 2.

Therefore we adopt the random forest model with *m*=2. The test MSE of the random forest model is 402.0724, a substantial decrease from the bagged model. Now `age` is no longer important. The variable importance measures show that `dem` and `rep` are actually the most important predictors to the model. They contribute to the largest amount of reduction in the RSS of the model. This means one's party affiliation is the single most important factor in determining one's feeling thermometer scores to Joe Biden when compared with other variables in the model. We can also observe that the total reduction of RSS associated with each variable is generally smaller using the random forest method compared to bagging - this is because of the variable restriction imposed when considering splits.

#### 6.Boosting approach

Last, we turn to use boosting approach to analyze the data.

``` r
set.seed(1234)
# boosting with different lambda
(bid_boo_s1 = gbm(biden ~ ., data = bid_split$train, n.trees = 10000, interaction.depth = 1, shrinkage = 0.01))
```

    ## Distribution not specified, assuming gaussian ...

    ## gbm(formula = biden ~ ., data = bid_split$train, n.trees = 10000, 
    ##     interaction.depth = 1, shrinkage = 0.01)
    ## A gradient boosted model with gaussian loss function.
    ## 10000 iterations were performed.
    ## There were 5 predictors of which 5 had non-zero influence.

``` r
(bid_boo_s2 = gbm(biden ~ ., data = bid_split$train, n.trees = 10000, interaction.depth = 1, shrinkage = 0.03))
```

    ## Distribution not specified, assuming gaussian ...

    ## gbm(formula = biden ~ ., data = bid_split$train, n.trees = 10000, 
    ##     interaction.depth = 1, shrinkage = 0.03)
    ## A gradient boosted model with gaussian loss function.
    ## 10000 iterations were performed.
    ## There were 5 predictors of which 5 had non-zero influence.

``` r
(bid_boo_s3 = gbm(biden ~ ., data = bid_split$train, n.trees = 10000, interaction.depth = 1, shrinkage = 0.05))
```

    ## Distribution not specified, assuming gaussian ...

    ## gbm(formula = biden ~ ., data = bid_split$train, n.trees = 10000, 
    ##     interaction.depth = 1, shrinkage = 0.05)
    ## A gradient boosted model with gaussian loss function.
    ## 10000 iterations were performed.
    ## There were 5 predictors of which 5 had non-zero influence.

``` r
# calculate mse

boost_mse <- function(model) {
  best.iter <- gbm.perf(model,method="OOB")
  f.predict <- predict(model,bid_split$test,best.iter)
  boo_mse = mean((f.predict - bid[bid_split$test[2]$idx, ]$biden)^2)
  print(best.iter)
  return(boo_mse)
}

boost_mse(bid_boo_s1)
```

![](PS8_files/figure-markdown_github/biden%20boosting-1.png)

    ## [1] 279

    ## [1] 407.7262

``` r
boost_mse(bid_boo_s2)
```

![](PS8_files/figure-markdown_github/biden%20boosting-2.png)

    ## [1] 206

    ## [1] 401.1384

``` r
boost_mse(bid_boo_s3)
```

![](PS8_files/figure-markdown_github/biden%20boosting-3.png)

    ## [1] 189

    ## [1] 399.6882

The boosting approach gives us different MSEs with different number of trees and at different shrinkage rates. Particularly, when shrinkage rate increases from 0.01 to 0.05, the test MSE with optimal trees decreases from 407.7262 to 399.6882. The results show that the boosting model with 189 trees and 0.05 shrinkage rate perfoms the best among all models we tested. It appears that the slower the boosting approach learns, the less number of trees it needs and the better performance it achieves.

Part 2: Modeling voter turnout \[3 points\]
===========================================

-   `vote96` - 1 if the respondent voted in the 1996 presidential election, 0 otherwise
-   `mhealth_sum` - index variable which assesses the respondent's mental health, ranging from 0 (an individual with no depressed mood) to 9 (an individual with the most severe depressed mood)[2]
-   `age` - age of the respondent
-   `educ` - Number of years of formal education completed by the respondent
-   `black` - 1 if the respondent is black, 0 otherwise
-   `female` - 1 if the respondent is female, 0 if male
-   `married` - 1 if the respondent is currently married, 0 otherwise
-   `inc10` - Family income, in $10,000s

#### 1.Five tree-based models.

To predict the voter turnout of the presidential election, traditional theories emphasizes the socio-economic factors such as age and educational level. In addition, an emerging theory indicates that an individual's mental health would influence his/her political participation. Guided by these theories, we first fit a decision tree model with the predictor of `mhealth_sum`, then we add traditional predictors `age` and `educ` to check if this fits the data better.

``` r
# split the data
set.seed(1234)
vote_data <- gss %>%
  mutate_each(funs(as.factor(.)), vote96, black, female, married) %>%
  na.omit

vote_split <- resample_partition(vote_data, p = c("test" = .3, "train" = .7))
train <- vote_split$train 
test <- vote_split$test 

# calculate error rate
err.rate.tree <- function(model, data) {
  data <- as_tibble(data)
  response <- as.character(model$terms[[2]])
  
  pred <- predict(model, newdata = data, type = "class")
  actual <- data[[response]]
  
  return(mean(pred != actual, na.rm = TRUE))
}

# estimate tree model 1
vote_model1 <- tree(vote96 ~ mhealth_sum, data = train,
                     control = tree.control(nobs = nrow(train),
                            mindev = 0))

err.rate.tree(vote_model1, test)
```

    ## [1] 0.3295129

``` r
# estimate tree model 2
vote_model2 <- tree(vote96 ~ mhealth_sum + age + educ, data = train,
                     control = tree.control(nobs = nrow(train),
                            mindev = 0))

err.rate.tree(vote_model2, test)
```

    ## [1] 0.3610315

We first split the data into 70% training set and 30% test set. Then we fit the two tree models and calculates the test error rates. The results show that mental health is indeed an important factor, and this single predictor can predict the voter turnout at an test error rate of 32.95%. After adding age and educational level into the tree model, the error rate increases to 36.10%. Recall that decision trees are highly susceptible to overfitting due to its natural complexity, we want to prune the tree to grow a large tree and preserve the most important branches or elements. We thus turn to use 10-fold cross-validation approach to determine the optimal terminal nodes.

``` r
set.seed(1234)
# generate 10-fold CV trees
vote_cv <- vote_data %>%
  na.omit() %>%
  crossv_kfold(k = 10) %>%
  mutate(tree = map(train, ~ tree(vote96 ~ mhealth_sum + age + educ, data = .,
     control = tree.control(nobs = nrow(vote_data),
                            mindev = 0))))

# calculate each possible prune result for each fold
vote_cv <- expand.grid(vote_cv$.id, 2:25) %>%
  as_tibble() %>%
  mutate(Var2 = as.numeric(Var2)) %>%
  rename(.id = Var1,
         k = Var2) %>%
  left_join(vote_cv) %>%
  mutate(prune = map2(tree, k, ~ prune.misclass(.x, best = .y)),
         mse = map2_dbl(prune, test, err.rate.tree))

vote_cv %>%
  group_by(k) %>%
  summarize(test_mse = mean(mse),
            sd = sd(mse, na.rm = TRUE)) %>%
  ggplot(aes(k, test_mse)) +
  geom_point() +
  geom_line() +
  labs(title = "Voter Turnout tree",
       subtitle = "all predictors",
       x = "Number of terminal nodes",
       y = "Test error rate")
```

![](PS8_files/figure-markdown_github/vote%20tree%20model3-1.png)

``` r
# tree pruning with optimal terminal nodes
vote_optimal <- prune.tree(vote_model2, best = 6)
mean(vote_cv$mse[vote_cv$k==6])
```

    ## [1] 0.2720085

The result shows that the decision tree model has the lowest test error rate with 6 terminal nodes, where the error rate decreases to 27.2%. This is a substantial decrease from the original model. But considering that trees can be non-robust and the test error may change dramatically between different splits. In order to reduce variance, we turn to use bagging approach to aggregate decision trees and average across them. We then use random forest approach to see if we can further improve performance.

``` r
set.seed(1234)
(vote_bag <- randomForest(vote96 ~ mhealth_sum + age + educ, data = train,
                             mtry = 3))
```

    ## 
    ## Call:
    ##  randomForest(formula = vote96 ~ mhealth_sum + age + educ, data = train,      mtry = 3) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 32.23%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 122 139   0.5325670
    ## 1 124 431   0.2234234

``` r
(vote_rf1 <- randomForest(vote96 ~ mhealth_sum + age + educ, data = train, mtry=1))
```

    ## 
    ## Call:
    ##  randomForest(formula = vote96 ~ mhealth_sum + age + educ, data = train,      mtry = 1) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 1
    ## 
    ##         OOB estimate of  error rate: 28.55%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 103 158   0.6053640
    ## 1  75 480   0.1351351

``` r
(vote_rf2 <- randomForest(vote96 ~ mhealth_sum + age + educ, data = train, mtry=2))
```

    ## 
    ## Call:
    ##  randomForest(formula = vote96 ~ mhealth_sum + age + educ, data = train,      mtry = 2) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 32.11%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 114 147   0.5632184
    ## 1 115 440   0.2072072

The bagging approach gives us a test error rate (estimated by OOB) of 32.23%, this does not improve the model performance. But after we use random forest approach to ignore some random sets of predictors, the uncorrelated trees with a *m* = 1 indeed give us a lower OOB estimated test error rate of 28.55%. However, this is still slightly higher than the previous decision tree model with 6 terminal nodes. We thus select the optimal decision tree model to interpret the result.

``` r
#plot tree
tree_data <- dendro_data(vote_optimal)
ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Voter Turnout tree",
       subtitle = "mental health + age + education")
```

![](PS8_files/figure-markdown_github/vote%20tree%20optimal-1.png) The decision tree shows that for people who have a mental health index greater than 4.5, meaning a bad mental health, we proceed to the next internal node on the right branch and check if the person is older than 30.5. If yes, the model estimates that he would vote in the presidential election. If he is younger than 30.5, then the model estimates this person would not vote in the election. However, for people who have a mental health index smaller than 4.5 (meaning a good mental health), the model estimates that he would show up to vote in the election, no matter how old he is and how many years of education he has. Note that the branches split by predictors `age`&lt; 44.5 and `educ` in fact lead to the same outcome. This is because splitting the node leads to increased node purity where we are even more confident in our predictions. This decision tree model shows that one's mental health is important in determining his/her political participation, whereas age and educational level play a less important role.

#### 2. five SVM models.

Now we turn to use SVM models to fit this data. This time we use all predictors in the dataset and we will compare different SVM models by using 10-fold cv on the 70% training set. We start from a SVM model with linear kernel.

``` r
library(e1071)
set.seed(1234)
mh <- gss %>%
  mutate_each(funs(as.factor(.)), vote96, black, female, married) %>%
  na.omit

mh_split <- resample_partition(mh, p = c("test" = .3, "train" = .7))

mh_lin_tune <- tune(svm, vote96 ~ ., data = as_tibble(mh_split$train),
                    kernel = "linear",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(mh_lin_tune)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost
    ##   0.1
    ## 
    ## - best performance: 0.2905601 
    ## 
    ## - Detailed performance results:
    ##    cost     error dispersion
    ## 1 1e-03 0.3199940 0.06774801
    ## 2 1e-02 0.3224330 0.06831178
    ## 3 1e-01 0.2905601 0.05658162
    ## 4 1e+00 0.2991418 0.06115980
    ## 5 5e+00 0.2991418 0.06115980
    ## 6 1e+01 0.2991418 0.06115980
    ## 7 1e+02 0.2991418 0.06115980

``` r
mh_lin <- mh_lin_tune$best.model
summary(mh_lin)
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
    ##        cost:  0.1 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  511
    ## 
    ##  ( 256 255 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_lin, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

# draw ROC line
roc_line <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
plot(roc_line)
```

![](PS8_files/figure-markdown_github/SVM%20linear-1.png)

``` r
# calculate AUC 
auc(roc_line)
```

    ## Area under the curve: 0.7423

The 10-fold cross-validation shows that the optimal cost parameter is 0.1 for the linear SVM (i.e. support vector classifier). It produces the lowest CV error rate of 0.2905. So we use that model for estimating model fit using a ROC curve. The AUC score is 0.7423, which essentially indicates the model sustantively improves the performance of the classifier from 0.5, an AUC score of a random guess of voter turnout.

However, how does this linear SVM compare to a polynomial kernel SVM? We now turn to fit a polynomial SVM to the data.

``` r
mh_poly_tune <- tune(svm, vote96 ~ ., data = as_tibble(mh_split$train),
                    kernel = "polynomial",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(mh_poly_tune)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost
    ##     5
    ## 
    ## - best performance: 0.2965221 
    ## 
    ## - Detailed performance results:
    ##    cost     error dispersion
    ## 1 1e-03 0.3197230 0.05361547
    ## 2 1e-02 0.3197230 0.05361547
    ## 3 1e-01 0.3136104 0.05484807
    ## 4 1e+00 0.3025444 0.05686011
    ## 5 5e+00 0.2965221 0.04544425
    ## 6 1e+01 0.2989461 0.04354621
    ## 7 1e+02 0.3185787 0.03467297

``` r
mh_poly <- mh_poly_tune$best.model
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
    ##        cost:  5 
    ##      degree:  3 
    ##       gamma:  0.125 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  495
    ## 
    ##  ( 258 237 )
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
plot(roc_poly)
```

![](PS8_files/figure-markdown_github/SVM%20poly-1.png)

``` r
auc(roc_poly)
```

    ## Area under the curve: 0.7413

The 10-fold cv shows that the optimal cost parameter is 5 for the polynomial SVM. It produces the lowest 10-fold CV error rate of 0.2965. However, this model is not quite as good as the linear SMV. The associated CV error rate is higher than the linear kernel and the resulting test AUC is 0.7413, slightly smaller than the linear kernel.

How does this compare with the radial kernel? We move on to fit a radial SVM model with the data.

``` r
mh_rad_tune <- tune(svm, vote96 ~ ., data = as_tibble(mh_split$train),
                    kernel = "radial",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(mh_rad_tune)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost
    ##     1
    ## 
    ## - best performance: 0.2965372 
    ## 
    ## - Detailed performance results:
    ##    cost     error dispersion
    ## 1 1e-03 0.3197832 0.05763695
    ## 2 1e-02 0.3197832 0.05763695
    ## 3 1e-01 0.3051189 0.07012804
    ## 4 1e+00 0.2965372 0.05518606
    ## 5 5e+00 0.3002559 0.06230836
    ## 6 1e+01 0.3039295 0.06379227
    ## 7 1e+02 0.3186992 0.06912742

``` r
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
    ## Number of Support Vectors:  509
    ## 
    ##  ( 265 244 )
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
plot(roc_rad)
```

![](PS8_files/figure-markdown_github/svm%20radial-1.png)

``` r
auc(roc_rad)
```

    ## Area under the curve: 0.7349

The results show that the radial SVM model works worse than both the polynomial and the linear SVMs. The 10-fold cv shows that the radial SVM has the lowest CV error rate of 0.2965 with the best cost parameter of 1. While the cost parameter is the smallest, this cv error rate is almost the same with the polynomial SVM, and the test AUC is 0.7349, smaller than both linear and polynomial SVMs.

The above three SVMs use all predictors to fit the model. However, based on our previous analysis, some predictors are actually unimportant to the model. Therefore, we now fit a linear SVM with some selected predictors.

``` r
mh_lin_tune2 <- tune(svm, vote96 ~ mhealth_sum + educ + age + inc10, data = as_tibble(mh_split$train),
                    kernel = "linear",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(mh_lin_tune2)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost
    ##     5
    ## 
    ## - best performance: 0.3003764 
    ## 
    ## - Detailed performance results:
    ##    cost     error dispersion
    ## 1 1e-03 0.3200994 0.06627120
    ## 2 1e-02 0.3213038 0.06265516
    ## 3 1e-01 0.3090184 0.05355423
    ## 4 1e+00 0.3004065 0.04413528
    ## 5 5e+00 0.3003764 0.04234539
    ## 6 1e+01 0.3003764 0.04234539
    ## 7 1e+02 0.3003764 0.04234539

``` r
mh_lin2 <- mh_lin_tune2$best.model
summary(mh_lin2)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ mhealth_sum + educ + 
    ##     age + inc10, data = as_tibble(mh_split$train), ranges = list(cost = c(0.001, 
    ##     0.01, 0.1, 1, 5, 10, 100)), kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  5 
    ##       gamma:  0.25 
    ## 
    ## Number of Support Vectors:  507
    ## 
    ##  ( 254 253 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted2 <- predict(mh_lin2, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_lin2 <- roc(as_tibble(mh_split$test)$vote96, fitted2$decision.values)
plot(roc_lin2)
```

![](PS8_files/figure-markdown_github/SVM%20selected-1.png)

``` r
auc(roc_lin2)
```

    ## Area under the curve: 0.7429

Again, the 10-fold cv suggests that the optimal cost parameter is 5 for the revised linear SVM. It produces the lowest 10-fold CV error rate of 0.2917. This error rate is smaller than the polynomial and radial SVMs, but it is slightly larger than the original linear SVM. Nonetheless, the test AUC score of this model is 0.7429, larger than the original radial SVM. This indicates that in general the revised linear SVM model does improve the performance from all previous models.

Lastly, we want to check if the model performance will further improve at different degree levels.

``` r
mh_lin_tune3 <- tune(svm, vote96 ~ mhealth_sum + educ + age + inc10, data = as_tibble(mh_split$train),
                    kernel = "linear",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100), degree = c(3, 4, 5)))
summary(mh_lin_tune3)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost degree
    ##     5      3
    ## 
    ## - best performance: 0.2917946 
    ## 
    ## - Detailed performance results:
    ##     cost degree     error dispersion
    ## 1  1e-03      3 0.3199639 0.05254198
    ## 2  1e-02      3 0.3212135 0.05616123
    ## 3  1e-01      3 0.2954682 0.03917463
    ## 4  1e+00      3 0.2930292 0.03756251
    ## 5  5e+00      3 0.2917946 0.03897218
    ## 6  1e+01      3 0.2917946 0.03897218
    ## 7  1e+02      3 0.2917946 0.03897218
    ## 8  1e-03      4 0.3199639 0.05254198
    ## 9  1e-02      4 0.3212135 0.05616123
    ## 10 1e-01      4 0.2954682 0.03917463
    ## 11 1e+00      4 0.2930292 0.03756251
    ## 12 5e+00      4 0.2917946 0.03897218
    ## 13 1e+01      4 0.2917946 0.03897218
    ## 14 1e+02      4 0.2917946 0.03897218
    ## 15 1e-03      5 0.3199639 0.05254198
    ## 16 1e-02      5 0.3212135 0.05616123
    ## 17 1e-01      5 0.2954682 0.03917463
    ## 18 1e+00      5 0.2930292 0.03756251
    ## 19 5e+00      5 0.2917946 0.03897218
    ## 20 1e+01      5 0.2917946 0.03897218
    ## 21 1e+02      5 0.2917946 0.03897218

``` r
mh_lin3 <- mh_lin_tune3$best.model
summary(mh_lin3)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ mhealth_sum + educ + 
    ##     age + inc10, data = as_tibble(mh_split$train), ranges = list(cost = c(0.001, 
    ##     0.01, 0.1, 1, 5, 10, 100), degree = c(3, 4, 5)), kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  5 
    ##       gamma:  0.25 
    ## 
    ## Number of Support Vectors:  507
    ## 
    ##  ( 254 253 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted3 <- predict(mh_lin3, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_lin3 <- roc(as_tibble(mh_split$test)$vote96, fitted3$decision.values)
plot(roc_lin3)
```

![](PS8_files/figure-markdown_github/svm%20degrees-1.png)

``` r
auc(roc_lin3)
```

    ## Area under the curve: 0.7429

The 10-fold cv shows that, when using a linear kernel with selected predictors, the linear SVM achieves the best performance with a cost parameter of 1 and a degree level at 3. The 10-fold cv error rate is 0.2867 -- this is the lowest error rate among all five models. Also, the test AUC score is 0.743, larger than all previous models. Both measures indicate that this is the best model so far. We conclude that this revised linear SVM model performs better than other models.

We now put together all five models and visualize the ROC lines.

``` r
plot(roc_line, print.auc = TRUE, col = "blue")
plot(roc_poly, print.auc = TRUE, col = "green", print.auc.y = .4, add = TRUE)
plot(roc_rad, print.auc = TRUE, col = "orange", print.auc.y = .3, add = TRUE)
plot(roc_lin2, print.auc = TRUE, col = "pink", print.auc.y = .2, add = TRUE)
plot(roc_lin3, print.auc = TRUE, col = "red", print.auc.y = .1, add = TRUE)
```

![](PS8_files/figure-markdown_github/svm%20plot-1.png) It shows that the SVM with the highest AUC is the two revised linear SVM models with selected predictors, followed by the linear and then the polynomial SVMs with all predictors. The radial SVM produces the worst AUC. Moreover, the linear SVM with selected predictors and with a cost parameter of 1 and at degree of 3 had the lowest 10-fold cv error rate. We conclude that this is the best model.

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

#### 1. Relationship between race and belief of OJ Simpson's guilt.

The response and the predictors are all binary variables. This suggests that their relationship is less likely to be linear. Since we only have two predictors, a decision tree would provide simple and intuitive interpretation. We therefore use a decision tree to explore the relationship between race and belief of OJ Simpson's guilt. We start from partitioning the data into 70% traning set and 30% test set. Then we fit a decision tree with the training data. Next we use 10-fold cv to determine the optimal terminal nodes.

``` r
# split the data
set.seed(1234)
sim <- sim %>%
  mutate_each(funs(as.factor(.)), guilt, dem, rep, ind, black, female, hispanic) %>%
  na.omit

sim_split <- resample_partition(sim, p = c("test" = .3, "train" = .7))

# estimate decision tree model
sim_race <- tree(guilt ~ black + hispanic, data = sim_split$train,
                     control = tree.control(nobs = nrow(sim_split$train),
                            mindev = 0))

# 10-fold cv determines the optimal terminal nodes
sim_cv <- sim %>%
  na.omit() %>%
  crossv_kfold(k = 10) %>%
  mutate(tree = map(train, ~ tree(guilt ~ black + hispanic, data = .,
     control = tree.control(nobs = nrow(sim),
                            mindev = 0))))

# calculate each possible prune result for each fold
sim_cv <- expand.grid(sim_cv$.id, 2:5) %>%
  as_tibble() %>%
  mutate(Var2 = as.numeric(Var2)) %>%
  rename(.id = Var1,
         k = Var2) %>%
  left_join(sim_cv) %>%
  mutate(prune = map2(tree, k, ~ prune.misclass(.x, best = .y)),
         mse = map2_dbl(prune, test, err.rate.tree))

sim_cv %>%
  group_by(k) %>%
  summarize(test_mse = mean(mse),
            sd = sd(mse, na.rm = TRUE)) %>%
  ggplot(aes(k, test_mse)) +
  geom_point() +
  geom_line() +
  labs(title = "Belief of Simpson's Guilty tree",
       subtitle = "race",
       x = "Number of terminal nodes",
       y = "Test error rate")
```

![](PS8_files/figure-markdown_github/simpson%20binary-1.png)

``` r
mean(sim_cv$mse[sim_cv$k==2])
```

    ## [1] 0.1843622

The 10-fold cv shows that different numbers of terminal nodes actually produce exactly the same test error rate of 0.184362, indicating the tree model is stable and robust with different number of nodes. The test error rate is pretty good, indicating that there is only 18.43% cross-validation error rate when we use race as predictors. We go for parsimony and prune the tree model to two terminal nodes.

``` r
# tree pruning with optimal terminal nodes
sim_optimal <- prune.tree(sim_race, best = 2)

#plot tree
tree_data <- dendro_data(sim_optimal)
ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Belief of Simpson Guilty tree",
       subtitle = "race")
```

![](PS8_files/figure-markdown_github/simpson%20pruning-1.png) The tree model shows that `black` is the single most important factor to predict one's belief of Simpson's guilty. Particularly, if one is black, the tree model estimates that this person would think OJ Simpson to be probably not guilty. By contrast, if one is not black, the tree model predicts that this person is likely to think OJ Simpson is probably guilty. The predictor `hispanic` seems to be not important to the model, as the test error rate remain the same with more terminal nodes.

#### 2. Predict belief of OJ Simpson's guilt.

Now we turn to fit a full model to predict people's belief of OJ Simpson's guilt. Based on previous analysis, we know that `black` is an important factor influencing people's belief. We suspect that one's gender, educational level and income will also influence one's belief. Specifically, OJ Simpson was suspected of nurduring his wife. We thus assume that women would feel symphathetic to the wife and would tend to believe that Simpson was guilty. Nevertheless, people with more education and higher income may be more likely to believe the law and insitutions, and would tend to agree with the Court's decision. In order to testify our assumptions, we now fit a logistic regression model to the training data.

``` r
#logit model fitting with the traning set
sim_log <- glm(guilt ~ black + female + educ + income, data = sim_split$train, family = binomial)
summary(sim_log)
```

    ## 
    ## Call:
    ## glm(formula = guilt ~ black + female + educ + income, family = binomial, 
    ##     data = sim_split$train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.3493  -0.5439   0.5601   0.6540   2.0479  
    ## 
    ## Coefficients:
    ##                                      Estimate Std. Error z value Pr(>|z|)
    ## (Intercept)                           1.83287    0.26804   6.838 8.02e-12
    ## black1                               -3.01764    0.22103 -13.652  < 2e-16
    ## female1                              -0.29012    0.17560  -1.652  0.09850
    ## educHIGH SCHOOL GRAD                 -0.30171    0.22301  -1.353  0.17609
    ## educNOT A HIGH SCHOOL GRAD           -0.49098    0.33095  -1.484  0.13792
    ## educREFUSED                          13.32483  489.68587   0.027  0.97829
    ## educSOME COLLEGE(TRADE OR BUSINESS)  -0.04037    0.24111  -0.167  0.86703
    ## income$30,000-$50,000                -0.05979    0.22584  -0.265  0.79121
    ## income$50,000-$75,000                 0.17564    0.29032   0.605  0.54519
    ## incomeOVER $75,000                    0.86119    0.39535   2.178  0.02938
    ## incomeREFUSED/NO ANSWER              -1.01522    0.34937  -2.906  0.00366
    ## incomeUNDER $15,000                  -0.39908    0.28196  -1.415  0.15696
    ##                                        
    ## (Intercept)                         ***
    ## black1                              ***
    ## female1                             .  
    ## educHIGH SCHOOL GRAD                   
    ## educNOT A HIGH SCHOOL GRAD             
    ## educREFUSED                            
    ## educSOME COLLEGE(TRADE OR BUSINESS)    
    ## income$30,000-$50,000                  
    ## income$50,000-$75,000                  
    ## incomeOVER $75,000                  *  
    ## incomeREFUSED/NO ANSWER             ** 
    ## incomeUNDER $15,000                    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1229.06  on 991  degrees of freedom
    ## Residual deviance:  910.43  on 980  degrees of freedom
    ## AIC: 934.43
    ## 
    ## Number of Fisher Scoring iterations: 13

``` r
# PRE
PRE <- function(model){
  y <- model$y
  y.hat <- round(model$fitted.values)
  E1 <- sum(y != median(y))
  E2 <- sum(y != y.hat)
  PRE <- (E1 - E2) / E1
  return(PRE)
}

PRE_sim <- PRE(sim_log)

# model accuracy for traning and test sets
getProb <- function(model, data){
  data <- data %>% 
    add_predictions(model) %>% 
    mutate(prob = exp(pred) / (1 + exp(pred)),
           pred_bi = as.numeric(prob > .5))
  return(data)
}

test_acc <- getProb(sim_log, as.data.frame(sim_split$test))

test_acc_rate <- mean(test_acc$guilt == test_acc$pred_bi, na.rm = TRUE)

# AUC
library(pROC)
AUC_test <- auc(test_acc$guilt, test_acc$pred_bi)
```

The test error rate of the logistic model is 0.8160377, the PRE is 0.4058442, and AUC is 0.7330417, which are pretty good.

The logistic regression results show that `black` has a statistically significant relationship with the response with a p value smaller than 8.02e-12. Specifically, `black` has an estimated parameter of about -3.02, which can be tanslated to an odds-ratio of 0.0488012, meaning that the odds of believing OJ Simpson was guilty are just 0.0488 times as high for black people when compared with non-black. `income` is also significantly associated with the response, but to a lesser degree than `black`. Again, this model shows that race is the most important indicator to predict one's belief of OJ Simpson's guilt.

\`\`\`

[1] Feeling thermometers are a common metric in survey research used to gauge attitudes or feelings of warmth towards individuals and institutions. They range from 0-100, with 0 indicating extreme coldness and 100 indicating extreme warmth.

[2] The variable is an index which combines responses to four different questions: "In the past 30 days, how often did you feel: 1) so sad nothing could cheer you up, 2) hopeless, 3) that everything was an effort, and 4) worthless?" Valid responses are none of the time, a little of the time, some of the time, most of the time, and all of the time.
