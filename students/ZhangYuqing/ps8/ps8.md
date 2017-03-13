ps8
================
Yuqing Zhang
3/4/2017

-   [Part 1: Sexy Joe Biden (redux)](#part-1-sexy-joe-biden-redux)
    -   [1 Split the data](#split-the-data)
    -   [2 Fit a decision tree to the training data](#fit-a-decision-tree-to-the-training-data)
    -   [3 Fit another tree to the training data](#fit-another-tree-to-the-training-data)
    -   [4 Use the bagging approach](#use-the-bagging-approach)
-   [Part 2 Modeling voter turnout](#part-2-modeling-voter-turnout)
    -   [1 Use cross-validation techniques and standard measures of model fit to compare and evaluate at least five tree-based models of voter turnout](#use-cross-validation-techniques-and-standard-measures-of-model-fit-to-compare-and-evaluate-at-least-five-tree-based-models-of-voter-turnout)
    -   [2 Use cross-validation techniques and standard measures of model fit to compare and evaluate at least five SVM models of voter turnout](#use-cross-validation-techniques-and-standard-measures-of-model-fit-to-compare-and-evaluate-at-least-five-svm-models-of-voter-turnout)
-   [Part3 OJ Simpson](#part3-oj-simpson)
    -   [1 Develop a robust statistical learning model and use this model to explain the impact of an individual's race on their beliefs about OJ Simpson's guilt](#develop-a-robust-statistical-learning-model-and-use-this-model-to-explain-the-impact-of-an-individuals-race-on-their-beliefs-about-oj-simpsons-guilt)
    -   [2 Develop a robust statistical learning model to predict whether individuals believe OJ Simpson to be either probably guilty or probably not guilty and demonstrate the effectiveness of this model](#develop-a-robust-statistical-learning-model-to-predict-whether-individuals-believe-oj-simpson-to-be-either-probably-guilty-or-probably-not-guilty-and-demonstrate-the-effectiveness-of-this-model)

Part 1: Sexy Joe Biden (redux)
------------------------------

``` r
mse <- function(model,data) {
  x<- modelr:::residuals(model, data)
  mean(x^2, na.rm = TRUE)
}
```

### 1 Split the data

``` r
biden<-read.csv('biden.csv')
set.seed(1234)
biden_split <- resample_partition(biden, c(test = 0.3, train = 0.7))
```

### 2 Fit a decision tree to the training data

![](ps8_files/figure-markdown_github/fit%20decision%20tree-1.png) The test MSE is 406.417.

### 3 Fit another tree to the training data

![](ps8_files/figure-markdown_github/fit%20decision%20tree%20w/%20full%20nodes-1.png)

    ## [1] 481

The test MSE without pruning is 481.49

``` r
# generate 10-fold CV trees
biden_cv <- crossv_kfold(biden, k = 10) %>%
  mutate(tree = map(train, ~ tree(biden ~ ., data =biden_split$train,
     control = tree.control(nobs = nrow(biden_split$train),
                            mindev = 0))))

# calculate each possible prune result for each fold
biden_cv <- expand.grid(biden_cv$.id, 3:15) %>%
  as_tibble() %>%
  mutate(Var2 = as.numeric(Var2)) %>%
  rename(.id = Var1,
         k = Var2) %>%
  left_join(biden_cv) %>%
  mutate(prune = map2(tree, k, ~ prune.tree(.x, best = .y)),
         mse = map2_dbl(prune, test, mse))
```

    ## Joining, by = ".id"

``` r
biden_cv %>%
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

![](ps8_files/figure-markdown_github/tenfold-1.png)

``` r
mod <- prune.tree(biden_tree, best = 12)

# plot tree
tree_data <- dendro_data(mod)
ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro()
```

![](ps8_files/figure-markdown_github/prune-1.png)

### 4 Use the bagging approach

``` r
# generate sample index
samp <- data_frame(x = seq.int(1000))

# generate bootstrap sample and count proportion of observations in each draw
prop_drawn <- bootstrap(samp, n = nrow(samp)) %>%
  mutate(strap = map(strap, as_tibble)) %>%
  unnest(strap) %>%
  mutate(drawn = TRUE) %>%
  complete(.id, x, fill = list(drawn = FALSE)) %>%
  distinct %>%
  group_by(x) %>%
  mutate(n_drawn = cumsum(drawn),
         .id = as.numeric(.id),
         n_prop = n_drawn / .id)

ggplot(prop_drawn, aes(.id, n_prop, group = x)) +
  geom_line(alpha = .05) +
  labs(x = "b-th bootstrap sample ",
       y = "Proportion i-th observation in samples 1:b")
```

![](ps8_files/figure-markdown_github/bagged-1.png)

``` r
biden_rf_data <- biden %>%
    #select(-age, -educ) %>%
    mutate_each(funs(as.factor(.)),female,dem,rep) %>%
    na.omit

(biden_bag <- randomForest(biden ~ ., data = biden,
                             mtry = 5, ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden, mtry = 5, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##           Mean of squared residuals: 492
    ##                     % Var explained: 10.6

``` r
data_frame(var = rownames(importance(biden_bag)),
           MeanDecreaseGini = importance(biden_bag)[,1]) %>%
  mutate(var = fct_reorder(var, MeanDecreaseGini, fun = median)) %>%
  ggplot(aes(var, MeanDecreaseGini)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting biden score",
       subtitle = "Bagging",
       x = NULL,
       y = "Average decrease in the Gini Index")
```

![](ps8_files/figure-markdown_github/bagged-2.png)

For classification trees, larger values are better. So for the biden bagged model, being a democrats, and education are the most important predictors, whereas being a republicants and gender are relatively unimportant.

``` r
(biden_rf <- randomForest(biden ~ ., data = biden_rf_data,
                            ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden_rf_data, ntree = 500) 
    ##                Type of random forest: regression
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 1
    ## 
    ##           Mean of squared residuals: 405
    ##                     % Var explained: 26.3

``` r
seq.int(biden_rf$ntree) %>%
  map_df(~ getTree(biden_rf, k = ., labelVar = TRUE)[1,]) %>%
  count(`split var`) #%>%
```

    ## # A tibble: 5 × 2
    ##   `split var`     n
    ##         <chr> <int>
    ## 1         age    99
    ## 2         dem   100
    ## 3        educ    98
    ## 4      female    99
    ## 5         rep   104

``` r
  #knitr::kable(caption = "Variable used to generate the first split in each tree")#,
               #col.names = c("Variable used to split", "Number of training observations"))
```

``` r
data_frame(var = rownames(importance(biden_rf)),
           `Random forest` = importance(biden_rf)[,1]) %>%
  left_join(data_frame(var = rownames(importance(biden_rf)),
           Bagging = importance(biden_bag)[,1])) %>%
  mutate(var = fct_reorder(var, Bagging, fun = median)) %>%
  gather(model, gini, -var) %>%
  ggplot(aes(var, gini, color = model)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting biden score",
       x = NULL,
       y = "Average decrease in the Gini Index",
       color = "Method")
```

    ## Joining, by = "var"

![](ps8_files/figure-markdown_github/rf-1.png)

``` r
set.seed(1234)
biden_boost <- gbm(as.numeric(biden) - 1 ~ ., data = biden_split$train, n.trees = 10000, interaction.depth = 1)
```

    ## Distribution not specified, assuming gaussian ...

``` r
yhat.boost = predict(biden_boost, newdata = biden_split$test, n.trees = 10000)

mean((yhat.boost - biden[biden_split$test[2]$idx, ]$biden)^2)
```

    ## [1] 402

``` r
mses <- numeric(4)
shrinkages <- numeric(4)
for (s in 1:4){
  shrinkages[s] <- 10^(-s)
  biden_boost <- gbm(biden ~ ., data = biden_split$train, n.trees = 10000, interaction.depth = 1, shrinkage = shrinkages[s])
  yhat.boost = predict(biden_boost, newdata = biden_split$test, n.trees = 10000)
  mses[s] <- mean((yhat.boost - biden[biden_split$test[2]$idx, ]$biden)^2)
}
```

    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...

``` r
data_frame(mse = mses, shrinkage = shrinkages) %>% 
  ggplot(aes(shrinkage, mse)) +
  geom_point() +
  geom_line() +
  labs(title = "Predicting Biden Score",
       subtitle = "boosting",
       x = "Shrinkage",
       y = "Test MSE")
```

![](ps8_files/figure-markdown_github/boosting-1.png)

With boosting, the test MSE becomes 402, and it is lower than that of the other model, which indicates a potentially better model. The value of the shrinkage parameter goes down from 0.1, 0.01, 0.001 to 0.0001, and the test MSE goes down first from 420 to 408 to 400, and goes up again from 400 to 444. The best shrinkage level in this case is around 0.001, which produces a test MSE around 400.

Part 2 Modeling voter turnout
-----------------------------

### 1 Use cross-validation techniques and standard measures of model fit to compare and evaluate at least five tree-based models of voter turnout

``` r
(mental <- read_csv("mental_health.csv") %>% 
  mutate_each(funs(as.factor(.)),vote96,black,female,married) %>%
  na.omit)
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

    ## # A tibble: 1,165 × 8
    ##    vote96 mhealth_sum   age  educ  black female married inc10
    ##    <fctr>       <dbl> <dbl> <dbl> <fctr> <fctr>  <fctr> <dbl>
    ## 1       1           0    60    12      0      0       0  4.81
    ## 2       1           1    36    12      0      0       1  8.83
    ## 3       0           7    21    13      0      0       0  1.74
    ## 4       0           6    29    13      0      0       0 10.70
    ## 5       1           1    41    15      1      1       1  8.83
    ## 6       1           2    48    20      0      0       1  8.83
    ## 7       0           9    20    12      0      1       0  7.22
    ## 8       0          12    27    11      0      1       0  1.20
    ## 9       1           2    28    16      0      0       1  7.22
    ## 10      1           0    72    14      0      0       1  4.01
    ## # ... with 1,155 more rows

``` r
set.seed(1234)
mental_split <- resample_partition(mental, c(test = 0.3, train = 0.7))
```

![](ps8_files/figure-markdown_github/mental%20fit%20decision%20tree-1.png)

``` r
#mse
mental_testerr <- err.rate.rf(mental_tree, mental_split$test)
mental_testerr
```

    ## [1] 0.304

``` r
#ROC/AUC
fitted <- predict(mental_tree, as_tibble(mental_split$test), type = "class")

roc_td <- roc(as.numeric(as_tibble(mental_split$test)$vote96), as.numeric(fitted))
plot(roc_td)
```

![](ps8_files/figure-markdown_github/mental%20evaluate1-1.png)

``` r
auc(roc_td)
```

    ## Area under the curve: 0.56

``` r
#PRE
real <- as.numeric(na.omit(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mental_testerr
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.094

The decision tree with default setting and all predictor variables has a test error rate 30.4%. The AUC is 0.56 and the PRE is 9.4%, meaning that when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by only 9.4%.

![](ps8_files/figure-markdown_github/mental%20fit%20decision%20tree%20w/%20full%20nodes-1.png)

``` r
#mse
mental_nodes_testerr <- err.rate.rf(mental_tree_nodes, mental_split$test)
mental_nodes_testerr
```

    ## [1] 0.335

``` r
#ROC/AUC
fitted <- predict(mental_tree_nodes, as_tibble(mental_split$test), type = "class")

roc_td <- roc(as.numeric(as_tibble(mental_split$test)$vote96), as.numeric(fitted))
plot(roc_td)
```

![](ps8_files/figure-markdown_github/mental%20evaluate2-1.png)

``` r
auc(roc_td)
```

    ## Area under the curve: 0.629

``` r
#PRE
real <- as.numeric(na.omit(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mental_nodes_testerr
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0

The decision tree with full nodes has a test error rate 33.5%, and is a little bit higher than the decision tree with default setting, which indicates a potential overfitting problem. The AUC is 0.629 and the PRE is 0, meaning when compared to the NULL model, estimating all with the median data value, this model did not decrease the error rate.

``` r
set.seed(1234)
(mental_bag <- randomForest(vote96 ~ ., data = mental,
                             mtry = 5, ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = vote96 ~ ., data = mental, mtry = 5, ntree = 500) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 5
    ## 
    ##         OOB estimate of  error rate: 31.9%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 165 213       0.563
    ## 1 159 628       0.202

``` r
data_frame(var = rownames(importance(mental_bag)),
           MeanDecreaseGini = importance(mental_bag)[,1]) %>%
  mutate(var = fct_reorder(var, MeanDecreaseGini, fun = median)) %>%
  ggplot(aes(var, MeanDecreaseGini)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting Voter Turnout",
       subtitle = "Bagging",
       x = NULL,
       y = "Average decrease in the Gini Index")
```

![](ps8_files/figure-markdown_github/mental%20bagged-1.png)

``` r
#ROC/AUC
fitted <- predict(mental_bag, na.omit(as_tibble(mental_split$test)), type = "prob")[,2]

roc_b <- roc(na.omit(as_tibble(mental_split$test))$vote96, fitted)
plot(roc_b)
```

![](ps8_files/figure-markdown_github/mental%20evaluate3-1.png)

``` r
auc(roc_b)
```

    ## Area under the curve: 1

``` r
#PRE
real <- as.numeric(na.omit(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.314
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.0634

For the bagged model, mhealth\_sum, age, educ and inc10 are the most important predictors. The test error rate 31.9% (estimated by out-of-bag error estimate), which is higher than the default one and indicates a potential overfitting problem. Also, the AUC is 1 and the PRE is 0.0634, meaning when compared to the NULL model, estimating all with the median data value, this model decrease the error rate by 6.34%.

``` r
(mental_rf <- randomForest(vote96 ~ ., data = mental,
                            ntree = 500))
```

    ## 
    ## Call:
    ##  randomForest(formula = vote96 ~ ., data = mental, ntree = 500) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 30.6%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 131 247       0.653
    ## 1 109 678       0.139

``` r
seq.int(mental_rf$ntree) %>%
  map_df(~ getTree(mental_rf, k = ., labelVar = TRUE)[1,]) %>%
  count(`split var`) 
```

    ## # A tibble: 7 × 2
    ##   `split var`     n
    ##        <fctr> <int>
    ## 1         age   127
    ## 2       black    18
    ## 3        educ    94
    ## 4      female     3
    ## 5       inc10    92
    ## 6     married    41
    ## 7 mhealth_sum   125

``` r
data_frame(var = rownames(importance(mental_rf)),
           `Random forest` = importance(mental_rf)[,1]) %>%
  left_join(data_frame(var = rownames(importance(mental_rf)),
           Bagging = importance(mental_bag)[,1])) %>%
  mutate(var = fct_reorder(var, Bagging, fun = median)) %>%
  gather(model, gini, -var) %>%
  ggplot(aes(var, gini, color = model)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting Voter Turnout",
       x = NULL,
       y = "Average decrease in the Gini Index",
       color = "Method")
```

    ## Joining, by = "var"

![](ps8_files/figure-markdown_github/rf1-1.png)

``` r
#ROC
fitted <- predict(mental_rf, na.omit(as_tibble(mental_split$test)), type = "prob")[,2]

roc_rf <- roc(na.omit(as_tibble(mental_split$test))$vote96, fitted)
plot(roc_rf)
```

![](ps8_files/figure-markdown_github/mental%20evaluate%204-1.png)

``` r
auc(roc_rf)
```

    ## Area under the curve: 0.999

``` r
#PRE
real <- as.numeric(na.omit(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.294
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.123

Using random forest, with all predictor variables, we can also observe that the average decrease in the Gini index associated with each variable is generally smaller using the random forest method compared to bagging. The test error rate 29.4% (estimated by out-of-bag error estimate), which is lower than the default one. Also, the AUC us 0.998 and the PRE is 0.123, meaning when compared to the NULL model, estimating all with the median data value, this model even decreases the error rate by 12.3%.

``` r
set.seed(1234)

#Grow tree
mental_tree_two <- tree(vote96 ~ age + inc10, data = mental_split$train)

#Plot tree
tree_data_two <- dendro_data(mental_tree_two)

ggplot(segment(tree_data_two)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), alpha = 0.5) +
  geom_text(data = label(tree_data_two), aes(x = x, y = y, label = label), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data_two), aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Voter turnout tree",
       subtitle = "age + inc10")
```

![](ps8_files/figure-markdown_github/mental%20two%20predictors-1.png)

``` r
#Mse
mental_tree_two_testerr <- err.rate.rf(mental_tree_two, mental_split$test)
mental_tree_two_testerr
```

    ## [1] 0.315

``` r
#ROC
fitted <- predict(mental_tree_two, as_tibble(mental_split$test), type = "class")

roc_t <- roc(as.numeric(as_tibble(mental_split$test)$vote96), as.numeric(fitted))
plot(roc_t)
```

![](ps8_files/figure-markdown_github/mental%20two%20predictors-2.png)

``` r
auc(roc_t)
```

    ## Area under the curve: 0.572

``` r
#PRE
real <- as.numeric(na.omit(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mental_tree_two_testerr
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.0598

The decision tree with two predictors has a test error rate 31.5%. The AUC is 0.572 and the PRE is 5.98%, meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 5.98%.

A quick wrapup: The best model is random forest model, with the test error rate 29.4% and a PRE as 12.3%, which decreases the most error rate.

### 2 Use cross-validation techniques and standard measures of model fit to compare and evaluate at least five SVM models of voter turnout

I chose linear kernel, 2-degree polynomial, 3-degree polynomial, radial kernel, and sigmoid kernel as my five SVM models. For each of them I used 10-fold cross-validation to determine the optimal cost parameter.

``` r
#linear kernel
mh_lin_tune <- tune(svm, vote96 ~ ., data = as_tibble(mental_split$train),
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
    ##     1
    ## 
    ## - best performance: 0.286 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.320     0.0440
    ## 2 1e-02 0.319     0.0454
    ## 3 1e-01 0.293     0.0415
    ## 4 1e+00 0.286     0.0254
    ## 5 5e+00 0.289     0.0289
    ## 6 1e+01 0.289     0.0289
    ## 7 1e+02 0.289     0.0289

``` r
mh_lin <- mh_lin_tune$best.model
summary(mh_lin)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mental_split$train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  1 
    ##       gamma:  0.125 
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
fitted <- predict(mh_lin, as_tibble(mental_split$test), decision.values = TRUE) %>%
  attributes
roc_line <- roc(as_tibble(mental_split$test)$vote96, fitted$decision.values)
#AUC
auc(roc_line)
```

    ## Area under the curve: 0.746

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.286
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.147

Using linear kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 1 and with the lowest 10-fold CV error rate 0.286. Also, the AUC us 0.746 and the PRE is 14.7% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 14.7%.

``` r
#polynomial kernel
set.seed(1234)

mh_poly_tune <- tune(svm, vote96 ~ ., data = as_tibble(mental_split$train),
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
    ## - best performance: 0.302 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.320     0.0440
    ## 2 1e-02 0.320     0.0440
    ## 3 1e-01 0.318     0.0492
    ## 4 1e+00 0.308     0.0364
    ## 5 5e+00 0.302     0.0294
    ## 6 1e+01 0.315     0.0340
    ## 7 1e+02 0.325     0.0407

``` r
#Best
mh_poly <- mh_poly_tune$best.model
summary(mh_poly)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mental_split$train), 
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
#ROC
fitted <- predict(mh_poly, as_tibble(mental_split$test), decision.values = TRUE) %>%
  attributes

roc_poly <- roc(as_tibble(mental_split$test)$vote96, fitted$decision.values)
auc(roc_poly)
```

    ## Area under the curve: 0.741

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.302
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.0992

Using polynomial kernel, with all predictor variables, default degree level (3), and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 5 and has a 10-fold CV error rate 0.302. Also, the AUC us 0.741 and the PRE is 9.92% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 9.92%.

``` r
#radial kernel
set.seed(1234)
mh_rad_tune <- tune(svm, vote96 ~ ., data = as_tibble(mental_split$train),
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
    ## - best performance: 0.292 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.320     0.0440
    ## 2 1e-02 0.320     0.0440
    ## 3 1e-01 0.306     0.0445
    ## 4 1e+00 0.292     0.0305
    ## 5 5e+00 0.295     0.0400
    ## 6 1e+01 0.293     0.0339
    ## 7 1e+02 0.320     0.0313

``` r
mh_rad <- mh_rad_tune$best.model
summary(mh_rad)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mental_split$train), 
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
fitted <- predict(mh_rad, as_tibble(mental_split$test), decision.values = TRUE) %>%
  attributes

#ROC
roc_rad <- roc(as_tibble(mental_split$test)$vote96, fitted$decision.values)
auc(roc_rad)
```

    ## Area under the curve: 0.735

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.292
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.129

Using radial kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 1with the lowerst 10-fold CV error rate of 29.2%. Also, the AUC us 0.735 and the PRE is 12.9% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 12.9%.

``` r
set.seed(1234)

mh_sig_tune <- tune(svm, vote96 ~ ., data = as_tibble(mental_split$train),
                    kernel = "sigmoid",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(mh_sig_tune)
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
    ## - best performance: 0.319 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.320     0.0440
    ## 2 1e-02 0.320     0.0440
    ## 3 1e-01 0.319     0.0454
    ## 4 1e+00 0.348     0.0294
    ## 5 5e+00 0.370     0.0323
    ## 6 1e+01 0.371     0.0392
    ## 7 1e+02 0.376     0.0353

``` r
mh_sig <- mh_sig_tune$best.model
summary(mh_sig)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mental_split$train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "sigmoid")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  sigmoid 
    ##        cost:  0.1 
    ##       gamma:  0.125 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  525
    ## 
    ##  ( 264 261 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_sig, as_tibble(mental_split$test), decision.values = TRUE) %>%
  attributes

#ROC
roc_sig <- roc(as_tibble(mental_split$test)$vote96, fitted$decision.values)
auc(roc_sig)
```

    ## Area under the curve: 0.73

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.319
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.0485

Using sigmoid kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 0.1 with the lowerst 10-fold CV error rate of 0.319. Also, the AUC us 0.73 and the PRE is 4.85% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 4.85%.

``` r
#2-degree polynomial kernel
mh_poly2_tune <- tune(svm, vote96 ~ ., data = as_tibble(mental_split$train),
                    kernel = "polynomial",
                    degree = 2,
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(mh_poly2_tune)
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
    ## - best performance: 0.295 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.320     0.0573
    ## 2 1e-02 0.320     0.0573
    ## 3 1e-01 0.316     0.0572
    ## 4 1e+00 0.295     0.0487
    ## 5 5e+00 0.300     0.0499
    ## 6 1e+01 0.304     0.0476
    ## 7 1e+02 0.298     0.0498

``` r
mh_poly2 <- mh_poly2_tune$best.model
summary(mh_poly2)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mental_split$train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "polynomial", degree = 2)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  polynomial 
    ##        cost:  1 
    ##      degree:  2 
    ##       gamma:  0.125 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  507
    ## 
    ##  ( 259 248 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
fitted <- predict(mh_poly2, as_tibble(mental_split$test), decision.values = TRUE) %>%
  attributes

roc_poly2 <- roc(as_tibble(mental_split$test)$vote96, fitted$decision.values)
auc(roc_poly2)
```

    ## Area under the curve: 0.74

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(mental_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.293
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.126

Using polynomial kernel, with all predictor variables, I used 2-degree levels, and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 5 with the lowerst 10-fold CV error rate of 29.3%. Also, the AUC us 0.749 and the PRE is 12.6% (the model MSE is estimated by the 10-fold error rate).

A quick summary:

``` r
plot(roc_line, print.auc = TRUE, col = "blue")
plot(roc_poly2, print.auc = TRUE, col = "purple", print.auc.y = .4, add = TRUE)
plot(roc_poly, print.auc = TRUE, col = "red", print.auc.y = .3, add = TRUE)
plot(roc_rad, print.auc = TRUE, col = "orange", print.auc.y = .2, add = TRUE)
plot(roc_sig, print.auc = TRUE, col = "green", print.auc.y = .1, add = TRUE)
```

![](ps8_files/figure-markdown_github/summary%20of%20svms-1.png) The above graph shows their ROC curves.

Among these five models, 3-degree polynomial kernel has the best performance since it has low error rate and largest PRE, meaning that this model has certain accuracy and fit the test data well.

Part3 OJ Simpson
----------------

### 1 Develop a robust statistical learning model and use this model to explain the impact of an individual's race on their beliefs about OJ Simpson's guilt

I choose logistic, decision tree, random forest models and polynomial kernel SVM generating from the training data to fit the test data. I also split the data into 30% testing and 70% training sets for cross validating their fittness.

``` r
oj <-read_csv('simpson.csv') %>%
  na.omit() %>%
  mutate_each(funs(as.factor(.)), guilt, dem, rep, ind,
              female, black, hispanic, educ, income) 
```

    ## Parsed with column specification:
    ## cols(
    ##   guilt = col_double(),
    ##   dem = col_double(),
    ##   rep = col_double(),
    ##   ind = col_double(),
    ##   age = col_double(),
    ##   educ = col_character(),
    ##   female = col_double(),
    ##   black = col_double(),
    ##   hispanic = col_double(),
    ##   income = col_character()
    ## )

``` r
set.seed(1234)
getProb <- function(model, data){
  data <- data %>% 
    add_predictions(model) %>% 
    mutate(prob = exp(pred) / (1 + exp(pred)),
           pred_bi = as.numeric(prob > .5))
  return(data)
}
oj_split <- resample_partition(oj, c(test = 0.3, train = 0.7))
```

``` r
oj_logistic <- glm(guilt ~ black + hispanic, data = oj_split$train, family = binomial)
summary(oj_logistic)
```

    ## 
    ## Call:
    ## glm(formula = guilt ~ black + hispanic, family = binomial, data = oj_split$train)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -1.835  -0.607   0.641   0.641   2.018  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)   1.4789     0.0934   15.84   <2e-16 ***
    ## black1       -3.0789     0.2165  -14.22   <2e-16 ***
    ## hispanic1    -0.2966     0.3167   -0.94     0.35    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1229.06  on 991  degrees of freedom
    ## Residual deviance:  947.18  on 989  degrees of freedom
    ## AIC: 953.2
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
fitted1 <- predict(oj_logistic, as_tibble(oj_split$test), type = "response")

#error
oj_logit_err <- mean(as_tibble(oj_split$test)$guilt != round(fitted1))
oj_logit_err
```

    ## [1] 0.184

``` r
#ROC
oj_roc_logit <- roc(as_tibble(oj_split$test)$guilt, fitted1)
auc(oj_roc_logit)
```

    ## Area under the curve: 0.744

As for the logistic model, the test error rate is 18.4% and AUC is 0.744 AUC.

According to the p-values of the independent variables, only black in the model has statistically significant relationships with the guilt, with (p-value &lt; 2e-16) at a 99.9% confidence level.

``` r
#decision tree
set.seed(1234)

#Grow tree
oj_tree_default <- tree(guilt ~ black + hispanic, data = oj_split$train)

#Plot tree
tree_data <- dendro_data(oj_tree_default)

ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), alpha = 0.5) +
  geom_text(data = label(tree_data), aes(x = x, y = y, label = label), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Simpson guilt opinion tree",
       subtitle = "black + hispanic")
```

![](ps8_files/figure-markdown_github/oj%20decision%20tree-1.png)

``` r
#ROC
fitted <- predict(oj_tree_default, as_tibble(oj_split$test), type = "class")

roc_t <- roc(as.numeric(as_tibble(oj_split$test)$guilt), as.numeric(fitted))
auc(roc_t)
```

    ## Area under the curve: 0.733

``` r
#Accuracy
pred_bi <- predict(oj_tree_default, newdata = oj_split$test, type = "class")
df_logistic_test <- getProb(oj_logistic, as.data.frame(oj_split$test))
accuracy <- mean(df_logistic_test$guilt == pred_bi, na.rm = TRUE)
accuracy
```

    ## [1] 0.816

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(oj_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 1 - accuracy
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.418

As for the decision tree model with default setting, the test error rate is 0.816, PRE is 0.418, and a 0.733 AUC. The test error rate is very high, indicating that it might not be a good model.

``` r
set.seed(1234)

simpson_rf <- randomForest(guilt ~ black + hispanic, data = na.omit(as_tibble(oj_split$train)), ntree = 500)
simpson_rf
```

    ## 
    ## Call:
    ##  randomForest(formula = guilt ~ black + hispanic, data = na.omit(as_tibble(oj_split$train)),      ntree = 500) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 1
    ## 
    ##         OOB estimate of  error rate: 18.4%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 156 152      0.4935
    ## 1  31 653      0.0453

``` r
data_frame(var = rownames(importance(simpson_rf)),
           MeanDecreaseRSS = importance(simpson_rf)[,1]) %>%
  mutate(var = fct_reorder(var, MeanDecreaseRSS, fun = median)) %>%
  ggplot(aes(var, MeanDecreaseRSS)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting opinion on Simpson guilty",
       subtitle = "Random forest",
       x = NULL,
       y = "Average decrease in the Gini Index")
```

![](ps8_files/figure-markdown_github/oj%20random%20forest-1.png)

``` r
#ROC
fitted <- predict(simpson_rf, na.omit(as_tibble(oj_split$test)), type = "prob")[,2]

roc_rf <- roc(na.omit(as_tibble(oj_split$test))$guilt, fitted)
auc(roc_rf)
```

    ## Area under the curve: 0.732

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(oj_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.1843
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.417

As for the random forest model, it gives us a 18.4% test error rate and a 41.7% PRE, both are worse than the previous two models. However, the random forest model has a 0.745 AUC at a similar level as the previous two models do. From the above graph we can also see that the black has a way higher average decrease in the Gini index than hispanic, which indicates black's importance and confirms the results from the previous two models.

I'll choose the logistic model as my final model and redo the logistic model with a 100-time 10-fold cross validation to examine its robustness.

``` r
fold_model_mse <- function(df, k){
  cv10_data <- crossv_kfold(df, k = k)
  cv10_models <- map(cv10_data$train, ~ glm(guilt ~ black + hispanic, family = binomial, data = .))
  cv10_prob <- map2(cv10_models, cv10_data$train, ~getProb(.x, as.data.frame(.y)))
  cv10_mse <- map(cv10_prob, ~ mean(.$guilt != .$pred_bi, na.rm = TRUE))
  return(data_frame(cv10_mse))
}

set.seed(1234)
mses <- rerun(100, fold_model_mse(oj, 10)) %>%
  bind_rows(.id = "id")

ggplot(data = mses, aes(x = "MSE (100 times 10-fold)", y = as.numeric(cv10_mse))) +
  geom_boxplot() +
  labs(title = "Boxplot of MSEs - logistic model",
       x = element_blank(),
       y = "MSE value")
```

![](ps8_files/figure-markdown_github/oj%20validation-1.png)

``` r
mse_100cv10 <- mean(as.numeric(mses$cv10_mse))
mseSd_100cv10 <- sd(as.numeric(mses$cv10_mse))
mse_100cv10 = mse_100cv10*100
mseSd_100cv10
```

    ## [1] 0.00345

The model gets a 18.432% average error rate, which is still pretty good, the std of the err or rate is also very low, 0.003.

### 2 Develop a robust statistical learning model to predict whether individuals believe OJ Simpson to be either probably guilty or probably not guilty and demonstrate the effectiveness of this model

``` r
#decision tree
oj_tree1 <- tree(guilt ~ dem + rep + age + educ + female + black + hispanic + income, 
                data = oj_split$train,
                control = tree.control(nrow(oj_split$train),
                                       mindev = 0))
oj_tree_results <- data_frame(terms = 2:50,
           model = map(terms, ~ prune.tree(oj_tree1, k = NULL, best = .)), error = map_dbl(model, ~ err.rate.rf(., data = oj_split$test)))
ggplot(oj_tree_results, aes(terms, error)) +
  geom_line() +
  labs(title = "Comparing Tree Complexity",
       subtitle = "Using validation set",
       x = "Terminal Nodes",
       y = "Test Error Rate")
```

![](ps8_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
auc_best <- function(model) {
  fitted <- predict(model, as_tibble(oj_split$test), type = 'class')
  roc1 <- roc(as.numeric(as_tibble(oj_split$test)$guilt), as.numeric(fitted))
  auc(roc1)
}

oj_tree_results2 <- data_frame(terms = 2:50,
           model = map(terms, ~ prune.tree(oj_tree1, k = NULL, best = .)),
           AUC = map_dbl(model, ~ auc_best(.)))

ggplot(oj_tree_results2, aes(terms, AUC)) +
  geom_line() +
  labs(title = "Comparing Tree Complexity",
       subtitle = "Using validation set",
       x = "Terminal Nodes",
       y = "AUC")
```

![](ps8_files/figure-markdown_github/unnamed-chunk-1-2.png)

``` r
#decision tree with prune
oj_tree <- prune.tree(oj_tree1, best = 10)
fitted2 <- predict(oj_tree, as_tibble(oj_split$test), type = "class")
oj_tree_err <- min(oj_tree_results$error)
oj_roc_tree <- roc(as.numeric(as_tibble(oj_split$test)$guilt), as.numeric(fitted2))

#bagging
oj_bag <- randomForest(guilt ~ dem + rep + age + educ + female + black + hispanic + income, data = as_tibble(oj_split$train), mtry = 2)
fitted3 <- predict(oj_bag, as_tibble(oj_split$test), type = "prob")[,2]
oj_bag_err <- 0.194
oj_roc_bag <- roc(as_tibble(oj_split$test)$guilt, fitted3)

#linear kernel
simpson_lin_tune <- tune(svm, guilt ~ dem + rep + age + educ + female + black + hispanic + income, data = na.omit(as_tibble(oj_split$train)),
                    kernel = "linear",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
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
    ## - best performance: 0.184 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.310     0.0495
    ## 2 1e-02 0.184     0.0468
    ## 3 1e-01 0.184     0.0468
    ## 4 1e+00 0.184     0.0468
    ## 5 5e+00 0.184     0.0468
    ## 6 1e+01 0.184     0.0468
    ## 7 1e+02 0.184     0.0468

``` r
simpson_lin <- simpson_lin_tune$best.model
simpson_lin_err <- simpson_lin_tune$best.performance
fitted4 <- predict(simpson_lin, as_tibble(oj_split$test), decision.values = TRUE) %>%
  attributes
oj_roc_line <- roc(as.numeric(as_tibble(oj_split$test)$guilt), as.numeric(fitted4$decision.values))

#polynomial kernel
oj_poly_tune <- tune(svm, guilt ~ dem + rep + age + educ + female + black + hispanic + income, data = as_tibble(oj_split$train), kernel = "polynomial", range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
oj_poly <- oj_poly_tune$best.model
oj_poly_err <- oj_poly_tune$best.performance
fitted5 <- predict(oj_poly, as_tibble(oj_split$test), decision.values = TRUE) %>%
  attributes
oj_roc_poly <- roc(as.numeric(as_tibble(oj_split$test)$guilt), as.numeric(fitted5$decision.values))

plot(oj_roc_line, print.auc = TRUE, col = "blue", print.auc.x = .2)
plot(oj_roc_poly, print.auc = TRUE, col = "red", print.auc.x = .2, print.auc.y = .4, add = TRUE)
plot(oj_roc_bag, print.auc = TRUE, col = "green", print.auc.x = .2, print.auc.y = .2, add = TRUE)
plot(oj_roc_tree, print.auc = TRUE, col = "black", print.auc.x = .2, print.auc.y = .3, add = TRUE)
```

![](ps8_files/figure-markdown_github/unnamed-chunk-1-3.png)

Above are the ROC curves of the models that I used, which are SVM with linear, polynomial, and bagging, and decision tree. As one can tell, decision treehas lowest error rate, and larger AUC, indicating that it is by far the best model.
