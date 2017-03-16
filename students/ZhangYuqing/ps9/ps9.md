ps9
================
Yuqing Zhang
3/15/2017

-   [Attitudes towards feminists](#attitudes-towards-feminists)
    -   [1. Split the data into a training and test set (70/30%)](#split-the-data-into-a-training-and-test-set-7030)
    -   [Calculate the test MSE for KNN models](#calculate-the-test-mse-for-knn-models)
    -   [3 Calculate the test MSE for weighted KNN models](#calculate-the-test-mse-for-weighted-knn-models)
    -   [4 Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before](#compare-the-test-mse-for-the-best-knnwknn-models-to-the-test-mse-for-the-equivalent-linear-regression-decision-tree-boosting-and-random-forest-methods-using-the-same-combination-of-variables-as-before)
-   [2 Voter turnout and depression](#voter-turnout-and-depression)
    -   [1 Split the data into a training and test set (70/30)](#split-the-data-into-a-training-and-test-set-7030-1)
    -   [2 Calculate the test error rate for KNN models with *K* = 1, 2, …, 10](#calculate-the-test-error-rate-for-knn-models-with-k-12dots10)
    -   [3 Calculate the test error rate for weighted KNN models with *K* = 1, 2, …, 10](#calculate-the-test-error-rate-for-weighted-knn-models-with-k-12dots10)
    -   [4 Compare the test error rate for the best KNN/wKNN model(s) to the test error rate for the equivalent logistic regression, decision tree, boosting, random forest, and SVM methods](#compare-the-test-error-rate-for-the-best-knnwknn-models-to-the-test-error-rate-for-the-equivalent-logistic-regression-decision-tree-boosting-random-forest-and-svm-methods)
-   [3 Colleges](#colleges)
    -   [Perform PCA analysis on the college dataset and plot the first two principal components. Describe the results.](#perform-pca-analysis-on-the-college-dataset-and-plot-the-first-two-principal-components.-describe-the-results.)
-   [Clustering States](#clustering-states)
    -   [1 Perform PCA on the dataset and plot the observations on the first and second principal components](#perform-pca-on-the-dataset-and-plot-the-observations-on-the-first-and-second-principal-components)
    -   [2 Perform *K*-means clustering with *K* = 2.](#perform-k-means-clustering-with-k2.)
    -   [3 Perform *K*-means clustering with *K* = 4](#perform-k-means-clustering-with-k4)
    -   [4 Perform *K*-means clustering with *K* = 3.](#perform-k-means-clustering-with-k3.)
    -   [5 Perform *K*-means clustering with *K* = 3 on the first two principal components score vectors](#perform-k-means-clustering-with-k3-on-the-first-two-principal-components-score-vectors)
    -   [6 Hierarchical clustering](#hierarchical-clustering)
    -   [7 Cut the dendrogram at a height that results in three distinct clusters](#cut-the-dendrogram-at-a-height-that-results-in-three-distinct-clusters)
    -   [8 Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation 1](#hierarchically-cluster-the-states-using-complete-linkage-and-euclidean-distance-after-scaling-the-variables-to-have-standard-deviation-1)

Attitudes towards feminists
---------------------------

### 1. Split the data into a training and test set (70/30%)

``` r
feminist <- read_csv("feminist.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   feminist = col_integer(),
    ##   female = col_integer(),
    ##   age = col_integer(),
    ##   educ = col_integer(),
    ##   income = col_integer(),
    ##   dem = col_integer(),
    ##   rep = col_integer()
    ## )

``` r
set.seed(1234)
feminist_split <- resample_partition(feminist, c(test = 0.3, train = 0.7))
feminist_train <- as_tibble(feminist_split$train)
feminist_test <- as_tibble(feminist_split$test)
```

### Calculate the test MSE for KNN models

``` r
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}

mse_knn <- data_frame(k = seq(5, 100, by = 5), 
                      knn = map(k, ~ knn.reg(select(feminist_train, -age, -income, -female), y = feminist_train$feminist, test = select(feminist_test, -age, -income, -female), k = .)), 
                      mse = map_dbl(knn, ~ mean((feminist_test$feminist - .$pred)^2))) 

ggplot(mse_knn, aes(k, mse)) +
  geom_line() +
  geom_point() +
  labs(title = "KNN on Feminist Score data",
       x = "K",
       y = "Test mean squared error") +
  expand_limits(y = 0)
```

![](ps9_files/figure-markdown_github/KNN-1.png)

From the model we can see that as K increases, test MSE increases as well. The lowest MSE is when k=5 and this is the best model among all KNN models.

### 3 Calculate the test MSE for weighted KNN models

``` r
sim_pred_wknn <- data_frame(k=seq(5,100,by=5),
                            wknn = map(k, ~ kknn(feminist ~ age + female + income, train                                     = feminist_train, test = feminist_test, k = .)), 
                            mse_wknn = map_dbl(wknn, ~ mean((feminist_test$feminist -                                                .$fitted.values)^2))) %>%
                 left_join(mse_knn, by = "k") %>%
                 mutate(mse_knn = mse)

sim_pred_wknn %>%
  select(k, mse_knn, mse_wknn) %>%
  gather(method, mse, -k) %>%
  mutate(method = str_replace(method, "mse_", "")) %>%
  mutate(method = factor(method, levels = c("knn", "wknn"),
                         labels = c("KNN", "Weighted KNN"))) %>%
  ggplot(aes(k, mse, color = method)) +
  geom_line() +
  geom_point() +
  labs(title = "Test MSE for KNN",
       subtitle = "Traditional and weighted KNN",
       x = "K",
       y = "Test mean squared error",
       method = NULL) +
  expand_limits(y = 0) +
  theme(legend.position = "bottom")
```

![](ps9_files/figure-markdown_github/weighted%20KNN-1.png) Using the same combination of variables as before, it is obvious from the above graph that the tradional KNN has an overall lower test MSE than the weighted KNN method. When using weighted KNN, test MSE decreases as K increases. When K=100 there is the lowest MSE for this model.

### 4 Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before

``` r
set.seed(1234)
#Linear regression
lm<-lm(feminist ~ ., data = feminist_train)
mse_lm <- mse(lm,feminist_test)

#decision tree
tree <- tree(feminist ~ age + female +income, data = feminist_train)
tree_data <- dendro_data(tree)
mse_tree <- mse(tree, feminist_test)

# Random Forests:
rf<- randomForest(feminist ~ age + female +income, data = feminist_train, ntree = 500)
mse_rf <- mse(rf, feminist_test)

#Boosting
mse_boost <-function(model, test, tree_number) {
  yhat.boost <- predict (model, newdata = test, n.trees=tree_number)
  mse <- mean((yhat.boost - (as_tibble(test))$feminist)^2)
  return (mse)
}
boosting <- gbm(feminist ~ age + female +income, data = feminist_train, distribution = 'gaussian',n.trees = 2000, interaction.depth = 2)
mse_boosting <- mse_boost(boosting,feminist_test,2000)

comparision <- data_frame("model" = c("KNN (k=5)", "Weighted KNN (k=100)", "Decision Tree", "Random Forest", "Boosting","OLS"),
                  "test MSE" = c(min(mse_knn$mse), min(sim_pred_wknn$mse_wknn), mse_tree, mse_rf,mse_boosting, mse_lm))
comparision
```

    ## # A tibble: 6 × 2
    ##                  model `test MSE`
    ##                  <chr>      <dbl>
    ## 1            KNN (k=5)      0.181
    ## 2 Weighted KNN (k=100)    454.524
    ## 3        Decision Tree    451.784
    ## 4        Random Forest    446.340
    ## 5             Boosting    445.347
    ## 6                  OLS    435.111

According to the table above we can conclude that the traditional KNN model gives the lowest test MSE and thus yields the best result. All the other four models have pretty similar test MSEs. KNN works better than OLS is because the non-parametric method relaxes the linear assumption and thus can better reflect the real structural of the data. In my opinion, it is better than other non-parametric models because the traditional KNN can largely avoid some overfitting problems, which influece the test MSE.

2 Voter turnout and depression
------------------------------

### 1 Split the data into a training and test set (70/30)

``` r
mh <- read_csv('mental_health.csv')
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
mh_rm_na <- mh %>%
  select(vote96, age, inc10, educ, mhealth_sum)%>%
  na.omit()
set.seed(1234)
mh_split <- resample_partition(mh_rm_na, c(test = 0.3, train = 0.7))
mh_train <- as_tibble(mh_split$train)
mh_test <- as_tibble(mh_split$test)
```

### 2 Calculate the test error rate for KNN models with *K* = 1, 2, …, 10

For this problem I will use: income, age, mental\_health\_sum and education as my variables.

``` r
mse_knn_2 <- data_frame(k = 1:10,
                      knn_train = map(k, ~ class::knn(select(mh_train, -vote96),
                                                test = select(mh_train, -vote96),
                                                cl = mh_train$vote96, k = .)),
                      knn_test = map(k, ~ class::knn(select(mh_train, -vote96),
                                                test = select(mh_test, -vote96),
                                                cl = mh_train$vote96, k = .)),
                      mse_train = map_dbl(knn_train, ~ mean(mh_test$vote96 != .)),
                      mse_test = map_dbl(knn_test, ~ mean(mh_test$vote96 != .)))

ggplot(mse_knn_2, aes(k, mse_test)) +
  geom_line() +
  labs(x = "K",
       y = "Test error rate") +
  expand_limits(y = 0)
```

![](ps9_files/figure-markdown_github/KNN%20test%20error%20rate-1.png)

``` r
knn_mse_2<-min(mse_knn_2$mse_test)
```

Overall there is a decreasing trend in the graph, so the best model is when K=10 and it yields the lowest MSE which is, 0.287.

### 3 Calculate the test error rate for weighted KNN models with *K* = 1, 2, …, 10

``` r
mse_wknn_2 <- data_frame(k = 1:10,
                      wknn = map(k, ~ kknn(vote96 ~., train = mh_train, test = mh_test, k =.)),
                      mse_test_wknn = map_dbl(wknn, ~ mean(mh_test$vote96 != as.numeric(.$fitted.values > 0.5))))

mse_wknn_mh <- min(mse_wknn_2$mse_test_wknn)

err<-mse_wknn_2 %>%
  left_join(mse_knn_2, by = "k") %>%
  select(k, mse_test_wknn, mse_test) %>%
  gather(method,mse, -k) %>%
  mutate(method = factor(method, levels =c("mse_test_wknn","mse_test"), labels = c("Weighted KNN","KNN")))

err %>%
  ggplot(aes(k, mse, color = method)) +
  geom_line() +
  geom_point() +
  labs(title = "Test MSE for KNN, on Vote Turnout",
       subtitle = "Traditional and weighted KNN",
       x = "K",
       y = "Test mean squared error",
       method = NULL) +
  expand_limits(y = 0) +
  theme(legend.position = "bottom")
```

![](ps9_files/figure-markdown_github/weighted%20KNN%20test%20error%20rate-1.png) The change of error rate for the weighted KNN is not that much compared to traditional KNN model.The best weighted KNN model is when k = 5,the test error rate is 0.338.

### 4 Compare the test error rate for the best KNN/wKNN model(s) to the test error rate for the equivalent logistic regression, decision tree, boosting, random forest, and SVM methods

``` r
set.seed(1234)
err.rate.tree <- function(model, data) {
  data <- as_tibble(data)
  response <- as.character(model$terms[[2]])

  pred <- predict(model, newdata = data, type = "class") 
  actual <- data[[response]]

  return(mean(pred != actual, na.rm = TRUE))
}
#Logistic regression
logis_mh <- glm(vote96 ~ ., data=mh_train, family=binomial)
logistic_mh <- mh_test %>%
  add_predictions(logis_mh) %>%
  mutate(prob = exp(pred) / (1 + exp(pred))) %>%
  mutate(pred_bi = as.numeric(prob > .5))

err_logistic_mh <- mean(mh_test$vote96 != logistic_mh$pred_bi)

#Decision Tree
mh_fac<- mh_rm_na %>%
  mutate (vote96 = factor(vote96, levels = 0:1, label =c("no_vote", "vote")))

mh_split_tree <- resample_partition(mh_fac, c(test = 0.3, train = 0.7))
mh_train_tree <- as_tibble(mh_split_tree$train)
mh_test_tree <- as_tibble(mh_split$test)
tree_mh <- tree(vote96 ~ age + inc10 + mhealth_sum + educ, data = mh_train_tree)
tree_data <- dendro_data(tree_mh)

error_tree <- err.rate.tree(tree_mh, mh_test_tree)

#Random Forest
rf<- randomForest(vote96 ~ age + inc10 + mhealth_sum + educ, data = mh_train, ntree = 500)
error_rf <- err.rate.tree(rf, mh_test)

#Boosting
boost_mh <- gbm(as.character(vote96) ~ age + inc10 + mhealth_sum + educ, data=mh_train, n.trees=500)
```

    ## Distribution not specified, assuming bernoulli ...

``` r
yhat.boost <- predict(boost_mh, newdata=mh_test, n.trees=500)
yhat.boost_bi <- as.numeric(yhat.boost > .5)
err_boost_mh <- mean(yhat.boost_bi != mh_test$vote96)

#SVM
svmlin_mh <- tune(svm, vote96 ~ age + inc10 + mhealth_sum + educ, data = as_tibble(mh_split$train),kernel = "linear",range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
mh_lin_best <- svmlin_mh$best.model
summary(svmlin_mh)
```

    ## 
    ## Parameter tuning of 'svm':
    ## 
    ## - sampling method: 10-fold cross validation 
    ## 
    ## - best parameters:
    ##  cost
    ##    10
    ## 
    ## - best performance: 0.265 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.286     0.0376
    ## 2 1e-02 0.269     0.0360
    ## 3 1e-01 0.265     0.0358
    ## 4 1e+00 0.265     0.0358
    ## 5 5e+00 0.265     0.0358
    ## 6 1e+01 0.265     0.0358
    ## 7 1e+02 0.265     0.0358

``` r
comparision_mh <- data_frame("model" = c("KNN (k=10)", "Weighted KNN (k=5)", "Decision Tree", "Random Forest", "Boosting","GLM", "SVM"),
                  "test_MSE" = c(knn_mse_2, mse_wknn_mh,error_tree,error_rf, err_boost_mh, err_logistic_mh , 0.265 ))

comparision_mh%>%
  ggplot(aes(model, test_MSE))+
  geom_bar(stat = "identity", width=.5)+
  theme_bw()+
  labs(title = "Test Error Rate for various models (Vote Turnout)",
       x = "model",
       y = "Test Error Rate")+
  coord_flip()
```

![](ps9_files/figure-markdown_github/comparison%20mh-1.png)

``` r
comparision_mh
```

    ## # A tibble: 7 × 2
    ##                model test_MSE
    ##                <chr>    <dbl>
    ## 1         KNN (k=10)    0.287
    ## 2 Weighted KNN (k=5)    0.338
    ## 3      Decision Tree    1.000
    ## 4      Random Forest    1.000
    ## 5           Boosting    0.289
    ## 6                GLM    0.275
    ## 7                SVM    0.265

According to the above graph, we can say that SVM yields the best model, it has the lowest test error rate, which is 0.265, as we got from the summary above.I think it is better is because by adjusting the cost value, this method may have much more flexibility than the other methods.

3 Colleges
----------

### Perform PCA analysis on the college dataset and plot the first two principal components. Describe the results.

``` r
college <-read_csv('college.csv') %>%
  na.omit() %>%
  mutate(Private = as.numeric(factor(Private))-1) %>%
  {.} -> coldata
```

    ## Parsed with column specification:
    ## cols(
    ##   Private = col_character(),
    ##   Apps = col_double(),
    ##   Accept = col_double(),
    ##   Enroll = col_double(),
    ##   Top10perc = col_double(),
    ##   Top25perc = col_double(),
    ##   F.Undergrad = col_double(),
    ##   P.Undergrad = col_double(),
    ##   Outstate = col_double(),
    ##   Room.Board = col_double(),
    ##   Books = col_double(),
    ##   Personal = col_double(),
    ##   PhD = col_double(),
    ##   Terminal = col_double(),
    ##   S.F.Ratio = col_double(),
    ##   perc.alumni = col_double(),
    ##   Expend = col_double(),
    ##   Grad.Rate = col_double()
    ## )

``` r
pr.col <- prcomp(coldata, scale = TRUE)
biplot(pr.col, scale=0, cex=.6)
```

![](ps9_files/figure-markdown_github/PCA-1.png) The above graph is hard to interpret, so we just plot the first two principal components.

``` r
pr.col$rotation[, 1]
```

    ##     Private        Apps      Accept      Enroll   Top10perc   Top25perc 
    ##     -0.0890     -0.1996     -0.1538     -0.1178     -0.3603     -0.3448 
    ## F.Undergrad P.Undergrad    Outstate  Room.Board       Books    Personal 
    ##     -0.0941      0.0175     -0.3277     -0.2665     -0.0572      0.0719 
    ##         PhD    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ##     -0.3033     -0.3039      0.2103     -0.2367     -0.3330     -0.2731

From the above table we can see that PhD, Terminal, Outstate, Top10perc, Top25perc and Expend have the highest magnitude. Thus, the percent of faculty with PhD's or with terminal degrees, percent of the student body in the top 25% or 10% of their high school class, the percent of the student body from out of state, the cost of the university are strongly correlated together.

``` r
pr.col$rotation[, 2]
```

    ##     Private        Apps      Accept      Enroll   Top10perc   Top25perc 
    ##      0.3459     -0.3436     -0.3726     -0.3997      0.0162     -0.0177 
    ## F.Undergrad P.Undergrad    Outstate  Room.Board       Books    Personal 
    ##     -0.4107     -0.2931      0.1915      0.0940     -0.0573     -0.1928 
    ##         PhD    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ##     -0.1162     -0.1042     -0.2044      0.1941      0.0703      0.1178

From the above table we can see that Private, Apps, Accept, Enroll, F.Undergrad have the highest magnitude. Thus, whether the university is private or not, the number of apps received, the number of new students accepted, the number of new students enrolled, the number of full-time undergraduates are strongly correlated together.

Clustering States
-----------------

### 1 Perform PCA on the dataset and plot the observations on the first and second principal components

``` r
arrests <- read.csv('USArrests.csv')
arr.k <- select(arrests, -State)
pr.arr <- prcomp(x = arr.k, scale = TRUE)
biplot(pr.arr, scale=0, cex=.6)
```

![](ps9_files/figure-markdown_github/PCA%20arrest-1.png)

``` r
pr.data <- select(as_data_frame(pr.arr$x), PC1:PC2) %>%
  mutate(State = arrests$State)
```

The first principal component puts approximately equal weight on Assault, Murder, and Rape, with much less weight on UrbanPop while the second principal component puts much more weight on urbanpop(the percent of the population in each state living in urban areas).

### 2 Perform *K*-means clustering with *K* = 2.

``` r
pr.data %>%
  add_column(., k.2 = factor(kmeans(arr.k,2)$cluster)) %>%
  ggplot(aes(PC1, PC2, color = k.2, label = State)) +
  geom_text()+
  labs(title = "K-means clustering with K = 2",
       color = "Clusters")
```

![](ps9_files/figure-markdown_github/unnamed-chunk-1-1.png) States are partitioned into two distinct groups. It seems like it followed the first component vector. One can tell that the blue states on the right, like Hawaii and West Virginia have lower rates of violent crimes, while the red states on the left, like Califonia and Florida have higher rates of violent crimes.

### 3 Perform *K*-means clustering with *K* = 4

``` r
pr.data %>%
  add_column(., k.4 = factor(kmeans(arr.k,4)$cluster)) %>%
  ggplot(aes(PC1, PC2, color = k.4, label = State)) +
  geom_text()+
  labs(title = "K-means clustering with K = 4",
       color = "Clusters")
```

![](ps9_files/figure-markdown_github/unnamed-chunk-2-1.png)

This time the states are grouped into four. The clusters are still following first component vector. The plot seems more accurate/detailed than when k=2. The states in green like South Dakota and Maine seem to have the lowest crime rate while the states in red like California have the highest crime rate.

### 4 Perform *K*-means clustering with *K* = 3.

``` r
pr.data %>%
  add_column(., k.3 = factor(kmeans(arr.k,3)$cluster)) %>%
  ggplot(aes(PC1, PC2, color = k.3, label = State)) +
  geom_text()+
  labs(title = "K-means clustering with K = 3",
       color = "Clusters")
```

![](ps9_files/figure-markdown_github/unnamed-chunk-3-1.png) This time the states are grouped into three. The clusters are still following first component vector. The plot seems more accurate/detailed than when k=2 and less detailed when k=4. The states in red like South Dakota and Maine seem to have the lowest crime rate while the states in blue like California have the highest crime rate.

### 5 Perform *K*-means clustering with *K* = 3 on the first two principal components score vectors

``` r
pr.data %>%
  add_column(., k.3.2 = factor(kmeans(select(., -State),3)$cluster)) %>%
  ggplot(aes(PC1, PC2, color = k.3.2, label = State)) +
  geom_text() +
  labs(title = "K-means clustering with K = 3 on 1st and 2nd Component Vectors",
       color = "Clusters")
```

![](ps9_files/figure-markdown_github/k%203%20score%20vection-1.png)

Clustering with K=3 on the first two principal components score vectors, the groups all seem to be more clustered than when clustering with K=3 on the raw data. Each cluster is more distinct from the others. The states in green like South Dakota and Maine seem to have the lowest crime rate while the states in red like California have the highest crime rate. The blue states in between have medium crime rates.

### 6 Hierarchical clustering

``` r
arrests_hier <- column_to_rownames(arrests, var = "State")
hc_complete <- hclust(dist(arrests_hier), method = "complete")
hc_gg <-ggdendrogram(hc_complete, labels = TRUE) + 
  labs(title = 'Hierarchical Clustering',
       y = 'Euclidean Distance')
hc_gg
```

![](ps9_files/figure-markdown_github/hierarchical-1.png) Each leaf represents one of the 50 observations of states. The leafs in the same branch correspond to observations that are similar to each other, for example, Vermont and North Dakota are very close to each other in this clustering and actually the states have lower crime level and lower percent urban population

### 7 Cut the dendrogram at a height that results in three distinct clusters

``` r
sort(unique(cophenetic(hc_complete)))
```

    ##  [1]   2.29   3.83   3.93   6.24   6.64   7.36   8.03   8.54  10.86  11.46
    ## [11]  12.42  12.61  12.78  13.04  13.30  13.35  13.90  14.50  15.41  15.45
    ## [21]  15.63  15.89  16.98  18.26  19.44  19.90  21.17  22.37  22.77  24.89
    ## [31]  25.09  28.64  29.25  31.48  31.62  32.72  36.73  36.85  38.53  41.49
    ## [41]  48.73  53.59  57.27  64.99  68.76  87.33 102.86 168.61 293.62

``` r
hc_gg+ geom_hline(yintercept=105, linetype=2)+
  geom_hline(yintercept=168, linetype=2)
```

![](ps9_files/figure-markdown_github/cut-1.png)

``` r
cutree(hc_complete, 3)
```

    ##        Alabama         Alaska        Arizona       Arkansas     California 
    ##              1              1              1              2              1 
    ##       Colorado    Connecticut       Delaware        Florida        Georgia 
    ##              2              3              1              1              2 
    ##         Hawaii          Idaho       Illinois        Indiana           Iowa 
    ##              3              3              1              3              3 
    ##         Kansas       Kentucky      Louisiana          Maine       Maryland 
    ##              3              3              1              3              1 
    ##  Massachusetts       Michigan      Minnesota    Mississippi       Missouri 
    ##              2              1              3              1              2 
    ##        Montana       Nebraska         Nevada  New Hampshire     New Jersey 
    ##              3              3              1              3              2 
    ##     New Mexico       New York North Carolina   North Dakota           Ohio 
    ##              1              1              1              3              3 
    ##       Oklahoma         Oregon   Pennsylvania   Rhode Island South Carolina 
    ##              2              2              3              2              1 
    ##   South Dakota      Tennessee          Texas           Utah        Vermont 
    ##              3              2              2              3              3 
    ##       Virginia     Washington  West Virginia      Wisconsin        Wyoming 
    ##              2              2              3              3              2

I found that cut height at 105 and 168 can divide the data into 3 distinct groups. In fact, this classification result is very similar to that of the PCA/k-means clustering approach (k=3).Florida, Carolina, Alabama in the left most group are those states with higher criminal rates, while states, like North Dakota, Vermont, in the right most group are those states with lower criminal rates by the PCA/k-means clustering approah.

### 8 Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation 1

``` r
hc_complete_s <- hclust(dist(scale(arr.k)), method = 'complete')
hc2 <- ggdendrogram(hc_complete_s, labels = TRUE) + 
  labs(title = '50 states crime statistics hierarchical clustering with scaling',
       subtitle = "complete linkage with Euclidean distance")

sort(unique(cophenetic(hc_complete_s)))
```

    ##  [1] 0.206 0.350 0.429 0.494 0.530 0.535 0.594 0.646 0.704 0.711 0.739
    ## [12] 0.772 0.778 0.787 0.798 0.829 0.841 0.846 0.982 0.997 1.012 1.035
    ## [23] 1.071 1.080 1.092 1.131 1.183 1.197 1.212 1.250 1.272 1.333 1.399
    ## [34] 1.467 1.623 1.645 1.659 1.854 1.865 2.263 2.295 2.337 2.446 2.475
    ## [45] 3.088 3.255 4.401 4.420 6.077

``` r
grid.arrange(hc_gg,hc2, ncol = 1, nrow = 2 ) 
```

![](ps9_files/figure-markdown_github/scaling-1.png) After scaling, the Euclidean distance is much smaller, ranging from 0.206 to 6.077.The above graph is cutting the tree at a height of 3, which generated 6 groups.And the scaling increases the weights of Murder and Rape variables, which have lower original value and lower variance compared with Assault in previous model. Even though these two results obtained through hierarchical clustering are quite similar, but we could find that some members change their membership after rescaling in higher level branch.

I would say that we should scale these variables beforehand because before scaling, variables may have different levels of standard deviations. Data features have different units and variables may have more or less weight on the analysis based on their units.
