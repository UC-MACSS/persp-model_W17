Problem set \#8: tree-based methods and support vector machines
================
Tong Ju
**March 5th 2017**

-   [Part 1: Sexy Joe Biden (redux times two) \[3 points\]](#part-1-sexy-joe-biden-redux-times-two-3-points)
    -   [1.Split the data into a training set (70%) and a validation set (30%). **Be sure to set your seed prior to this part of your code to guarantee reproducibility of results.**](#split-the-data-into-a-training-set-70-and-a-validation-set-30.-be-sure-to-set-your-seed-prior-to-this-part-of-your-code-to-guarantee-reproducibility-of-results.)
    -   [2.Fit a decision tree to the training data, with `biden` as the response variable and the other variables as predictors. Plot the tree and interpret the results. What is the test MSE?](#fit-a-decision-tree-to-the-training-data-with-biden-as-the-response-variable-and-the-other-variables-as-predictors.-plot-the-tree-and-interpret-the-results.-what-is-the-test-mse)
    -   [3.Now fit another tree to the training data with the following `control` options:](#now-fit-another-tree-to-the-training-data-with-the-following-control-options)
    -   [4. Use the bagging approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results.](#use-the-bagging-approach-to-analyze-this-data.-what-test-mse-do-you-obtain-obtain-variable-importance-measures-and-interpret-the-results.)
    -   [5. Use the random forest approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results. Describe the effect of \(m\), the number of variables considered at each split, on the error rate obtained.](#use-the-random-forest-approach-to-analyze-this-data.-what-test-mse-do-you-obtain-obtain-variable-importance-measures-and-interpret-the-results.-describe-the-effect-of-m-the-number-of-variables-considered-at-each-split-on-the-error-rate-obtained.)
    -   [6. Use the boosting approach to analyze the data. What test MSE do you obtain? How does the value of the shrinkage parameter \(\lambda\) influence the test MSE?](#use-the-boosting-approach-to-analyze-the-data.-what-test-mse-do-you-obtain-how-does-the-value-of-the-shrinkage-parameter-lambda-influence-the-test-mse)
-   [Part 2: Modeling voter turnout \[3 points\]](#part-2-modeling-voter-turnout-3-points)
    -   [1. Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five tree-based models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)](#use-cross-validation-techniques-and-standard-measures-of-model-fit-e.g.-test-error-rate-pre-roc-curvesauc-to-compare-and-evaluate-at-least-five-tree-based-models-of-voter-turnout.-select-the-best-model-and-interpret-the-results-using-whatever-methods-you-see-fit-graphs-tables-model-fit-statistics-predictions-for-hypothetical-observations-etc.)
    -   [2. Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five SVM models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)](#use-cross-validation-techniques-and-standard-measures-of-model-fit-e.g.-test-error-rate-pre-roc-curvesauc-to-compare-and-evaluate-at-least-five-svm-models-of-voter-turnout.-select-the-best-model-and-interpret-the-results-using-whatever-methods-you-see-fit-graphs-tables-model-fit-statistics-predictions-for-hypothetical-observations-etc.)
-   [Part 3: OJ Simpson \[4 points\]](#part-3-oj-simpson-4-points)
    -   [1. What is the relationship between race and belief of OJ Simpson's guilt? Develop a robust statistical learning model and use this model to explain the impact of an individual's race on their beliefs about OJ Simpson's guilt.](#what-is-the-relationship-between-race-and-belief-of-oj-simpsons-guilt-develop-a-robust-statistical-learning-model-and-use-this-model-to-explain-the-impact-of-an-individuals-race-on-their-beliefs-about-oj-simpsons-guilt.)
    -   [2. How can you predict whether individuals believe OJ Simpson to be guilty of these murders? Develop a robust statistical learning model to predict whether individuals believe OJ Simpson to be either probably guilty or probably not guilty and demonstrate the effectiveness of this model using methods we have discussed in class.](#how-can-you-predict-whether-individuals-believe-oj-simpson-to-be-guilty-of-these-murders-develop-a-robust-statistical-learning-model-to-predict-whether-individuals-believe-oj-simpson-to-be-either-probably-guilty-or-probably-not-guilty-and-demonstrate-the-effectiveness-of-this-model-using-methods-we-have-discussed-in-class.)

Part 1: Sexy Joe Biden (redux times two) \[3 points\]
=====================================================

### 1.Split the data into a training set (70%) and a validation set (30%). **Be sure to set your seed prior to this part of your code to guarantee reproducibility of results.**

``` r
set.seed(1234)
# Factorize some string variables in the dataset
biden_fact<-biden %>%
  mutate (female = factor(female, levels =0:1, labels = c("male", "female")),
          dem = factor (dem, levels =0:1, labels = c("non-dem","dem")),
          rep = factor (rep, levels =0:1, labels = c("non-rep", "redp")))

#split the data set to training/test set (70%:30%) as required:
biden_split <- resample_partition(biden_fact, c(test = 0.3, train = 0.7))
```

### 2.Fit a decision tree to the training data, with `biden` as the response variable and the other variables as predictors. Plot the tree and interpret the results. What is the test MSE?

    * Leave the control options for `tree()` at their default values

``` r
# define the MSE() function:
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}

# Make the tree model of Biden data (full tree model)
tree_biden <- tree(biden ~ ., data = biden_split$train)

# Plot the tree:
tree_data <- dendro_data(tree_biden)

ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = 'Decision Tree for Warmth towards Biden ', 
       subtitle = 'Default controls, all predictors as independent vairables')
```

<img src="ps8-trees-svm_files/figure-markdown_github/1-tree1-1.png" style="display: block; margin: auto;" />

``` r
leaf_vals <- leaf_label(tree_data)$yval
test_mse <- mse(tree_biden, biden_split$test)
```

Based on the training data, the full tree model by using the `biden` as the responsive variable, and the other variables as independent variables is established, with the control parameters defaulted.

There are three nodes ((internal and terminal) in this decision tree,in which the braches of the tree are dependent are the political affilication of the respendents (Democratic, Republican, and neither Democratic nor Republican). The result of this tree model could be simply interpreted as below:

If the respondent affiliated with Democratic party, then the model estimates the warmth toward Biden to be 74.513;

If the respondent not affiliated with Democratic party, then we proceed down the left branch to the next internal node:

If he/she is Republican, then the model estimates the warmth toward Biden to be 43.226; Otherwise, if neither Republican nor Democratic, the estimated warmth is predicted to be 57.597.

In addition, the MSE score of this full tree model is 406.417

### 3.Now fit another tree to the training data with the following `control` options:

Use cross-validation to determine the optimal level of tree complexity, plot the optimal tree, and interpret the results. Does pruning the tree improve the test MSE?

First, under the controlled condition, the full tree based on the training data is shown below, with 192 terminal nodes. And the MSE value is 481.

``` r
tree_biden <- tree(biden ~ ., data = biden_split$train, 
                   control = tree.control(nobs = nrow(biden_split$train),
                            mindev = 0))

tree_data<-dendro_data(tree_biden)
ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = 'Decision Tree for Warmth towards Biden (Before Optimization)', 
       subtitle = 'Under controlled condition, all predictors as independent vairables')
```

<img src="ps8-trees-svm_files/figure-markdown_github/1-tree-full-1.png" style="display: block; margin: auto;" />

``` r
full_mse <- mse(tree_biden, biden_split$test)
```

Then, by using the cross-validation appraoch, I plot all the MSE value for the trees with various terminal nodes number. As a result, we find the minimal cross-validated test MSE is for 4 terminal nodes.

``` r
# in this step, the MSE is based on the 10 fold-validation using the whole dataset
# then figure out the optimized number for terminal nodes

# generate 10-fold CV trees
biden_cv <- crossv_kfold(biden_fact, k = 10) %>%
  mutate(tree = map(train, ~ tree(biden ~ ., data = .,
     control = tree.control(nobs = nrow(biden_fact),
                            mindev = 0))))

# calculate each possible prune result for each fold
biden_cv <- expand.grid(biden_cv$.id, 2:15) %>%
  as_tibble() %>%
  mutate(Var2 = as.numeric(Var2)) %>%
  rename(.id = Var1,k = Var2) %>%
  left_join(biden_cv, by = ".id") %>%
  mutate(prune = map2(tree, k, ~ prune.tree(.x, best = .y)),
         mse = map2_dbl(prune, test, mse)) 


biden_cv %>%
  select(k, mse) %>%
  group_by(k) %>%
  summarize(test_mse = mean(mse),
            sd = sd(mse, na.rm = TRUE)) %>%
  ggplot(aes(k, test_mse)) +
  geom_point() +
  geom_line() +
  labs(x = "Number of terminal nodes",
       y = "Test MSE",
       title = "MSE vs. Terminal Node Number (Cross-validation Approach)")
```

<img src="ps8-trees-svm_files/figure-markdown_github/1-tree-kfold-1.png" style="display: block; margin: auto;" />

Therefore, under the controlled condition, I established the optimized tree for the training data with 4 terminal nodes, as follows.

``` r
# the tree before the pruning
tree_biden <- tree(biden ~ ., data = biden_split$train, 
                   control = tree.control(nobs = nrow(biden_split$train),
                            mindev = 0))
# set the node number as 4
mod <- prune.tree(tree_biden, best = 4)

tree_data<-dendro_data(mod)

ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), 
               alpha = 0.5) +
  geom_text(data = label(tree_data), 
            aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), 
            aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = 'Optimized Decision Tree for Warmth towards Biden', 
       subtitle = 'Under controlled condition, all predictors as independent vairables')
```

<img src="ps8-trees-svm_files/figure-markdown_github/1-tree-optimized-1.png" style="display: block; margin: auto;" />

``` r
opt_mse <- mse(mod, biden_split$test)
```

In conclusion, the pruning indeed improve the MSE score: before the pruning, the full model has the MSE as 481, while it is reduced to 407 after the optimization (with 4 nodes). In addition, we can interpret this tree as below:

If the respondent affiliated with Democratic party, then we proceed down to right branch to the next internal node: if the respondent is younger than 53.5, the estimated warmth score toward Biden is approximaely 78.64; otherwise, it is 71.86.

If the respondent not affiliated with Democratic party, then we proceed down the left branch to the next internal node:if he/she is Republican, then the model estimates the warmth toward Biden to be 43.23; Otherwise, if neither Republican nor Democratic, the estimated warmth is predicted to be 57.60.

### 4. Use the bagging approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results.

``` r
biden_bag <- randomForest(biden ~ ., data = biden_split$train,
                          mtry=5,
                          ntree=500)

bagging_mse <- mse(biden_bag , biden_split$test)

data_frame(var = rownames(importance(biden_bag)),
           MeanDecreaseGini = importance(biden_bag)[,1]) %>%
  mutate(var = fct_reorder(var, MeanDecreaseGini, fun = median)) %>%
  ggplot(aes(var, MeanDecreaseGini)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting Warmth Score toward Biden",
       subtitle = "Bagging Appraoch",
       x = NULL,
       y = "Average decrease in the Gini Index")
```

<img src="ps8-trees-svm_files/figure-markdown_github/1-bagging-1.png" style="display: block; margin: auto;" />

The model established by using the bagged approach (trees number = 500, and all predictors should be considered for each split of the tree) can explain only 9.16 variance in the responsive variable considering all the independent variables. In addition, the MSE score is 486, much higher than all the MSE score in the optimized single tree model.

Although interpreting a bagged model is much more difficult than interpreting a single decision tree, from the above plots of variable importance, we can still clearly see how the responsive variable is influenced by the independent variables in terms of the reduction in Gini index. For the bagged model, age, democratic affliation are the most important predictors. The variable education also has some impact on the warmth score towards Biden. However, the republican affiliation and gender are the unimportant predictors.

### 5. Use the random forest approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results. Describe the effect of \(m\), the number of variables considered at each split, on the error rate obtained.

``` r
biden_rf <- randomForest(biden ~ ., data = biden_split$train, 
                          ntree = 500)

rf_mse <- mse(biden_rf , biden_split$test)

seq.int(biden_bag$ntree) %>%
  map_df(~ getTree(biden_bag, k = ., labelVar = TRUE)[1,]) %>%
  count(`split var`) %>%
  knitr::kable(caption = "Variable used to generate the first split in each tree (Bagging)",
               col.names = c("Variable used to split", "Number of training observations"))
```

| Variable used to split |  Number of training observations|
|:-----------------------|--------------------------------:|
| dem                    |                              494|
| rep                    |                                6|

``` r
seq.int(biden_rf$ntree) %>%
  map_df(~ getTree(biden_rf, k = ., labelVar = TRUE)[1,]) %>%
  count(`split var`) %>%
  knitr::kable(caption = "Variable used to generate the first split in each tree (Random Forest)",
               col.names = c("Variable used to split", "Number of training observations"))
```

| Variable used to split |  Number of training observations|
|:-----------------------|--------------------------------:|
| age                    |                              101|
| dem                    |                              105|
| educ                   |                              102|
| female                 |                               79|
| rep                    |                              113|

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
  labs(title = "Predicting Warmth Score toward Biden",
       x = NULL,
       y = "Average decrease in the Gini Index",
       color = "Method")
```

    ## Joining, by = "var"

<img src="ps8-trees-svm_files/figure-markdown_github/1-bagging-sample-number-1.png" style="display: block; margin: auto;" />

Compared with the bagged model, which considered all the predictors in the splitting (m = p), the random forest model with m = p/3 has much lower MSE, 409 (while bagging model's test MSE score is 486).In addition this random forest model can explain 25.8 % of the variance (also much higher than the bagged model's 9.16%).

In the above plot, we can conclude that, compared with the bagged model, age is no longer important in the random forest model, while the affilication with republilcan will be much more important. Except `rep`, we can also observe that the average decrease in the Gini index associated with each variable is generally smaller using the random forest method compared to bagging - this is because of the variable restriction imposed when considering splits. To sum up, the choice m = p/3 gave a significant improvement over bagging (m = p) in this example.

### 6. Use the boosting approach to analyze the data. What test MSE do you obtain? How does the value of the shrinkage parameter \(\lambda\) influence the test MSE?

In this section, first, I explore the optimal number of boosting iterations for the boosting models with various interaction depth 1, 2, or 4. Then by using the optimized number, I calculate out the MSE for each boosting model. It appears that at the optimized number of boosting iterations, both boosting model with depth2 and depth4 have the similar and small enough MSE. Therefore, in the next step, I determine to set the `interaction.depth` as 4 and `n.trees` as 2080, and subsequently investigate how the \(\lambda\) influence the test MSE.

``` r
set.seed(1234)


# Define a function to calculate the MSE for each bossting model
mse_biden_boost <-function(model, test, tree_number) {
  yhat.boost <- predict (model, newdata = test, n.trees=tree_number)
  mse <- mean((yhat.boost - (as_tibble(test))$biden)^2)
  return (mse)
}

biden_models <- list("boosting_depth1" = gbm(biden ~ .,
                                               data = biden_split$train,
                                               n.trees = 5000, interaction.depth = 1),
                       "boosting_depth2" = gbm(biden - 1 ~ .,
                                               data = biden_split$train,
                                               n.trees = 5000, interaction.depth = 2),
                       "boosting_depth4" = gbm(biden~ .,
                                               data = biden_split$train,
                                               n.trees = 5000, interaction.depth = 4))
```

    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...

``` r
data_frame(depth = c(1, 2, 4),
           model = biden_models[c("boosting_depth1", "boosting_depth2", "boosting_depth4")],
           optimal = map_dbl(model, gbm.perf, plot.it = FALSE),
           mse = c(mse_biden_boost(biden_models$boosting_depth1,biden_split$test, 3248),
                   mse_biden_boost(biden_models$boosting_depth2,biden_split$test, 2551),
                   mse_biden_boost(biden_models$boosting_depth4,biden_split$test, 2080))) %>%
  select(-model) %>%
  knitr::kable(caption = "Optimal number of boosting iterations",
               col.names = c("Depth", "Optimal number of iterations", "MSE"))
```

    ## Using OOB method...
    ## Using OOB method...
    ## Using OOB method...

|  Depth|  Optimal number of iterations|  MSE|
|------:|-----------------------------:|----:|
|      1|                          3316|  406|
|      2|                          2611|  406|
|      4|                          2073|  405|

The default shrinkage number is 0.001, so I try different shrinkage number from 0.00025 to 0.4, and plot the MSE vs. the shrinkage number. As the blot below, I found that test MSE score first decrease to reach its minimal value = 402 (when \(\lambda\) = 0.002), then increases as the \(\lambda\) increases. Since \(\lambda\), essentially, is a shrinkage parameter which controls the rate at which boosting learns, very small \(\lambda\) will require very large value of B in order to achieve the good performance. In our case, given the tree number, if the lambda is too small, then the B is not big enough for the model to achieve good result.However, in our case, when the \(\lambda\) to a rather large number (larger than 0.1), the MSE also increases, the accuracy of the model decreases.

In conclusion, based on the optimized number of boosting iterations, I choose the best boosting model as B = 2080, depth =4, \(\lambda\) = 0.002, which gave the MSE as 402, the lowest MSE in all the models.

``` r
s <- c(0.00025, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4)

MSE<-list()
for (i in s) {
  boost <- gbm(biden~ .,data = biden_split$train,n.trees = 2080, interaction.depth = 4, shrinkage = i)
  MSE <- append(MSE, mse_biden_boost(boost,biden_split$test, 2080))
}
```

    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...

``` r
MSE_lambda<-data_frame (shrinkage = s,
            MSE = unlist(MSE))

pander(MSE_lambda)
```

<table style="width:24%;">
<colgroup>
<col width="16%" />
<col width="6%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">shrinkage</th>
<th align="center">MSE</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">0.00025</td>
<td align="center">459</td>
</tr>
<tr class="even">
<td align="center">0.00050</td>
<td align="center">423</td>
</tr>
<tr class="odd">
<td align="center">0.00100</td>
<td align="center">405</td>
</tr>
<tr class="even">
<td align="center">0.00200</td>
<td align="center">402</td>
</tr>
<tr class="odd">
<td align="center">0.00400</td>
<td align="center">407</td>
</tr>
<tr class="even">
<td align="center">0.00600</td>
<td align="center">411</td>
</tr>
<tr class="odd">
<td align="center">0.00800</td>
<td align="center">415</td>
</tr>
<tr class="even">
<td align="center">0.01000</td>
<td align="center">418</td>
</tr>
<tr class="odd">
<td align="center">0.02000</td>
<td align="center">429</td>
</tr>
<tr class="even">
<td align="center">0.04000</td>
<td align="center">436</td>
</tr>
<tr class="odd">
<td align="center">0.06000</td>
<td align="center">451</td>
</tr>
<tr class="even">
<td align="center">0.08000</td>
<td align="center">460</td>
</tr>
<tr class="odd">
<td align="center">0.10000</td>
<td align="center">473</td>
</tr>
<tr class="even">
<td align="center">0.20000</td>
<td align="center">506</td>
</tr>
<tr class="odd">
<td align="center">0.40000</td>
<td align="center">565</td>
</tr>
</tbody>
</table>

``` r
ggplot(MSE_lambda, aes(x=shrinkage, y=MSE)) +
  geom_line()+
  labs(x = "Shrinkage parameter",
       y = "test MSE",
       title = "MSE vs. Shrinkage parameter (0.00025 ~ 0.4) for Boosting",
       subtitle ="num. of trees = 2080, interaction depth = 4")
```

<img src="ps8-trees-svm_files/figure-markdown_github/1-boosting-shrinkage-1.png" style="display: block; margin: auto;" />

Part 2: Modeling voter turnout \[3 points\]
===========================================

### 1. Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five tree-based models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)

In this section, I have built up 5 trees/forests: 1) singel tree by using all independent variables; 2) bagging approach; 3) random forest; 4) single tree by using the age as the independent variabes; 5) single tree with full nodes.

##### Model 1: simple tree with all independent variables

``` r
set.seed(1234)
# factorize the original data
mh <- read_csv("data/mental_health.csv") %>% 
  mutate(vote96 = factor(vote96, levels = 0:1, labels = c("xvote", "vote")), black = factor(black), female = factor(female), married = factor(married))
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
na.omit(mh)
```

    ## # A tibble: 1,165 × 8
    ##    vote96 mhealth_sum   age  educ  black female married inc10
    ##    <fctr>       <dbl> <dbl> <dbl> <fctr> <fctr>  <fctr> <dbl>
    ## 1    vote           0    60    12      0      0       0  4.81
    ## 2    vote           1    36    12      0      0       1  8.83
    ## 3   xvote           7    21    13      0      0       0  1.74
    ## 4   xvote           6    29    13      0      0       0 10.70
    ## 5    vote           1    41    15      1      1       1  8.83
    ## 6    vote           2    48    20      0      0       1  8.83
    ## 7   xvote           9    20    12      0      1       0  7.22
    ## 8   xvote          12    27    11      0      1       0  1.20
    ## 9    vote           2    28    16      0      0       1  7.22
    ## 10   vote           0    72    14      0      0       1  4.01
    ## # ... with 1,155 more rows

``` r
#split the data
mh_split <- resample_partition(mh, p = c("test" = .3, "train" = .7))
# Define the error rate function 
err.rate.tree <- function(model, data) {
  data <- as_tibble(data)
  response <- as.character(model$terms[[2]])

  pred <- predict(model, newdata = data, type = "class")
  actual <- data[[response]]

  return(mean(pred != actual, na.rm = TRUE))
}
##########################################################

# Model 1) simple tree with all the independent variables
tree_default <- tree(vote96 ~ ., data = mh_split$train)

#Plot tree
tree_data <- dendro_data(tree_default)

ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), alpha = 0.5) +
  geom_text(data = label(tree_data), aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Voter turnout tree",
       subtitle = "mhealth_sum + age + educ + black + female + married + inc10")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-1-1.png" style="display: block; margin: auto;" />

``` r
#ROC
fitted <- predict(tree_default, as_tibble(mh_split$test), type = "class")

roc_td <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_td, main = "ROC for model 1")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-1-2.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as.numeric(as_tibble(mh_split$test)$vote96),     predictor = as.numeric(fitted))
    ## 
    ## Data: as.numeric(fitted) in 233 controls (as.numeric(as_tibble(mh_split$test)$vote96) 1) < 561 cases (as.numeric(as_tibble(mh_split$test)$vote96) 2).
    ## Area under the curve: 0.638

``` r
auc1 <-auc(roc_td)

#Mse
mse1 <- err.rate.tree(tree_default, mh_split$test)


#PRE
real <- as.numeric(na.omit(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mse1 
PRE1 <- (E1 - E2) / E1
```

This decision tree is established with full nodes by using all the independent variables. Test error rate is 28.1%, AUC is 0.638 and the PRE is 4.29%, meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by only 4.29%. The tree model stress the age, education, income and mental health status. The tree model can be explained as below:

If age &lt; 44.5, then we proceed down the left branch to the next internal node.

If educ &lt; 12.5 (year), then the model estimates the person don't vote. If educ &gt;= 12.5, then we proceed down the right branch to the next internal node. If the mental health index &lt; 4.5, then the model estimates the person vote. If the mental health index &gt;= 4.5, then the model estimates the person don't vote. If age &gt;= 44.5, then we proceed down the right branch to the next internal node.

If inc10 &lt; 1.08335 (in $10,000s), then the model estimates the person vote. If inc10 &gt; 1.08335, the we proceed down the right branch to the next internal node. If the mental health index &lt; 4.5, then the model estimates the person vote. If the mental health index &gt;= 4.5, then the model estimates the person vote.

##### Model 2: Bagging approach

``` r
set.seed(1234)

# Model 2) bagging
bagging <- randomForest(vote96 ~ ., data = na.omit(as_tibble(mh_split$train)), mtry = 7, ntree = 500)

data_frame(var = rownames(importance(bagging)),
           MeanDecreaseRSS = importance(bagging)[,1]) %>%
  mutate(var = fct_reorder(var, MeanDecreaseRSS, fun = median)) %>%
  ggplot(aes(var, MeanDecreaseRSS)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting voter turnout",
       subtitle = "Bagging",
       x = NULL,
       y = "Average decrease in the Gini Index")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-2-1.png" style="display: block; margin: auto;" />

``` r
#ROC
fitted <- predict(bagging, as_tibble(mh_split$test), type = "prob")[,2]

roc_td <- roc(as.numeric(as_tibble(mh_split$test)$vote96), fitted)
plot(roc_td, main = "ROC for model 2")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-2-2.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as.numeric(as_tibble(mh_split$test)$vote96),     predictor = fitted)
    ## 
    ## Data: fitted in 109 controls (as.numeric(as_tibble(mh_split$test)$vote96) 1) < 250 cases (as.numeric(as_tibble(mh_split$test)$vote96) 2).
    ## Area under the curve: 0.73

``` r
auc2 <-auc(roc_td)

#Mse
mse2 <- err.rate.tree(bagging, mh_split$test)


#PRE
real <- as.numeric(na.omit(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mse2
PRE2 <- (E1 - E2) / E1
```

Through bagging approach, with all predictor variables, the model emphasizes age, inc10, educ, and mental health and has a test error rate 31.5% (estimated by out-of-bag error estimate), which is higher than the default one and indicates a potential overfitting problem. Also, the AUC us 0.73 and the PRE is -7.26%, meaning when compared to the NULL model, estimating all with the median data value, this model even increases the error rate by 7.26%.

##### Model 3: Random Forest

``` r
set.seed(1234)

# Model 3) bagging
rf<- randomForest(vote96 ~ ., data = na.omit(as_tibble(mh_split$train)), ntree = 500)

data_frame(var = rownames(importance(rf)),
           MeanDecreaseRSS = importance(rf)[,1]) %>%
  mutate(var = fct_reorder(var, MeanDecreaseRSS, fun = median)) %>%
  ggplot(aes(var, MeanDecreaseRSS)) +
  geom_point() +
  coord_flip() +
  labs(title = "Predicting voter turnout",
       subtitle = "Random Forest",
       x = NULL,
       y = "Average decrease in the Gini Index")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-3-1.png" style="display: block; margin: auto;" />

``` r
#ROC
fitted <- predict(rf, as_tibble(mh_split$test), type = "prob")[,2]

roc_td <- roc(as.numeric(as_tibble(mh_split$test)$vote96), fitted)
plot(roc_td, main = "ROC for model 3")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-3-2.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as.numeric(as_tibble(mh_split$test)$vote96),     predictor = fitted)
    ## 
    ## Data: fitted in 109 controls (as.numeric(as_tibble(mh_split$test)$vote96) 1) < 250 cases (as.numeric(as_tibble(mh_split$test)$vote96) 2).
    ## Area under the curve: 0.766

``` r
auc3 <-auc(roc_td)

#Mse
mse3 <- err.rate.tree(rf, mh_split$test)


#PRE
real <- as.numeric(na.omit(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mse3
PRE3 <- (E1 - E2) / E1
```

Similar as in the bagging appraoch, the random forest also stresses age, income, education and mental health status, with the test error rate as 28.4%, AUC as 0.766, PRE as 3.18%.

##### Model 4: Simple tree with only age

``` r
tree_age <- tree(vote96 ~ age, data = mh_split$train)

#Plot tree
tree_data <- dendro_data(tree_age)

ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), alpha = 0.5) +
  geom_text(data = label(tree_data), aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Voter turnout tree",
       subtitle = " vote96 ~ age ")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-4-1.png" style="display: block; margin: auto;" />

``` r
#ROC
fitted <- predict(tree_default, as_tibble(mh_split$test), type = "class")

roc_td <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_td, main = "ROC for model 4")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-4-2.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as.numeric(as_tibble(mh_split$test)$vote96),     predictor = as.numeric(fitted))
    ## 
    ## Data: as.numeric(fitted) in 233 controls (as.numeric(as_tibble(mh_split$test)$vote96) 1) < 561 cases (as.numeric(as_tibble(mh_split$test)$vote96) 2).
    ## Area under the curve: 0.638

``` r
auc4 <-auc(roc_td)

#Mse
mse4 <- err.rate.tree(tree_age, mh_split$test)


#PRE
real <- as.numeric(na.omit(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mse4
PRE4<- (E1 - E2) / E1
```

In this model, with the default condition, I only include the education as the independent variable. The results show that the error rate is 0.293, PRE is 0 (indicating the model has a performance equals to the NULL model), and AUC as 0.638.

##### Model 5: Simple tree with controlled condition

``` r
tree_control <- tree(vote96 ~ ., data = mh_split$train, control = tree.control(nobs = nrow(mh_split$train), mindev = 0))

#Plot tree
tree_data <- dendro_data(tree_control)

ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), alpha = 0.5) +
  geom_text(data = label(tree_data), aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Voter turnout tree",
       subtitle = " all the independent variables, under controlled condition ")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-5-1.png" style="display: block; margin: auto;" />

``` r
#ROC
fitted <- predict(tree_control, as_tibble(mh_split$test), type = "class")

roc_td <- roc(as.numeric(as_tibble(mh_split$test)$vote96), as.numeric(fitted))
plot(roc_td, main = "ROC for model 5")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-tree-5-2.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as.numeric(as_tibble(mh_split$test)$vote96),     predictor = as.numeric(fitted))
    ## 
    ## Data: as.numeric(fitted) in 233 controls (as.numeric(as_tibble(mh_split$test)$vote96) 1) < 561 cases (as.numeric(as_tibble(mh_split$test)$vote96) 2).
    ## Area under the curve: 0.619

``` r
auc5 <-auc(roc_td)

#Mse
mse5 <- err.rate.tree(tree_control, mh_split$test)


#PRE
real <- as.numeric(na.omit(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- mse5
PRE5<- (E1 - E2) / E1
```

In this model, with the default condition, I only include the education as the independent variable. The results show that the error rate is 0.293, PRE is 0 (indicating the model has a performance equals to the NULL model), and AUC as 0.614.

To sum up, according to the error rate and PRE, the first single tree model (model 1) is the bset in all five models, sicne it has the lowest error rate and highest PRE value. However, the AUC number of model 1 is not the highest in all five models. From this aspect, the best model is the model 3 (random forests). Nevertheless, considering the AUC with single turning point in the sigle tree model, the AUC is not so informative, thus is not so imprtant when comparing the models.

``` r
sum<-data_frame("model"=c("model1","model2","model3","model4","model5"),
           "Error Rate"=c(mse1,mse2,mse3,mse4,mse5),
           "PRE" =c(PRE1,PRE2,PRE3,PRE4,PRE5),
           "AUC" =c(auc1,auc2,auc3,auc4,auc5)
           )

pander(sum)
```

<table style="width:49%;">
<colgroup>
<col width="11%" />
<col width="18%" />
<col width="12%" />
<col width="6%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">model</th>
<th align="center">Error Rate</th>
<th align="center">PRE</th>
<th align="center">AUC</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">model1</td>
<td align="center">0.281</td>
<td align="center">0.04292</td>
<td align="center">0.638</td>
</tr>
<tr class="even">
<td align="center">model2</td>
<td align="center">0.315</td>
<td align="center">-0.07263</td>
<td align="center">0.730</td>
</tr>
<tr class="odd">
<td align="center">model3</td>
<td align="center">0.284</td>
<td align="center">0.03179</td>
<td align="center">0.766</td>
</tr>
<tr class="even">
<td align="center">model4</td>
<td align="center">0.293</td>
<td align="center">0.00000</td>
<td align="center">0.638</td>
</tr>
<tr class="odd">
<td align="center">model5</td>
<td align="center">0.296</td>
<td align="center">-0.00858</td>
<td align="center">0.619</td>
</tr>
</tbody>
</table>

### 2. Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five SVM models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)

#### Model 1: Linear Kernel with age

``` r
library(e1071)
set.seed(1234)

mh_split <- resample_partition(na.omit(mh), p = c(test = 0.3, train = 0.7))

mh_lin_tune <- tune(svm, vote96 ~ age, data =  as_tibble(mh_split$train),
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
    ##   cost
    ##  0.001
    ## 
    ## - best performance: 0.32 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03  0.32     0.0677
    ## 2 1e-02  0.32     0.0677
    ## 3 1e-01  0.32     0.0677
    ## 4 1e+00  0.32     0.0677
    ## 5 5e+00  0.32     0.0677
    ## 6 1e+01  0.32     0.0677
    ## 7 1e+02  0.32     0.0677

``` r
# Best 
mh_lin <- mh_lin_tune$best.model
summary(mh_lin)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ age, data = as_tibble(mh_split$train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  0.001 
    ##       gamma:  1 
    ## 
    ## Number of Support Vectors:  523
    ## 
    ##  ( 262 261 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  xvote vote

``` r
# ROC

fitted <- predict(mh_lin, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_line <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
plot(roc_line, main = "ROC of Voter Turnout - Linear Kernel, Partial Model")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-SVM-1-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(mh_split$test)$vote96, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 117 controls (as_tibble(mh_split$test)$vote96 xvote) > 232 cases (as_tibble(mh_split$test)$vote96 vote).
    ## Area under the curve: 0.691

``` r
auc1<-auc(roc_line)

#PRE
real <- na.omit(as.numeric(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.292
PRE1 <- (E1 - E2) / E1
```

Using linear kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 1 to 7 and has a 10-fold CV error rate 29.2%. In addition, the AUC us 0.691 and the PRE is 12.9% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 12.9%.

#### Model 2: Linear Kernel with all independent variables

``` r
set.seed(1234)
mh_lin_tune2 <- tune(svm, vote96 ~ ., data =  as_tibble(mh_split$train),
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
# Best 
mh_lin2 <- mh_lin_tune2$best.model
summary(mh_lin2)
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
    ##  xvote vote

``` r
# ROC

fitted <- predict(mh_lin2, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_line2 <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
plot(roc_line2, main = "ROC of Voter Turnout - Linear Kernel, Full Model")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-SVM-2-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(mh_split$test)$vote96, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 117 controls (as_tibble(mh_split$test)$vote96 xvote) < 232 cases (as_tibble(mh_split$test)$vote96 vote).
    ## Area under the curve: 0.746

``` r
auc2<-auc(roc_line2)

#PRE
real <- na.omit(as.numeric(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.264
PRE2 <- (E1 - E2) / E1
```

Using linear kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 3 to 6 and has a 10-fold CV error rate 26.4 %. In addition, the AUC us 0.733 and the PRE is 21.3% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 21.3%. Based on PRE, AUC and error rate, this full model is better than the partial model above.

#### Model 3: Polynomial kernel (full model)

``` r
set.seed(1234)

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
    ##  xvote vote

``` r
# ROC

fitted <- predict(mh_poly, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_poly <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
plot(roc_poly, main = "ROC of Voter Turnout - Polinominal Kernel, Full Model")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-SVM-3-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(mh_split$test)$vote96, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 117 controls (as_tibble(mh_split$test)$vote96 xvote) < 232 cases (as_tibble(mh_split$test)$vote96 vote).
    ## Area under the curve: 0.741

``` r
auc3<-auc(roc_poly)

#PRE
real <- na.omit(as.numeric(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.263
PRE3 <- (E1 - E2) / E1
```

Using linear kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 4 and has a 10-fold CV error rate 26.3 %. In addition, the AUC us 0.702 and the PRE is 21.5% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 9.92%. It appears this polynominal kernal is worse than the linear one.

#### Model 4: Radial kernel (full model)

``` r
set.seed(1234)


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
    ##  xvote vote

``` r
# ROC

fitted <- predict(mh_rad, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_rad <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
plot(roc_rad, main = "ROC of Voter Turnout - Radial Kernel, Full Model")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-SVM-4-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(mh_split$test)$vote96, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 117 controls (as_tibble(mh_split$test)$vote96 xvote) < 232 cases (as_tibble(mh_split$test)$vote96 vote).
    ## Area under the curve: 0.735

``` r
auc4<-auc(roc_rad)

#PRE
real <- na.omit(as.numeric(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.22
PRE4 <- (E1 - E2) / E1
```

Using linear kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 4 and has a 10-fold CV error rate 22.0 %. In addition, the AUC us 0.738 and the PRE is 12.9% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 34.4%. This model is stil not as good as the linear full model.

#### Model 5: Sigmoid kernel (full model)

``` r
set.seed(1234)


mh_sig_tune <- tune(svm, vote96 ~ ., data = as_tibble(mh_split$train),
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
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mh_split$train), 
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
    ##  xvote vote

``` r
# ROC

fitted <- predict(mh_sig, as_tibble(mh_split$test), decision.values = TRUE) %>%
  attributes

roc_sig <- roc(as_tibble(mh_split$test)$vote96, fitted$decision.values)
plot(roc_sig, main = "ROC of Voter Turnout - Radial Kernel, Full Model")
```

<img src="ps8-trees-svm_files/figure-markdown_github/2-SVM-5-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(mh_split$test)$vote96, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 117 controls (as_tibble(mh_split$test)$vote96 xvote) < 232 cases (as_tibble(mh_split$test)$vote96 vote).
    ## Area under the curve: 0.73

``` r
auc5<-auc(roc_sig)

#PRE
real <- na.omit(as.numeric(as_tibble(mh_split$test)$vote96))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.247
PRE5 <- (E1 - E2) / E1
```

Using linear kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 3 and has a 10-fold CV error rate 26.3 %. In addition, the AUC us 0.603 and the PRE is 24.7% (the model MSE is estimated by the 10-fold error rate), meaning when compared to the NULL model, estimating all with the median data value, this model decreases the error rate by 24.7%. This model is even worse than the radial kernal.

``` r
#plot(roc_line2, print.auc = TRUE, col = "red", main ="ROC of Voter Turnout - 5 models")
#plot(roc_poly, print.auc = TRUE, col = "blue", print.auc.y = .4, add = TRUE)
#plot(roc_rad, print.auc = TRUE, col = "orange", print.auc.y = .3, add = TRUE)
#plot(roc_sig, print.auc = TRUE, col = "green", print.auc.y = .2, add = TRUE)
#plot(roc_line, print.auc = TRUE, col = "black",print.auc.y = .1, add = TRUE)


sum2<-data_frame("model"=c("model1","model2","model3","model4","model5"),
           "Error Rate"=c(0.329, 0.297, 0.302, 0.292, 0.319),
           "PRE" =c(PRE1,PRE2,PRE3,PRE4,PRE5),
           "AUC" =c(auc1,auc2,auc3,auc4,auc5)
           )

pander(sum2)
```

<table style="width:46%;">
<colgroup>
<col width="11%" />
<col width="18%" />
<col width="8%" />
<col width="8%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">model</th>
<th align="center">Error Rate</th>
<th align="center">PRE</th>
<th align="center">AUC</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">model1</td>
<td align="center">0.329</td>
<td align="center">0.129</td>
<td align="center">0.691</td>
</tr>
<tr class="even">
<td align="center">model2</td>
<td align="center">0.297</td>
<td align="center">0.213</td>
<td align="center">0.746</td>
</tr>
<tr class="odd">
<td align="center">model3</td>
<td align="center">0.302</td>
<td align="center">0.215</td>
<td align="center">0.741</td>
</tr>
<tr class="even">
<td align="center">model4</td>
<td align="center">0.292</td>
<td align="center">0.344</td>
<td align="center">0.735</td>
</tr>
<tr class="odd">
<td align="center">model5</td>
<td align="center">0.319</td>
<td align="center">0.263</td>
<td align="center">0.730</td>
</tr>
</tbody>
</table>

![](graphics/1.png)

Based on the data summary in the above table, model 2 has the highest AUC value. However, model 4 has a larger PRE than model 2,and the model 4 has the lowest error rate. From the plot above, it is also obvious that for model 2, there are more area under the curve than the other curves. So it will be difficult to determine whehter model 2 or model 4 is the best model in five models. If only concerning the AUC, I bet the model 2 will be much better than model4. Therefore, the model 2(linear full) is the best one, then model 4(radir), then the model 3 (polynominal), the model 5(sigmodial), the worse one is the model 1(partial linear). However, the SVM models is good at prediction rather than interpretation and do not provide clear ways for interpreting the relative importance and influence of individual predictors on the separating hyperplane.

Part 3: OJ Simpson \[4 points\]
===============================

You can make full use of any of the statistical learning techniques to complete this part of the assignment:

-   Linear regression
-   Logistic regression
-   Generalized linear models
-   Non-linear linear models
-   Tree-based models
-   Support vector machines
-   Resampling methods

Select methods that are appropriate for each question and **justify the use of these methods**.

### 1. What is the relationship between race and belief of OJ Simpson's guilt? Develop a robust statistical learning model and use this model to explain the impact of an individual's race on their beliefs about OJ Simpson's guilt.

In this section, I plan to employ the logistic regression, single tree and random forest models, for their ability of providing clearer interpretations about beliefs of OJ Simpson's guilt explained by an individual's race (include black and hispanic). I also split the data into 30% testing and 70% training sets for cross validating their fittness.

#### Single Tree

``` r
set.seed(1234)
simpson <- read_csv("data/simpson.csv") %>%
  mutate_each(funs(as.factor(.)), guilt, dem, rep, ind, female, black, hispanic, educ, income)
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
#Split data
simpson_split <- resample_partition(simpson, c(test = 0.3, train = 0.7))

#Grow tree
simpson_tree_default <- tree(guilt ~ black + hispanic, data = simpson_split$train)

#Plot tree
tree_data <- dendro_data(simpson_tree_default)

ggplot(segment(tree_data)) +
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend), alpha = 0.5) +
  geom_text(data = label(tree_data), aes(x = x, y = y, label = label_full), vjust = -0.5, size = 3) +
  geom_text(data = leaf_label(tree_data), aes(x = x, y = y, label = label), vjust = 0.5, size = 3) +
  theme_dendro() +
  labs(title = "Simpson guilt opinion tree",
       subtitle = "black + hispanic")
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-race-tree-1.png" style="display: block; margin: auto;" />

``` r
#ROC
fitted <- predict(simpson_tree_default, as_tibble(simpson_split$test), type = "class")

roc_t <- roc(as.numeric(as_tibble(simpson_split$test)$guilt), as.numeric(fitted))
plot(roc_t)
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-race-tree-2.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as.numeric(as_tibble(simpson_split$test)$guilt),     predictor = as.numeric(fitted))
    ## 
    ## Data: as.numeric(fitted) in 129 controls (as.numeric(as_tibble(simpson_split$test)$guilt) 1) < 300 cases (as.numeric(as_tibble(simpson_split$test)$guilt) 2).
    ## Area under the curve: 0.744

``` r
auc1<-auc(roc_t)

#Accuracy
MSE1 <- err.rate.tree(simpson_tree_default, simpson_split$test)

#PRE
real <- na.omit(as.numeric(as_tibble(simpson_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- MSE1
PRE1 <- (E1 - E2) / E1
```

As for the single tree model with default setting, it gives us a 17.02% test error rate, a 0.4341 PRE, and a 0.744 AUC, exactly the same as we got from the logistic model. Basically, the tree model uses only black to estimate the guilt belief.

If the person is not black, the model estimates that the person would believe Simpson is guilty. If the person is black, the model estimates that the person would believe Simpson is not guilty.

#### Random Forest

``` r
set.seed(1234)

simpson_rf <- randomForest(guilt ~ black + hispanic, data = na.omit(as_tibble(simpson_split$train)), ntree = 500)
simpson_rf
```

    ## 
    ## Call:
    ##  randomForest(formula = guilt ~ black + hispanic, data = na.omit(as_tibble(simpson_split$train)),      ntree = 500) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 1
    ## 
    ##         OOB estimate of  error rate: 19.1%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 156 157       0.502
    ## 1  31 643       0.046

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

<img src="ps8-trees-svm_files/figure-markdown_github/3-race-forest-1.png" style="display: block; margin: auto;" />

``` r
#ROC
fitted <- predict(simpson_rf, na.omit(as_tibble(simpson_split$test)), type = "prob")[,2]

roc_rf <- roc(na.omit(as_tibble(simpson_split$test))$guilt, fitted)
plot(roc_rf)
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-race-forest-2.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = na.omit(as_tibble(simpson_split$test))$guilt,     predictor = fitted)
    ## 
    ## Data: fitted in 129 controls (na.omit(as_tibble(simpson_split$test))$guilt 0) < 300 cases (na.omit(as_tibble(simpson_split$test))$guilt 1).
    ## Area under the curve: 0.745

``` r
auc2<-auc(roc_rf)

#PRE
real <- na.omit(as.numeric(as_tibble(simpson_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.1843
MSE2<-E2
PRE2 <- (E1 - E2) / E1
```

As for the random forest model with default setting and 500 trees, it gives us a 19.1% test error rate (estimated by out-of-bag error estimate) and a 38.71% PRE, both are worse than the previous two models. However, the random forest model has a 0.745 AUC at a similar level as the previous model doed. Regarding the predictor importance, the black has a way higher average decrease in the Gini index than hispanic, which indicates black's importance and confirms the results from the previous model.

#### Logistic model

``` r
set.seed(1234)

getProb <- function(model, data){
  data <- data %>% 
    add_predictions(model) %>% 
    mutate(prob = exp(pred) / (1 + exp(pred)),
           pred_bi = as.numeric(prob > .5))
  return(data)
}



model_logistic <- glm(guilt ~ black + hispanic, data = simpson_split$train, family = binomial)
summary(model_logistic)
```

    ## 
    ## Call:
    ## glm(formula = guilt ~ black + hispanic, family = binomial, data = simpson_split$train)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -1.826  -0.608   0.647   0.647   2.119  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)   1.4587     0.0938   15.55   <2e-16 ***
    ## black1       -3.0529     0.2169  -14.07   <2e-16 ***
    ## hispanic1    -0.5388     0.2846   -1.89    0.058 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1233.12  on 986  degrees of freedom
    ## Residual deviance:  956.83  on 984  degrees of freedom
    ##   (112 observations deleted due to missingness)
    ## AIC: 962.8
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
df_logistic_test <- getProb(model_logistic, as.data.frame(simpson_split$test))

#ROC
auc3 <- auc(df_logistic_test$guilt, df_logistic_test$pred_bi)


#Accuracy
accuracy3 <- mean(df_logistic_test$guilt == df_logistic_test$pred_bi, na.rm = TRUE)


#PRE
real <- na.omit(as.numeric(as_tibble(simpson_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 1 - accuracy3
MSE3<-E2
PRE3 <- (E1 - E2) / E1
```

In the logistic model by using the `black` and `haspanic` as two independent variables, it gives the 17.00% test error rate, a 0.434 PRE, and a 0.744 AUC. As shown above, both two independent variables included in the model have statistically significant relationships with the guilt, with black (p-value &lt; 2e-16) at a 99.9% confidence level and hispanic (p-value = 0.058) at a 90% confidence level. Both of them have negative relationship with the responsive variable, which means that when the respondent is black (parameter est. = -3.0529) or hispanic (parameter est. = -0.5388), the predected probability of believing Simpson's guilty has a differential increase. Black has a even stronger power than hispanic as it has a smaller p-value and a larger parameter absolute value. The amount of the change in the probability depends on the initial value of the changing independent variable. To interpret the co-effecient in this model: being as a black people, the odds of thinking Simpson guilty will only be exp(-3.0529)=4.72% than those non-black people; being as a haspanic people, the odds of thinking Simpson guilty will only be exp(-0.5388)=58.3% than those non-haspanic people.

``` r
logistic_grid <- as.data.frame(simpson_split$test) %>%
  data_grid(black, hispanic) %>%
  add_predictions(model_logistic) %>% 
  mutate(prob = exp(pred) / (1 + exp(pred)))

ggplot(logistic_grid, aes(black, pred, group = factor(hispanic), color = factor(hispanic))) +
  geom_line() +
  scale_color_discrete(name = "Hispanic or not (hispanic = 1)") +
  labs(title = "Log-odds of guilt belief",
       subtitle = "by race",
       x = "Black or not (black = 1)",
       y = "Log-odds of voter turnout")
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-race-logistic2-1.png" style="display: block; margin: auto;" />

``` r
ggplot(logistic_grid, aes(black, prob, group = factor(hispanic), color = factor(hispanic))) +
  geom_line() +
  scale_color_discrete(name = "Hispanic or not (hispanic = 1)") +
  labs(title = "Predicted probability of guilt belief",
       subtitle = "by race",
       x = "Black or not (black = 1)",
       y = "Predicted probability of voter turnout")
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-race-logistic2-2.png" style="display: block; margin: auto;" />

The above graphs illustrate the relationship between black, hispanic status, and the belief of Simpson's guilty. In the graph Log-odds of guilt belief, we observe the mentioned negative relationship between race and guilt belief log-odds. The log-odds goes down when people are black, as the lines with a negative slope. In addition, hispanic people have a line below the non-hispanic people. That is, The log-odds of guilt belief is lower for hispanic people. This could be because the the similarity between the respondents and Simpson in terms of race.

While both the logistic model and tree model perform well, I'll choose the logistic model as my final model, since its interpretability in terms of single relationship direction, the strength of effect, and the specific amount of effect in estimation. I thus redo the logistic model with a 100-time 10-fold cross validation to examine its robustness.

``` r
sum3<-data_frame("model"=c("model1","model2","model3"),
           "Error Rate"=c(MSE1, MSE2, MSE3),
           "PRE" =c(PRE1,PRE2,PRE3),
           "AUC" =c(auc1,auc2,auc3)
           )

pander(sum3)
```

<table style="width:46%;">
<colgroup>
<col width="11%" />
<col width="18%" />
<col width="8%" />
<col width="8%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">model</th>
<th align="center">Error Rate</th>
<th align="center">PRE</th>
<th align="center">AUC</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">model1</td>
<td align="center">0.170</td>
<td align="center">0.434</td>
<td align="center">0.744</td>
</tr>
<tr class="even">
<td align="center">model2</td>
<td align="center">0.184</td>
<td align="center">0.387</td>
<td align="center">0.745</td>
</tr>
<tr class="odd">
<td align="center">model3</td>
<td align="center">0.170</td>
<td align="center">0.434</td>
<td align="center">0.744</td>
</tr>
</tbody>
</table>

``` r
fold_model_mse <- function(df, k){
  cv10_data <- crossv_kfold(df, k = k)
  cv10_models <- map(cv10_data$train, ~ glm(guilt ~ black + hispanic, family = binomial, data = .))
  cv10_prob <- map2(cv10_models, cv10_data$train, ~getProb(.x, as.data.frame(.y)))
  cv10_mse <- map(cv10_prob, ~ mean(.$guilt != .$pred_bi, na.rm = TRUE))
  return(data_frame(cv10_mse))
}

set.seed(1234)
mses <- rerun(100, fold_model_mse(simpson, 10)) %>%
  bind_rows(.id = "id")

ggplot(data = mses, aes(x = "MSE (100 times 10-fold)", y = as.numeric(cv10_mse))) +
  geom_boxplot() +
  labs(title = "Boxplot of MSEs - logistic model",
       x = element_blank(),
       y = "MSE value")
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-race-validation-1.png" style="display: block; margin: auto;" />

``` r
mse_100cv10 <- mean(as.numeric(mses$cv10_mse))
mseSd_100cv10 <- sd(as.numeric(mses$cv10_mse))
mse_100cv10
```

    ## [1] 0.184

``` r
mseSd_100cv10
```

    ## [1] 0.00336

In the final step, by using the cross validation, I intend to examine the robustness of my logistic model. The model gets a 18.432% average error rate, which is still pretty good, with a small std of the error rate at 0.003.

### 2. How can you predict whether individuals believe OJ Simpson to be guilty of these murders? Develop a robust statistical learning model to predict whether individuals believe OJ Simpson to be either probably guilty or probably not guilty and demonstrate the effectiveness of this model using methods we have discussed in class.

In this section, the SVM with linear, polynomimal, radial, sigmoid kernel and single tree models were examined for their ability to predict the beliefs of OJ Simpson's guilt explained by all the available predictors.

#### SVM, Linear Kernel

``` r
set.seed(1234)

simpson_lin_tune <- tune(svm, guilt ~ dem + rep + age + educ + female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)),
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
    ## - best performance: 0.191 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.317     0.0349
    ## 2 1e-02 0.191     0.0288
    ## 3 1e-01 0.191     0.0288
    ## 4 1e+00 0.191     0.0288
    ## 5 5e+00 0.191     0.0288
    ## 6 1e+01 0.191     0.0288
    ## 7 1e+02 0.191     0.0288

``` r
simpson_lin <- simpson_lin_tune$best.model
summary(simpson_lin)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  0.01 
    ##       gamma:  0.0625 
    ## 
    ## Number of Support Vectors:  644
    ## 
    ##  ( 331 313 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#Best
simpson_lin <- simpson_lin_tune$best.model
summary(simpson_lin)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  0.01 
    ##       gamma:  0.0625 
    ## 
    ## Number of Support Vectors:  644
    ## 
    ##  ( 331 313 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#ROC
fitted <- predict(simpson_lin, as_tibble(simpson_split$test), decision.values = TRUE) %>%
  attributes

roc_line <- roc(as_tibble(simpson_split$test)$guilt, fitted$decision.values)
plot(roc_line)
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-2-0-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(simpson_split$test)$guilt, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 129 controls (as_tibble(simpson_split$test)$guilt 0) < 300 cases (as_tibble(simpson_split$test)$guilt 1).
    ## Area under the curve: 0.796

``` r
auc(roc_line)
```

    ## Area under the curve: 0.796

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(simpson_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.1905
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.366

Using linear kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100), the model gets the best cost level at 0.01, and a 19.1% 10-fold CV error rate. Also, the AUC us 0.796 and the PRE is 36.6% (the model MSE is estimated by the 10-fold error rate).

#### SVM, polynominal Kernel

``` r
set.seed(1234)

simpson_poly_tune <- tune(svm, guilt ~ dem + rep + age + educ + female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)),
                    kernel = "polynomial",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(simpson_poly_tune)
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
    ## - best performance: 0.199 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.317     0.0349
    ## 2 1e-02 0.317     0.0349
    ## 3 1e-01 0.317     0.0349
    ## 4 1e+00 0.315     0.0361
    ## 5 5e+00 0.211     0.0239
    ## 6 1e+01 0.199     0.0227
    ## 7 1e+02 0.199     0.0289

``` r
simpson_poly <- simpson_poly_tune$best.model
summary(simpson_poly)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "polynomial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  polynomial 
    ##        cost:  10 
    ##      degree:  3 
    ##       gamma:  0.0625 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  561
    ## 
    ##  ( 308 253 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#Best
simpson_poly <- simpson_poly_tune$best.model
summary(simpson_poly)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "polynomial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  polynomial 
    ##        cost:  10 
    ##      degree:  3 
    ##       gamma:  0.0625 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  561
    ## 
    ##  ( 308 253 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#ROC
fitted <- predict(simpson_poly, as_tibble(simpson_split$test), decision.values = TRUE) %>%
  attributes

roc_poly <- roc(as_tibble(simpson_split$test)$guilt, fitted$decision.values)
plot(roc_poly)
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-2-1-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(simpson_split$test)$guilt, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 129 controls (as_tibble(simpson_split$test)$guilt 0) < 300 cases (as_tibble(simpson_split$test)$guilt 1).
    ## Area under the curve: 0.766

``` r
auc(roc_poly)
```

    ## Area under the curve: 0.766

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(simpson_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.1986
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.34

Using polynomial kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100) and different degree levels (3, 4, and 5), the model gets the best cost level at 10, degree level at 3, and a 19.9% 10-fold CV error rate. Also, the AUC as 0.766 and the PRE is 34.00% (the model MSE is estimated by the 10-fold error rate). Generally, the model is slightly worse than the linear one.

#### SVM, radial Kernel

``` r
set.seed(1234)

simpson_rad_tune <- tune(svm, guilt ~ dem + rep + age + educ + female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)),
                    kernel = "radial",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(simpson_rad_tune)
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
    ## - best performance: 0.191 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.317     0.0349
    ## 2 1e-02 0.317     0.0349
    ## 3 1e-01 0.270     0.0436
    ## 4 1e+00 0.191     0.0288
    ## 5 5e+00 0.191     0.0288
    ## 6 1e+01 0.192     0.0298
    ## 7 1e+02 0.205     0.0313

``` r
simpson_rad <- simpson_rad_tune$best.model
summary(simpson_rad)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "radial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  1 
    ##       gamma:  0.0625 
    ## 
    ## Number of Support Vectors:  489
    ## 
    ##  ( 261 228 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#Best
simpson_rad <- simpson_rad_tune$best.model
summary(simpson_rad)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "radial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  1 
    ##       gamma:  0.0625 
    ## 
    ## Number of Support Vectors:  489
    ## 
    ##  ( 261 228 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#ROC
fitted <- predict(simpson_rad, as_tibble(simpson_split$test), decision.values = TRUE) %>%
  attributes

roc_rad <- roc(as_tibble(simpson_split$test)$guilt, fitted$decision.values)
plot(roc_rad)
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-2-2-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(simpson_split$test)$guilt, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 129 controls (as_tibble(simpson_split$test)$guilt 0) < 300 cases (as_tibble(simpson_split$test)$guilt 1).
    ## Area under the curve: 0.771

``` r
auc(roc_rad)
```

    ## Area under the curve: 0.771

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(simpson_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.1905
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.366

Using polynomial kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100) and different degree levels (3, 4, and 5), the model gets the best cost level at 10, degree level at 3, and a 19.86% 10-fold CV error rate. Also, the AUC as 0.771 and the PRE is 36.6% (the model MSE is estimated by the 10-fold error rate). Generally, the model is slightly worse than the linear one.

#### SVM, sigmodial Kernel

``` r
set.seed(1234)

simpson_sig_tune <- tune(svm, guilt ~ dem + rep + age + educ + female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)),
                    kernel = "sigmoid",
                    range = list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(simpson_sig_tune)
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
    ## - best performance: 0.191 
    ## 
    ## - Detailed performance results:
    ##    cost error dispersion
    ## 1 1e-03 0.317     0.0349
    ## 2 1e-02 0.317     0.0349
    ## 3 1e-01 0.317     0.0349
    ## 4 1e+00 0.191     0.0288
    ## 5 5e+00 0.195     0.0255
    ## 6 1e+01 0.215     0.0306
    ## 7 1e+02 0.283     0.0508

``` r
simpson_sig <- simpson_sig_tune$best.model
summary(simpson_sig)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "sigmoid")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  sigmoid 
    ##        cost:  1 
    ##       gamma:  0.0625 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  451
    ## 
    ##  ( 229 222 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#Best
simpson_sig <- simpson_sig_tune$best.model
summary(simpson_sig)
```

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = guilt ~ dem + rep + age + educ + 
    ##     female + black + hispanic + income, data = na.omit(as_tibble(simpson_split$train)), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "sigmoid")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  sigmoid 
    ##        cost:  1 
    ##       gamma:  0.0625 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  451
    ## 
    ##  ( 229 222 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

``` r
#ROC
fitted <- predict(simpson_sig, as_tibble(simpson_split$test), decision.values = TRUE) %>%
  attributes

roc_sig <- roc(as_tibble(simpson_split$test)$guilt, fitted$decision.values)
plot(roc_sig)
```

<img src="ps8-trees-svm_files/figure-markdown_github/3-2-sig-1.png" style="display: block; margin: auto;" />

    ## 
    ## Call:
    ## roc.default(response = as_tibble(simpson_split$test)$guilt, predictor = fitted$decision.values)
    ## 
    ## Data: fitted$decision.values in 129 controls (as_tibble(simpson_split$test)$guilt 0) < 300 cases (as_tibble(simpson_split$test)$guilt 1).
    ## Area under the curve: 0.782

``` r
auc(roc_sig)
```

    ## Area under the curve: 0.782

``` r
#PRE
real <- na.omit(as.numeric(as_tibble(simpson_split$test)$guilt))
E1 <- mean(as.numeric(real != median(real)))
E2 <- 0.191
PRE <- (E1 - E2) / E1
PRE
```

    ## [1] 0.365

Using polynomial kernel, with all predictor variables and tested at different cost levels (0.001, 0.01, 0.1, 1, 5, 10, and 100). the model gets the best cost level at 4 with a 19.1% 10-fold CV error rate. Also, the AUC us 0.782 and the PRE is 36.5% (the model MSE is estimated by the 10-fold error rate). Generally, the model is still worse than the linear one.

``` r
# plot all the ROC curves:
#plot(roc_line, print.auc = TRUE, col = "blue", print.auc.x = .2, main="ROC for Guilt of OJ Simpson - 4 models ")
#plot(roc_poly, print.auc = TRUE, col = "red", print.auc.x = .2, print.auc.y = .4, add = TRUE)
#plot(roc_rad, print.auc = TRUE, col = "orange", print.auc.x = .2, print.auc.y = .3, add = TRUE)
#plot(roc_sig, print.auc = TRUE, col = "green", print.auc.x = .2, print.auc.y = .2, add = TRUE)


# summarize:
sum4<-data_frame("model"=c("model1","model2","model3","model4"),
           "Error Rate"=c(0.1905, 0.1986, 0.1905, 0.1910),
           "PRE" =c(0.366, 0.340, 0.366, 0.365),
           "AUC" =c(0.796, 0.766, 0.771, 0.782)
           )

pander(sum4)
```

<table style="width:46%;">
<colgroup>
<col width="11%" />
<col width="18%" />
<col width="8%" />
<col width="8%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">model</th>
<th align="center">Error Rate</th>
<th align="center">PRE</th>
<th align="center">AUC</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">model1</td>
<td align="center">0.191</td>
<td align="center">0.366</td>
<td align="center">0.796</td>
</tr>
<tr class="even">
<td align="center">model2</td>
<td align="center">0.199</td>
<td align="center">0.340</td>
<td align="center">0.766</td>
</tr>
<tr class="odd">
<td align="center">model3</td>
<td align="center">0.191</td>
<td align="center">0.366</td>
<td align="center">0.771</td>
</tr>
<tr class="even">
<td align="center">model4</td>
<td align="center">0.191</td>
<td align="center">0.365</td>
<td align="center">0.782</td>
</tr>
</tbody>
</table>

![](graphics/2.png)

According to the error rate, PRE and AUC, I think the model1 (linear kernel) is the best model I got. It has the lowest Error rate, and highest PRE as well as the highest AUC, which indicates this model has the best performance among all the four models.
