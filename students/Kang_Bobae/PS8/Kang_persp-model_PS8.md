Problem set \#8: tree-based methods and support vector machines
================
Bobae Kang
March 6, 2017

-   [Part 1: Sexy Joe Biden (redux times two) \[3 points\]](#part-1-sexy-joe-biden-redux-times-two-3-points)
    -   [Fit a decision tree](#fit-a-decision-tree)
    -   [Fit another decision tree](#fit-another-decision-tree)
    -   [Cross-validation](#cross-validation)
    -   [Bagging](#bagging)
    -   [Random Forest](#random-forest)
    -   [Boosting](#boosting)
-   [Part 2: Modeling voter turnout \[3 points\]](#part-2-modeling-voter-turnout-3-points)
    -   [Fit tree-based models](#fit-tree-based-models)
    -   [Compare tree-based models](#compare-tree-based-models)
    -   [Fit SVM models](#fit-svm-models)
    -   [Compare SVM models](#compare-svm-models)
-   [Part 3: OJ Simpson \[4 points\]](#part-3-oj-simpson-4-points)
    -   [Explain](#explain)
    -   [Predict](#predict)

Part 1: Sexy Joe Biden (redux times two) \[3 points\]
=====================================================

In this part, I use and compare a variety of tree-based models on the `biden.csv` data. In doing so, I use the cross-validation approach, splitting the original data randomly into a training set (70% of all observations) and a test/validation set (30% of all observations).

Fit a decision tree
-------------------

Fist, I grow a decision tree using the training data. `biden` is the response variable and other variables are predictors. I have set seed to be 0 for reproducibility. Without any input for control argument, the algorithm chose a model with three terminal nodes. The model predicts that, if an observation is democrat, the themometer score is 74.49. For observations that are not not democrat, the model predicts that the themometer score is 44.17 if an observation is republican and 57.42 otherwise. Its mean squared error (MSE) on the test set is 387.9136.

    ## 
    ## Regression tree:
    ## tree(formula = biden ~ ., data = biden_train)
    ## Variables actually used in tree construction:
    ## [1] "dem" "rep"
    ## Number of terminal nodes:  3 
    ## Residual mean deviance:  408.6 = 515600 / 1262 
    ## Distribution of residuals:
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ## -74.490 -14.170   2.581   0.000  12.580  50.830

    ## [1] "Test MSE:"

    ## [1] 387.9136

![](Kang_persp-model_PS8_files/figure-markdown_github/Part%201:%20a%20decision%20tree-1.png)

Fit another decision tree
-------------------------

What if I let the model to grow more branches? In this second tree with different `control` options, there are 196 terminal nodes. Again, being democrat is responsible for the first split as in the previous tree. However, in the plot, it is difficult to identify what the predictios are at terminal nodes. The MSE is 528.6294, which is significantly larger than the MSE of the previous tree. More splits seemingly lead to worse performance.

    ## 
    ## Regression tree:
    ## tree(formula = biden ~ ., data = biden_train, control = tree.control(nobs = nrow(biden_train), 
    ##     mindev = 0))
    ## Number of terminal nodes:  196 
    ## Residual mean deviance:  344.4 = 368200 / 1069 
    ## Distribution of residuals:
    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -60.0000 -10.0000  -0.5556   0.0000  12.0000  47.8600

    ## [1] "Test MSE:"

    ## [1] 528.6294

![](Kang_persp-model_PS8_files/figure-markdown_github/Part%201:%20another%20tree-1.png)

Cross-validation
----------------

To find out the optimal number of terminal nodes, I try the 10-fold cross-validation approach. The plot illustrates the MSE values for different number of terminal nodes. We find that a tree with three terminal nodes has the least MSE value. This is the first tree I fitted above! The MSE score then increases with more terminal nodes. ![](Kang_persp-model_PS8_files/figure-markdown_github/Part%201:%20CV-1.png)

Bagging
-------

Here, I use bagging approach, growing total 5000 trees. At each split, all five predictors are considered. The MSE score on the test set is 504.1033. The plot shows the effectiveness of all predictors at decreasing the gini index score. Overall, `age` has contributed the most to decreasing the gini index of the model.

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden_test, mtry = 5,      ntree = 5000) 
    ##                Type of random forest: regression
    ##                      Number of trees: 5000
    ## No. of variables tried at each split: 5
    ## 
    ##           Mean of squared residuals: 480.431
    ##                     % Var explained: 13.32

    ## [1] "Test MSE:"

    ## [1] 143.7562

![](Kang_persp-model_PS8_files/figure-markdown_github/Part%201:%20bagging-1.png)

Random Forest
-------------

Now, I turn to the random forest approach. Again, I grow 5000 tress. In this case, only one randomly selected predictor is considered at each split. Its test MSE score, 398.9227, is notably lower than that of the bagging model. The plot compares baggning and random forest in terms of the contribution of each predictor to reducing the Gini Index. While `age` makes the greatest contribution in the bagging model, in the random forest model, `dem` makes the greatest contribution.

    ## 
    ## Call:
    ##  randomForest(formula = biden ~ ., data = biden_test, ntree = 5000) 
    ##                Type of random forest: regression
    ##                      Number of trees: 5000
    ## No. of variables tried at each split: 1
    ## 
    ##           Mean of squared residuals: 394.3393
    ##                     % Var explained: 28.86

    ## [1] "Test MSE:"

    ## [1] 357.0435

![](Kang_persp-model_PS8_files/figure-markdown_github/Part%201:%20random%20forest-1.png)

Boosting
--------

Here, I fit three models with boosting approach, growing 5000 trees for each model. Each of these models have a different value for the shrinkage parmeter, *λ*. The first model has *λ* = 0.001, the second model has *λ* = 0.01, and the final model has *λ* = 0.1. The plot shows the MSE values for each tree with different number of trees. The plot shows changes in test MSE values by the number of trees. With the *λ* = 0.001 model the test MSE keeps decreasing smoothly, even at n.tree = 5000. On the other hand test MSE for the *λ* = 0.1 model remains almost unchanged after n.tree &gt; 2000. Overall, however, the test MSE of the *λ* = 0.01 model is lower at all points than that of the *λ* = 0.001 model. Finally, the *λ* = 0.1 model shows a very distinct pattern: with the very low number of trees, its test MSE reaches its minimum (which, by the way, is the lowest of all for all three models) and continues to increase.

    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...
    ## Distribution not specified, assuming gaussian ...

![](Kang_persp-model_PS8_files/figure-markdown_github/Part%201:%20boosting-1.png)

Part 2: Modeling voter turnout \[3 points\]
===========================================

In this part, I fit five tree-based models and five support vector machine(SVM)-based models to the `mhealth.csv` data. In doing so, I use the cross-validation approach and split the original data into a training set (70% of all observations) and a test/validation set (30% of all observations.)

Fit tree-based models
---------------------

Here I fit and compare the following five tree-based models:

-   A pruned decision tree (the best number of terminal nodes = 5 is chosen using the 10-fold cv)
-   Bagging (5000 trees)
-   Random forest (5000 trees; 2 predictors tried at each split)
-   Boosting (5000 trees; shrinkage parameter = 0.001)
-   Boosting (5000 trees; shrinkage parameter = 0.1)

<!-- -->

    ## 
    ## Classification tree:
    ## snip.tree(tree = mhealth_tree, nodes = c(8L, 9L, 6L, 5L, 7L))
    ## Variables actually used in tree construction:
    ## [1] "age"         "educ"        "mhealth_sum" "inc10"      
    ## Number of terminal nodes:  5 
    ## Residual mean deviance:  1.09 = 883.9 / 811 
    ## Misclassification error rate: 0.2782 = 227 / 816

    ## 
    ## Call:
    ##  randomForest(formula = vote96 ~ ., data = mhealth_train, mtry = 7,      ntree = 5000) 
    ##                Type of random forest: classification
    ##                      Number of trees: 5000
    ## No. of variables tried at each split: 7
    ## 
    ##         OOB estimate of  error rate: 32.48%
    ## Confusion matrix:
    ##      No Yes class.error
    ## No  119 148   0.5543071
    ## Yes 117 432   0.2131148

    ## 
    ## Call:
    ##  randomForest(formula = vote96 ~ ., data = mhealth_train, ntree = 5000) 
    ##                Type of random forest: classification
    ##                      Number of trees: 5000
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 30.15%
    ## Confusion matrix:
    ##      No Yes class.error
    ## No  110 157   0.5880150
    ## Yes  89 460   0.1621129

    ## Distribution not specified, assuming bernoulli ...

    ## gbm(formula = vote96 ~ ., data = mhealth_train2, n.trees = 5000, 
    ##     shrinkage = 0.001)
    ## A gradient boosted model with bernoulli loss function.
    ## 5000 iterations were performed.
    ## There were 7 predictors of which 7 had non-zero influence.

    ## Distribution not specified, assuming bernoulli ...

    ## gbm(formula = vote96 ~ ., data = mhealth_train2, n.trees = 5000, 
    ##     shrinkage = 0.1)
    ## A gradient boosted model with bernoulli loss function.
    ## 5000 iterations were performed.
    ## There were 7 predictors of which 7 had non-zero influence.

Compare tree-based models
-------------------------

I now compare these tree-based models using 1) error rate and 2) ROC/AUC. When I compare test error rates of the five models, the boosting model with shrinkage = 0.001 appears to be the best approach, with the lowest test error rate = 0.2664756. The worst model was bagging, with the highest test error rate = 0.3065903.

    ##        Tree_model Test_error_rate
    ## 1     Pruned tree       0.2865330
    ## 2         Bagging       0.3065903
    ## 3   Random Forest       0.2865330
    ## 4 Boosting, 0.001       0.2664756
    ## 5    Boosing, 0.1       0.2865330

Then I compare the area under the curve for the ROC curves of the five tree-based models. The following plot shows the ROC curves of all five models, with the corresponding AUC scores. Here, again, boosting with shrinkage = 0.001 (green) appears to be the best model with the highest AUC score = 0.743. The second best is boosting with shrinkage = 0.1 (purple) with the AUC score = 0.734. The worst is the pruned tree model (red), with the AUC score = 0.593. ![](Kang_persp-model_PS8_files/figure-markdown_github/Part%202:%20compare%20tree-based%20model,%20roc%20auc-1.png)

Fit SVM models
--------------

Here I fit and compare the following six SVM models:

-   SVM, kernel = 'linear'
-   sVM, kernel = 'polynomial'
-   SVM, kernel = 'radial'
-   Tuned SVM, kernel = 'linear'
-   Tuned sVM, kernel = 'polynomial'
-   Tuned SVM, kernel = 'radial'

<!-- -->

    ## 
    ## Call:
    ## svm(formula = vote96 ~ ., data = mhealth_train, kernel = "linear", 
    ##     cost = 1, scale = FALSE)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  1 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  503

    ## 
    ## Call:
    ## svm(formula = vote96 ~ ., data = mhealth_train, kernel = "polynomial", 
    ##     cost = 1, scale = FALSE)
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
    ## Number of Support Vectors:  327

    ## 
    ## Call:
    ## svm(formula = vote96 ~ ., data = mhealth_train, kernel = "radial", 
    ##     cost = 1, scale = FALSE)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  1 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  718

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mhealth_train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)), 
    ##     kernel = "linear")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  5 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  504

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mhealth_train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)), kernel = "polynomial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  polynomial 
    ##        cost:  10 
    ##      degree:  3 
    ##       gamma:  0.125 
    ##      coef.0:  0 
    ## 
    ## Number of Support Vectors:  476

    ## 
    ## Call:
    ## best.tune(method = svm, train.x = vote96 ~ ., data = as_tibble(mhealth_train), 
    ##     ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100)), kernel = "radial")
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  1 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  508

Compare SVM models
------------------

Now I compare SVM models using ROC/AUC. The following plot shows that the tuned model with radial kenel (cyan) is the best model with the highest AUC score: 0.737. The second best model is a tie: both models using the linear kernel (red and green) have the second highest AUC score: 0.736. In fact, as we have seen just above, they are the same model. That is, the best model with the linear kernel is the one with cost = 1 and 491 support vectors, which is identical to the first SVM model. The worst model is the untuned SVM with the polynomial kernel, with the lowest AUC score = 0.601.
![](Kang_persp-model_PS8_files/figure-markdown_github/Part%202:%20compare%20SVM%20models-1.png)

Part 3: OJ Simpson \[4 points\]
===============================

In this part, I use two different approaches to the `simpson.csv` data: explanation and prediction. The goal of explanation is to understand the correlation between the response variable and the predictors. The goal of prediction, on the other hand, is to get best predctions for new observations.

Explain
-------

Here, the goal is to explain an individual's race on their beliefs about OJ Simpson's guilt. I use logistic regression for this task because 1) `guilt` is a binary variable with two possible outcomes: guilt or not guilt and 2) logistic regression provides the coefficient for each independent variable, making it easier to understand the precise relationship between the dependent and independent variables. I fit two different logistic regression models where: 1. Independent variables only include race-related variables: `black` and `hispanic` 2. Independent variables include all variables other than the dependent/response variable, `guilt`

In the first model, only `black` appears statistically significant, as the extremely low p-value for the coefficient (&lt;2e-16) suggests. The coefficient for `black` is -3.11438, in terms of log-odds. In terms of odds, the exponentiating the coefficient gives 0.04440603. This indicates that, holding other variables constant, being black leads to an average change in the odds that the responsdent thinks OJ Simpson was "probabilty guilty" by a multiplicative factor of 0.04440603. In terms of predicted probabilities, this corresponds to a multiplicative factor of 0.04440603 / (1 + 0.04440603) = 0.04251798 for being black holding other variables constant. That if the respondent is black, she is on avergae 4.25% more likely to think that OJ Simpson is "probably guilty" than a non-black respondent. Therefore, although the coefficient is statistically significant, it may not be substantively significant.

    ## 
    ## Call:
    ## glm(formula = guilt ~ black + hispanic, family = "binomial", 
    ##     data = (simpson %>% na.omit()))
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

In the second model, coefficients for the following variables are statistically significant: `rep`, `age`, `educHigh School Grad`, `educNOT A HIGH SCHOOL GRAD`, `female`, `black` and `incomeREFUSED/NO ANSWER` with p-values &lt; 0.05. The coefficient for `black` is -2.923476, in terms of log-odds. In terms of odds, the exponentiating the coefficient gives 0.05374654. This indicates that, holding other variables constant, being black leads to an average change in the odds that the responsdent thinks OJ Simpson was "probabilty guilty" by a multiplicative factor of 0.05374654. In terms of predicted probabilities, this corresponds to a multiplicative factor of 0.05374654 / (1 + 0.05374654) = 0.05100519 for being black holding other variables constant. That if the respondent is black, she is on avergae 5.1% more likely to think that OJ Simpson is "probably guilty" than a non-black respondent. Therefore, although the coefficient is statistically significant, it may not be substantively significant. The AIC score for the current model (1303.1) is lower than that of the previous model (1355.8). The lower AIC values makes the second regression model more preferable.

    ## 
    ## Call:
    ## glm(formula = guilt ~ ., family = "binomial", data = (simpson %>% 
    ##     na.omit()))
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5152  -0.5298   0.5031   0.6697   2.3807  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                                       Estimate Std. Error z value Pr(>|z|)
    ## (Intercept)                           0.965141   0.345218   2.796  0.00518
    ## dem                                   0.074649   0.224194   0.333  0.73916
    ## rep                                   0.513657   0.234447   2.191  0.02846
    ## ind                                         NA         NA      NA       NA
    ## age                                   0.019306   0.004423   4.365 1.27e-05
    ## educHIGH SCHOOL GRAD                 -0.425116   0.190683  -2.229  0.02578
    ## educNOT A HIGH SCHOOL GRAD           -1.131850   0.282606  -4.005 6.20e-05
    ## educREFUSED                          13.322001 478.030491   0.028  0.97777
    ## educSOME COLLEGE(TRADE OR BUSINESS)  -0.288948   0.200376  -1.442  0.14929
    ## female                               -0.358785   0.149088  -2.407  0.01610
    ## black                                -2.923476   0.193334 -15.121  < 2e-16
    ## hispanic                             -0.196967   0.259145  -0.760  0.44722
    ## income$30,000-$50,000                -0.122000   0.190129  -0.642  0.52109
    ## income$50,000-$75,000                 0.178103   0.247478   0.720  0.47173
    ## incomeOVER $75,000                    0.521352   0.311864   1.672  0.09458
    ## incomeREFUSED/NO ANSWER              -0.906884   0.315046  -2.879  0.00399
    ## incomeUNDER $15,000                  -0.180074   0.239591  -0.752  0.45230
    ##                                        
    ## (Intercept)                         ** 
    ## dem                                    
    ## rep                                 *  
    ## ind                                    
    ## age                                 ***
    ## educHIGH SCHOOL GRAD                *  
    ## educNOT A HIGH SCHOOL GRAD          ***
    ## educREFUSED                            
    ## educSOME COLLEGE(TRADE OR BUSINESS)    
    ## female                              *  
    ## black                               ***
    ## hispanic                               
    ## income$30,000-$50,000                  
    ## income$50,000-$75,000                  
    ## incomeOVER $75,000                  .  
    ## incomeREFUSED/NO ANSWER             ** 
    ## incomeUNDER $15,000                    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1758.1  on 1415  degrees of freedom
    ## Residual deviance: 1271.1  on 1400  degrees of freedom
    ## AIC: 1303.1
    ## 
    ## Number of Fisher Scoring iterations: 13

Predict
-------

Now, I fit and compare multiple models for prediciton. For this part, I split the data into training (70%) and test (30%) sets. The models I use here are the following:

-   Logistic regression
-   Random forest (n.trees = 5000)
-   Boosting (n.trees = 5000)
-   SVM (kernel = 'linear')
-   SVM (kernel = 'radial')

<!-- -->

    ## Distribution not specified, assuming bernoulli ...

To find out the best model, I compare ROC/AUC scores of the models. Thefollowing plot shows both the ROC curves and the corresponding AUC scores of all six models. Based on the AUC scores, The best model is boosting with the AUC score: 0.826. The logistic regression model is the second best model with only a slightly lower AUC score: 0.823. The model with the lowest AUC score is the SVM with radial kenel: 0.773.

![](Kang_persp-model_PS8_files/figure-markdown_github/Part%203:%20predict%20--%20ROC%20AUC-1.png)
