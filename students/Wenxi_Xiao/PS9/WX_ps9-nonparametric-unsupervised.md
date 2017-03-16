Problem set \#9: nonparametric methods and unsupervised learning
================
Wenxi Xiao
**Due Wednesday March 13th at 11:59pm**

-   [Attitudes towards feminists: estimate a series of models explaining/predicting attitudes towards feminists.](#attitudes-towards-feminists-estimate-a-series-of-models-explainingpredicting-attitudes-towards-feminists.)
    -   [1. Split the data into a training and test set (70/30%).](#split-the-data-into-a-training-and-test-set-7030.)
    -   [1. Calculate the test MSE for KNN models with *K* = 5, 10, 15, ..., 100, using whatever combination of variables you see fit. Which model produces the lowest test MSE?](#calculate-the-test-mse-for-knn-models-with-k-5-10-15-...-100-using-whatever-combination-of-variables-you-see-fit.-which-model-produces-the-lowest-test-mse)
    -   [1. Calculate the test MSE for weighted KNN models with *K* = 5, 10, 15, …, 100 using the same combination of variables as before. Which model produces the lowest test MSE?](#calculate-the-test-mse-for-weighted-knn-models-with-k-5-10-15-dots-100-using-the-same-combination-of-variables-as-before.-which-model-produces-the-lowest-test-mse)
    -   [1. Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?](#compare-the-test-mse-for-the-best-knnwknn-models-to-the-test-mse-for-the-equivalent-linear-regression-decision-tree-boosting-and-random-forest-methods-using-the-same-combination-of-variables-as-before.-which-performs-the-best-why-do-you-think-this-method-performed-the-best-given-your-knowledge-of-how-it-works)
-   [Voter turnout and depression](#voter-turnout-and-depression)
    -   [1. Split the data into a training and test set (70/30).](#split-the-data-into-a-training-and-test-set-7030.-1)
    -   [1. Calculate the test error rate for KNN models with *K* = 1, 2, …, 10, using whatever combination of variables you see fit. Which model produces the lowest test MSE?](#calculate-the-test-error-rate-for-knn-models-with-k-12dots10-using-whatever-combination-of-variables-you-see-fit.-which-model-produces-the-lowest-test-mse)
    -   [1. Calculate the test error rate for weighted KNN models with *K* = 1, 2, …, 10 using the same combination of variables as before. Which model produces the lowest test error rate?](#calculate-the-test-error-rate-for-weighted-knn-models-with-k-12dots10-using-the-same-combination-of-variables-as-before.-which-model-produces-the-lowest-test-error-rate)
    -   [1. Compare the test error rate for the best KNN/wKNN model(s) to the test error rate for the equivalent logistic regression, decision tree, boosting, random forest, and SVM methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?](#compare-the-test-error-rate-for-the-best-knnwknn-models-to-the-test-error-rate-for-the-equivalent-logistic-regression-decision-tree-boosting-random-forest-and-svm-methods-using-the-same-combination-of-variables-as-before.-which-performs-the-best-why-do-you-think-this-method-performed-the-best-given-your-knowledge-of-how-it-works)
-   [Colleges](#colleges)
-   [Clustering states](#clustering-states)
    -   [1. Perform PCA on the dataset and plot the observations on the first and second principal components.](#perform-pca-on-the-dataset-and-plot-the-observations-on-the-first-and-second-principal-components.)
    -   [1. Perform *K*-means clustering with *K* = 2. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.](#perform-k-means-clustering-with-k2.-plot-the-observations-on-the-first-and-second-principal-components-and-color-code-each-state-based-on-their-cluster-membership.-describe-your-results.)
    -   [1. Perform *K*-means clustering with *K* = 4. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.](#perform-k-means-clustering-with-k4.-plot-the-observations-on-the-first-and-second-principal-components-and-color-code-each-state-based-on-their-cluster-membership.-describe-your-results.)
    -   [1. Perform *K*-means clustering with *K* = 3. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.](#perform-k-means-clustering-with-k3.-plot-the-observations-on-the-first-and-second-principal-components-and-color-code-each-state-based-on-their-cluster-membership.-describe-your-results.)
    -   [1. Perform *K*-means clustering with *K* = 3 on the first two principal components score vectors, rather than the raw data. Describe your results and compare them to the clustering results with *K* = 3 based on the raw data.](#perform-k-means-clustering-with-k3-on-the-first-two-principal-components-score-vectors-rather-than-the-raw-data.-describe-your-results-and-compare-them-to-the-clustering-results-with-k3-based-on-the-raw-data.)
    -   [1. Using hierarchical clustering with complete linkage and Euclidean distance, cluster the states.](#using-hierarchical-clustering-with-complete-linkage-and-euclidean-distance-cluster-the-states.)
    -   [1. Cut the dendrogram at a height that results in three distinct clusters. Which states belong to which clusters?](#cut-the-dendrogram-at-a-height-that-results-in-three-distinct-clusters.-which-states-belong-to-which-clusters)
    -   [1. Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation 1. What effect does scaling the variables have on the hierarchical clustering obtained? In your opinion, should the variables be scaled before the inter-observation dissimilarities are computed? Provide a justification for your answer.](#hierarchically-cluster-the-states-using-complete-linkage-and-euclidean-distance-after-scaling-the-variables-to-have-standard-deviation-1.-what-effect-does-scaling-the-variables-have-on-the-hierarchical-clustering-obtained-in-your-opinion-should-the-variables-be-scaled-before-the-inter-observation-dissimilarities-are-computed-provide-a-justification-for-your-answer.)

Attitudes towards feminists: estimate a series of models explaining/predicting attitudes towards feminists.
===========================================================================================================

1. Split the data into a training and test set (70/30%).
--------------------------------------------------------

1. Calculate the test MSE for KNN models with *K* = 5, 10, 15, ..., 100, using whatever combination of variables you see fit. Which model produces the lowest test MSE?
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

I will used all the available variables, `female`, `age`, `educ`, `income`, `dem`, and `rep` as predictors.

    ## # A tibble: 20 <U+00D7> 3
    ##        k          knn      mse
    ##    <dbl>       <list>    <dbl>
    ## 1      5 <S3: knnReg> 546.9755
    ## 2     10 <S3: knnReg> 493.8420
    ## 3     15 <S3: knnReg> 480.0165
    ## 4     20 <S3: knnReg> 466.7502
    ## 5     25 <S3: knnReg> 463.6671
    ## 6     30 <S3: knnReg> 459.2396
    ## 7     35 <S3: knnReg> 461.1312
    ## 8     40 <S3: knnReg> 459.5793
    ## 9     45 <S3: knnReg> 455.7123
    ## 10    50 <S3: knnReg> 457.2871
    ## 11    55 <S3: knnReg> 457.7437
    ## 12    60 <S3: knnReg> 457.5016
    ## 13    65 <S3: knnReg> 457.5636
    ## 14    70 <S3: knnReg> 458.7949
    ## 15    75 <S3: knnReg> 459.1128
    ## 16    80 <S3: knnReg> 457.4027
    ## 17    85 <S3: knnReg> 457.2196
    ## 18    90 <S3: knnReg> 456.9237
    ## 19    95 <S3: knnReg> 456.9098
    ## 20   100 <S3: knnReg> 456.0094

From the above table I see that the lowest MSE is 455.7123 when k = 45.

1. Calculate the test MSE for weighted KNN models with *K* = 5, 10, 15, …, 100 using the same combination of variables as before. Which model produces the lowest test MSE?
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ## # A tibble: 20 <U+00D7> 3
    ##        k        knn      mse
    ##    <dbl>     <list>    <dbl>
    ## 1      5 <S3: kknn> 546.1298
    ## 2     10 <S3: kknn> 489.6123
    ## 3     15 <S3: kknn> 470.6933
    ## 4     20 <S3: kknn> 462.0196
    ## 5     25 <S3: kknn> 457.3337
    ## 6     30 <S3: kknn> 452.5241
    ## 7     35 <S3: kknn> 449.6870
    ## 8     40 <S3: kknn> 447.5103
    ## 9     45 <S3: kknn> 445.9645
    ## 10    50 <S3: kknn> 444.7194
    ## 11    55 <S3: kknn> 443.4669
    ## 12    60 <S3: kknn> 442.4137
    ## 13    65 <S3: kknn> 441.4353
    ## 14    70 <S3: kknn> 440.5221
    ## 15    75 <S3: kknn> 439.6563
    ## 16    80 <S3: kknn> 439.1449
    ## 17    85 <S3: kknn> 438.5183
    ## 18    90 <S3: kknn> 438.0937
    ## 19    95 <S3: kknn> 437.6844
    ## 20   100 <S3: kknn> 437.3657

From the above table I see that the lowest MSE is 437.3657 when k = 100.

1. Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ## 
    ## Call:
    ## lm(formula = feminist ~ ., data = feminist_split_train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -66.834 -10.820  -2.065  11.484  52.825 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 52.381200   3.774772  13.877  < 2e-16 ***
    ## female       6.183232   1.205850   5.128 3.39e-07 ***
    ## age         -0.007772   0.035868  -0.217 0.828484    
    ## educ         0.172570   0.258506   0.668 0.504530    
    ## income      -0.147707   0.109635  -1.347 0.178137    
    ## dem          7.070944   1.306056   5.414 7.36e-08 ***
    ## rep         -6.005882   1.636477  -3.670 0.000253 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 20.64 on 1270 degrees of freedom
    ## Multiple R-squared:  0.0846, Adjusted R-squared:  0.08027 
    ## F-statistic: 19.56 on 6 and 1270 DF,  p-value: < 2.2e-16

    ## [1] 435.1107

    ## 
    ## Regression tree:
    ## tree(formula = feminist ~ ., data = feminist_split_train)
    ## Variables actually used in tree construction:
    ## [1] "dem"    "female"
    ## Number of terminal nodes:  4 
    ## Residual mean deviance:  430.8 = 548400 / 1273 
    ## Distribution of residuals:
    ##     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
    ## -66.4700 -10.8300  -0.8264   0.0000  13.1800  49.1700

    ## [1] 436.2068

    ## Distribution not specified, assuming gaussian ...

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/c-1.png)

    ##           var     rel.inf
    ## dem       dem 61.98851684
    ## female female 19.48180352
    ## rep       rep 18.44841881
    ## income income  0.08126084
    ## age       age  0.00000000
    ## educ     educ  0.00000000

    ## [1] 448.7998

    ##                 Length Class  Mode     
    ## call               4   -none- call     
    ## type               1   -none- character
    ## predicted       1277   -none- numeric  
    ## mse              500   -none- numeric  
    ## rsq              500   -none- numeric  
    ## oob.times       1277   -none- numeric  
    ## importance         6   -none- numeric  
    ## importanceSD       0   -none- NULL     
    ## localImportance    0   -none- NULL     
    ## proximity          0   -none- NULL     
    ## ntree              1   -none- numeric  
    ## mtry               1   -none- numeric  
    ## forest            11   -none- list     
    ## coefs              0   -none- NULL     
    ## y               1277   -none- numeric  
    ## test               0   -none- NULL     
    ## inbag              0   -none- NULL     
    ## terms              3   terms  call

    ## [1] 437.7946

The MSE of linear regression is 435.1107. The MSE of decision tree is 436.2068. The MSE of boosting is 448.7998. The MSE of random forest is 437.7946. Recall that the lowest MSE for weighted KNN models is 437.3657 when k = 100, and the lowest MSE for KNN models is 455.7123 when k = 45. I think linear regression performs the best with the lowest MSE of 436.2068, which suggests that there is a linear relationship between the predictors and the dependent variable. It could be that non-parametric methods overfitted the data.

Voter turnout and depression
============================

1. Split the data into a training and test set (70/30).
-------------------------------------------------------

1. Calculate the test error rate for KNN models with *K* = 1, 2, …, 10, using whatever combination of variables you see fit. Which model produces the lowest test MSE?
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

I used all possible predictors (i.e., `mhealth_sum,`age`,`educ`,`black`,`female`,`married`,`inc10`) to explain`vote96\`:

    ## # A tibble: 10 <U+00D7> 3
    ##        k          knn       err
    ##    <int>       <list>     <dbl>
    ## 1      1 <fctr [349]> 0.3237822
    ## 2      2 <fctr [349]> 0.3667622
    ## 3      3 <fctr [349]> 0.3123209
    ## 4      4 <fctr [349]> 0.3237822
    ## 5      5 <fctr [349]> 0.3323782
    ## 6      6 <fctr [349]> 0.3209169
    ## 7      7 <fctr [349]> 0.3266476
    ## 8      8 <fctr [349]> 0.3123209
    ## 9      9 <fctr [349]> 0.3037249
    ## 10    10 <fctr [349]> 0.3008596

From the above table I see that the lowest error rate is 0.3008596 when k = 10.

1. Calculate the test error rate for weighted KNN models with *K* = 1, 2, …, 10 using the same combination of variables as before. Which model produces the lowest test error rate?
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ## # A tibble: 10 <U+00D7> 3
    ##        k        knn     err
    ##    <int>     <list>   <dbl>
    ## 1      1 <S3: kknn> 0.35817
    ## 2      2 <S3: kknn> 0.59026
    ## 3      3 <S3: kknn> 0.68481
    ## 4      4 <S3: kknn> 0.73066
    ## 5      5 <S3: kknn> 0.77937
    ## 6      6 <S3: kknn> 0.79370
    ## 7      7 <S3: kknn> 0.82235
    ## 8      8 <S3: kknn> 0.83954
    ## 9      9 <S3: kknn> 0.87106
    ## 10    10 <S3: kknn> 0.89112

From the above table I see that the lowest error rate is 0.35817 when k = 1.

1. Compare the test error rate for the best KNN/wKNN model(s) to the test error rate for the equivalent logistic regression, decision tree, boosting, random forest, and SVM methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ## 
    ## Call:
    ## glm(formula = vote96 ~ ., family = binomial, data = mentalFactor_train)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -2.521  -0.989   0.475   0.822   2.235  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -4.82741    0.63055   -7.66  1.9e-14 ***
    ## mhealth_sum -0.09317    0.02828   -3.29  0.00099 ***
    ## age          0.04541    0.00591    7.68  1.6e-14 ***
    ## educ         0.25355    0.03646    6.95  3.6e-12 ***
    ## black1       0.41331    0.24918    1.66  0.09719 .  
    ## female1     -0.03865    0.17081   -0.23  0.82098    
    ## married1     0.26086    0.18591    1.40  0.16056    
    ## inc10        0.10670    0.03500    3.05  0.00230 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1027.35  on 815  degrees of freedom
    ## Residual deviance:  842.28  on 808  degrees of freedom
    ## AIC: 858.3
    ## 
    ## Number of Fisher Scoring iterations: 5

    ## [1] 0.27221

    ## 
    ## Classification tree:
    ## tree(formula = vote96 ~ ., data = mentalFactor_train)
    ## Variables actually used in tree construction:
    ## [1] "age"         "educ"        "mhealth_sum" "married"     "inc10"      
    ## Number of terminal nodes:  8 
    ## Residual mean deviance:  1.02 = 828 / 808 
    ## Misclassification error rate: 0.27 = 220 / 816

    ## [1] 0.34384

    ## Distribution not specified, assuming bernoulli ...

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/compare-1.png)

    ##                     var rel.inf
    ## age                 age 44.8100
    ## mhealth_sum mhealth_sum 27.8798
    ## educ               educ 25.6822
    ## inc10             inc10  1.6281
    ## black             black  0.0000
    ## female           female  0.0000
    ## married         married  0.0000

    ## [1] 0.30946

    ##                 Length Class  Mode     
    ## call               4   -none- call     
    ## type               1   -none- character
    ## predicted        816   factor numeric  
    ## err.rate        1500   -none- numeric  
    ## confusion          6   -none- numeric  
    ## votes           1632   matrix numeric  
    ## oob.times        816   -none- numeric  
    ## classes            2   -none- character
    ## importance         7   -none- numeric  
    ## importanceSD       0   -none- NULL     
    ## localImportance    0   -none- NULL     
    ## proximity          0   -none- NULL     
    ## ntree              1   -none- numeric  
    ## mtry               1   -none- numeric  
    ## forest            14   -none- list     
    ## y                816   factor numeric  
    ## test               0   -none- NULL     
    ## inbag              0   -none- NULL     
    ## terms              3   terms  call

    ## [1] 0.3553

    ## 
    ## Call:
    ## svm(formula = vote96 ~ ., data = mentalFactor_train, kernel = "linear", 
    ##     cost = 5)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  linear 
    ##        cost:  5 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  495
    ## 
    ##  ( 249 246 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

    ## [1] 0.29513

    ## 
    ## Call:
    ## svm(formula = vote96 ~ ., data = mentalFactor_train, kernel = "polynomial", 
    ##     cost = 5)
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
    ## Number of Support Vectors:  482
    ## 
    ##  ( 251 231 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

    ## [1] 0.28653

    ## 
    ## Call:
    ## svm(formula = vote96 ~ ., data = mentalFactor_train, kernel = "radial", 
    ##     cost = 5)
    ## 
    ## 
    ## Parameters:
    ##    SVM-Type:  C-classification 
    ##  SVM-Kernel:  radial 
    ##        cost:  5 
    ##       gamma:  0.125 
    ## 
    ## Number of Support Vectors:  476
    ## 
    ##  ( 247 229 )
    ## 
    ## 
    ## Number of Classes:  2 
    ## 
    ## Levels: 
    ##  0 1

    ## [1] 0.30372

The MSE of logistic regression is 0.27221. The MSE of decision tree is 0.34384. The MSE of boosting is 0.30946. The MSE of random forest is 0.3553. The MSE of SVM with a linear kernel is 0.29513. The MSE of SVM with a polynomial kernel is 0.28653. The MSE of SVM with a radial kernel is 0.30372. Recall that the lowest test error rate for weighted KNN models is 0.35817 when k = 1, and the lowest test error rate for KNN models is 0.3008596 when k = 10. I think logistic regression performs the best with the lowest error rate of 0.2722063, which suggests that there is a logistic relationship between the predictors and the dependent variable. Again, it could be that non-parametric methods overfitted the data.

Colleges
========

Perform PCA analysis on the college dataset and plot the first two principal components. Describe the results. What variables appear strongly correlated on the first principal component? What about the second principal component?

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/college_pca-1.png)

    ##     Private        Apps      Accept      Enroll   Top10perc   Top25perc 
    ## -0.08900986 -0.19963015 -0.15379708 -0.11779674 -0.36034940 -0.34475068 
    ## F.Undergrad P.Undergrad    Outstate  Room.Board       Books    Personal 
    ## -0.09408770  0.01748305 -0.32766424 -0.26653375 -0.05718904  0.07190001 
    ##         PhD    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ## -0.30325418 -0.30386831  0.21026024 -0.23665864 -0.33301113 -0.27308629

    ##     Private        Apps      Accept      Enroll   Top10perc   Top25perc 
    ##  0.34587868 -0.34362075 -0.37255665 -0.39969665  0.01623782 -0.01772991 
    ## F.Undergrad P.Undergrad    Outstate  Room.Board       Books    Personal 
    ## -0.41073159 -0.29306437  0.19151794  0.09397936 -0.05733827 -0.19275549 
    ##         PhD    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ## -0.11619109 -0.10419229 -0.20439519  0.19406065  0.07029054  0.11783035

The biplot visualizes the relationship between the first two principal components for the dataset, including both the scores and the loading vectors. The first principal component (the herizontal axis) places approximately equal weight on `Top10perc`, `Top25perc`, `Outstate,`PhD`,`Terminal`, and`Expend`. We can tell this because these vectors??? length on the first principal component dimension are roughly the same, whereas the length for other vectors is smaller.`Top10perc`,`Top25perc`,` Outstate, `PhD`, `Terminal`, and `Expend` appear to strongly correlate among one and another on the first principal component. Conversely, the second principal component (the vertical axis) places more emphasis on `Private`, `Apps`, `Accept`, `Enroll`, and `F.Undergrad`. `Private`, `Apps`, `Accept`, `Enroll`, and `F.Undergrad` appear to strongly correlate among one and another on the second principal component. We can also interpret the plot for idividual colleges based on their positions along the two dimensions. Colleges with large positive values on the first principal component have high `Top10perc`, `Top25perc`, `Outstate,`PhD`,`Terminal`, and`Expend`, while colleges with large negative values have low`Top10perc`,`Top25perc`,` Outstate, `PhD`, `Terminal`, and `Expend`; colleges with large positive values on the second principal component have high levels of `Private`, `Apps`, `Accept`, `Enroll`, and `F.Undergrad` while colleges with large negative values have low levels of `Private`, `Apps`, `Accept`, `Enroll`, and `F.Undergrad`.

Clustering states
=================

1. Perform PCA on the dataset and plot the observations on the first and second principal components.
-----------------------------------------------------------------------------------------------------

    ##                 PC1        PC2        PC3         PC4
    ## Murder   -0.5358995  0.4181809 -0.3412327  0.64922780
    ## Assault  -0.5831836  0.1879856 -0.2681484 -0.74340748
    ## UrbanPop -0.2781909 -0.8728062 -0.3780158  0.13387773
    ## Rape     -0.5434321 -0.1673186  0.8177779  0.08902432

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_pca-1.png)

The principal component score vectors have length n=50 (i.e.,50 states) and the principal component loading vectors have length p=4. The biplot visualizes the relationship between the first two principal components for the dataset, including both the scores and the loading vectors. The first principal component (the horizontal axis) places approximately equal weight on `murder`, `assault`, and `rape`. We can tell this because these vectors??? length on the first principal component dimension are roughly the same, whereas the length for urban population is smaller. Conversely, the second principal component (the vertical axis) places more emphasis on `urban population`. We can also interpret the plot for individual states based on their positions along the two dimensions. States with large positive values on the first principal component have high crime rates while states with large negative values have low crime rates; states with large positive values on the second principal component have high levels of urbanization while states with large negative values have low levels of urbanization.

1. Perform *K*-means clustering with *K* = 2. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc2-1.png)![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc2-2.png)

From this plot, we can see that the two clusterings are separate by PC1 (i.e., `Rape`, `Murder`, and `Assault`). The first clustering featured states that are mostly in the southwest. The second clustering featured states that are mostly in the northeast. The states in the first clustering with generally lower rates of `Rape`, `Murder`, and `Assault`, comparing to those in the second clustering.

1. Perform *K*-means clustering with *K* = 4. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc4-1.png)![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc4-2.png)

From this plot, we can clearly four clusterings and they appear to be separate by PC1, which are `Rape`, `Murder`, and `Assault`.

1. Perform *K*-means clustering with *K* = 3. Plot the observations on the first and second principal components and color-code each state based on their cluster membership. Describe your results.
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc3-1.png)![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc3-2.png)

From this plot, we can three clusterings and they appear to be separate by PC1, which are `Rape`, `Murder`, and `Assault`.

1. Perform *K*-means clustering with *K* = 3 on the first two principal components score vectors, rather than the raw data. Describe your results and compare them to the clustering results with *K* = 3 based on the raw data.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc3_vector-1.png)![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/states_kmc3_vector-2.png)

Performing *K*-means clustering with *K* = 3 on the first two principal components score vectors, I see that the three clusterings becomes more distinct and less overlapped among one and another. In addition to PC1, PC2 also seems to contribute to separate the three clusterings.

1. Using hierarchical clustering with complete linkage and Euclidean distance, cluster the states.
--------------------------------------------------------------------------------------------------

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/hierarchical_clustering-1.png)

1. Cut the dendrogram at a height that results in three distinct clusters. Which states belong to which clusters?
-----------------------------------------------------------------------------------------------------------------

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/cut-1.png)

    ## List of 2
    ##  $ axis.text.x    : list()
    ##   ..- attr(*, "class")= chr [1:2] "element_blank" "element"
    ##  $ legend.position: chr "none"
    ##  - attr(*, "class")= chr [1:2] "theme" "gg"
    ##  - attr(*, "complete")= logi FALSE
    ##  - attr(*, "validate")= logi TRUE

I see that Florida, North Carolina, Delaware, Alabama, Louisiana, Alaska, Mississippi, South Carolina, Maryland, Arizona, New Mexico, California, Illinois, New York, Michigan, and Nevada belong to the first cluster, which generally has high crime rates; Missouri, Arkansas, Tennessee, Georgia, Colorado, Texas, Rhode Island, Wyoming, Oregon, Oklahoma, Virginia, Washington, Massachusetts, and New Jersey belong to the second cluster, which generally has lower crime rates; and Ohio, Utah, Connecticut, Pennsylvania, Nebraska, Kentucky, Montana, Idaho, Indiana, Kansas, Hawaii, Minnesota, Wisconsin, Iowa, New Hampshire, West Virginia, Maine, South Dakota, North Dakota, and Vermont belong to the third cluster, which generally has the lowest crime rates.

1. Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation 1. What effect does scaling the variables have on the hierarchical clustering obtained? In your opinion, should the variables be scaled before the inter-observation dissimilarities are computed? Provide a justification for your answer.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

![](WX_ps9-nonparametric-unsupervised_files/figure-markdown_github/hierarchical_scaling-1.png)

    ## List of 2
    ##  $ axis.text.x    : list()
    ##   ..- attr(*, "class")= chr [1:2] "element_blank" "element"
    ##  $ legend.position: chr "none"
    ##  - attr(*, "class")= chr [1:2] "theme" "gg"
    ##  - attr(*, "complete")= logi FALSE
    ##  - attr(*, "validate")= logi TRUE

Scaling the variables made uplifted `murder` and `rape`, so that the height cut here (i.e., 4.41) can be much lower than the one in without scaling case (i.e., 150). Also, the tree looks has more branches in the lower level after scaling. I think the variables should be scaled before the inter-observation dissimilarities are computed. Scaling the variables to have standard deviation of 1 makes each variable weight equally in the hierarchical clustering. However, we saw after PCA that `murder`, `rape`, and `assault` belong to PCA1 while `urbanPop` is PCA2, so these four variables should not be weighted equally for hierarchical clustering analysis, and we see that scaling makes the structure of the lower level of the tree more complicated and hard to interpret.
