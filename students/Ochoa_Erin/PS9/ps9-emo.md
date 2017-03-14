MACS 30100 PS9
================
Erin M. Ochoa
2017 March 13

-   [Attitudes towards feminists \[3 points\]](#attitudes-towards-feminists-3-points)
-   [Voter turnout and depression \[2 points\]](#voter-turnout-and-depression-2-points)
-   [Colleges](#colleges)
-   [Clustering states \[3 points\]](#clustering-states-3-points)

``` r
mse = function(model, data) {
  x = modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}
```

We read in the data:

``` r
df = read.csv('data/feminist.csv')
```

Next, we split the dataset into training and validation sets in a ratio of 7:3:

``` r
set.seed(1234)

fem_split7030 = resample_partition(df, c(test = 0.3, train = 0.7))
fem_train70 = fem_split7030$train %>%
              tbl_df()
fem_test30 = fem_split7030$test %>%
             tbl_df()
```

Attitudes towards feminists \[3 points\]
========================================

![](https://tbmwomenintheworld2016.files.wordpress.com/2016/11/rtx2pdge.jpg?w=800)

`feminist.csv` contains a selection of variables from the [2008 American National Election Studies survey](http://www.electionstudies.org/) that allow you to test competing factors that may influence attitudes towards feminists. The variables are coded as follows:

-   `feminist` - feeling thermometer ranging from 0-100[1]
-   `female` - 1 if respondent is female, 0 if respondent is male
-   `age` - age of respondent in years
-   `dem` - 1 if respondent is a Democrat, 0 otherwise
-   `rep` - 1 if respondent is a Republican, 0 otherwise
-   `educ` - number of years of formal education completed by respondent
    -   `17` - 17+ years (aka first year of graduate school and up)
-   `income` - ordinal variable indicating respondent's income

        1. A. None or less than $2,999
        2. B. $3,000 -$4,999
        3. C. $5,000 -$7,499
        4. D. $7,500 -$9,999
        5. E. $10,000 -$10,999
        6. F. $11,000-$12,499
        7. G. $12,500-$14,999
        8. H. $15,000-$16,999
        9. J. $17,000-$19,999
        10. K. $20,000-$21,999
        11. M. $22,000-$24,999
        12. N. $25,000-$29,999
        13. P. $30,000-$34,999
        14. Q. $35,000-$39,999
        15. R. $40,000-$44,999
        16. S. $45,000-$49,999
        17. T. $50,000-$59,999
        18. U. $60,000-$74,999
        19. V. $75,000-$89,999
        20. W. $90,000-$99,999
        21. X. $100,000-$109,999
        22. Y. $110,000-$119,999
        23. Z. $120,000-$134,999
        24. AA. $135,000-$149,999
        25. BB. $150,000 and over

Estimate a series of models explaining/predicting attitudes towards feminists.

1.  Split the data into a training and test set (70/30%).
2.  Calculate the test MSE for KNN models with *K* = 5, 10, 15, …, 100, using whatever combination of variables you see fit. Which model produces the lowest test MSE?
3.  Calculate the test MSE for weighted KNN models with *K* = 5, 10, 15, …, 100 using the same combination of variables as before. Which model produces the lowest test MSE?
4.  Compare the test MSE for the best KNN/wKNN model(s) to the test MSE for the equivalent linear regression, decision tree, boosting, and random forest methods using the same combination of variables as before. Which performs the best? Why do you think this method performed the best, given your knowledge of how it works?

Voter turnout and depression \[2 points\]
=========================================

The 1998 General Social Survey included several questions about the respondent's mental health. `mental_health.csv` reports several important variables from this survey.

-   `vote96` - 1 if the respondent voted in the 1996 presidential election, 0 otherwise
-   `mhealth_sum` - index variable which assesses the respondent's mental health, ranging from 0 (an individual with no depressed mood) to 9 (an individual with the most severe depressed mood)[2]
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

``` r
college = College
college$Private = as.numeric(college$Private)
```

``` r
college_pca = prcomp(college, scale = TRUE)
college_pca$rotation
```

    ##                 PC1     PC2      PC3     PC4     PC5      PC6       PC7
    ## Private     -0.0890  0.3459 -0.15139 -0.2311  0.0443 -0.03145  0.178345
    ## Apps        -0.1996 -0.3436 -0.00482 -0.3050 -0.0230 -0.00862 -0.061047
    ## Accept      -0.1538 -0.3726  0.02535 -0.3175  0.0314  0.01307 -0.015474
    ## Enroll      -0.1178 -0.3997  0.02758 -0.2048 -0.0657 -0.04306 -0.008237
    ## Top10perc   -0.3603  0.0162 -0.00468  0.1075 -0.3882 -0.05992 -0.144406
    ## Top25perc   -0.3448 -0.0177  0.05895  0.1463 -0.4098  0.02061 -0.079889
    ## F.Undergrad -0.0941 -0.4107  0.01681 -0.1430 -0.0466 -0.04574  0.000282
    ## P.Undergrad  0.0175 -0.2931 -0.14937  0.0978  0.3280 -0.19678  0.162965
    ## Outstate    -0.3277  0.1915 -0.06636 -0.1288  0.2033 -0.02030  0.094948
    ## Room.Board  -0.2665  0.0940 -0.18211 -0.1824  0.5263  0.18303  0.145142
    ## Books       -0.0572 -0.0573 -0.66231  0.0893 -0.1586  0.64831 -0.141371
    ## Personal     0.0719 -0.1928 -0.46956  0.2906 -0.1880 -0.34706  0.609167
    ## PhD         -0.3033 -0.1162  0.20849  0.4646  0.2059  0.07189  0.031347
    ## Terminal    -0.3039 -0.1042  0.14672  0.4604  0.2656  0.13832  0.003847
    ## S.F.Ratio    0.2103 -0.2044  0.29225  0.0749 -0.0515  0.46937  0.275797
    ## perc.alumni -0.2367  0.1941  0.15944 -0.0100 -0.2151 -0.05166  0.286361
    ## Expend      -0.3330  0.0703 -0.21732 -0.0072  0.0571 -0.28447 -0.280168
    ## Grad.Rate   -0.2731  0.1178  0.17262 -0.2682 -0.1412  0.22185  0.501653
    ##                  PC8     PC9    PC10    PC11     PC12    PC13      PC14
    ## Private     -0.03231  0.0850 -0.2596  0.6798 -0.24569  0.4006 -1.01e-02
    ## Apps         0.09918 -0.0841 -0.0581 -0.0104 -0.04181  0.0374  5.95e-01
    ## Accept       0.05423 -0.1653 -0.0895  0.1426 -0.00455 -0.0736  2.93e-01
    ## Enroll      -0.06326 -0.1168 -0.0741  0.1058  0.03426  0.0591 -4.46e-01
    ## Top10perc    0.09714  0.3557 -0.0594 -0.0221  0.01518  0.0444  9.19e-05
    ## Top25perc    0.07806  0.4224 -0.0436  0.1382  0.23123 -0.0998  2.38e-02
    ## F.Undergrad -0.08393 -0.0498 -0.0504  0.0784  0.06250  0.0697 -5.25e-01
    ## P.Undergrad -0.56989  0.5432  0.2295  0.0451 -0.12114 -0.0228  1.26e-01
    ## Outstate     0.00412 -0.0114 -0.1973  0.0643 -0.25544 -0.8037 -1.28e-01
    ## Room.Board   0.24867  0.2442 -0.1899 -0.2808  0.48583  0.2014 -7.23e-02
    ## Books       -0.22503 -0.1333  0.0758  0.0204 -0.03719 -0.0249  1.18e-02
    ## Personal     0.30143 -0.1216 -0.1169 -0.0468  0.02038 -0.0379  4.01e-02
    ## PhD          0.07055 -0.1629  0.0560  0.1705 -0.09081  0.1172  1.25e-01
    ## Terminal     0.00463 -0.2332  0.0212  0.1784  0.01067  0.0544 -5.91e-02
    ## S.F.Ratio    0.09450  0.2845 -0.4477 -0.2162 -0.42475  0.0584 -1.92e-02
    ## perc.alumni -0.64039 -0.2905 -0.3463 -0.2600  0.21823  0.0916  1.03e-01
    ## Expend       0.03757 -0.0562 -0.0330 -0.4550 -0.54829  0.2993 -9.86e-02
    ## Grad.Rate    0.07773 -0.0226  0.6600 -0.1031 -0.15593  0.0729 -7.00e-02
    ##                PC15     PC16      PC17      PC18
    ## Private     -0.0232 -0.00537 -0.044796  0.007795
    ## Apps        -0.0807  0.13375 -0.458603  0.363283
    ## Accept      -0.0333 -0.14635  0.512188 -0.547462
    ## Enroll       0.0815  0.02848  0.403907  0.607174
    ## Top10perc    0.1062  0.69742  0.148018 -0.146308
    ## Top25perc   -0.1487 -0.61690 -0.050194  0.080606
    ## F.Undergrad  0.0534  0.00984 -0.569725 -0.408938
    ## P.Undergrad -0.0196  0.02068  0.050648  0.008963
    ## Outstate     0.0572  0.04195 -0.078622  0.048023
    ## Room.Board   0.0560  0.00351  0.028890  0.000368
    ## Books        0.0677 -0.00927 -0.001728  0.000603
    ## Personal    -0.0266 -0.00304  0.012911 -0.001211
    ## PhD          0.6864 -0.11269 -0.035842  0.015128
    ## Terminal    -0.6746  0.15786  0.020277  0.007143
    ## S.F.Ratio   -0.0449 -0.02172  0.014827 -0.001297
    ## perc.alumni  0.0268 -0.00815 -0.000483 -0.019705
    ## Expend      -0.0809 -0.22876  0.038317 -0.034863
    ## Grad.Rate   -0.0372 -0.00327  0.006996 -0.013507

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

[1] Feeling thermometers are a common metric in survey research used to gauge attitudes or feelings of warmth towards individuals and institutions. They range from 0-100, with 0 indicating extreme coldness and 100 indicating extreme warmth.

[2] The variable is an index which combines responses to four different questions: "In the past 30 days, how often did you feel: 1) so sad nothing could cheer you up, 2) hopeless, 3) that everything was an effort, and 4) worthless?" Valid responses are none of the time, a little of the time, some of the time, most of the time, and all of the time.
