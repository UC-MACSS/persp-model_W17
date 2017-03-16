MACSS 30100 PS\#9
================
Alice Mee Seon Chung
3/10/2017

Attitudes towards feminists
===========================

![](ps9_files/figure-markdown_github/1-2-1.png)

1.  For KNN for Feminist graph, we can see that as K become larger, the test MSE decreases overall. When KNN models with K = 45 and 100, we get the lowest test MSE, 456.

![](ps9_files/figure-markdown_github/1-3,%20weighted%20KNN-1.png)

1.  For Weighted KNN models, the trend as similart as previous question, that is as K become larger, the test MSE decreases overall. The weighted KNN models with K = 85 and 100, the test MSE is the lowest, 455.

<!-- -->

    ## # A tibble: 1 × 5
    ##   measures `linear regression` `decision tree` boosting `random forest`
    ##      <chr>               <dbl>           <dbl>    <dbl>           <dbl>
    ## 1      MSE                 451             452      455             450

1.  Comparing to the MSE for the best KNN/ wKNN models to the test MSE for the equivalent models, the test MSE of random forest model is lowest, 450 and it performs the best. However comparing to other models, the differences are still small so it is hard to say explicitely this model is the best model. In this case, randome forest model performs the best. I think this is because random forests intentionally ignores a random set of variables so it reduces the probability that single dominant predictor in the dataset affects the results. So the variable restriction imposed when random forest approach considers splits. So The work process of random forest makes the method performed the best.

Voter turnout and depression
============================

![](ps9_files/figure-markdown_github/2-2-1.png)

1.  Above graph of test error rate with KNN for different numbers of K, the test error graph draws big differences between at first 5 intervals, but overall we can say that test error decreases as K increses. KNN model with 8 produces the lowest test error rate and the error rate is 0.295.

![](ps9_files/figure-markdown_github/2-3-1.png)

1.  Above graph of test error rate with Weighted KNN for different numbers of K, the test error graph shows that test error rate decreases as K increses. Weighted KNN model with 10 produces the lowest test error rate and the error rate is 0.278.

<!-- -->

    ## Distribution not specified, assuming bernoulli ...

    ## # A tibble: 1 × 6
    ##     measures `logistic regression` `decision tree` boosting
    ##        <chr>                 <dbl>           <dbl>    <dbl>
    ## 1 Error rate                 0.272           0.304    0.298
    ## # ... with 2 more variables: `random forest` <dbl>, svm <dbl>

    ## # A tibble: 1 × 6
    ##     measures `logistic regression` `decision tree` boosting
    ##        <chr>                 <dbl>           <dbl>    <dbl>
    ## 1 Error rate                 0.272           0.304    0.298
    ## # ... with 2 more variables: `random forest` <dbl>, svm <dbl>

1.  Comparing to the MSE for the best KNN/ wKNN models to the test error rate for the equivalent models, the test error rate of logistic regression model is lowest, 0.272 and it performs the best. However comparing to other models, the differences are still small so it is hard to say explicitely this model is the best model. In this case, logistic regression model performs the best because logistics models the probability that the response belongs to a particular category and here vote96 is binary variables it is more suitable to predict using logistic regression model and also logistics regression performs well when the relationship is curvilinear and through last problem sets we already observed that the predictors and the reponse have curvilinear relationship.

3 College
=========

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

![](ps9_files/figure-markdown_github/3-1-1.png)

The bi-plot shows that the variable Private locates high in the second principle component. It is hard to see the plot so we have to see princial component more closely.

    ## [1] "First Principal Component"

    ##     Private        Apps      Accept      Enroll   Top10perc   Top25perc 
    ##     -0.0890     -0.1996     -0.1538     -0.1178     -0.3603     -0.3448 
    ## F.Undergrad P.Undergrad    Outstate  Room.Board       Books    Personal 
    ##     -0.0941      0.0175     -0.3277     -0.2665     -0.0572      0.0719 
    ##         PhD    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ##     -0.3033     -0.3039      0.2103     -0.2367     -0.3330     -0.2731

    ## [1] "Second Principal Component"

    ##     Private        Apps      Accept      Enroll   Top10perc   Top25perc 
    ##      0.3459     -0.3436     -0.3726     -0.3997      0.0162     -0.0177 
    ## F.Undergrad P.Undergrad    Outstate  Room.Board       Books    Personal 
    ##     -0.4107     -0.2931      0.1915      0.0940     -0.0573     -0.1928 
    ##         PhD    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ##     -0.1162     -0.1042     -0.2044      0.1941      0.0703      0.1178

For first principal components table, the variables with high figure are Top10perc, Top25perc, PhD, Terminal, Expend and all these are more than 0.3 level. These vatiabels are seems to locate upeer-right. So it seems that percent of faculty with Ph.D's, percent of faculty with terminal degress, instructuinal expenditure per student, percent of new students from top 10% or 25% of H.S Class are strongly correlated. When we look at the second principal components table, the variabels with high fugure are Apps, Accept, Enroll, F.Undergrad, P.Undergrad and all these variables are lower than -0.29 level. It seems that number of application received, number of application accepted, number of new students enrolled, number of full-time undergraduates and number of parttime undergraduates are stronly correlated. For first components table seems like put more emphasis on the aspects of expenditure of students and performance of students. Second components table seems like put emphasis on population size of colleges because if the population of college is large, than the number of applications received, accpeted, and enrolled full-time or part time studnets would be larger too.

Clustering states
=================

    ##             PC1    PC2    PC3    PC4
    ## Murder   -0.536  0.418 -0.341  0.649
    ## Assault  -0.583  0.188 -0.268 -0.743
    ## UrbanPop -0.278 -0.873 -0.378  0.134
    ## Rape     -0.543 -0.167  0.818  0.089

![](ps9_files/figure-markdown_github/4-1-1.png)

    ## [1] "First Principal Component"

    ##   Murder  Assault UrbanPop     Rape 
    ##   -0.536   -0.583   -0.278   -0.543

    ## [1] "Second Principal Component"

    ##   Murder  Assault UrbanPop     Rape 
    ##    0.418    0.188   -0.873   -0.167

1.  The first loading vector has the same size on Muder, Assault and Rape. For UnrbanPop, it is mush longer than other three and it means it lightly weighted. Also in the second principal components, UrbanPop has most weights on and the other thress variables hace relatively small weights on. From these fact, we can see that three variables(Murder, Assault, Rape) are crime-related and they are closely correlated. Thus UrbanPop is not related with Murder so its correlation is not significant.

![](ps9_files/figure-markdown_github/4-2-1.png)

1.  K-mean clustering with K=2 graph shows that it divides all states into two group regarding the level of PC1 socres. PC1 scores are related with crime-related variables, so we can say that right side group which includes Maine, West Virginia, Nebraska and so on is safe than left side group which includes Florida, California, Nevada and so on.

![](ps9_files/figure-markdown_github/4-3-1.png)

1.  K-mean clustering with K=4 graph shows that it divides all states into four group regarding level of PC1 socres. PC1 scores are related with crime-related variables, so we can say that first group with Florida, Nevada and California is the least safe cluster group, and the group with Georgia, Alabama, Texas is the third safe group, the group woth Idaho, Virginia , Wyoming is the second safe group and lastrly the right side group with South Dakota, Main, Wiscosin is the safest cluster group.

![](ps9_files/figure-markdown_github/4-4-1.png) 4. K-mean clustering with K=4 graph shows that it divides all states into three group regarding level of PC1 socres. PC1 scores are related with crime-related variables, so we can say that first group with Florida, Nevada and California is the least safe cluster group, and second group with Georgia, Colorado, Texas is the second safe group, the third group woth Indiana, Ohio , Wisconsin is the safest cluster group.

![](ps9_files/figure-markdown_github/4-5-1.png) 5. K-means clustering with K= 3 on the first two principal components score vectors divides all states into three group regarding PC1 and PC2. Unlike 4-4 question, this graphs shows differecnt trend. The first group with Florida, Nevada , California has low PC2 level and PC1 level and it means that those states are highly crime-related and urbanized. The group with South Carolina, Alabama, Alaska is less safe than first group and less urbanized. The last group with South Dakora, Connecticut, Washington is safe and urbanized. Compared to K-mean cluestering with K=3 with raw data, this clustering methods seems to consider PC2 level as well as PC1 level. ![](ps9_files/figure-markdown_github/4-6-1.png)

![](ps9_files/figure-markdown_github/4-7-1.png) 7. I put a cut-off at 120 and cut the dendrogram into three distinct clusters. First clustrs is unsafe group and Florida, South Carolina, Delaware, Alabama, Luisiana, Alaska, Misissippi, North Carolina, Maryland, New Mexico, California, Illinois, New York, Michigan and Nevada belong to this group. Second cluseter is more safe group and Missouri, Kansas, Tenessee, Georgia, Texas, Rode Island, Wyoming, Oregon, Alabama, Washington, Massachusetts, and New Jersey belong to this group. The last group is the safest group and Ohio, Utah, Conneticut, Pensylvania, Nebraska, Kentucky, Montana, Idaho, Indiana Kansas, Hawaii, Minessota, Wisconsin, Iowa, New Hapshire, West Virginia, Maine, South Dakota, North Dakora, Vermont belong to this group.

![](ps9_files/figure-markdown_github/4-8-1.png)

1.  After the scalling he variables to have standard deviation 1, Murder and Rape, and unrbanized factor have more weights on. So these variables used to have lower values and variance so its effects have more weights on at the same time. As a result, we can see that we obtained very similar to the cluster we input PC1, PC2. In this case I suggest that the variables be scaled before inter-observation dissimilarities are computed. Then each variables will have equal importace in the clustering and we can avoid over-weighting or under-weighting the variables in larger scale. In this dataset, Assault is twice in number than Murder so it can be over-fitted or exaggerated its effect compared to Murder in clustering because it has large variance. Other than just numbers, when we compare two variables, assault and murder, even if the number of murder is small and the number of assault is large, the meaning and effects of murder and assault can be generalized with the numbers.
