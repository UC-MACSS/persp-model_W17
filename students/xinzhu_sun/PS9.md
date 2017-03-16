PS9
================
Xinzhu Sun
3/15/2017

Attitudes towards feminists
===========================

1.Split the data
----------------

``` r
fem = read_csv('data/feminist.csv')
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
fem1<-fem %>%
mutate (dem = factor (dem, levels =0:1, labels = c("non-dem","dem")), 
        rep = factor (rep, levels =0:1, labels = c("non-rep", "redp")),
        inc = factor (income, levels = 1: 25, labels = c("0","3","5","7.5","10","11","12.5","15","17","20","22","25","30","35","40","45","50","60","75","90","100","110","120","135","150"))) %>%
mutate (inc=as.numeric(as.character(inc)))%>%
na.omit()

fem_split <- resample_partition(fem1, c(test = 0.3, train = 0.7))
fem_train <- as_tibble(fem_split$train)
fem_test <- as_tibble(fem_split$test)
```

2.Calculate the test MSE for KNN models
---------------------------------------

I choose the `inc`, `educ`,and `female` as my combination.

``` r
mse <- function(model, data) {
  x <- modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}
mse_lm <- lm(feminist ~ educ + female + inc , data = fem_train) %>%
mse(.,fem_test)

mse_knn <- data_frame(k = seq(5, 100, by = 5),
                      knn = map(k, ~ knn.reg(select(fem_train, -age, -income, -dem, -rep), 
                      y = fem_train$feminist, test = 
                      select(fem_test, -age, -income, -dem, -rep), k = .)), 
                      mse = map_dbl(knn, ~ mean((fem_test$feminist - .$pred)^2))) 
ggplot(mse_knn, aes(k, mse)) +
  geom_line() +
  geom_point() +
  labs(title = "Test MSE for KNN models",
       x = "K",
       y = "MSE") +
  expand_limits(y = 0)
```

<img src="PS9_files/figure-markdown_github/KNN-1.png" style="display: block; margin: auto;" />

``` r
knn_mse_fem<-min(mse_knn$mse)
```

In the KNN plot, we find as the K increases, the MSE increses, the larger the K, more likely the generated model overfitting acorss the training data, leading to the higher MSE. Thus, the lowest MSE is produced by k = 5.

3.Calculate the test MSE for weighted KNN models
------------------------------------------------

I choose the `inc`, `educ`,and `female` as before.

``` r
mse_knn_w <- data_frame(k = seq(5, 100, by = 5),
                        wknn = map(k, ~ kknn(feminist ~ educ + female + inc, 
                                             train = fem_train, test = fem_test, k = .)),
                        mse_wknn = map_dbl(wknn, ~ mean(
                          (fem_test$feminist - .$fitted.values)^2))) %>%
  left_join(mse_knn, by = "k") %>%
  mutate(mse_knn = mse)%>%
  select (k, mse_knn, mse_wknn) %>%
  gather(method,mse, -k) %>%
  mutate(method = str_replace(method, "mse_", ""))%>%
  mutate(method = factor (method, levels = c("knn","wknn"), 
                          labels = c("KNN","Weighted KNN"))) 
mse_knn_w %>%
  ggplot(aes(k, mse, color = method)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = mse_lm, linetype = 2) +
  labs(title = "Test MSE for KNN and weighted KNN models",
       x = "K",
       y = "MSE",
       method = NULL) +
  expand_limits(y = 0) +
  theme(legend.position = "bottom")
```

<img src="PS9_files/figure-markdown_github/weighted KNN-1.png" style="display: block; margin: auto;" /> In the KNN and weighed KNN plot, we find as the K increases, the MSE of weighted KNN decreses. Thus, the lowest MSE is produced by k = 100.

Voter turnout and depression
============================

1. Split the data
-----------------

``` r
set.seed(1234)
mh = read_csv('data/mental_health.csv')
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
mh_split <- resample_partition(mh_rm_na, c(test = 0.3, train = 0.7))
mh_train <- as_tibble(mh_split$train)
mh_test <- as_tibble(mh_split$test)
```

2. Calculate the test error rate for KNN models
-----------------------------------------------

I choose the `inc10`, `educ`, `mhealth_sum`,and `age` as my combination.

``` r
logit2prob <- function(x){
  exp(x) / (1 + exp(x))
}
mh_glm <- glm(vote96 ~ age + inc10 + mhealth_sum + educ, data = mh_train, family = binomial) 

# estimate the error rate for this model:
x<- mh_test %>%
  add_predictions(mh_glm) %>%
  mutate (pred = logit2prob(pred),
          prob = pred,
          pred = as.numeric(pred > 0.5))
err.rate.glm <-mean(x$vote96 != x$pred)

# estimate the MSE for KNN
mse_knn <- data_frame(k = 1:10,
                      knn_train = map(k, ~ class::knn(select(mh_train, -vote96),
                                                test = select(mh_train, -vote96),
                                                cl = mh_train$vote96, k = .)),
                      knn_test = map(k, ~ class::knn(select(mh_train, -vote96),
                                                test = select(mh_test, -vote96),
                                                cl = mh_train$vote96, k = .)),
                      mse_train = map_dbl(knn_train, ~ mean(mh_test$vote96 != .)),
                      mse_test = map_dbl(knn_test, ~ mean(mh_test$vote96 != .)))

ggplot(mse_knn, aes(k, mse_test)) +
  geom_line() +
  geom_hline(yintercept = err.rate.glm, linetype = 2) +
  labs(x = "K",
       y = "Test error rate",
       title = "Test error rate for KNN models") +
  expand_limits(y = 0)
```

<img src="PS9_files/figure-markdown_github/KNN 2-1.png" style="display: block; margin: auto;" />

``` r
hm_knn_mse<-min(mse_knn$mse_test)
```

The lowest MSE is produced by k = 10.

3.Calculate the test error rate for weighted KNN models
-------------------------------------------------------

I choose the `inc10`, `educ`, `mhealth_sum`,and `age` as before.

``` r
mse_wknn <- data_frame(k = 1:10,
                  wknn = map(k, ~ kknn(vote96 ~., train = mh_train, test = mh_test, k =.)),
                  mse_test_wknn = map_dbl(wknn, ~ mean(mh_test$vote96 != as.numeric(.$fitted.values > 0.5))))
mse_wknn_mh <- min(mse_wknn$ mse_test_wknn)

err<-mse_wknn %>%
  left_join(mse_knn, by = "k") %>%
  select(k, mse_test_wknn, mse_test) %>%
  gather(method,mse, -k) %>%
  mutate(method = factor(method, levels =c("mse_test_wknn","mse_test"), labels = c("Weighted KNN","KNN")))

err %>%
  ggplot(aes(k, mse, color = method)) +
  geom_line() +
  geom_point() +
  geom_hline(yintercept = err.rate.glm, linetype = 2) +
  labs(title = "Test error rate for weighted KNN models",
       x = "K",
       y = "Test error rate",
       method = NULL) +
  expand_limits(y = 0) +
  theme(legend.position = "bottom")
```

<img src="PS9_files/figure-markdown_github/weighted KNN 2-1.png" style="display: block; margin: auto;" /> The lowest MSE is produced by k = 5.

4.Compare the test MSE for the best KNN/wKNN model(s).
------------------------------------------------------

Colleges
========

``` r
college = read_csv('data/College.csv')
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
c <- college %>%
  mutate(Private = ifelse (Private =="Yes",1,0 ) )
pr.out <- prcomp(c, scale = TRUE)
biplot(pr.out, scale = 0, cex = .6)
```

<img src="PS9_files/figure-markdown_github/PCA-1.png" style="display: block; margin: auto;" />

``` r
pr.out <- prcomp(college[,2:18], scale = TRUE)
print('First Principal Component')
```

    ## [1] "First Principal Component"

``` r
pr.out$rotation[, 1]
```

    ##        Apps      Accept      Enroll   Top10perc   Top25perc F.Undergrad 
    ##     0.24877     0.20760     0.17630     0.35427     0.34400     0.15464 
    ## P.Undergrad    Outstate  Room.Board       Books    Personal         PhD 
    ##     0.02644     0.29474     0.24903     0.06476    -0.04253     0.31831 
    ##    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ##     0.31706    -0.17696     0.20508     0.31891     0.25232

``` r
print('Second Principal Component')
```

    ## [1] "Second Principal Component"

``` r
pr.out$rotation[, 2]
```

    ##        Apps      Accept      Enroll   Top10perc   Top25perc F.Undergrad 
    ##    -0.33160    -0.37212    -0.40372     0.08241     0.04478    -0.41767 
    ## P.Undergrad    Outstate  Room.Board       Books    Personal         PhD 
    ##    -0.31509     0.24964     0.13781    -0.05634    -0.21993    -0.05831 
    ##    Terminal   S.F.Ratio perc.alumni      Expend   Grad.Rate 
    ##    -0.04643    -0.24667     0.24660     0.13169     0.16924

Looking at the first principal component, the variables with the highest magnitude loadings are `PhD`, `Terminal`, `Top10perc`, `Top25perc`, `Outstate`, `Expend` and `Grad.Rate`. Thus, it seems that the percent of faculty with PhD's or with terminal degrees, percent of the student body in the top 25% or 10% of their high school class, the percent of the student body from out of state, the cost of the university, and the graduation rate of the university seem to move together, i.e. they are correlated.

Looking at the Second Principal Component, the variables with the highest magnitude loadings are `Private`, `Apps`, `Accept`, `Enroll`, `F.Undergrad`, and `P.Undergrad`. Thus, it seems that whether the university is private or not, the number of apps received, the number of new students accepted, the number of new students enrolled, the number of full-time undergraduates, and the percent of full-time undergraduates seem to move together, i.e. they are correlated.

Clustering states
=================

1. Perform PCA
--------------

``` r
usar = read_csv('data/USArrests.csv')
```

    ## Parsed with column specification:
    ## cols(
    ##   State = col_character(),
    ##   Murder = col_double(),
    ##   Assault = col_integer(),
    ##   UrbanPop = col_integer(),
    ##   Rape = col_double()
    ## )

``` r
pr.out <- prcomp(usar[, 2:5], scale = TRUE)
pr.out$rotation
```

    ##              PC1     PC2     PC3      PC4
    ## Murder   -0.5359  0.4182 -0.3412  0.64923
    ## Assault  -0.5832  0.1880 -0.2681 -0.74341
    ## UrbanPop -0.2782 -0.8728 -0.3780  0.13388
    ## Rape     -0.5434 -0.1673  0.8178  0.08902

``` r
biplot(pr.out, scale = 0, cex = .6,  xlabs= usar$State)
```

<img src="PS9_files/figure-markdown_github/PCA 2-1.png" style="display: block; margin: auto;" />

``` r
print('First Principal Component')
```

    ## [1] "First Principal Component"

``` r
pr.out$rotation[, 1]
```

    ##   Murder  Assault UrbanPop     Rape 
    ##  -0.5359  -0.5832  -0.2782  -0.5434

``` r
print('Second Principal Component')
```

    ## [1] "Second Principal Component"

``` r
pr.out$rotation[, 2]
```

    ##   Murder  Assault UrbanPop     Rape 
    ##   0.4182   0.1880  -0.8728  -0.1673

2. Perform *K*-means clustering with *K* = 2
--------------------------------------------

``` r
set.seed(1234)
autoplot(kmeans(usar[, 2:5], 2), data = usar) +
  geom_text(vjust=-1, label=usar$State, size = 1.8) +
  labs(title = 'K-means Clustering with K=2')
```

<img src="PS9_files/figure-markdown_github/k-means clustering 2-1.png" style="display: block; margin: auto;" /> As shown in the plot above, states are classified into two distinct groups. It seems that this partition is according the first principle components. According to the PCA, we can know that the blue states with positive PC1 values, are those states with higher criminal rate, while those red states are those with lower criminal rate.

3. Perform *K*-means clustering with *K* = 4
--------------------------------------------

``` r
set.seed(1234)
autoplot(kmeans(usar[, 2:5], 4), data = usar) +
  geom_text(vjust=-1, label=usar$State, size = 1.8) +
  labs(title = 'K-means Clustering with K=4')
```

<img src="PS9_files/figure-markdown_github/k-means clustering 4-1.png" style="display: block; margin: auto;" /> As shown in the plot above, states are classified into 4 distinct subgroups. According to the first principle component, which emphasized the overall rates of serious crimes, this classification reflects the criminal rate from the lower to higher. In cluster 4, Vermont, North Dakota have the lowest criminal rate across all the 50 states, while in the cluster 2 Florida, Califonia have the highest criminal rates.

4. Perform *K*-means clustering with *K* = 3
--------------------------------------------

``` r
set.seed(1234)
autoplot(kmeans(usar[, 2:5], 3), data = usar) +
  geom_text(vjust=-1, label=usar$State, size = 1.8) +
  labs(title = 'K-means Clustering with K=3')
```

<img src="PS9_files/figure-markdown_github/k-means clustering 3-1.png" style="display: block; margin: auto;" /> As shown in the plot above, states are classified into 3 distinct subgroups. According to the first principle component, which emphasized the overall rates of serious crimes, this classification reflects the criminal rate from the lower to higher. In the cluster 3, Vermont, North Dakota have the lowest criminal rate across all the 50 states, while in the cluster 2 the states like Florida, Califonia have the highest criminal rates. In addition, the criminal rates of the states in cluster 1, like, New Jersey, Arkansas, and so on, are not so radical as the cluster 2 and 3.

5.Perform *K*-means clustering with *K* = 3 on the first two principal components score vectors.
------------------------------------------------------------------------------------------------

``` r
set.seed(1234)
autoplot(kmeans(usar[, 2:5], 3), data = usar, loadings = TRUE, loadings.colour = 'black') +
  geom_text(vjust=-1, label=usar$State, size = 1.8) +
  labs(title = 'K-means Clustering with K=3')
```

<img src="PS9_files/figure-markdown_github/k-means clustering 3 on score vectors-1.png" style="display: block; margin: auto;" />

``` r
p1<-data_frame(x1= usar$Murder, x2=usar$Assault)
p2<-data_frame(x1= usar$Murder, x2=usar$UrbanPop) 
p3<-data_frame(x1= usar$Murder, x2=usar$Rape)
p4<-data_frame(x1= usar$Assault, x2=usar$UrbanPop)
p5<-data_frame(x1= usar$Assault, x2=usar$Rape)
p6<-data_frame(x1= usar$UrbanPop, x2=usar$Rape)

p1.out<-p1 %>%
  mutate(k3 = kmeans (p1, 3, nstart = 20)[[1]]) %>%
  mutate(k3 = as.character(k3)) %>%
  ggplot (aes (x1, x2, color = k3))+
  geom_point()+
  theme_bw()+
  labs (x="Murder", y ="Assault")

p2.out<-p2 %>%
  mutate(k3 = kmeans (p2, 3, nstart = 20)[[1]]) %>%
  mutate(k3 = as.character(k3)) %>%
  ggplot (aes (x1, x2, color = k3))+
  geom_point()+
  theme_bw()+
  labs (x="Murder", y ="UrbanPop")

p3.out<-p3 %>%
  mutate(k3 = kmeans (p3, 3, nstart = 20)[[1]]) %>%
  mutate(k3 = as.character(k3)) %>%
  ggplot (aes (x1, x2, color = k3))+
  geom_point()+
  theme_bw()+
  labs (x="Murder", y ="Rape")

p4.out<-p4 %>%
  mutate(k3 = kmeans (p4, 3, nstart = 20)[[1]]) %>%
  mutate(k3 = as.character(k3)) %>%
  ggplot (aes (x1, x2, color = k3))+
  geom_point()+
  theme_bw()+
  labs (x="Assault", y ="UrbanPop")

p5.out<-p5 %>%
  mutate(k3 = kmeans (p5, 3, nstart = 20)[[1]])  %>%
  mutate(k3 = as.character(k3)) %>%
  ggplot (aes (x1, x2, color = k3))+
  geom_point()+
  theme_bw()+
  labs (x="Assault", y ="Rape")

p6.out<-p6 %>%
  mutate(k3 = kmeans (p6, 3, nstart = 20)[[1]])  %>%
  mutate(k3 = as.character(k3)) %>%
  ggplot (aes (x1, x2, color = k3))+
  geom_point()+
  theme_bw()+
  labs (x="UrbanPop", y ="Rape")

grid.arrange(p1.out,p2.out,p3.out,p4.out,p5.out,p6.out, ncol = 3, nrow = 2 )
```

<img src="PS9_files/figure-markdown_github/k-means clustering 3 on score vectors-2.png" style="display: block; margin: auto;" /> As shown above, if plotting the k-means clustering on the raw data, we need 6 combinations of each two variables from total 4 variables. Compred to the PCA approach, it is rather difficult to interpret the clusterings on 6 sub-plots. In addition, we do not know which combinations are really statistically significant, and therefore we do not know which clustering represents the major feature of this data structure. On the other hand, by performing the PCA, the dimension of the data has been reduced. It is rather easy and convient to interpret the data in the first two principle component vectors. As what we have done in the above analysis, we find the Assualt, Rape and Murder cand be viewd together as in the same component vector, while the UrbanPop as the second principle component, representing the unbarnization. This dimension reduction make our intepretation for the clustering much easier than that on the raw data.

6.Cluster the states
--------------------

``` r
set.seed(1234)
dd <- dist(usar[, 2:5], method = "euclidean")
hc <- hclust(dd, method = "complete")
hcdata <- dendro_data (hc)
hclabs <- label(hcdata) %>%
  left_join (data_frame (label = as.factor (seq.int(nrow(usar))),
                         cl = as.factor (usar$State)))
```

    ## Joining, by = "label"

``` r
ggdendrogram(hc, labels =FALSE) +
  geom_text(data = hclabs,
            aes(label = cl, x = x, y = 0),
            hjust = 0.5, vjust=-0.1, angle = 90, size = 2.0) +
  theme(axis.text.x = element_blank(),
        legend.position = "none") +
  labs(title = "Hierarchical clustering")
```

<img src="PS9_files/figure-markdown_github/hierarchical clustering-1.png" style="display: block; margin: auto;" />

7.Cut the dendrogram at a height that results in three distinct clusters
------------------------------------------------------------------------

``` r
set.seed(1234)
hclabs <- label(hcdata) %>%
  left_join (data_frame (label = as.factor (seq.int(nrow(usar))),
                         state = as.factor (usar$State),
                         cl = as.factor(cutree(hc, h = 150))))
```

    ## Joining, by = "label"

``` r
ggdendrogram(hc, labels =FALSE) +
  geom_text(data = hclabs,
            aes(label = state, x = x, y = 0, color = cl),
            hjust = 0.5, vjust=-0.1, angle = 90, size = 2.0) +
  theme(axis.text.x = element_blank(),
        legend.position = "none") +
  geom_hline(yintercept = 150, linetype = 2) +
  labs(title = "Hierarchical clustering with 3 distinct clusters")
```

<img src="PS9_files/figure-markdown_github/cut the dendrogram-1.png" style="display: block; margin: auto;" />

``` r
sum<- data_frame ( "group "= c("red","green","blue"),
                   "States" = c(paste ((hclabs %>% select (state, cl) %>% 
                                          filter (cl == 1))$state, collapse=', '),
                                paste ((hclabs %>% select (state, cl) %>% 
                                          filter (cl == 2))$state, collapse=', '),
                                paste ((hclabs %>% select (state, cl) %>% 
                                          filter (cl == 3))$state, collapse=', ')))
pander (sum)
```

<table style="width:53%;">
<colgroup>
<col width="11%" />
<col width="41%" />
</colgroup>
<thead>
<tr class="header">
<th align="center">group</th>
<th align="center">States</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="center">red</td>
<td align="center">Florida, North Carolina, Delaware, Alabama, Louisiana, Alaska, Mississippi, South Carolina, Maryland, Arizona, New Mexico, California, Illinois, New York, Michigan, Nevada</td>
</tr>
<tr class="even">
<td align="center">green</td>
<td align="center">Missouri, Arkansas, Tennessee, Georgia, Colorado, Texas, Rhode Island, Wyoming, Oregon, Oklahoma, Virginia, Washington, Massachusetts, New Jersey</td>
</tr>
<tr class="odd">
<td align="center">blue</td>
<td align="center">Ohio, Utah, Connecticut, Pennsylvania, Nebraska, Kentucky, Montana, Idaho, Indiana, Kansas, Hawaii, Minnesota, Wisconsin, Iowa, New Hampshire, West Virginia, Maine, South Dakota, North Dakota, Vermont</td>
</tr>
</tbody>
</table>

See the table which summarizes the states and their clusters.

8.Hierarchically cluster the states
-----------------------------------

``` r
set.seed(1234)
dd_scale <- dist(scale(usar[, 2:5]), method = "euclidean") 
hc_scale <- hclust(dd_scale, method = "complete")
hcdata2 <- dendro_data (hc_scale)
hclabs2 <- label(hcdata2) %>%
  left_join (data_frame (label = as.factor (seq.int(nrow(usar))),
                         state = as.factor (usar$State),
                         cl = as.factor(cutree(hc_scale , h = 4.41))))
```

    ## Joining, by = "label"

``` r
g2<-ggdendrogram(hc_scale, labels =FALSE) +
  geom_text(data = hclabs2,
            aes(label = state, x = x, y = 0, color = cl),
            hjust = 0.5, vjust=-0.1, angle = 90, size = 2.0) +
  theme(axis.text.x = element_blank(),
        legend.position = "none") +
  geom_hline(yintercept = 4.41, linetype = 2) +
  labs(title = "Hierarchical clustering After Scaling")

g1<-ggdendrogram(hc, labels =FALSE) +
  geom_text(data = hclabs,
            aes(label = state, x = x, y = 0, color = cl),
            hjust = 0.5, vjust=-0.1, angle = 90, size = 2.0) +
  theme(axis.text.x = element_blank(),
        legend.position = "none") +
  geom_hline(yintercept = 150, linetype = 2) +
  labs(title = "Hierarchical clustering")

grid.arrange(g1,g2, ncol = 2, nrow = 1 )
```

<img src="PS9_files/figure-markdown_github/hierarchical clustering 2-1.png" style="display: block; margin: auto;" /> Looking at the above two plots, scaling the variables has two noticeable effects. Firstly, the y-axis, the Euclidean distance from the complete linkage method, is much smaller with scaled variables. Secondly, some of the clusterings are different, Alaska merges with Mississippi and South Carolina without scaling the variables, but with Alabama, Louisiana, Georgia, Tennessee, North Carolina, Mississippi, and South Carolina when the variables are scaled. (Some clusterings stay the same though).

In my opinion, the variables should be scaled before inter-observation dissimilarities are computed. Unless the variables have the same standard deviation, those variables with larger and smaller standard deviations will have, respectively, exaggerated and diminished effects on the dissimilarity measure. For instance, if there are two variables, the first with a standard deviation of 1000, and the second with a standard deviation of 10, and under complete linkage the dissimilarity between a given two clusters is 200 with respect to the first variable, and 20, with respect to the second, in reality, the difference between the two clusters in terms of the first variable is actually quite small relative to the standard deviation of that variable, while the difference in terms of the second variable is quite large, twice the size of the standard deviation of that variable. However, without scaling, the dissimilarity contributed by the difference in the first variable will be much larger than that of the second, which does not reflect the reality of the closeness in the 1st variable, and the dissimilarity in the second variable! Under scaling, this issue would not occur, as dissimilarity is taken with respect to the standard deviation of each variable.
