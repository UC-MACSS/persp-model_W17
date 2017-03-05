MACS 30100 PS8
================
Erin M. Ochoa
2017 March 6

-   [Part 1: Joe Biden (redux times two) \[3 points\]](#part-1-joe-biden-redux-times-two-3-points)
-   [Part 2: Modeling voter turnout \[3 points\]](#part-2-modeling-voter-turnout-3-points)
-   [Part 3: OJ Simpson \[4 points\]](#part-3-oj-simpson-4-points)
-   [Submission instructions](#submission-instructions)
    -   [If you use R](#if-you-use-r)
    -   [If you use Python](#if-you-use-python)

``` r
mse = function(model, data) {
  x = modelr:::residuals(model, data)
  mean(x ^ 2, na.rm = TRUE)
}

#boltoff's function
# hackish function to get line segment coordinates for ggplot
partition.tree.data = function (tree, label = "yval", add = FALSE, ordvars, ...) 
{
  ptXlines = function(x, v, xrange, xcoord = NULL, ycoord = NULL, 
                       tvar, i = 1L) {
    if (v[i] == "<leaf>") {
      y1 = (xrange[1L] + xrange[3L])/2
      y2 = (xrange[2L] + xrange[4L])/2
      return(list(xcoord = xcoord, ycoord = c(ycoord, y1, 
                                              y2), i = i))
    }
    if (v[i] == tvar[1L]) {
      xcoord = c(xcoord, x[i], xrange[2L], x[i], xrange[4L])
      xr = xrange
      xr[3L] = x[i]
      ll2 = Recall(x, v, xr, xcoord, ycoord, tvar, i + 
                      1L)
      xr = xrange
      xr[1L] = x[i]
      return(Recall(x, v, xr, ll2$xcoord, ll2$ycoord, tvar, 
                    ll2$i + 1L))
    }
    else if (v[i] == tvar[2L]) {
      xcoord = c(xcoord, xrange[1L], x[i], xrange[3L], 
                  x[i])
      xr = xrange
      xr[4L] = x[i]
      ll2 = Recall(x, v, xr, xcoord, ycoord, tvar, i + 
                      1L)
      xr = xrange
      xr[2L] = x[i]
      return(Recall(x, v, xr, ll2$xcoord, ll2$ycoord, tvar, 
                    ll2$i + 1L))
    }
    else stop("wrong variable numbers in tree.")
  }
  if (inherits(tree, "singlenode")) 
    stop("cannot plot singlenode tree")
  if (!inherits(tree, "tree")) 
    stop("not legitimate tree")
  frame = tree$frame
  leaves = frame$var == "<leaf>"
  var = unique(as.character(frame$var[!leaves]))
  if (length(var) > 2L || length(var) < 1L) 
    stop("tree can only have one or two predictors")
  nlevels = sapply(attr(tree, "xlevels"), length)
  if (any(nlevels[var] > 0L)) 
    stop("tree can only have continuous predictors")
  x = rep(NA, length(leaves))
  x[!leaves] = as.double(substring(frame$splits[!leaves, "cutleft"], 
                                    2L, 100L))
  m = model.frame(tree)
  if (length(var) == 1L) {
    x = sort(c(range(m[[var]]), x[!leaves]))
    if (is.null(attr(tree, "ylevels"))) 
      y = frame$yval[leaves]
    else y = frame$yprob[, 1L]
    y = c(y, y[length(y)])
    if (add) {
      # lines(x, y, type = "s", ...)
    }
    else {
      a = attributes(attr(m, "terms"))
      yvar = as.character(a$variables[1 + a$response])
      xo = m[[yvar]]
      if (is.factor(xo)) 
        ylim = c(0, 1)
      else ylim = range(xo)
      # plot(x, y, ylab = yvar, xlab = var, type = "s", ylim = ylim,
      #      xaxs = "i", ...)
    }
    data_frame(x = x, y = y)
  }
  else {
    if (!missing(ordvars)) {
      ind = match(var, ordvars)
      if (any(is.na(ind))) 
        stop("unmatched names in vars")
      var = ordvars[sort(ind)]
    }
    lab = frame$yval[leaves]
    if (is.null(frame$yprob)) 
      lab = format(signif(lab, 3L))
    else if (match(label, attr(tree, "ylevels"), nomatch = 0L)) 
      lab = format(signif(frame$yprob[leaves, label], 
                           3L))
    rx = range(m[[var[1L]]])
    rx = rx + c(-0.025, 0.025) * diff(rx)
    rz = range(m[[var[2L]]])
    rz = rz + c(-0.025, 0.025) * diff(rz)
    xrange = c(rx, rz)[c(1, 3, 2, 4)]
    xcoord = NULL
    ycoord = NULL
    xy = ptXlines(x, frame$var, xrange, xcoord, ycoord, 
                   var)
    xx = matrix(xy$xcoord, nrow = 4L)
    yy = matrix(xy$ycoord, nrow = 2L)

    return(list(data_frame(xmin = xx[1L,],
                           ymin = xx[2L,],
                           xmax = xx[3L,],
                           ymax = xx[4L,]),
                data_frame(x = yy[1L,],
                           y = yy[2L,],
                           label = lab)))
    # if (!add) 
    #   plot(rx, rz, xlab = var[1L], ylab = var[2L], type = "n", 
    #        xaxs = "i", yaxs = "i", ...)
    # segments(xx[1L, ], xx[2L, ], xx[3L, ], xx[4L, ])
    # text(yy[1L, ], yy[2L, ], as.character(lab), ...)
  }
}
```

Part 1: Joe Biden (redux times two) \[3 points\]
================================================

We read in the data and create a categorical three-level variable for Party.

``` r
df = read.csv('data/biden.csv')
df$Party[df$dem == 1] = 'Democrat'
df$Party[df$dem == 0 & df$rep == 0] = 'No Affiliation'
df$Party[df$rep == 1] = 'Republican'
```

Next, we split the dataset into training and validation sets in a ratio of 7:3, then estimate a regression tree:

``` r
set.seed(1234)

biden_split7030 = resample_partition(df, c(test = 0.3, train = 0.7))
biden_train70 = biden_split7030$train %>%
                tbl_df()
biden_test30 = biden_split7030$test %>%
               tbl_df()

# estimate model
biden_tree1 = tree(biden ~ female + age + educ + dem + rep, data = biden_train70)
mse_test30 = mse(biden_tree1,biden_test30)
```

We evaluate the model with the testing data and find that the mean squared error is 406.417. Next, we plot the tree:

![](ps8-emo_files/figure-markdown_github/plot_biden_tree1-1.png)

The model shows that identifying as a Democrat is the strongest predictor of feelings of warmth toward Vice President Biden, and identifying as a Republican is the second-strongest predictor. Together, these splits indicate that party affiliation (whether Democratic, Republican, or neither) is the most important factor when it comes to predicting an individual subject's feelings of warmth toward Mr. Biden.

We now fit another, more complex tree model:

``` r
biden_tree2 = tree(biden ~ female + age + educ + dem + rep, data = biden_train70,
                   control = tree.control(nobs = nrow(biden_train70), mindev = .0000001))
```

We prune the tree 49 different times, increasing the number of branches from 2 to 50 and storing the MSE for each pruned tree:

``` r
rounds = 50

mse_list_biden_50 = vector("numeric", rounds - 1)
branch_list_biden_50 = vector("numeric", rounds - 1)

set.seed(1234)

for(i in 2:rounds) {
    biden_mod = prune.tree(biden_tree2, best=i)

    mse_val = mse(biden_mod,biden_test30)
    mse_list_biden_50[[i-1]] = mse_val
    branch_list_biden_50[[i-1]] = i
}

mse_df_biden_50 = as.data.frame(mse_list_biden_50)
mse_df_biden_50$branches = branch_list_biden_50
```

We plot the MSE for each tree vs. the number of branches:

![](ps8-emo_files/figure-markdown_github/plot%20biden_50_mse-1.png)

We can clearly see that the lowest MSE is 401.075 for a tree with 11 branches. We plot that tree:

``` r
biden_pruned11 <- prune.tree(biden_tree2, best=11)
mse_test30_2 = mse(biden_pruned11,biden_test30)

plot(biden_pruned11, col='darkturquoise', lwd=2.5)
title("Regression Tree (Best 11) for Warmth Toward Joe Biden (2008)\n", sub = "Validation Dataset")
text(biden_pruned11, col='deeppink')
```

![](ps8-emo_files/figure-markdown_github/plotting_optimal_biden_tree-1.png)

The tree indicates that for Democrats, age is the next most important variable and education after that, but that gender is not important. For unaffiliated voters, gender is important; for women, education and age are both important, but not so for men. Among Republican voters, age is important within the whole group and education is important, but only for voters between ages 44 and 47; gender is not an important predictor of feelings of warmth toward Joe Biden among Republican voters.

Pruning the tree reduces the MSE from 406.417 to 401.075.

``` r
bag_biden = bagging(biden ~ female + age + educ + dem + rep, biden_train70, nbagg=500, coob=TRUE)
pred_bag_biden = predict(bag_biden, biden_test30, type="prob")
mse_bag_biden = mse(bag_biden, biden_test30)
```

1.  Use the bagging approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results.

2.  Use the random forest approach to analyze this data. What test MSE do you obtain? Obtain variable importance measures and interpret the results. Describe the effect of *m*, the number of variables considered at each split, on the error rate obtained.

3.  Use the boosting approach to analyze the data. What test MSE do you obtain? How does the value of the shrinkage parameter *λ* influence the test MSE?

Part 2: Modeling voter turnout \[3 points\]
===========================================

An important question in American politics is why do some people participate in the political process, while others do not? Participation has a direct impact on outcomes -- if you fail to participate in politics, the government and political officials are less likely to respond to your concerns. Typical explanations focus on a resource model of participation -- individuals with greater resources, such as time, money, and civic skills, are more likely to participate in politics. One area of importance is understanding voter turnout, or why people participate in elections. Using the resource model of participation as a guide, we can develop several expectations. First, women, who more frequently are the primary caregiver for children and earn a lower income, are less likely to participate in elections than men. Second, older Americans, who typically have more time and higher incomes available to participate in politics, should be more likely to participate in elections than younger Americans. Finally, individuals with more years of education, who are generally more interested in politics and understand the value and benefits of participating in politics, are more likely to participate in elections than individuals with fewer years of education.

While these explanations have been repeatedly tested by political scientists, an emerging theory assesses an individual's mental health and its effect on political participation.[1] Depression increases individuals' feelings of hopelessness and political efficacy, so depressed individuals will have less desire to participate in politics. More importantly to our resource model of participation, individuals with depression suffer physical ailments such as a lack of energy, headaches, and muscle soreness which drain an individual's energy and requires time and money to receive treatment. For these reasons, we should expect that individuals with depression are less likely to participate in election than those without symptoms of depression.

The 1998 General Social Survey included several questions about the respondent's mental health. `mental_health.csv` reports several important variables from this survey.

-   `vote96` - 1 if the respondent voted in the 1996 presidential election, 0 otherwise
-   `mhealth_sum` - index variable which assesses the respondent's mental health, ranging from 0 (an individual with no depressed mood) to 9 (an individual with the most severe depressed mood)[2]
-   `age` - age of the respondent
-   `educ` - Number of years of formal education completed by the respondent
-   `black` - 1 if the respondent is black, 0 otherwise
-   `female` - 1 if the respondent is female, 0 if male
-   `married` - 1 if the respondent is currently married, 0 otherwise
-   `inc10` - Family income, in $10,000s

1.  Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five tree-based models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)
2.  Use cross-validation techniques and standard measures of model fit (e.g. test error rate, PRE, ROC curves/AUC) to compare and evaluate at least five SVM models of voter turnout. Select the best model and interpret the results using whatever methods you see fit (graphs, tables, model fit statistics, predictions for hypothetical observations, etc.)

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

1.  What is the relationship between race and belief of OJ Simpson's guilt? Develop a robust statistical learning model and use this model to explain the impact of an individual's race on their beliefs about OJ Simpson's guilt.
2.  How can you predict whether individuals believe OJ Simpson to be guilty of these murders? Develop a robust statistical learning model to predict whether individuals believe OJ Simpson to be either probably guilty or probably not guilty and demonstrate the effectiveness of this model using methods we have discussed in class.

You can make full use of any of the statistical learning techniques to complete this part of the assignment:

-   Linear regression
-   Logistic regression
-   Generalized linear models
-   Non-linear linear models
-   Tree-based models
-   Support vector machines
-   Resampling methods

Select methods that are appropriate for each question and **justify the use of these methods**.

Submission instructions
=======================

Assignment submission will work the same as earlier assignments. Submit your work as a pull request before the start of class on Monday. Store it in the same locations as you've been using. However the format of your submission should follow the procedures outlined below.

If you use R
------------

Submit your assignment as a single [R Markdown document](http://rmarkdown.rstudio.com/). R Markdown is similar to Juptyer Notebooks and compiles all your code, output, and written analysis in a single reproducible file.

If you use Python
-----------------

Either:

1.  Submit your assignment following the same procedures as required by Dr. Evans. Submit a Python script containing all your code, plus a $\\LaTeX$ generated PDF document with your results and substantive analysis.
2.  Submit your assignment as a single Jupyter Notebook with your code, output, and written analysis compiled there.

[1] [Ojeda, C. (2015). Depression and political participation. *Social Science Quarterly*, 96(5), 1226-1243.](http://onlinelibrary.wiley.com.proxy.uchicago.edu/doi/10.1111/ssqu.12173/abstract)

[2] The variable is an index which combines responses to four different questions: "In the past 30 days, how often did you feel: 1) so sad nothing could cheer you up, 2) hopeless, 3) that everything was an effort, and 4) worthless?" Valid responses are none of the time, a little of the time, some of the time, most of the time, and all of the time.
