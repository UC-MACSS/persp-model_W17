\documentclass[]{article}
\usepackage{lmodern}
\usepackage{amssymb,amsmath}
\usepackage{ifxetex,ifluatex}
\usepackage{fixltx2e} % provides \textsubscript
\ifnum 0\ifxetex 1\fi\ifluatex 1\fi=0 % if pdftex
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
\else % if luatex or xelatex
  \ifxetex
    \usepackage{mathspec}
  \else
    \usepackage{fontspec}
  \fi
  \defaultfontfeatures{Ligatures=TeX,Scale=MatchLowercase}
\fi
% use upquote if available, for straight quotes in verbatim environments
\IfFileExists{upquote.sty}{\usepackage{upquote}}{}
% use microtype if available
\IfFileExists{microtype.sty}{%
\usepackage{microtype}
\UseMicrotypeSet[protrusion]{basicmath} % disable protrusion for tt fonts
}{}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\hypersetup{unicode=true,
            pdftitle={PS6},
            pdfauthor={Yuqing Zhang},
            pdfborder={0 0 0},
            breaklinks=true}
\urlstyle{same}  % don't use monospace font for urls
\usepackage{color}
\usepackage{fancyvrb}
\newcommand{\VerbBar}{|}
\newcommand{\VERB}{\Verb[commandchars=\\\{\}]}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{commandchars=\\\{\}}
% Add ',fontsize=\small' for more characters per line
\usepackage{framed}
\definecolor{shadecolor}{RGB}{248,248,248}
\newenvironment{Shaded}{\begin{snugshade}}{\end{snugshade}}
\newcommand{\KeywordTok}[1]{\textcolor[rgb]{0.13,0.29,0.53}{\textbf{{#1}}}}
\newcommand{\DataTypeTok}[1]{\textcolor[rgb]{0.13,0.29,0.53}{{#1}}}
\newcommand{\DecValTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{{#1}}}
\newcommand{\BaseNTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{{#1}}}
\newcommand{\FloatTok}[1]{\textcolor[rgb]{0.00,0.00,0.81}{{#1}}}
\newcommand{\ConstantTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{{#1}}}
\newcommand{\CharTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{{#1}}}
\newcommand{\SpecialCharTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{{#1}}}
\newcommand{\StringTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{{#1}}}
\newcommand{\VerbatimStringTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{{#1}}}
\newcommand{\SpecialStringTok}[1]{\textcolor[rgb]{0.31,0.60,0.02}{{#1}}}
\newcommand{\ImportTok}[1]{{#1}}
\newcommand{\CommentTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textit{{#1}}}}
\newcommand{\DocumentationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{{#1}}}}}
\newcommand{\AnnotationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{{#1}}}}}
\newcommand{\CommentVarTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{{#1}}}}}
\newcommand{\OtherTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{{#1}}}
\newcommand{\FunctionTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{{#1}}}
\newcommand{\VariableTok}[1]{\textcolor[rgb]{0.00,0.00,0.00}{{#1}}}
\newcommand{\ControlFlowTok}[1]{\textcolor[rgb]{0.13,0.29,0.53}{\textbf{{#1}}}}
\newcommand{\OperatorTok}[1]{\textcolor[rgb]{0.81,0.36,0.00}{\textbf{{#1}}}}
\newcommand{\BuiltInTok}[1]{{#1}}
\newcommand{\ExtensionTok}[1]{{#1}}
\newcommand{\PreprocessorTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textit{{#1}}}}
\newcommand{\AttributeTok}[1]{\textcolor[rgb]{0.77,0.63,0.00}{{#1}}}
\newcommand{\RegionMarkerTok}[1]{{#1}}
\newcommand{\InformationTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{{#1}}}}}
\newcommand{\WarningTok}[1]{\textcolor[rgb]{0.56,0.35,0.01}{\textbf{\textit{{#1}}}}}
\newcommand{\AlertTok}[1]{\textcolor[rgb]{0.94,0.16,0.16}{{#1}}}
\newcommand{\ErrorTok}[1]{\textcolor[rgb]{0.64,0.00,0.00}{\textbf{{#1}}}}
\newcommand{\NormalTok}[1]{{#1}}
\usepackage{graphicx,grffile}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}
\makeatother
% Scale images if necessary, so that they will not overflow the page
% margins by default, and it is still possible to overwrite the defaults
% using explicit options in \includegraphics[width, height, ...]{}
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
\IfFileExists{parskip.sty}{%
\usepackage{parskip}
}{% else
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 2pt minus 1pt}
}
\setlength{\emergencystretch}{3em}  % prevent overfull lines
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{0}
% Redefines (sub)paragraphs to behave more like sections
\ifx\paragraph\undefined\else
\let\oldparagraph\paragraph
\renewcommand{\paragraph}[1]{\oldparagraph{#1}\mbox{}}
\fi
\ifx\subparagraph\undefined\else
\let\oldsubparagraph\subparagraph
\renewcommand{\subparagraph}[1]{\oldsubparagraph{#1}\mbox{}}
\fi

%%% Use protect on footnotes to avoid problems with footnotes in titles
\let\rmarkdownfootnote\footnote%
\def\footnote{\protect\rmarkdownfootnote}

%%% Change title format to be more compact
\usepackage{titling}

% Create subtitle command for use in maketitle
\newcommand{\subtitle}[1]{
  \posttitle{
    \begin{center}\large#1\end{center}
    }
}

\setlength{\droptitle}{-2em}
  \title{PS6}
  \pretitle{\vspace{\droptitle}\centering\huge}
  \posttitle{\par}
  \author{Yuqing Zhang}
  \preauthor{\centering\large\emph}
  \postauthor{\par}
  \predate{\centering\large\emph}
  \postdate{\par}
  \date{2/18/2017}


\begin{document}
\maketitle

\section{Modeling voter turnout}\label{modeling-voter-turnout}

\subsection{Describe the data}\label{describe-the-data}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{mental_health<-}\KeywordTok{read.csv}\NormalTok{(}\StringTok{'mental_health.csv'}\NormalTok{)}
\end{Highlighting}
\end{Shaded}

\subsection{Including Plots}\label{including-plots}

\includegraphics{PS6_files/figure-latex/voter turnout-1.pdf}

\begin{enumerate}
\def\labelenumi{\arabic{enumi}.}
\tightlist
\item
  The unconditional probability of a given individual turning out to
  vote is: 63\%
\end{enumerate}

\includegraphics{PS6_files/figure-latex/scatterplot-1.pdf} 2. The graph
tells us that the worse one person's mental condition is, the less
likely he or she is going to vote. The problem with the linear line is
that the only possible values for voter turnout are 0 and 1. Yet the
linear regression model gives us predicted values such as .75 and .25.

\subsection{Basic Model}\label{basic-model}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{vote_mental <-}\StringTok{ }\KeywordTok{glm}\NormalTok{(vote96 ~}\StringTok{ }\NormalTok{mhealth_sum, }\DataTypeTok{data =} \NormalTok{mental_health, }\DataTypeTok{family =} \NormalTok{binomial)}
\KeywordTok{summary}\NormalTok{(vote_mental)}
\end{Highlighting}
\end{Shaded}

\begin{verbatim}
## 
## Call:
## glm(formula = vote96 ~ mhealth_sum, family = binomial, data = mental_health)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.6834  -1.2977   0.7452   0.8428   1.6911  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  1.13921    0.08444  13.491  < 2e-16 ***
## mhealth_sum -0.14348    0.01969  -7.289 3.13e-13 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1672.1  on 1321  degrees of freedom
## Residual deviance: 1616.7  on 1320  degrees of freedom
##   (1510 observations deleted due to missingness)
## AIC: 1620.7
## 
## Number of Fisher Scoring iterations: 4
\end{verbatim}

1.The relationship between mental health and voter turnout is
statistically significant because p-value is almost 0. The coefficient
is -.14348, which means increasing by 1 on the mental health scale,
decreases the likelihood of voting by almost -86. It indicates the
relationship is substantive.

\begin{enumerate}
\def\labelenumi{\arabic{enumi}.}
\setcounter{enumi}{1}
\item
\end{enumerate}

\begin{Shaded}
\begin{Highlighting}[]
\KeywordTok{tidy}\NormalTok{(vote_mental)}
\end{Highlighting}
\end{Shaded}

\begin{verbatim}
##          term   estimate  std.error statistic      p.value
## 1 (Intercept)  1.1392097 0.08444019 13.491321 1.759191e-41
## 2 mhealth_sum -0.1434752 0.01968511 -7.288516 3.133883e-13
\end{verbatim}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{log_odds <-}\StringTok{ }\NormalTok{vote_mental$coefficients[}\DecValTok{2}\NormalTok{]}
\end{Highlighting}
\end{Shaded}

For every one-unit increase in mental\_health score, we expect the
log-odds of voter turnout to decrease by \texttt{\{r\ param\}log\_odds}

\includegraphics{PS6_files/figure-latex/log_odds graph-1.pdf}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{mental_health_score <-}\StringTok{ }\NormalTok{mental_health %>%}
\StringTok{  }\KeywordTok{add_predictions}\NormalTok{(vote_mental) %>%}
\StringTok{  }\CommentTok{# predicted values are in the log-odds form - convert to probabilities}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{prob =} \KeywordTok{logit2prob}\NormalTok{(pred))}
\NormalTok{prob2odds <-}\StringTok{ }\NormalTok{function(x)\{}
  \NormalTok{x /}\StringTok{ }\NormalTok{(}\DecValTok{1} \NormalTok{-}\StringTok{ }\NormalTok{x)}
\NormalTok{\}}
\NormalTok{mental_health_score <-}\StringTok{ }\NormalTok{mental_health_score %>%}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{odds =} \KeywordTok{prob2odds}\NormalTok{(prob))}
\end{Highlighting}
\end{Shaded}

\begin{enumerate}
\def\labelenumi{\arabic{enumi}.}
\setcounter{enumi}{2}
\item
\end{enumerate}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{exp_odds =}\StringTok{ }\KeywordTok{exp}\NormalTok{(log_odds)}
\end{Highlighting}
\end{Shaded}

The odds ratio associated with a one unit increase in mhealth\_sum is
0.8663423

\includegraphics{PS6_files/figure-latex/odds_graph-1.pdf}

4.a

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{prob =}\StringTok{ }\KeywordTok{logit2prob}\NormalTok{(log_odds)}
\end{Highlighting}
\end{Shaded}

A one-unit increase in mental health index is associated with 0.4641926
decrease in probability of voting on average.

\includegraphics{PS6_files/figure-latex/prob_graph-1.pdf} 4.b

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{fd_1_2 =}\StringTok{ }\KeywordTok{logit2prob}\NormalTok{(}\DecValTok{2}\NormalTok{) -}\StringTok{ }\KeywordTok{logit2prob}\NormalTok{(}\DecValTok{1}\NormalTok{)}
\NormalTok{fd_5_6 =}\StringTok{ }\KeywordTok{logit2prob}\NormalTok{(}\DecValTok{6}\NormalTok{) -}\StringTok{ }\KeywordTok{logit2prob}\NormalTok{(}\DecValTok{5}\NormalTok{)}
\end{Highlighting}
\end{Shaded}

The first difference for an increase in the mental health index from 1
to 2 is: 0.1497385. The first difference for an increase in the mental
health index from 5 to 6 is: 0.0042202.

\begin{enumerate}
\def\labelenumi{\arabic{enumi}.}
\setcounter{enumi}{4}
\item
\end{enumerate}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{mh_accuracy <-}\StringTok{ }\NormalTok{mental_health %>%}
\StringTok{  }\KeywordTok{add_predictions}\NormalTok{(vote_mental) %>%}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{pred =} \KeywordTok{logit2prob}\NormalTok{(pred),}
         \DataTypeTok{pred =} \KeywordTok{as.numeric}\NormalTok{(pred >}\StringTok{ }\NormalTok{.}\DecValTok{5}\NormalTok{))}

\NormalTok{accr_rate =}\StringTok{ }\KeywordTok{mean}\NormalTok{(mh_accuracy$vote96 ==}\StringTok{ }\NormalTok{mh_accuracy$pred, }\DataTypeTok{na.rm =} \OtherTok{TRUE}\NormalTok{)}
\end{Highlighting}
\end{Shaded}

\begin{Shaded}
\begin{Highlighting}[]
\CommentTok{# create a function to calculate the modal value of a vector}
\NormalTok{getmode <-}\StringTok{ }\NormalTok{function(v) \{}
   \NormalTok{uniqv <-}\StringTok{ }\KeywordTok{unique}\NormalTok{(v)}
   \NormalTok{uniqv[}\KeywordTok{which.max}\NormalTok{(}\KeywordTok{tabulate}\NormalTok{(}\KeywordTok{match}\NormalTok{(v, uniqv)))]}
\NormalTok{\}}
\CommentTok{# function to calculate PRE for a logistic regression model}
\NormalTok{PRE <-}\StringTok{ }\NormalTok{function(model)\{}
  \CommentTok{# get the actual values for y from the data}
  \NormalTok{y <-}\StringTok{ }\NormalTok{model$y}
  
  \CommentTok{# get the predicted values for y from the model}
  \NormalTok{y.hat <-}\StringTok{ }\KeywordTok{round}\NormalTok{(model$fitted.values)}
  
  \CommentTok{# calculate the errors for the null model and your model}
  \NormalTok{E1 <-}\StringTok{ }\KeywordTok{sum}\NormalTok{(y !=}\StringTok{ }\KeywordTok{median}\NormalTok{(y))}
  \NormalTok{E2 <-}\StringTok{ }\KeywordTok{sum}\NormalTok{(y !=}\StringTok{ }\NormalTok{y.hat)}
  
  \CommentTok{# calculate the proportional reduction in error}
  \NormalTok{PRE <-}\StringTok{ }\NormalTok{(E1 -}\StringTok{ }\NormalTok{E2) /}\StringTok{ }\NormalTok{E1}
  \KeywordTok{return}\NormalTok{(PRE)}
\NormalTok{\}}
\NormalTok{pre =}\StringTok{ }\KeywordTok{PRE}\NormalTok{(vote_mental)}
\end{Highlighting}
\end{Shaded}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{mh_accuracy <-}\StringTok{ }\NormalTok{mental_health %>%}
\StringTok{  }\KeywordTok{add_predictions}\NormalTok{(vote_mental) %>%}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{pred =} \KeywordTok{logit2prob}\NormalTok{(pred),}
         \DataTypeTok{prob =} \NormalTok{pred,}
         \DataTypeTok{pred =} \KeywordTok{as.numeric}\NormalTok{(pred >}\StringTok{ }\NormalTok{.}\DecValTok{5}\NormalTok{))}
\NormalTok{auc_x <-}\StringTok{ }\KeywordTok{auc}\NormalTok{(mh_accuracy$vote96, mh_accuracy$prob)}
\end{Highlighting}
\end{Shaded}

The accuracy rate is: 67.78\% and the proportional reduction in error
is: 1.62\%. The AUC is 0.6243087.

\section{Multiple variable model}\label{multiple-variable-model}

\subsection{1. Three components}\label{three-components}

\begin{enumerate}
\def\labelenumi{\alph{enumi})}
\item
  A random component specifying the conditional distribution of the
  response variable, Yi, given the values of the predictor variables in
  the model. Each individual vote turnout is either 0(not vote) or
  1(vote), so each one is a bernoulli trial. Thus the response variable
  vote96, ,which is a collection of each individual vote turnout, is
  distributed as a binomial random variable.
\item
  The linear predictor is:
  \[vote96_{i} = \beta_{0} + \beta_{1}mhealth\_sum_{i} + \beta_{2}age_{i} + \beta_{3}educ_{i} + \beta_{4}black_{i} + \beta_{5}female_{i} + \beta_{6}married_{i} + \beta_{7}inc10_{i}\]
\item
  The link function is
  \[g(vote96_i) = \frac{e^{vote96_i}}{1 + e^{vote96_i}}\]
\end{enumerate}

\subsection{2,3 Estimate the model and report your
results.}\label{estimate-the-model-and-report-your-results.}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{vote_all <-}\StringTok{ }\KeywordTok{glm}\NormalTok{(vote96 ~}\StringTok{ }\NormalTok{., }\DataTypeTok{data =} \NormalTok{mental_health,}
                         \DataTypeTok{family =} \NormalTok{binomial)}
\NormalTok{vote_all_grid <-}\StringTok{ }\NormalTok{mental_health %>%}
\StringTok{  }\CommentTok{#data_grid(.) %>%}
\StringTok{  }\KeywordTok{add_predictions}\NormalTok{(vote_all) %>%}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{pred =} \KeywordTok{logit2prob}\NormalTok{(pred))}

\KeywordTok{summary}\NormalTok{(vote_all)}
\end{Highlighting}
\end{Shaded}

\begin{verbatim}
## 
## Call:
## glm(formula = vote96 ~ ., family = binomial, data = mental_health)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -2.4843  -1.0258   0.5182   0.8428   2.0758  
## 
## Coefficients:
##              Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -4.304103   0.508103  -8.471  < 2e-16 ***
## mhealth_sum -0.089102   0.023642  -3.769 0.000164 ***
## age          0.042534   0.004814   8.835  < 2e-16 ***
## educ         0.228686   0.029532   7.744 9.65e-15 ***
## black        0.272984   0.202585   1.347 0.177820    
## female      -0.016969   0.139972  -0.121 0.903507    
## married      0.296915   0.153164   1.939 0.052557 .  
## inc10        0.069614   0.026532   2.624 0.008697 ** 
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 1468.3  on 1164  degrees of freedom
## Residual deviance: 1241.8  on 1157  degrees of freedom
##   (1667 observations deleted due to missingness)
## AIC: 1257.8
## 
## Number of Fisher Scoring iterations: 4
\end{verbatim}

The results table shows that four predictors are statistically
significant. Mental health score and coefficient is -0.089102, meaning
that for every one-unit increase in mental\_health score, we expect the
log-odds of voter turnout to decrease by 0.089102; age and coefficient
is 0.042534 for age, meaning that for every one-unit increase in
mental\_health score, we expect the log-odds of voter turnout to
increase by0.042534; education and coefficient is 0.228686,meaning that
for every one-unit increase in education, we expect the log-odds of
voter turnout to increase by 0.228686; income and coefficient is
0.069614,meaning that for every one-unit increase in income, we expect
the log-odds of voter turnout to increase by 0.069614.

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{mh_accuracy_all <-}\StringTok{ }\NormalTok{mental_health %>%}
\StringTok{  }\KeywordTok{add_predictions}\NormalTok{(vote_all) %>%}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{pred =} \KeywordTok{logit2prob}\NormalTok{(pred),}
         \DataTypeTok{prob =} \NormalTok{pred,}
         \DataTypeTok{pred =} \KeywordTok{as.numeric}\NormalTok{(pred >}\StringTok{ }\NormalTok{.}\DecValTok{5}\NormalTok{))}
\NormalTok{accr_rate_all =}\StringTok{ }\KeywordTok{mean}\NormalTok{(mh_accuracy_all$vote96 ==}\StringTok{ }\NormalTok{mh_accuracy_all$pred, }\DataTypeTok{na.rm =} \OtherTok{TRUE}\NormalTok{)}
\NormalTok{auc_all <-}\StringTok{ }\KeywordTok{auc}\NormalTok{(mh_accuracy_all$vote96, mh_accuracy_all$prob)}
\NormalTok{pre_all <-}\StringTok{ }\KeywordTok{PRE}\NormalTok{(vote_all)}
\end{Highlighting}
\end{Shaded}

The accuracy rate,72.36\%, proportional reduction in error (PRE),14.81\%
and area under the curve (AUC),0.759624 of the current model indicate
that the model is better than the ``simple'' logistic regression model.
Nonetheless, even with more predictors, the current logistic regression
model shows a rather poor performance.

\section{Estimate a regression model}\label{estimate-a-regression-model}

\subsection{1. Three components}\label{three-components-1}

\begin{enumerate}
\def\labelenumi{\alph{enumi})}
\tightlist
\item
  The response variable tvhours is distributed as a poisson random
  variable.
  \[Pr(tvhours = k|\lambda) = \frac{\lambda^{k}e^{-\lambda}}{k!}\]
\item
  The linear predictor is:
  \[tvhours_{i} = \beta_{0} + \beta_{1}age + \beta_{2}childs + \beta_{3}educ + \beta_{4}female + \beta_{5}grass + \beta_{6}hrsrelax + \beta_{7}black + \beta_{8}social_connect + \beta_{9}voted04 + \beta_{10}xmovie + \beta_{11}zodiac + \beta_{12}dem + \beta_{13}rep + \beta_{14}ind\]
\item
  The link function is \[\mu_{i} = \ln(tvhours_{i})\] \#\# 2,3 Estimate
  the model and report your results. In this model, the number of hours
  watching TV per day is the response variable. I chose to estimate
  these predictors: age, number of children, education, social\_connect.
\end{enumerate}

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{tv_consumption<-}\KeywordTok{read.csv}\NormalTok{(}\StringTok{'gss2006.csv'}\NormalTok{)}
\NormalTok{tv_pred <-}\StringTok{ }\KeywordTok{glm}\NormalTok{(tvhours ~}\StringTok{ }\NormalTok{age+childs+educ+hrsrelax, }\DataTypeTok{data =} \NormalTok{tv_consumption,}
                         \DataTypeTok{family =} \StringTok{'poisson'}\NormalTok{)}
\NormalTok{tv_pred_grid <-}\StringTok{ }\NormalTok{tv_consumption %>%}
\StringTok{  }\CommentTok{#data_grid(.) %>%}
\StringTok{  }\KeywordTok{add_predictions}\NormalTok{(tv_pred) %>%}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{pred =} \KeywordTok{logit2prob}\NormalTok{(pred))}

\KeywordTok{summary}\NormalTok{(tv_pred)}
\end{Highlighting}
\end{Shaded}

\begin{verbatim}
## 
## Call:
## glm(formula = tvhours ~ age + childs + educ + hrsrelax, family = "poisson", 
##     data = tv_consumption)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -3.0735  -0.8218  -0.1922   0.4025   6.5412  
## 
## Coefficients:
##               Estimate Std. Error z value Pr(>|z|)    
## (Intercept)  1.4014163  0.1195231  11.725  < 2e-16 ***
## age         -0.0003867  0.0016140  -0.240    0.811    
## childs      -0.0010728  0.0144411  -0.074    0.941    
## educ        -0.0462130  0.0072351  -6.387 1.69e-10 ***
## hrsrelax     0.0417919  0.0061889   6.753 1.45e-11 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for poisson family taken to be 1)
## 
##     Null deviance: 1328.9  on 1111  degrees of freedom
## Residual deviance: 1239.9  on 1107  degrees of freedom
##   (3398 observations deleted due to missingness)
## AIC: 4094.9
## 
## Number of Fisher Scoring iterations: 5
\end{verbatim}

At the p\textless{}.001 level,the results table shows that of the four
predictors I chose, two of them are statistically significant. Education
and coefficient is -0.0429390, meaning that for every one-unit increase
in education, we expect the log-odds of hours of watching tv to decrease
by 0.0429390; education and coefficient is 0.228686,meaning that for
every one-unit increase in education, we expect the log-odds of voter
turnout to increase by 0.228686; hours of relaxing and coefficient is
0.0417919,meaning that for every one-unit increase in social\_connect,
we expect the log-odds of hours of watching tv to increase by 0.0417919.

This model pretty makes sense. More educated one person is, less time
they spend on watching tv. This may because of the fact that instead of
`wasting time on watching tv',they have more of other things to do.The
relationship between hours of relax make sense too. As hours of relax
increases, number of hours watching television also increases.

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{tv_accuracy <-}\StringTok{ }\NormalTok{tv_consumption %>%}
\StringTok{  }\KeywordTok{add_predictions}\NormalTok{(tv_pred) %>%}
\StringTok{  }\KeywordTok{mutate}\NormalTok{(}\DataTypeTok{pred =} \KeywordTok{logit2prob}\NormalTok{(pred),}
         \DataTypeTok{prob =} \NormalTok{pred,}
         \DataTypeTok{pred =} \KeywordTok{as.numeric}\NormalTok{(pred >}\StringTok{ }\NormalTok{.}\DecValTok{5}\NormalTok{))}
\NormalTok{accr_rate_tv =}\StringTok{ }\KeywordTok{mean}\NormalTok{(tv_accuracy$tvhours ==}\StringTok{ }\NormalTok{tv_accuracy$pred, }\DataTypeTok{na.rm =} \OtherTok{TRUE}\NormalTok{)}
\NormalTok{auc_tv <-}\StringTok{ }\KeywordTok{auc}\NormalTok{(tv_accuracy$tvhours, tv_accuracy$prob)}
\NormalTok{pre_tv <-}\StringTok{ }\KeywordTok{PRE}\NormalTok{(tv_pred)}
\end{Highlighting}
\end{Shaded}

The accuracy rate,25\%, proportional reduction in error (PRE),-4.56\%
and area under the curve (AUC),0.483906 of the current model indicate
that the model is not a very good model.

\begin{Shaded}
\begin{Highlighting}[]
\NormalTok{tv_pred_quasi <-}\StringTok{ }\KeywordTok{glm}\NormalTok{(tvhours ~}\StringTok{ }\NormalTok{age+childs+educ+hrsrelax, }\DataTypeTok{data =} \NormalTok{tv_consumption,}
                         \DataTypeTok{family =} \StringTok{'quasipoisson'}\NormalTok{)}
\KeywordTok{summary}\NormalTok{(tv_pred_quasi)}
\end{Highlighting}
\end{Shaded}

\begin{verbatim}
## 
## Call:
## glm(formula = tvhours ~ age + childs + educ + hrsrelax, family = "quasipoisson", 
##     data = tv_consumption)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -3.0735  -0.8218  -0.1922   0.4025   6.5412  
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  1.4014163  0.1351933  10.366  < 2e-16 ***
## age         -0.0003867  0.0018256  -0.212    0.832    
## childs      -0.0010728  0.0163345  -0.066    0.948    
## educ        -0.0462130  0.0081837  -5.647 2.07e-08 ***
## hrsrelax     0.0417919  0.0070003   5.970 3.19e-09 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for quasipoisson family taken to be 1.2794)
## 
##     Null deviance: 1328.9  on 1111  degrees of freedom
## Residual deviance: 1239.9  on 1107  degrees of freedom
##   (3398 observations deleted due to missingness)
## AIC: NA
## 
## Number of Fisher Scoring iterations: 5
\end{verbatim}

I used quasipoisson model to see if the model is under or
over-dispersion. From the table summary above, dispersion parameter for
quasipoisson family is 1.2794,higher than 1, which indicates that the
model is over-dispersed (the true variance of the distribution is
greater than its mean).


\end{document}
