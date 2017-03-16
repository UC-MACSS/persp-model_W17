import pandas
import sklearn
import sklearn.neighbors
import sklearn.ensemble
import sklearn.tree
import sklearn.svm

import numpy as np
import warnings

import matplotlib.pyplot as plt
import seaborn

import os
import sys
import collections

np.random.seed(seed=1234)
outFormat = 'pdf'
outputDir = 'images'

femnistFname = 'data/feminist.csv'
mentalFname = 'data/mental_health.csv'

def getholdOut(df, split = .7):
    sf = df.reindex(np.random.permutation(df.index))
    return sf[:int(len(df) * split)].copy(), sf[int(len(df) * split):].copy()

def mse(model, testDf, yName, xNames):
    return np.mean((model.predict(testDf[xNames]) - testDf[yName]) ** 2)

def getPredictions(model, testDf, yName, xNames):
    results = []
    for i, row in testDf.iterrows():
        xVal = [row[n] for n in xNames]
        yVal = row[yName]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(int(model.predict(xVal)))
    return results

def getResults(models, test, xVars, yVar, fname, stringNames = False):
    results = {
        'error-rate' : [],
        'auc' : [],
        'PRE' : [],
        'AP' : [],
        'RE' : [],
        }
    if stringNames:
        results['model'] = []
    else:
        results['num-neighbors'] = []
    fig, ax = plt.subplots()
    for name, model in models.items():
        if stringNames:
            results['model'].append(name)
        else:
            results['num-neighbors'].append(name)
        test[name] = getPredictions(model, test, yVar, xVars)
        results['auc'].append(sklearn.metrics.roc_auc_score(test[name], test[yVar]))
        results['AP'].append(sklearn.metrics.average_precision_score(test[name], test[yVar]))
        results['PRE'].append(sklearn.metrics.precision_score(test[name], test[yVar]))
        results['RE'].append(sklearn.metrics.recall_score(test[name], test[yVar]))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test[name], test[yVar])
        results['error-rate'].append(1 -  sklearn.metrics.accuracy_score(test[name], test[yVar]))
        ax.plot(fpr, tpr, label = "{} (AUC ${:.3f}$)".format(name, results['auc'][-1]))
    resultsDf = pandas.DataFrame(results)
    if stringNames:
        resultsDf = resultsDf[['model', 'error-rate', 'auc', 'PRE', 'AP', 'RE']]
    else:
        resultsDf = resultsDf[['num-neighbors', 'error-rate', 'auc', 'PRE', 'AP', 'RE']]
    ax.set_title('Receiver Operating Characteristics')
    plt.plot([0,1], [0,1], color = 'k')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc = 'lower right')
    plt.savefig("{}/{}_roc.{}".format(outputDir, fname, outFormat), format = outFormat)
    plt.close()
    fig, ax = plt.subplots()
    if stringNames:
        resultsDf.plot(x = 'model', y ='error-rate', ax = ax, legend = False, logy = False)
        ax.set_title('{}: Test error rate vs model'.format(fname))
        ax.set_xlabel('Model')
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 15)
    else:
        resultsDf.plot(x = 'num-neighbors', y ='error-rate', ax = ax, legend = False, logy = False, xticks = resultsDf['num-neighbors'])
        ax.set_title('{}: Test error rate vs number of neighbors'.format(fname))
        ax.set_xlabel('Number of neighbors')
    ax.set_ylabel('Error rate')
    plt.savefig("{}/{}.{}".format(outputDir, fname, outFormat), format = outFormat)
    plt.close()
    return resultsDf

def getMSEs(models, test, xVars, yVar, fname, stringNames = False):
    results = {'MSE' : []}
    if stringNames:
        results['model'] = []
    else:
        results['num-neighbors'] = []
    indices = []
    for name, model in models.items():
        m = mse(model, test, yVar, xVars)
        if stringNames:
            results['model'].append(name)
        else:
            results['num-neighbors'].append(name)
        results['MSE'].append(m)
    resultsDF = pandas.DataFrame(results)
    if stringNames:
        resultsDF = resultsDF[['model','MSE']]
    else:
        resultsDF = resultsDF[['num-neighbors','MSE']]
    fig, ax = plt.subplots()
    if stringNames:
        resultsDF.plot(x = 'model', y ='MSE', ax = ax, legend = False, logy = False)
        ax.set_title('{}: Test $MSE$ vs Model'.format(fname))
        ax.set_xlabel('Model')
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(),rotation = 15)
    else:
        resultsDF.plot(x = 'num-neighbors', y ='MSE', ax = ax, legend = False, logy = False, xticks = resultsDF['num-neighbors'])
        ax.set_title('{}: Test $MSE$ vs number of neighbors'.format(fname))
        ax.set_xlabel('Number of neighbors')
    ax.set_ylabel('MSE')
    plt.savefig("{}/{}.{}".format(outputDir, fname, outFormat), format = outFormat)
    plt.close()
    return resultsDF

def q1part2(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1) * 5,
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=(i + 1) * 5, weights='uniform')) for i in range(20)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getMSEs(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: unweighted MSE table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minumum MSE is {:.2f}, at {} neighbors".format(np.min(results['MSE']), results['num-neighbors'][np.argmin(results['MSE'])]))

def q1part3(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1) * 5,
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=(i + 1) * 5, weights='distance')) for i in range(20)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getMSEs(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: weighted MSE table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minumum MSE is {:.2f}, at {} neighbors".format(np.min(results['MSE']), results['num-neighbors'][np.argmin(results['MSE'])]))

def q1part4(train, test, xVars, yVar):
    models = collections.OrderedDict([
        ('30-NN, unweighted', sklearn.neighbors.KNeighborsClassifier(n_neighbors= 30, weights = 'uniform')),
        ('30-NN, weighted', sklearn.neighbors.KNeighborsClassifier(n_neighbors= 30, weights = 'distance')),
        ('linear', sklearn.linear_model.LinearRegression()),
        ('linearSVM', sklearn.svm.LinearSVC()),
        ('RandomForest', sklearn.ensemble.RandomForestClassifier()),
        ('DecisionTree', sklearn.tree.DecisionTreeClassifier()),
        ('logit', sklearn.linear_model.LogisticRegression()),
        ('AdaBoost', sklearn.ensemble.AdaBoostClassifier()),
        ])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getMSEs(models, test, xVars, yVar, sys._getframe().f_code.co_name, stringNames = True)
    print("{}: MSE table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minumum MSE is {:.2f}, at {} neighbors".format(np.min(results['MSE']), results['model'][np.argmin(results['MSE'])]))


def q2part2(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1),
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=(i + 1), weights='uniform')) for i in range(10)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getResults(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: results table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minumum error rate is {:.2f}, at {} neighbors".format(np.min(results['error-rate']), results['num-neighbors'][np.argmin(results['error-rate'])]))

def q2part3(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1),
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=(i + 1), weights='distance')) for i in range(10)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getResults(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: unweighted results table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minumum error rate is {:.2f}, at {} neighbors".format(np.min(results['error-rate']), results['num-neighbors'][np.argmin(results['error-rate'])]))

def q2part4(train, test, xVars, yVar):
    models = collections.OrderedDict([
        ('10-NN, unweighted', sklearn.neighbors.KNeighborsClassifier(n_neighbors= 10, weights = 'uniform')),
        ('10-NN, weighted', sklearn.neighbors.KNeighborsClassifier(n_neighbors= 10, weights = 'distance')),
        ('100-NN, weighted', sklearn.neighbors.KNeighborsClassifier(n_neighbors= 100, weights = 'distance')),
        ('linearSVM', sklearn.svm.LinearSVC()),
        ('RandomForest', sklearn.ensemble.RandomForestClassifier()),
        ('DecisionTree', sklearn.tree.DecisionTreeClassifier()),
        ('logit', sklearn.linear_model.LogisticRegression()),
        ('AdaBoost', sklearn.ensemble.AdaBoostClassifier()),
        ])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getResults(models, test, xVars, yVar, sys._getframe().f_code.co_name, stringNames = True)
    print("{}: weighted results table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'left'))
    print("The minumum error rate is {:.2f}, for {}".format(np.min(results['error-rate']), results['model'][np.argmin(results['error-rate'])]))

def question1():
    xVars = ['female', 'age', 'educ', 'income', 'dem', 'rep']
    yVar = 'feminist'
    df = pandas.read_csv(femnistFname)
    train, test = getholdOut(df, .7)
    q1part2(test, train, xVars, yVar)
    q1part3(train, test, xVars, yVar)
    q1part4(train, test, xVars, yVar)

def question2():
    xVars = ['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'inc10']
    yVar = 'vote96'
    df = pandas.read_csv(mentalFname).dropna(how = 'any')
    df[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'vote96']] = df[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'vote96']].astype('int')
    train, test = getholdOut(df, .7)
    q2part2(train, test, xVars, yVar)
    q2part3(train, test, xVars, yVar)
    q2part4(train, test, xVars, yVar)

def main():
    os.makedirs(outputDir, exist_ok = True)
    question1()
    question2()

if __name__ == '__main__':
    main()
