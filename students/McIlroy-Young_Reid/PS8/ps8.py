import pandas
import sklearn
import numpy as np
import graphviz
import warnings
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics
import sklearn.svm

from ggplot import *

import matplotlib.pyplot as plt
import seaborn

np.random.seed(seed=1234)
outFormat = 'pdf'
outputDir = 'images'

bidenFname = 'data/biden.csv'

mentalFname = 'data/mental_health.csv'

def getholdOut(df, split = .7):
    sf = df.reindex(np.random.permutation(df.index))
    return sf[:int(len(df) * split)].copy(), sf[:int(len(df) *(1 - split))].copy()

def saveTree(t, outName):
    sklearn.tree.export_graphviz(t, out_file = '{}.dot'.format(outName))
    gS = sklearn.tree.export_graphviz(t, out_file = None)
    dot = graphviz.Source(gS)
    dot.render(outName)

def mse(model, testDf, yName, xNames):
    results = []
    for i, row in testDf.iterrows():
        xVal = [row[n] for n in xNames]
        yVal = row[yName]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append((model.predict(xVal) - yVal) ** 2)
    return np.mean(results)

def getPredictions(model, testDf, yName, xNames):
    results = []
    for i, row in testDf.iterrows():
        xVal = [row[n] for n in xNames]
        yVal = row[yName]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(int(model.predict(xVal)))
    return results

def q1Partb(train, test):
    clf1 = sklearn.tree.DecisionTreeRegressor()
    clf1 = clf1.fit(train[['dem', 'rep']], train['biden'])
    saveTree(clf1, 'images/partb')
    m = mse(clf1, test, 'biden', ['dem', 'rep'])
    print('mse is {}'.format(m))

def q1Partc(train, test):
    clf1 = sklearn.tree.DecisionTreeRegressor()
    clf1 = clf1.fit(train[['female', 'age', 'educ', 'dem', 'rep']], train['biden'])
    saveTree(clf1, 'images/partc')
    m = mse(clf1, test, 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
    print('mse is {}'.format(m))

def q1Partd(train, test):
    estCounts = np.linspace(2, 100, 10)
    mses = []
    for c in estCounts:
        clf3 = sklearn.ensemble.BaggingRegressor(n_estimators = int(c))
        clf3 = clf3.fit(train[['female', 'age', 'educ', 'dem', 'rep']],train['biden'])
        m = mse(clf3, test, 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
        mses.append(m)
        print('mse is {}'.format(m))
    plt.plot(estCounts, mses)
    plt.show()

def q1Parte(train, test):
    estCounts = np.linspace(2, 100, 10)
    mses = []
    for c in estCounts:
        clf = sklearn.ensemble.RandomForestRegressor(n_estimators = int(c))
        clf = clf.fit(train[['female', 'age', 'educ', 'dem', 'rep']],train['biden'])
        m = mse(clf, test, 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
        mses.append(m)
        print('mse is {}, importances are:\n{}'.format(m, clf.feature_importances_))
    plt.plot(estCounts, mses)
    plt.show()

def q1Partf(train, test):
    learningRates = np.linspace(0.001, 1, 10)
    mses = []
    for c in learningRates:
        clf = sklearn.ensemble.GradientBoostingRegressor(learning_rate = c)
        clf = clf.fit(train[['female', 'age', 'educ', 'dem', 'rep']],train['biden'])
        m = mse(clf, test, 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
        mses.append(m)
        print('mse is {}, importances are:\n{}'.format(m, clf.feature_importances_))
    plt.plot(learningRates, mses)
    plt.show()

def question1():
    df = pandas.read_csv(bidenFname)
    train, test = getholdOut(df, .7)
    q1Partb(train, test)
    q1Partc(train, test)
    q1Partd(train, test)
    q1Parte(train, test)
    q1Partf(train, test)

def q2Part1(train, test):
    indices = []
    results = {
        'auc' : [],
        'PRE' : [],
        'AP' : [],
        'RE' : [],
        }
    models = {
        'DecisionTreeReduced' : sklearn.tree.DecisionTreeClassifier(),
        'DecisionTree' : sklearn.tree.DecisionTreeClassifier(),
        'AdaBoost' : sklearn.ensemble.AdaBoostClassifier(),
        'Bagging' : sklearn.ensemble.BaggingClassifier(),
        'RandomForest' : sklearn.ensemble.RandomForestClassifier(),
    }
    for name, model in models.items():
        print("Fitting {}".format(name))
        if name == 'DecisionTreeClassifierReduced':
            model.fit(train[['mhealth_sum', 'age']],train['vote96'])
        else:
            model.fit(train[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'inc10']],train['vote96'])

    fig, ax = plt.subplots()
    print("Proccessing")
    for name, model in models.items():
        indices.append(name)
        if name == 'DecisionTreeClassifierReduced':
            test[name] = getPredictions(model, test, 'vote96', ['mhealth_sum', 'age'])
        else:
            test[name] = getPredictions(model, test, 'vote96', ['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'inc10'])

        results['auc'].append(sklearn.metrics.roc_auc_score(test[name], test['vote96']))
        results['AP'].append(sklearn.metrics.average_precision_score(test[name], test['vote96']))
        results['PRE'].append(sklearn.metrics.precision_score(test[name], test['vote96']))
        results['RE'].append(sklearn.metrics.recall_score(test[name], test['vote96']))

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test[name], test['vote96'])
        ax.plot(fpr, tpr, label = "{} (AUC ${:.3f}$)".format(name, results['auc'][-1]))
    resultsDf = pandas.DataFrame(results, index = indices)
    print(resultsDf)

    ax.set_title('Decision Trees Receiver Operating Characteristics')
    plt.plot([0,1], [0,1], color = 'k')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc = 'lower right')
    plt.savefig("{}/q2part1.{}".format(outputDir, outFormat), format = outFormat)
    #plt.show()
    plt.close()

def q2Part2(train, test):
    indices = []
    results = {
        'auc' : [],
        'PRE' : [],
        'AP' : [],
        'RE' : [],
        }
    models = {
        'linearSVM' : sklearn.svm.LinearSVC(),
        'radialSVM' : sklearn.svm.SVC(kernel = 'rbf'),
        'poly2SVM' : sklearn.svm.SVC(kernel = 'poly', degree = 2),
        'nuPoly2SVM' : sklearn.svm.SVC(kernel = 'poly', degree = 2),
        'nuRadialSVM' : sklearn.svm.NuSVC(),
    }
    for name, model in models.items():
        print("Fitting {}".format(name))
        model.fit(train[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'inc10']],train['vote96'])

    fig, ax = plt.subplots()
    print("Proccessing")
    for name, model in models.items():
        indices.append(name)
        test[name] = getPredictions(model, test, 'vote96', ['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'inc10'])

        results['auc'].append(sklearn.metrics.roc_auc_score(test[name], test['vote96']))
        results['AP'].append(sklearn.metrics.average_precision_score(test[name], test['vote96']))
        results['PRE'].append(sklearn.metrics.precision_score(test[name], test['vote96']))
        results['RE'].append(sklearn.metrics.recall_score(test[name], test['vote96']))

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test[name], test['vote96'])
        ax.plot(fpr, tpr, label = "{} (AUC ${:.3f}$)".format(name, results['auc'][-1]))
    resultsDf = pandas.DataFrame(results, index = indices)
    print(resultsDf)

    ax.set_title('SVMs Receiver Operating Characteristics')
    plt.plot([0,1], [0,1], color = 'k')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc = 'lower right')
    plt.savefig("{}/q2part2.{}".format(outputDir, outFormat), format = outFormat)
    #plt.show()
    plt.close()

def question2():
    df = pandas.read_csv(mentalFname).dropna(how = 'any')
    df[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'vote96']] = df[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'vote96']].astype('int')
    train, test = getholdOut(df, .7)
    q2Part1(train, test)
    #q2Part2(train, test)

def main():
    #question1()
    question2()

if __name__ == '__main__':
    main()

"""
clf2 = tree.DecisionTreeRegressor()
clf2 = clf2.fit(df[['female', 'age', 'educ', 'dem', 'rep']],df['biden'])

clf3 = sklearn.ensemble.BaggingRegressor()
clf3 = clf3.fit(df[['female', 'age', 'educ', 'dem', 'rep']],df['biden'])

clf4 = sklearn.ensemble.RandomForestRegressor()
clf4 = clf4.fit(df[['female', 'age', 'educ', 'dem', 'rep']],df['biden'])

clf5 = sklearn.ensemble.GradientBoostingRegressor()
clf5 = clf5.fit(df[['female', 'age', 'educ', 'dem', 'rep']],df['biden'])
"""
