import pandas
import sklearn
import sklearn.neighbors
import sklearn.ensemble
import sklearn.tree
import sklearn.svm
import sklearn.decomposition
import sklearn.cluster

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
collegeFname = 'data/College.csv'
usaFname = 'data/USArrests.csv'

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

def makePCAPlot(df, fname):
    clf = sklearn.decomposition.PCA(n_components = 2)
    clf.fit(df)
    mapping = clf.transform(df)
    fig, ax = plt.subplots()
    pallet = seaborn.color_palette("coolwarm", len(df.columns))
    ax.scatter(mapping[:,0], mapping[:,1], color = 'g')
    max0 = np.max(mapping[:,0])
    max1 = np.max(mapping[:,1])
    randScaling = .3
    for i, c in enumerate(df.columns):
        a = np.zeros((len(df.columns),))
        a[i] = max(df[c])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec = clf.transform(a)
            ax.scatter(vec[:,0], vec[:,1], color = pallet[i], label = c)
            ax.annotate(c,
            xy=(vec[0,0], vec[0,1]),
            xytext=(vec[0,0] + max0 * np.random.uniform(-randScaling, randScaling),
            vec[0,1] + max1 * np.random.uniform(-randScaling, randScaling)),
            arrowprops = {'shrink' : .1, 'facecolor' : pallet[i], 'headwidth' : 4, 'headlength' : 5, 'width' : 2})
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("PCA 1")
    ax.set_xlabel("PCA 0")
    plt.savefig("{}/{}.{}".format(outputDir, fname, outFormat), format = outFormat)
    plt.close()
    return clf

def kclustering(k, fname, df, xNames):
    clf = sklearn.cluster.KMeans(n_clusters = k)
    fits = clf.fit_predict(df[xNames])
    fig, ax = plt.subplots()
    pallet = seaborn.color_palette("hls", k)
    colours = [pallet[c] for c in fits]
    for i, state in enumerate(df.index):
        ax.scatter(df['pca0'][i], df['pca1'][i], color = colours[i])
        ax.annotate(state, (df['pca0'][i], df['pca1'][i]))
    ax.set_ylabel("PCA 1")
    ax.set_xlabel("PCA 0")
    ax.set_title('{}: {}-nearests clusters'.format(fname, k))
    plt.savefig("{}/{}.{}".format(outputDir, fname, outFormat), format = outFormat)
    plt.close()

def q1part2(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1) * 5,
    sklearn.neighbors.KNeighborsRegressor(n_neighbors=(i + 1) * 5, weights='uniform')) for i in range(20)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getMSEs(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: unweighted MSE table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minimum MSE is {:.2f}, at {} neighbors".format(np.min(results['MSE']), results['num-neighbors'][np.argmin(results['MSE'])]))

def q1part3(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1) * 5,
    sklearn.neighbors.KNeighborsRegressor(n_neighbors=(i + 1) * 5, weights='distance')) for i in range(20)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getMSEs(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: weighted MSE table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minimum MSE is {:.2f}, at {} neighbors".format(np.min(results['MSE']), results['num-neighbors'][np.argmin(results['MSE'])]))

def q1part4(train, test, xVars, yVar):
    models = collections.OrderedDict([
        ('30-NN, unweighted', sklearn.neighbors.KNeighborsRegressor(n_neighbors= 30, weights = 'uniform')),
        ('30-NN, weighted', sklearn.neighbors.KNeighborsRegressor(n_neighbors= 30, weights = 'distance')),
        ('linear', sklearn.linear_model.LinearRegression()),
        ('linearSVM', sklearn.svm.LinearSVC()),
        ('RandomForest', sklearn.ensemble.RandomForestRegressor()),
        ('DecisionTree', sklearn.tree.DecisionTreeRegressor()),
        ('logit', sklearn.linear_model.LogisticRegression()),
        ('AdaBoost', sklearn.ensemble.AdaBoostRegressor()),
        ])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getMSEs(models, test, xVars, yVar, sys._getframe().f_code.co_name, stringNames = True)
    print("{}: MSE table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minimum MSE is {:.2f}, for the {} model".format(np.min(results['MSE']), results['model'][np.argmin(results['MSE'])]))

def q2part2(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1),
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=(i + 1), weights='uniform')) for i in range(10)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getResults(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: results table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minimum error rate is {:.2f}, at {} neighbors".format(np.min(results['error-rate']), results['num-neighbors'][np.argmin(results['error-rate'])]))

def q2part3(train, test, xVars, yVar):
    models = collections.OrderedDict([
    ((i + 1),
    sklearn.neighbors.KNeighborsClassifier(n_neighbors=(i + 1), weights='distance')) for i in range(10)])
    for name, model in models.items():
        model.fit(train[xVars], train[yVar])
    results = getResults(models, test, xVars, yVar, sys._getframe().f_code.co_name)
    print("{}: unweighted results table".format(sys._getframe().f_code.co_name))
    print(results.to_string(index=False, justify = 'right'))
    print("The minimum error rate is {:.2f}, at {} neighbors".format(np.min(results['error-rate']), results['num-neighbors'][np.argmin(results['error-rate'])]))

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
    print("The minimum error rate is {:.2f}, for the {} model".format(np.min(results['error-rate']), results['model'][np.argmin(results['error-rate'])]))

def q4part1(df):
    print("{}: Doing PCA".format(sys._getframe().f_code.co_name))
    return makePCAPlot(df, sys._getframe().f_code.co_name)

def q4part2(df, xNames):
    print("{}: Doing 2-NN clustering".format(sys._getframe().f_code.co_name))
    kclustering(2, sys._getframe().f_code.co_name, df, xNames)

def q4part3(df, xNames):
    print("{}: Doing 2-NN clustering".format(sys._getframe().f_code.co_name))
    kclustering(3, sys._getframe().f_code.co_name, df, xNames)

def q4part4(df, xNames):
    print("{}: Doing 4-NN clustering".format(sys._getframe().f_code.co_name))
    kclustering(4, sys._getframe().f_code.co_name, df, xNames)

def q4part5(df, xNames):
    print("{}: Doing 3-NN clustering on PCA space".format(sys._getframe().f_code.co_name))
    kclustering(3, sys._getframe().f_code.co_name, df, xNames)

def q4part6(df, xNames):
    clf = sklearn.cluster.AgglomerativeClustering(n_clusters = len(df), linkage = 'complete', affinity = 'euclidean')
    clf.fit(df[xNames])
    print("{}: Done hierarchical clustering ".format(sys._getframe().f_code.co_name))

def q4part7(df, xNames):
    print("{}: Doing hierarchical clustering ".format(sys._getframe().f_code.co_name))
    clf = sklearn.cluster.AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean')
    df['clusters'] = clf.fit_predict(df[xNames])
    print("The clusters for each state are:")
    print(df[['clusters']])

def q4part8(df, xNames):
    print("{}: Doing hierarchical clustering with scaled data ".format(sys._getframe().f_code.co_name))
    clf = sklearn.cluster.AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = 'euclidean')
    df['clusters'] = clf.fit_predict(sklearn.preprocessing.scale(df[xNames], with_mean = False))
    print("The clusters for each state are:")
    print(df[['clusters']])


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

def question3():
    print("{}: Doing PCA".format(sys._getframe().f_code.co_name))
    df = pandas.read_csv(collegeFname)
    df['Private'] = df['Private'].apply(lambda x: 1 if x == 'Yes' else 0)
    return makePCAPlot(df, sys._getframe().f_code.co_name)

def question4():
    xNames = ['Murder', 'Assault', 'UrbanPop', 'Rape']
    df = pandas.read_csv(usaFname, index_col='State')
    pca = q4part1(df)
    mapping = pca.transform(df)
    df['pca0'] = mapping[:, 0]
    df['pca1'] = mapping[:, 1]
    q4part2(df, xNames)
    q4part3(df, xNames)
    q4part4(df, xNames)
    q4part5(df, ['pca0', 'pca1'])
    q4part6(df, xNames)
    q4part7(df, xNames)
    q4part8(df, xNames)

def main():
    os.makedirs(outputDir, exist_ok = True)
    question1()
    question2()
    question3()
    question4()

if __name__ == '__main__':
    main()
