import pandas
import sklearn
import numpy as np
import graphviz
import warnings
import sklearn.ensemble
import sklearn.tree
import sklearn.metrics
import sklearn.svm
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.pipeline

from ggplot import *

import matplotlib.pyplot as plt
import seaborn

np.random.seed(seed=1234)
outFormat = 'pdf'
outputDir = 'images'

bidenFname = 'data/biden.csv'

mentalFname = 'data/mental_health.csv'

ojFName = 'data/simpson.csv'

def getholdOut(df, split = .7):
    sf = df.reindex(np.random.permutation(df.index))
    return sf[:int(len(df) * split)].copy(), sf[int(len(df) * split):].copy()

def saveTree(t, outName):
    sklearn.tree.export_graphviz(t, out_file = '{}/{}.dot'.format(outputDir, outName))
    gS = sklearn.tree.export_graphviz(t, out_file = None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dot = graphviz.Source(gS)
        dot.render('{}/{}'.format(outputDir, outName))

def mse(model, testDf, yName, xNames):
    results = []
    for i, row in testDf.iterrows():
        xVal = [row[n] for n in xNames]
        yVal = row[yName]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append((model.predict(xVal) - yVal) ** 2)
    return np.mean(results)

def getPredictions(model, testDf, yName, xNames, nonInt = False):
    results = []
    for i, row in testDf.iterrows():
        xVal = [row[n] for n in xNames]
        yVal = row[yName]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if nonInt:
                results.append(float(model.predict(xVal)))
            else:
                results.append(int(model.predict(xVal)))
    return results

def q1Part2(train, test):
    clf = sklearn.tree.DecisionTreeRegressor()
    clf = clf.fit(train[['dem', 'rep']], train['biden'])
    saveTree(clf, 'q1Part2_graph')
    m = mse(clf, test, 'biden', ['dem', 'rep'])
    print('q1 part 2: MSE is {:.2f}'.format(m))

def q1Part3(df):
    kFolds = 10
    splitSpace = np.linspace(2, 1800, 30)
    X = df[['female', 'age', 'educ', 'dem', 'rep']]
    Y = df['biden']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skf = sklearn.model_selection.StratifiedKFold(n_splits = kFolds)
        splits = [(train_index, test_index) for train_index, test_index in skf.split(X, Y)]

    fig, ax = plt.subplots()
    splitMSEs = []
    minMSE = 10000
    minModel = None
    for nSamples in splitSpace:
        sampleMSEs = []
        for train_index, test_index in splits:
            clf = sklearn.tree.DecisionTreeRegressor(min_samples_split = int(nSamples))
            clf.fit(X.ix[train_index], Y.ix[train_index])
            m = mse(clf, df.ix[test_index], 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
            sampleMSEs.append(m)
            if m < minMSE:
                minMSE = m
                minModel = clf
        ax.scatter([nSamples] * kFolds, sampleMSEs, s = 2)
        splitMSEs.append(np.mean(sampleMSEs))
    ax.plot(splitSpace, splitMSEs)
    print("q1 part 3: min MSE is {:.2f}".format(minMSE))
    ax.set_title('Q1P3: MSE vs minimum samples per leaf')
    ax.set_xlabel('minimum samples per leaf')
    ax.set_ylabel('MSE')
    plt.savefig("{}/q1part3.{}".format(outputDir, outFormat), format = outFormat)
    #plt.show()
    plt.close()
    saveTree(minModel, 'q1Part3_graph')

def q1Part4(train, test):
    estCounts = np.linspace(2, 100, 10).astype('int')
    mses = []
    for c in estCounts:
        clf3 = sklearn.ensemble.BaggingRegressor(n_estimators = c)
        clf3 = clf3.fit(train[['female', 'age', 'educ', 'dem', 'rep']],train['biden'])
        m = mse(clf3, test, 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
        mses.append(m)
    print('q1 part 4: The minumum MSE is {:.2f} at {} estimators'.format(min(mses),estCounts[np.argmin(mses)]))
    fig, ax = plt.subplots()
    ax.plot(estCounts, mses)
    ax.set_title('Q1P4: MSE vs Number of Estimators')
    ax.set_xlabel('Number of Estimators')
    ax.set_ylabel('MSE')
    plt.savefig("{}/q1part4.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()

def q1Part5(train, test):
    nFeatures = len(['female', 'age', 'educ', 'dem', 'rep'])
    featureCounts = np.linspace(1 / nFeatures, 1, nFeatures)
    mses = []
    minModel = None
    for c in featureCounts:
        clf = sklearn.ensemble.RandomForestRegressor(max_features = c, min_samples_split = 500)
        clf = clf.fit(train[['female', 'age', 'educ', 'dem', 'rep']],train['biden'])
        m = mse(clf, test, 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
        if len(mses) == 0 or m < min(mses):
            minModel = clf
        mses.append(m)
    print('q1 part 5: The minumum MSE is {:.2f} at {} features considered. It has the following variable importances:'.format(min(mses), int(nFeatures * featureCounts[np.argmin(mses)])))
    for i, vName in enumerate(['female', 'age', 'educ', 'dem', 'rep']):
        print("{}:  {:.4f}".format(vName.ljust(7, ' '), minModel.feature_importances_[i]))
    fig, ax = plt.subplots()
    ax.plot(featureCounts * nFeatures, mses)
    ax.set_title('Q1P5: MSE vs Number of Features Considered')
    ax.set_xlabel('Number of Features Considered')
    ax.set_ylabel('MSE')
    plt.savefig("{}/q1part5.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()

def q1Part6(train, test):
    learningRates = 10 ** (- np.linspace(0, 8, 9))
    mses = []
    minModel = None
    for c in learningRates:
        clf = sklearn.ensemble.GradientBoostingRegressor(learning_rate = c, min_samples_split = 500)
        clf = clf.fit(train[['female', 'age', 'educ', 'dem', 'rep']],train['biden'])
        m = mse(clf, test, 'biden', ['female', 'age', 'educ', 'dem', 'rep'])
        if len(mses) == 0 or m < min(mses):
            minModel = clf
        mses.append(m)
    print('q1 part 6: The minumum MSE is {:.2f} at lambda of {}. It has the following variable importances:'.format(min(mses), learningRates[np.argmin(mses)]))
    for i, vName in enumerate(['female', 'age', 'educ', 'dem', 'rep']):
        print("{}:  {:.4f}".format(vName.ljust(7, ' '), minModel.feature_importances_[i]))
    fig, ax = plt.subplots()
    ax.plot(learningRates, mses)
    ax.set_title('Q1P5: MSE vs Learning Rate')
    ax.set_xlabel('Learning Rate')
    ax.set_xscale("log")
    ax.set_ylabel('MSE')
    plt.savefig("{}/q1part6.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()

def question1():
    np.random.seed(seed=1234)
    df = pandas.read_csv(bidenFname)
    train, test = getholdOut(df, .7)
    q1Part2(train, test)
    q1Part3(df)
    q1Part4(train, test)
    q1Part5(train, test)
    q1Part6(train, test)

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
    print("The stats table for the models is:")
    print(resultsDf)
    for i, vName in enumerate(['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'inc10']):
        print("{}:  {:.4f}".format(vName.ljust(9, ' '), models['AdaBoost'].feature_importances_[i]))
    ax.set_title('Q2P1: Decision Trees Receiver Operating Characteristics')
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
        'linearSVM_reduced' : sklearn.svm.LinearSVC(),
        'radialSVM' : sklearn.svm.SVC(kernel = 'rbf'),
        'poly2SVM' : sklearn.svm.SVC(kernel = 'poly', degree = 2),
        'nuRadialSVM' : sklearn.svm.NuSVC(),
    }
    for name, model in models.items():
        print("Fitting {}".format(name))
        if name == 'linearSVM_reduced':
            model.fit(train[['mhealth_sum', 'age']],train['vote96'])
        else:
            model.fit(train[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'inc10']],train['vote96'])

    fig, ax = plt.subplots()
    print("Proccessing")
    for name, model in models.items():
        indices.append(name)
        if name == 'linearSVM_reduced':
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
    print("The stats table for the models is:")
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
    np.random.seed(seed=1234)
    df = pandas.read_csv(mentalFname).dropna(how = 'any')
    df[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'vote96']] = df[['mhealth_sum', 'age', 'educ', 'black', 'female', 'married', 'vote96']].astype('int')
    train, test = getholdOut(df, .7)
    q2Part1(train, test)
    q2Part2(train, test)

def q3part1(train, test):
    indices = []
    results = {
        'auc' : [],
        'PRE' : [],
        'AP' : [],
        'RE' : [],
        }
    models = {
        'linear' : sklearn.linear_model.LinearRegression(),
        'linear_LowThreshold' : sklearn.linear_model.LinearRegression(),
        'linear_highThreshold' : sklearn.linear_model.LinearRegression(),
        'logit' : sklearn.linear_model.LogisticRegression(),
        'linearSVM' : sklearn.svm.LinearSVC(),
        'DecisionTree' : sklearn.tree.DecisionTreeClassifier(),
        'RandomForest' : sklearn.ensemble.RandomForestClassifier(),
    }
    for name, model in models.items():
        print("Fitting {}".format(name))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[['black', 'hispanic']], train['guilt'])
    fig, ax = plt.subplots()
    print("Proccessing")
    for name, model in models.items():
        indices.append(name)
        if name in {'linear', 'logit', 'linear_LowThreshold', 'linear_highThreshold'}:
            test[name] = getPredictions(model, test, 'guilt', ['black', 'hispanic'], nonInt = True)
            if name == 'linear_LowThreshold':
                test[name] = test[name].apply(lambda x: 1 if x > min(test[name]) else 0)
            elif name == 'linear_highThreshold':
                test[name] = test[name].apply(lambda x: 1 if x >= max(test[name]) else 0)
            else:
                test[name] = test[name].apply(lambda x: 1 if x > 0.5 else 0)
        else:
            test[name] = getPredictions(model, test, 'guilt', ['black', 'hispanic'])
        results['auc'].append(sklearn.metrics.roc_auc_score(test[name], test['guilt']))
        results['AP'].append(sklearn.metrics.average_precision_score(test[name], test['guilt']))
        results['PRE'].append(sklearn.metrics.precision_score(test[name], test['guilt']))
        results['RE'].append(sklearn.metrics.recall_score(test[name], test['guilt']))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test[name], test['guilt'])
        ax.plot(fpr, tpr, label = "{} (AUC ${:.3f}$)".format(name, results['auc'][-1]))
    resultsDf = pandas.DataFrame(results, index = indices)
    print("The stats table for the models is:")
    print(resultsDf)
    print("The variable importances for the Random Forest are:")
    for i, vName in enumerate(['black', 'hispanic']):
        print("{}:  {:.4f}".format(vName.ljust(7, ' '), models['RandomForest'].feature_importances_[i]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pandas.DataFrame({
            'hispanic' : [int(models['RandomForest'].predict([i, 1])) for i in [1, 0]],
             'not hispanic' : [int(models['RandomForest'].predict([i, 0])) for i in [1, 0]]
             }, index = ['black', 'not black'])
    print("The predicted outcomes for all models are:")
    print(df)
    ax.set_title('SVMs Receiver Operating Characteristics')
    plt.plot([0,1], [0,1], color = 'k')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc = 'lower right')
    plt.savefig("{}/q3part1.{}".format(outputDir, outFormat), format = outFormat)
    #plt.show()
    plt.close()

def q3part2(train, test):
    xVars = ['dem', 'rep', 'ind', 'age', 'educ', 'female', 'black', 'hispanic', 'income']
    yVar = 'guilt'
    indices = []
    results = {
        'auc' : [],
        'PRE' : [],
        'AP' : [],
        'RE' : [],
        }
    models = {
        'linear' : sklearn.linear_model.LinearRegression(),
        'poly' : sklearn.pipeline.Pipeline([('poly', sklearn.preprocessing.PolynomialFeatures(degree=3)), ('linear', sklearn.linear_model.LinearRegression())]),
        'logit' : sklearn.linear_model.LogisticRegression(),
        'linearSVM' : sklearn.svm.LinearSVC(),
        'radialSVM' : sklearn.svm.SVC(kernel = 'rbf'),
        #'poly2SVM' : sklearn.svm.SVC(kernel = 'poly', degree = 2), #slow
        'DecisionTree' : sklearn.tree.DecisionTreeClassifier(),
        'RandomForest' : sklearn.ensemble.RandomForestClassifier(),
        'ridge' : sklearn.linear_model.RidgeClassifierCV(),
    }
    for name, model in models.items():
        print("Fitting {}".format(name))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[xVars],train[yVar])

    fig, ax = plt.subplots()
    print("Proccessing")
    for name, model in models.items():
        indices.append(name)
        if name in {'linear', 'logit', 'poly'}:
            test[name] = getPredictions(model, test, yVar, xVars, nonInt = True)
            test[name] = test[name].apply(lambda x: 1 if x > 0.5 else 0)
        else:
            test[name] = getPredictions(model, test, yVar, xVars)

        results['auc'].append(sklearn.metrics.roc_auc_score(test[name], test[yVar]))
        results['AP'].append(sklearn.metrics.average_precision_score(test[name], test[yVar]))
        results['PRE'].append(sklearn.metrics.precision_score(test[name], test[yVar]))
        results['RE'].append(sklearn.metrics.recall_score(test[name], test[yVar]))

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(test[name], test[yVar])
        ax.plot(fpr, tpr, label = "{} (AUC ${:.3f}$)".format(name, results['auc'][-1]))
    resultsDf = pandas.DataFrame(results, index = indices)
    print("The stats table for the models is:")
    print(resultsDf)
    print("The variable importances for the Random Forest are:")
    for i, vName in enumerate(xVars):
        print("{}:  {:.4f}".format(vName.ljust(9, ' '), models['RandomForest'].feature_importances_[i]))
    print("The full report of logit using the hold out data is:")
    print(sklearn.metrics.classification_report(test['logit'], test[yVar]))
    print("The full report of linearSVM using the hold out data is:")
    print(sklearn.metrics.classification_report(test['linearSVM'], test[yVar]))
    ax.set_title('Receiver Operating Characteristics')
    plt.plot([0,1], [0,1], color = 'k')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc = 'lower right')
    plt.savefig("{}/q3part2.{}".format(outputDir, outFormat), format = outFormat)
    #plt.show()
    plt.close()

def question3():
    np.random.seed(seed=1234)
    df = pandas.read_csv(ojFName).dropna(how = 'any')
    df[['guilt', 'dem', 'rep', 'ind', 'age', 'female', 'black', 'hispanic']] = df[['guilt', 'dem', 'rep', 'ind', 'age', 'female', 'black', 'hispanic']].astype('int')
    incMap = {
        'REFUSED/NO ANSWER' : 40,
        'UNDER $15,000' : 10,
        '$15,000-$30,000' : 22.5,
        '$30,000-$50,000' : 40,
        '$50,000-$75,000' : 62.5,
        'OVER $75,000' : 80,
        }
    educMAp = {
        'NOT A HIGH SCHOOL GRAD' : 0,
        'HIGH SCHOOL GRAD' : 1,
        'SOME COLLEGE(TRADE OR BUSINESS)' : 2,
        'COLLEGE GRAD AND BEYOND' : 3,
        'REFUSED' : 1,
        }
    df['income'] = df['income'].apply(lambda x: incMap[x])
    df['educ'] = df['educ'].apply(lambda x: educMAp[x])
    train, test = getholdOut(df, .7)
    q3part1(train, test)
    q3part2(train, test)


def main():
    question1()
    question2()
    question3()

if __name__ == '__main__':
    main()
