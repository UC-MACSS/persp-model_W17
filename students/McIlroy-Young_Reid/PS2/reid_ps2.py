#!/usr/local/bin/python3.5

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import seaborn
import os
import subprocess
import pandas
import warnings

seaborn.set_palette("pastel")

targetFile = 'incomes.txt'
targetFile2 = 'sick.txt'

outputDir = 'images'
outFormat = 'pdf'

templateFile = 'LaTeX_probset_template.tex'
outputPDF = "reid-PS2.pdf"

sigmaGiven = .3
muGiven = 9

def lognormPDF(x, mu, sigma):
    logNorm = sp.stats.lognorm(scale = np.exp(mu), s = sigma)
    return logNorm.pdf(x)

def lognormCDF(x, mu, sigma):
    logNorm = sp.stats.lognorm(scale = np.exp(mu), s = sigma)
    return logNorm.cdf(x)

def logNormLogLike(x, mu, sigma):
    log_lik_val = np.log(lognormPDF(x, mu, sigma)).sum()
    return log_lik_val

def crit(params, *args):
    mu, sigma = params
    xvals = args
    log_lik_val = logNormLogLike(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    return(neg_log_lik_val)

def estimate(muI, sigmaI, inAr):
    params_init = np.array([muI, sigmaI])
    mle_args = (inAr)
    bnds = ((None, None), (0.00001, None))
    results = sp.optimize.minimize(crit, params_init, method='SLSQP', args = (mle_args), bounds = bnds)
    with warnings.catch_warnings():
        #I want a hessian
        warnings.simplefilter("ignore")
        results = sp.optimize.minimize(crit, results.x, method='BFGS', args = (mle_args), bounds = bnds)
    return results

def getIncomes(target):
    with open(target) as f:
        values = [float(l) for l in f]
    return np.array(values)

def parta(inAr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(inAr, bins = 30, normed = True)
    ax.set_title("Income Data Histogram")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Income ($USD$)")
    plt.xlim([inAr.min(), inAr.max()])
    plt.savefig("{}/q1_a.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()
    return "# Part (a)\n\n![Question 1 (a) histogram]({}/q1_a.{})\n\n".format(outputDir, outFormat)

def partb(inAr):
    dist_pts = np.linspace(0, inAr.max(), 10000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(dist_pts, lognormPDF(dist_pts, muGiven, sigmaGiven))
    ax.set_title("Lognormal PDF with mu = {} and sigma = {}".format(muGiven, sigmaGiven))
    ax.set_xlabel("Income ($USD$)")
    ax.set_ylabel("Ratio")
    plt.savefig("{}/q1_b.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()
    llv = logNormLogLike(inAr, muGiven, sigmaGiven)
    retString = "# Part (b)\n\nThe Log-likelihood of mu = {:.2f} and sigma = {:.2f}, is {:.2f}".format(muGiven, sigmaGiven, llv)
    print(retString)

    return retString + "\n\n![Question 1 (c) histogram]({}/q1_b.{})\n\n".format(outputDir, outFormat)

def partc(inAr):
    dist_pts = np.linspace(0, inAr.max(), 10000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(inAr, bins = 30, normed = True, label="Provided data histogram")
    ax.set_title("Income Data Histogram")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Income ($USD$)")
    plt.xlim([0, inAr.max()])

    plt.plot(dist_pts, lognormPDF(dist_pts, muGiven, sigmaGiven), label = "mu = {:.2f}, sigma = {:.2f} lognormal PDF".format(muGiven, sigmaGiven))
    r = estimate(muGiven, sigmaGiven, inAr)
    plt.plot(dist_pts, lognormPDF(dist_pts, *r.x), label = "mu = {:.2f}, sigma = {:.2f} lognormal PDF".format(*r.x))
    ax.legend()
    plt.savefig("{}/q1_c.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()

    llv = logNormLogLike(inAr, r.x[0], r.x[1])
    hes = r['hess_inv']

    retString = "# Part (c)\n\nThe optimizer gives mu = {:.2f} and sigma = {:.2f}, with a log-likelihood of {:.2f}. The associated var-covar matrix is:\n\n    {}\n    {}".format(r.x[0], r.x[1], llv, hes[0], hes[1])

    print(retString)
    return r.x[0], r.x[1], retString + "\n\n![Question 1 (c) histogram]({}/q1_c.{})\n\n".format(outputDir, outFormat)

def partd(inAr, mu, sigma):
    log_lik_h0 = logNormLogLike(inAr, muGiven, sigmaGiven)
    log_lik_mle = logNormLogLike(inAr, mu, sigma)
    pval_h0 = 1.0 - sp.stats.chi2.cdf(2 * (log_lik_mle - log_lik_h0), 2)
    retString = "# Part (d)\n\nThe probabilty that the data came from the distrubution in Part (b) is: {:.10f}".format(pval_h0)

    print(retString)
    return retString + '\n\n'

def parte(mu, sigma):
    retString = "# Part (d)\n\nThe probabilty that I will earn more than 100000 USD is {:.3f} and that I will earn less than 75000 USD is {:.3f}".format(1 - lognormCDF(100000, mu, sigma), lognormCDF(75000, mu, sigma))

    print(retString)
    return retString + '\n\n'

def normPDF(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( -(x - mu)**2 / (2 * sigma**2))

def normLogLike(sick, age, children, temp, b0, b1, b2, b3, sigma):
    x = sick - b0 - b1 * age - b2 * children - b3 * temp
    log_lik_val = np.log(normPDF(x, 0, sigma)).sum()
    return log_lik_val

def crit2(params, *args):
    b0, b1, b2, b3, sigma = params
    sick, age, children, temp = args
    log_lik_val = normLogLike(sick, age, children, temp, b0, b1, b2, b3, sigma)
    neg_log_lik_val = - log_lik_val
    return(neg_log_lik_val)

def estimate2(inDF):
    params_init = np.array([1, 0, 0, 0, 1]) #sigma = .1 does not converge
    mle_args = (inDF['sick'], inDF['age'], inDF['children'], inDF['temp'])
    bnds = ((None, None), (None, None), (None, None), (None, None), (.000001, None))
    results = sp.optimize.minimize(crit2, params_init, method='SLSQP', args = mle_args, bounds = bnds)
    with warnings.catch_warnings():
        #I want a hessian
        warnings.simplefilter("ignore")
        results = sp.optimize.minimize(crit2, results.x, method='BFGS', args = mle_args, bounds = bnds)
    return results

def question2():
    incomeDF = pandas.read_csv(targetFile2, skiprows = 1, names = ['sick', 'age', 'children', 'temp'])
    r = estimate2(incomeDF)
    print(normLogLike(incomeDF['sick'], incomeDF['age'], incomeDF['children'], incomeDF['temp'], *r.x))
    partAString = """# Part (a)

The estimated values, with a log-likelihood of {:.3f} are:

    Beta_0 = {:.3f}
    Beta_1 = {:.3f}
    Beta_2 = {:.3f}
    Beta_3 = {:.3f}
    sigma = {:.3f}

""".format(normLogLike(incomeDF['sick'], incomeDF['age'], incomeDF['children'], incomeDF['temp'], *r.x), *r.x)

    partAString += "The var-covar matrix is:\n\n    {}".format('\n    '.join([str(r) for r in r['hess_inv']]))
    print(partAString)

    log_lik_h0 = normLogLike(incomeDF['sick'], incomeDF['age'], incomeDF['children'], incomeDF['temp'], 1, 0, 0, 0, 1)
    log_lik_mle = normLogLike(incomeDF['sick'], incomeDF['age'], incomeDF['children'], incomeDF['temp'], *r.x)

    pval_h0 = 1.0 - sp.stats.chi2.cdf(2 * (log_lik_mle - log_lik_h0), 2)

    partBString = "The probabilty that Beta_0 is 1, Beta_1, Beta_2 and Beta_3 are 0 and sigma is .1, is: {:.10f}".format(pval_h0)

    print(partBString)

    return partAString + '\n\n' + partBString

def main():
    os.makedirs(outputDir, exist_ok = True)
    incomeAr = np.loadtxt(targetFile)
    print("Making report")
    #Header in YAML for pandoc LaTeX template
    s = """---
title: Problem Set 2
author: Reid McIlroy-Young
date: January 22, 2017
---\n\n# Question 1\n\n"""

    s += parta(incomeAr)
    s += partb(incomeAr)
    muCalc, sigmaCalc, partCStr = partc(incomeAr)
    s += partCStr
    s += partd(incomeAr, muCalc, sigmaCalc)
    s += parte(muCalc, sigmaCalc)
    s += question2()
    try:
        p = subprocess.Popen(['pandoc', '-o', outputPDF, '--from=markdown', '--template={}'.format(templateFile)], stdin=subprocess.PIPE)
        p.communicate(input=s.encode('utf-8'))
    except FileNotFoundError:
        print("pandoc not found, cannot create PDF")

if __name__ == '__main__':
    main()
