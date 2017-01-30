#!/usr/local/bin/python3.5

import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate
import numpy as np
import seaborn
import os
import subprocess
import pandas
import warnings

import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt

seaborn.set_palette("pastel")

targetFile = 'incomes.txt'
targetFile2 = 'sick.txt'

outputDir = 'images'
outFormat = 'pdf'

templateFile = 'LaTeX_probset_template.tex'
outputPDF = "reid-PS3.pdf"

sigmaGiven = .3
muGiven = 9

lowIncome = 75000
highIncome = 100000

p_lowInit = p_midInit = p_highInit = 1/3

def lognormPDF(x, mu, sigma):
    logNorm = sp.stats.lognorm(scale = np.exp(mu), s = sigma)
    return logNorm.pdf(x)

def lognormCDF(x, mu, sigma):
    logNorm = sp.stats.lognorm(scale = np.exp(mu), s = sigma)
    return logNorm.cdf(x)

def data_moments(x):
    return np.array([[x.mean()], [x.var()]])

def model_moments(mu, sigma, cutoff):
    #return np.array([[logNormMean(mu, sigma)], [logNormSTD(mu, sigma)]])
    def xMoment(x):
        return x * lognormPDF(x, mu, sigma)

    def x2Moment(x):
        return np.square(x - mean_model) * lognormPDF(x, mu, sigma)

    mean_model, m_m_err = intgr.quad(xMoment, 0, cutoff, limit = 100)
    var_model, v_m_err = intgr.quad(x2Moment, 0,  cutoff, limit = 100)
    return np.array([[mean_model], [var_model]])

def data_moments3(x):
    n = len(x)
    p_low = p_mid = p_high = None
    #Since x is sorted
    for index, val in enumerate(x):
        if p_low is None and val > lowIncome:
            p_low = index
        elif val > highIncome:
            p_mid = index - p_low
            p_high = n - index
            break
    return np.array([[p_low], [p_mid], [p_high]]) / n

def model_moments3(mu, sigma):
    p_low = lognormCDF(lowIncome, mu, sigma)
    p_high = 1 - lognormCDF(highIncome, mu, sigma)
    p_mid = 1 - p_high - p_low

    return np.array([[p_low], [p_mid], [p_high]])

def err_vec(xvals, mu, sigma, threeMode = False):
    cutoff = xvals.max()
    if threeMode:
        moms_data = data_moments3(xvals)
        moms_model = model_moments3(mu, sigma)
    else:
        moms_data = data_moments(xvals)
        moms_model = model_moments(mu, sigma, cutoff)

    err_vec = (moms_model - moms_data) / moms_data
    return err_vec

def criterion(params, *args):
    mu, sigma = params
    x, W, threeMode = args
    err = err_vec(x, mu, sigma, threeMode = threeMode)
    crit_val = np.matmul(np.matmul(err.T, W), err)
    return crit_val

def estimate(x, W_hat, muStart, sigmaStart, threeMode = False, method = 'L-BFGS-B'):

    bnds = ((1e-10, None), (1e-10, None))
    params_init = (muStart, sigmaStart)
    gmm_args = (x, W_hat, threeMode)
    return sp.optimize.minimize(criterion, params_init, args=(gmm_args), method = method, bounds = bnds)

def plotHistEtc(inAr, fName, logNorms = None, title = ""):
    if logNorms is None:
        logNorms = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(inAr, bins = 30, normed = True)
    ax.set_title(title)
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Income ($USD$)")
    plt.xlim([0, inAr.max()])

    dist_pts = np.linspace(0, inAr.max(), 10000)
    for mu, sigma, label in logNorms:
        plt.plot(dist_pts, lognormPDF(dist_pts, mu, sigma), label = "mu = {:.2f}, sigma = {:.2f}\n{}".format(mu, sigma, label))
    if len(logNorms) > 0:
        ax.legend(loc = 'upper left')
    plt.savefig("{}/{}.{}".format(outputDir, fName, outFormat), format = outFormat)

    plt.close()

def parta(inAr):
    plotHistEtc(inAr, 'q1_a', title = 'Question 1 (a): Income Data Histogram')
    return "# Part (a)\n\n![Question 1 (a) histogram]({}/q1_a.{})\n\n".format(outputDir, outFormat)


def partb(inAr):
    r = estimate(inAr, np.eye(2), muGiven, sigmaGiven)

    plotHistEtc(inAr, 'q1_b', logNorms = [(*r.x, "one step estimate, two moments")], title = 'Question 1 (b): Income Data Histogram and One Step Estimated Distribution')

    retString = "# Part (b)\n\nThe the two moments, one step GMM estimate of mu = {:.2f} and sigma = {:.2f}, has a criterion function value of {:.2e}. ".format(*r.x, criterion(r.x, inAr, np.eye(2), False)[0][0])

    retString += "This gives a relatvely good fit, which is evdent both from the citerion function's low value, order of $10^{-13}$ and how well the PDF matchs the histogram."

    print(retString)

    return r.x[0], r.x[1], retString + "\n\n![Question 1 (b) plot]({}/q1_b.{})\n\n".format(outputDir, outFormat)

def partc(inAr, muStart, sigmaStart):
    err = err_vec(inAr, muStart, sigmaStart)
    Sigma_2 = np.matmul(err, err.T) / len(inAr)
    W_hat = lin.pinv(Sigma_2)
    r = estimate(inAr, W_hat, muStart, sigmaStart)

    plotHistEtc(inAr, 'q1_c', logNorms = [(*r.x, "two step estimate, two moments"), (muStart, sigmaStart, "one step estimate, two moments")], title = 'Question 1 (c): Income Data Histogram with One and Two Step Estimated Distribution')

    retString = "# Part (c)\n\nThe the two moments, two step GMM estimate of mu = {:.2f} and sigma = {:.2f}, has a criterion function value of {:.2e}. ".format(*r.x, criterion(r.x, inAr, W_hat, False)[0][0])

    dm = data_moments(inAr).flatten()
    mmOld = model_moments(muStart, sigmaStart, inAr.max()).flatten()
    mmNew = model_moments(*r.x, inAr.max()).flatten()

    retString += """This gives a Similar fit to part (b), although with a larger criterion, since this appears to be a strong minuma for both mu and sigma. The values of the moments from both approaches are:

Source   | mean   | std
---------|--------|------
data     | {:.3f} | {:.0f}
one step | {:.3f} | {:.0f}
two step | {:.3f} | {:.0f}
""".format(dm[0], dm[1], mmOld[0], mmOld[1], mmNew[0], mmNew[1])

    print(retString)
    return retString + "\n\n![Question 1 (c) plot]({}/q1_c.{})\n\n".format(outputDir, outFormat)


def partd(inAr, muStart, sigmaStart):
    r = estimate(inAr, np.eye(3), muStart, sigmaStart, threeMode = True)
    plotHistEtc(inAr, 'q1_d', logNorms = [(*r.x, "one step estimate, three moments")], title = 'Question 1 (d): Income Data Histogram and One Step Estimated Distribution')

    retString = "# Part (d)\n\nThe the three moments, one step GMM estimate of mu = {:.2f} and sigma = {:.2f}, has a criterion function value of {:.2e}. ".format(*r.x, criterion(r.x, inAr, np.eye(3), True)[0][0])

    retString += "This gives a relatvely good fit, which is evdent both from the citerion function's low value, order of $10^{-13}$ and how well the PDF matchs the histogram."

    print(retString)

    return r.x[0], r.x[1], retString + "\n\n![Question 1 (d) plot]({}/q1_d.{})\n\n".format(outputDir, outFormat)

def parte(inAr, muStart, sigmaStart):
    err = err_vec(inAr, muStart, sigmaStart, threeMode = True)
    Sigma_2 = np.matmul(err, err.T) / len(inAr)
    W_hat = lin.pinv(Sigma_2)
    r = estimate(inAr, W_hat, muStart, sigmaStart, threeMode = True)

    plotHistEtc(inAr, 'q1_e', logNorms = [(*r.x, "two step estimate, three moments"), (muStart, sigmaStart, "one step estimate, three moments")], title = 'Question 1 (e): Income Data Histogram with One and Two Step Estimated Distribution')

    retString = "# Part (e)\n\nThe the two moments, two step GMM estimate of mu = {:.2f} and sigma = {:.2f}, has a criterion function value of {:.2e}. ".format(*r.x, criterion(r.x, inAr, W_hat, True)[0][0])

    dm = data_moments3(inAr).flatten()
    mmOld = model_moments3(muStart, sigmaStart).flatten()
    mmNew = model_moments3(*r.x).flatten()

    print(dm)

    retString += """This gives a Similar fit to part (b), although with a larger criterion, since this appears to be a strong minuma for both mu and sigma. The values of the moments from both approaches are:

Source   | Fraction Low | Fraction Medium | Fraction High
---------|--------------|-----------------|------------
data     | {:.6f}       | {:.6f}          | {:.6f}
one step | {:.6f}       | {:.6f}          | {:.6f}
two step | {:.6f}       | {:.6f}          | {:.6f}
""".format(dm[0], dm[1], dm[2], mmOld[0], mmOld[1], mmOld[2], mmNew[0], mmNew[1], mmOld[2])

    print(retString)
    return retString + "\n\n![Question 1 (e) plot]({}/q1_e.{})\n\n".format(outputDir, outFormat)


def err_vec2(sick, age, children, temp, b0, b1, b2, b3, sigma):
    moms_data = sick
    moms_model = b0 + b1 * age + b2 * children + b3 * temp
    return moms_data - moms_model

def crit2(params, *args):
    b0, b1, b2, b3, sigma = params
    sick, age, children, temp = args
    err = err_vec2(sick, age, children, temp, b0, b1, b2, b3, sigma)
    crit_val = np.matmul(err.T, err) #Since Identity
    return crit_val

def estimate2(inDF):
    params_init = np.array([1, 0, 0, 0, 1]) #sigma = .1 does not converge
    mle_args = (inDF['sick'], inDF['age'], inDF['children'], inDF['temp'])
    bnds = ((None, None), (None, None), (None, None), (None, None), (.000001, None))
    results = sp.optimize.minimize(crit2, params_init, method='L-BFGS-B', args = mle_args, bounds = bnds)
    return results

def question2():
    incomeDF = pandas.read_csv(targetFile2, skiprows = 1, names = ['sick', 'age', 'children', 'temp'])
    r = estimate2(incomeDF)

    retString = """#Question 2

# Part (a)

The estimated values, with a criterion function value of {:.2e} are:

    Beta_0 = {:.3f}
    Beta_1 = {:.3f}
    Beta_2 = {:.3f}
    Beta_3 = {:.3f}
    sigma = {:.3f}

""".format(crit2(r.x, incomeDF['sick'], incomeDF['age'], incomeDF['children'], incomeDF['temp']), *r.x)

    print(retString)

    return retString

def main():
    os.makedirs(outputDir, exist_ok = True)
    incomeAr = np.loadtxt(targetFile)
    incomeAr.sort()
    print("Making report")
    #Header in YAML for pandoc LaTeX template
    s = """---
title: Problem Set 3
author: Reid McIlroy-Young
date: January 30, 2017
---\n\n# Question 1\n\n"""

    s += parta(incomeAr)
    mu_GMM1, sig_GMM1, sb = partb(incomeAr)
    s += sb
    s += partc(incomeAr, mu_GMM1, sig_GMM1)
    mu_GMM3, sig_GMM3, sd = partd(incomeAr, 12, .3) #Close to the minuma, but not too close
    s += sd
    s += parte(incomeAr, mu_GMM3, sig_GMM3)

    s += """\\newpage\n# Part (f)

All four methods I used result in criterion function values below $10^{-5}$ and thus in very similar estimations. This is, as noted before due to a strong minuma being found, determining if it is a global minuma is beyond the scope of the assignment. But if I had to choose a best fit to use I would use the one step, thee moment fit as it matchs the data to 6 decimal places and is much faster to compute than the two moment ones, although I believe removing the integrals and using an analytically derived CDF would increase the run time considerably.\n\n"""

    s += question2()
    try:
        p = subprocess.Popen(['pandoc', '-o', outputPDF, '--from=markdown'], stdin=subprocess.PIPE)
        p.communicate(input=s.encode('utf-8'))
    except FileNotFoundError:
        print("pandoc not found, cannot create PDF")

if __name__ == '__main__':
    main()
