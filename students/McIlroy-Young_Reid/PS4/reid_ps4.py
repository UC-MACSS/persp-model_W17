#!/usr/local/bin/python3.5

import matplotlib.pyplot as plt
import scipy as sp
import scipy.integrate
import numpy as np
import seaborn
import os
import subprocess
import pandas

import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt

seaborn.set_palette("deep")

targetFile = 'incomes.txt'

outputDir = 'images'
outFormat = 'pdf'

outputPDF = "reid-PS4.pdf"

sigmaGiven = .3
muGiven = 9

nGiven = 200
sGiven = 300

root2 = np.sqrt(2)

def LN_pdf(x, mu, sigma):
    return np.exp(- np.square((np.log(x) - mu) / (root2 * sigma))) / (x * sigma * np.sqrt(2 * np.pi))

def data_moments(x):
    try:
        return np.array([[x.mean(axis = 0).mean()], [x.std(axis = 0).mean()]])
    except IndexError:
        return np.array([[x.mean()], [x.std()]])

def err_vec(data_vals, sim_vals):
    moms_data = data_moments(data_vals)
    moms_model = data_moments(sim_vals)
    return (moms_model - moms_data) / moms_data

def genDist(mu, sigma, N, S):
    """
    seedCheck = input("Type 'np.random.seed(seed=1234)':\n")
    if seedCheck != 'np.random.seed(seed=1234)':
        raise KeyboardInterrupt("You typed it wrong")
    else:
        eval(seedCheck)
    """
    np.random.seed(seed=1234)

    return np.exp(sts.norm(mu, sigma).rvs((N, S)))

def criterion(params, *args):
    mu, sigma = params
    xvals, N, S, W_hat = args
    sim_vals = genDist(mu, sigma, N, S)
    err = err_vec(xvals, sim_vals)
    return np.dot(np.dot(err.T, W_hat), err)

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

    dist_pts = np.linspace(1, inAr.max(), 10000)
    for mu, sigma, label in logNorms:
        plt.plot(dist_pts, LN_pdf(dist_pts, mu, sigma), label = "mu = {:.2f}, sigma = {:.2f}\n{}".format(mu, sigma, label))
    if len(logNorms) > 0:
        ax.legend(loc = 'upper left')
    plt.savefig("{}/{}.{}".format(outputDir, fName, outFormat), format = outFormat)

    plt.close()

def parta(inAr):
    plotHistEtc(inAr, 'q1_a', title = 'Question 1 (a): Income Data Histogram')
    return "# Part (a)\n\n![Question 1 (a) histogram]({}/q1_a.{})\n\n\\newpage".format(outputDir, outFormat)

def partb():
    testbIn = np.array([[200.0, 270.0], [180.0, 195.5]])
    testbOut = LN_pdf(testbIn, 5, 1)

    s = "# Part (b)\n\nThe result of `LN_pdf()` being run on the test array, with mu = 5, sigma = 1, is:\n\n    {:.6f}  {:.6f}\n    {:.6f}  {:.6f}".format(*testbOut[0], *testbOut[1])
    print(s)
    return s + '\n\n'

def partc(inAr):
    params_init = np.array([muGiven, sigmaGiven])
    W_hat = np.eye(2)
    smm_args = (inAr, nGiven, sGiven, W_hat)
    r = opt.minimize(criterion, params_init, args=(smm_args),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
    plotHistEtc(inAr, 'q1_c', logNorms = [(*r.x, "one step SMM estimate, two moments")], title = 'Question 1 (c): Income Data Histogram and One Step SMM Estimated Distribution')

    dm = data_moments(inAr).flatten()
    mm = data_moments(genDist(*r.x, nGiven, sGiven)).flatten()

    retString = "# Part (c)\n\nThe the two moments, one step SMM estimate of mu = {:.2f} and sigma = {:.2f}, has a criterion function value of {:.2e}. ".format(*r.x, criterion(r.x, inAr, nGiven, sGiven, W_hat)[0][0])

    retString += """This gives a very good fit, with moments:

Source   | mean   | std
---------|--------|------
data     | {:.3f} | {:.0f}
one step | {:.3f} | {:.0f}
""".format(dm[0], dm[1], mm[0], mm[1])

    print(retString)

    return r.x[0], r.x[1], retString + "\n\n![Question 1 (c) plot]({}/q1_c.{})\n\n".format(outputDir, outFormat)

def partd(inAr, muStart, sigmaStart):
    err = err_vec(inAr, genDist(muStart, sigmaStart, nGiven, sGiven))
    Sigma_2 = np.matmul(err, err.T) / len(inAr)
    W_hat = lin.pinv(Sigma_2)
    #+.01 so the minimizer is happy, does make the result worse though
    params_init = np.array([muStart + .01, sigmaStart + .01])
    smm_args = (inAr, nGiven, sGiven, W_hat)
    r = opt.minimize(criterion, params_init, args=(smm_args),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))

    plotHistEtc(inAr, 'q1_d', logNorms = [(muStart, sigmaStart, "one step SMM estimate, two moments"),(*r.x, "two step SMM estimate, two moments")], title = 'Question 1 (d): Income Data Histogram and Two Step SMM Estimated Distribution')

    dm = data_moments(inAr).flatten()
    mm1 = data_moments(genDist(muStart, sigmaStart, nGiven, sGiven)).flatten()
    mm2 = data_moments(genDist(*r.x, nGiven, sGiven)).flatten()

    retString = "# Part (d)\n\nThe the two moments, two step SMM estimate of mu = {:.2f} and sigma = {:.2f}, has a criterion function value of {:.2e}. ".format(*r.x, criterion(r.x, inAr, nGiven, sGiven, W_hat)[0][0])

    retString += """This gives a worse fit than the first, with moments:

Source   | mean   | std
---------|--------|------
data     | {:.3f} | {:.0f}
one step | {:.3f} | {:.0f}
two step | {:.3f} | {:.0f}
""".format(dm[0], dm[1], mm1[0], mm1[1], mm2[0], mm2[1])

    print(retString)

    return retString + "\n\n![Question 1 (d) plot]({}/q1_d.{})\n\n".format(outputDir, outFormat)


def main():
    os.makedirs(outputDir, exist_ok = True)
    incomeAr = np.loadtxt(targetFile)
    incomeAr.sort()
    print("Making report")
    #Header in YAML for pandoc LaTeX template
    s = """---
title: Problem Set 4
author: Reid McIlroy-Young
date: February 6, 2017
---\n\n# Question 1\n\n"""

    s += parta(incomeAr)
    s += partb()
    mu_SMM1, sig_SMM1, sc = partc(incomeAr)
    s += sc
    s += partd(incomeAr, mu_SMM1, sig_SMM1)

    try:
        p = subprocess.Popen(['pandoc', '-o', outputPDF, '--from=markdown'], stdin=subprocess.PIPE)
        p.communicate(input=s.encode('utf-8'))
    except FileNotFoundError:
        print("pandoc not found, cannot create PDF")

if __name__ == '__main__':
    main()
