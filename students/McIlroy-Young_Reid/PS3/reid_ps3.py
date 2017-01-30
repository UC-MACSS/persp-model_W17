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
outputPDF = "reid-PS2.pdf"

sigmaGiven = .3
muGiven = 9

def lognormPDF(x, mu, sigma):
    logNorm = sp.stats.lognorm(scale = np.exp(mu), s = sigma)
    return logNorm.pdf(x)

def lognormCDF(x, mu, sigma):
    logNorm = sp.stats.lognorm(scale = np.exp(mu), s = sigma)
    return logNorm.cdf(x)

def data_moments(x):
    return np.array([[x.mean()], [x.var()]])

def model_moments(mu, sigma, cutoff):
    def xMoment(x):
        return x * lognormPDF(x, mu, sigma)

    def x2Moment(x):
        return np.square(x - mean_model) * lognormPDF(x, mu, sigma)

    mean_model, m_m_err = intgr.quad(xMoment, 0, cutoff, limit = 100)
    var_model, v_m_err = intgr.quad(x2Moment, 0,  cutoff, limit = 100)

    return np.array([[mean_model], [var_model]])

def err_vec(xvals, mu, sigma, simple = False):
    cutoff = xvals.max()
    moms_data = data_moments(xvals)
    moms_model = model_moments(mu, sigma, cutoff)

    err_vec = (moms_model - moms_data) / moms_data
    return err_vec


def criterion(params, *args):
    mu, sigma = params
    x, W = args
    err = err_vec(x, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

def estimate(x, W_hat, muStart, sigmaStart, method = 'L-BFGS-B'):
    bnds = ((1e-10, None), (1e-10, None))
    params_init = np.array([muGiven, sigmaGiven])
    gmm_args = (x, W_hat)
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
    for mu, sigma in logNorms:
        plt.plot(dist_pts, lognormPDF(dist_pts, mu, sigma), label = "mu = {:.2f}, sigma = {:.2f} lognormal PDF".format(mu, sigma))

    ax.legend()
    plt.show()
    plt.savefig("{}/{}.{}".format(outputDir, fName, outFormat), format = outFormat)
    plt.close()


def parta(inAr):
    plotHistEtc(inAr, 'q1_a', title = 'Question 1 (a): Income Data Histogram')
    return "# Part (a)\n\n![Question 1 (a) histogram]({}/q1_a.{})\n\n".format(outputDir, outFormat)


def partb(inAr):
    r = estimate(inAr, np.eye(2), muGiven, sigmaGiven)
    plotHistEtc(inAr, 'q1_b', logNorms = [r.x], title = 'Question 1 (b): Income Data Histogram and One Step Estimated Distribution')
    return r.x

def partc(inAr, muStart, sigmaStart):
    err = err_vec(inAr, muStart, sigmaStart)
    Sigma_2 = np.matmul(err, err.T) / len(inAr)
    W_hat = lin.pinv(Sigma_2)
    r = estimate(inAr, W_hat, muStart, sigmaStart)
    print(r)

    plotHistEtc(inAr, 'q1_c', logNorms = [r.x, (muStart, sigmaStart)], title = 'Question 1 (b): Income Data Histogram with One and Two Step Estimated Distribution')

def main():
    os.makedirs(outputDir, exist_ok = True)
    incomeAr = np.loadtxt(targetFile)
    print("Making report")
    #Header in YAML for pandoc LaTeX template
    s = """---
title: Problem Set 3
author: Reid McIlroy-Young
date: January 30, 2017
---\n\n# Question 1\n\n"""

    #parta(incomeAr)
    mu_GMM1, sig_GMM1 = partb(incomeAr)
    partc(incomeAr, mu_GMM1, sig_GMM1)
    """
    try:
        raise FileNotFoundError
        p = subprocess.Popen(['pandoc', '-o', outputPDF, '--from=markdown', '--template={}'.format(templateFile)], stdin=subprocess.PIPE)
        p.communicate(input=s.encode('utf-8'))
    except FileNotFoundError:
        print("pandoc not found, cannot create PDF")
    """
if __name__ == '__main__':
    main()
