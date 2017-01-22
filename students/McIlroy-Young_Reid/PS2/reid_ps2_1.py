import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import seaborn
import os

seaborn.set_palette("pastel")

targetFile = 'incomes.txt'

outputDir = 'images'
outFormat = 'pdf'

sigmaGiven = .3
muGiven = 9


def lognormPDF(x, mu, sigma):
    logNorm = sp.stats.lognorm(scale = np.exp(mu), s = sigma)
    return logNorm.pdf(x)

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

def partb(inAr):
    dist_pts = np.linspace(0, 100000, 10000)
    plt.plot(dist_pts, lognormPDF(dist_pts, muGiven, sigmaGiven))
    plt.savefig("{}/q1_b.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()
    llv = logNormLogLike(inAr, muGiven, sigmaGiven)
    print("Part B:\nThe Log-likelihood of \u03BC = {:.2f} and \u03C3 = {:.2f}, is {:.2f}".format(muGiven, sigmaGiven, llv))

def partc(inAr):
    dist_pts = np.linspace(0, inAr.max(), 10000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(inAr, bins = 30, normed = True)
    ax.set_title("Income Data Histogram")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Income ($USD$)")
    plt.xlim([0, inAr.max()])

    plt.plot(dist_pts, lognormPDF(dist_pts, muGiven, sigmaGiven))
    r = estimate(muGiven, sigmaGiven, inAr)
    plt.plot(dist_pts, lognormPDF(dist_pts, r.x[0], r.x[1]))
    plt.savefig("{}/q1_c.{}".format(outputDir, outFormat), format = outFormat)
    plt.close()

    llv = logNormLogLike(inAr, r.x[0], r.x[1])
    hes = r['hess_inv']
    print("Part C:\nThe optimizer gives \u03BC = {:.2f} and \u03C3 = {:.2f}, with a log-likelihood of {:.2f}.".format(r.x[0], r.x[1], llv))
    print("The associated var-covar matrix is:\n\t{}\n\t{}".format(hes[0], hes[1]))

    return r.x[0], r.x[1]

def partd(mu, sigma):
    pass

def main():
    os.makedirs(outputDir, exist_ok = True)
    incomeAr = np.loadtxt(targetFile)

    parta(incomeAr)
    partb(incomeAr)
    partc(incomeAr)

if __name__ == '__main__':
    main()
