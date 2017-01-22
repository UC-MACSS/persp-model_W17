import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas
import seaborn
import os

seaborn.set_palette("pastel")

targetFile = 'sick.txt'

outputDir = 'images'
outFormat = 'pdf'

def normPDF(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( -(x - mu)**2 / (2 * sigma**2))

def normLogLike(sick, age, children, temp, b0, b1, b2, b3, sigma):
    x = sick - b0 - b1 * age - b2 * children - b3 * temp
    log_lik_val = np.log(normPDF(x, 0, sigma)).sum()
    return log_lik_val

def crit(params, *args):
    b0, b1, b2, b3, sigma = params
    sick, age, children, temp = args
    log_lik_val = normLogLike(sick, age, children, temp, b0, b1, b2, b3, sigma)
    neg_log_lik_val = - log_lik_val
    return(neg_log_lik_val)

def estimate(inDF):
    params_init = np.array([1, 0, 0, 0, 1])
    mle_args = (inDF['sick'], inDF['age'], inDF['children'], inDF['temp'])
    bnds = ((None, None), (None, None), (None, None), (None, None), (.000001, None))
    results = sp.optimize.minimize(crit, params_init, method='SLSQP', args = mle_args, bounds = bnds)
    results = sp.optimize.minimize(crit, results.x, method='BFGS', args = mle_args, bounds = bnds)
    return results

def main():
    os.makedirs(outputDir, exist_ok = True)
    incomeDF = pandas.read_csv(targetFile, skiprows = 1, names = ['sick', 'age', 'children', 'temp'])
    #print(incomeDF)
    r = estimate(incomeDF)
    print(r['hess_inv'])

if __name__ == '__main__':
    main()
