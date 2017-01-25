#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:39:48 2017

@author: wli
"""

import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt


'''
--------------------------------------------------------------------
Excercise 1
--------------------------------------------------------------------
'''

# Parameters
pts = np.loadtxt('incomes.txt')
dist_pts = np.linspace(0, 150000, 500)
mu = 9 
sig = 0.3


cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
    
'''
--------------------------------------------------------------------
(a) Histogram
--------------------------------------------------------------------
'''

print("Excercise 1a")

fig, ax = plt.subplots()
weights = (100 / len(pts)) * np.ones(len(pts))
count, bins, ignored = plt.hist(pts, bins=30, normed=False, weights=weights)
plt.title('Distribution of annual income', fontsize=16)
plt.xlabel('Annual income(\$s)')
plt.ylabel('Percent of income (%)')
plt.ylim([0,9])

output_path = os.path.join(output_dir, 'Distribution of Annual Income')
plt.savefig(output_path)
plt.show()
plt.close()


'''
--------------------------------------------------------------------
(b) Lognormal PDF
--------------------------------------------------------------------
'''

def lognorm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the lognormally distributed random
             variable
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    FILES CREATED BY THIS FUNCTION: None
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.lognorm.cdf(cutoff, s = sigma, loc = mu, scale = np.exp(mu))
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut
    return pdf_vals
        
print("Excercise 1b")
    
# Plot
fig, ax = plt.subplots()
ax.grid()
plt.plot(dist_pts, lognorm_pdf(dist_pts, mu, sig, "None"),
         linewidth=2, color='r', label='x: $\mu$=9,$\sigma$=0.3')
plt.legend(loc='upper right')
plt.xlim(0,150000)
plt.xlabel('x')
plt.ylabel('$f(x|\mu=9,\sigma=.3)$')
plt.title('Log normal pdf. $\mu={:.3f}$, $\sigma={:.3f}$'
          .format(mu, sig))
plt.tight_layout()
output_path = os.path.join(output_dir, "Lognormal PDF")
plt.savefig(output_path)
plt.show()
plt.close()


# Log likelihood function for the normal distribution
def log_lik_norm(xvals, mu, sigma):
    log_lik_val = np.log(lognorm_pdf(xvals, mu, sigma, "None")).sum()
    return log_lik_val

print("Log likelihood is: ", log_lik_norm(pts, mu, sig))


'''
--------------------------------------------------------------------
(c) MLE
--------------------------------------------------------------------
'''

def crit(params, *args):
    mu, sig = params
    xvals, cutoff= args
    log_lik_val = log_lik_norm(xvals, mu, sig)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

# To estimate the results
params_init = np.array([mu, sig])
mle_args = (pts, pts.max())
results = opt.minimize(crit, params_init, args=(mle_args), method = "SLSQP", bounds=((None, None), (0.001, None)))

# Run the minimiser again using previous results as initial parameters and method "BFGS" to 
# get variance-covariance matrix
params_init = results.x
results = opt.minimize(crit, params_init, args=(mle_args), method = "BFGS")
mu_MLE, sig_MLE = results.x
vcv_mle = results.hess_inv

print("Excercise 1c")
print('Estimated mu: ', mu_MLE, 'sigma: ', sig_MLE)
print('VCV(MLE) = ', vcv_mle)
print('MLE estimation: ', log_lik_norm(pts, mu_MLE, sig_MLE))


# Compare three plots on the same graph
fig, ax = plt.subplots()
ax.grid()
hist_wgts = (1 / len(pts)) * np.ones(len(pts))
num_bins = 30
plt.hist(pts, num_bins, normed=True)
plt.title('Annual Income distribution for MACSS Graduates', fontsize=16)
plt.xlabel(r'Annual income')
plt.ylabel(r'Percent of students')

plt.plot(dist_pts, lognorm_pdf(dist_pts, mu, sig, 'None'), linewidth=2, color='r', label='Guess: $\mu$=9.00, $\sigma$=0.30')
plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_MLE, sig_MLE, 'None'), linewidth=2, color='g', label='MLE: $\mu$=11.33, $\sigma$=0.21')
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "1c")
plt.savefig(output_path)
plt.show()
plt.close()



'''
------------------------------------------------------------------------
(d) chi squared test
------------------------------------------------------------------------
'''
# Ratio test to determine the probability that the data came from the distribution 
# in part b
log_lik_h0 = log_lik_norm(pts, mu, sig)
log_lik_mle = log_lik_norm(pts, mu_MLE, sig_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)

print("Excercise 1d")
print("Probability that the data came from the distribution in part b = {:.4}: ".format(pval_h0))


'''
------------------------------------------------------------------------
(e) Predictions
------------------------------------------------------------------------
'''

def lognorm_cdf(x, mu, sigma):
    lognorm = sp.stats.lognorm(loc = mu_MLE, scale = np.exp(mu), s = sigma)
    return lognorm.cdf(x)

print("Excercise 1e")

part1 = 1-sts.lognorm.cdf(100000, sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
print("Probability that I will earn more than $100,000 is", round(part1,4)*100,'%')
part2 = sts.lognorm.cdf(75000, sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
print("Probability that I will earn less than $75,000 is", round(part2,4)*100,'%')




'''
--------------------------------------------------------------------
Excercise 2
--------------------------------------------------------------------
'''

sick= pd.read_csv('sick.txt', header = 0)
    
    
'''
--------------------------------------------------------------------
(a)
--------------------------------------------------------------------
'''
# Initialise parameters
beta0_init = 0
beta1_init = 0
beta2_init = 0
beta3_init = 0
sig_init = .1


def norm_pdf(xvals, mu, sigma):        
    pdf_vals    = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))) 
    
    return pdf_vals

# Define log likelihood function
def log_like_sick(data, b0, b1, b2, b3, sigma):
    err = data['﻿sick']- (b0+ b1*data['age'] + b2*data['children']+ b3*data['avgtemp_winter'])
    log_lik_val = np.log(norm_pdf(err, 0, sigma)).sum()
    
    return log_lik_val

# Set up criterion function
def crit2(params, *args):
    b0, b1, b2, b3, sigma = params
    data = args[0]
    log_lik_val = log_like_sick(data, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val
    
#Find MLE    
params_init = np.array([beta0_init, beta1_init, beta2_init, beta3_init, sig_init])
bnds = ((None, None),(None, None),(None, None),(None, None),(0.00001, None))

results = opt.minimize(crit2, params_init, args=sick, method = "SLSQP", bounds=bnds)
b0, b1, b2, b3, sigma = results.x

results = opt.minimize(crit2,  results.x, args=sick, method = "L-BFGS-B", bounds=bnds)
b0, b1, b2, b3, sigma = results.x

print("Excercise 2a")
print('beta0 =', b0, ' beta1 =', b1, 'beta2 =', b2, 'beta3 =', b3, 'sigma =', sigma)
print('The variance-covariance matrix is: ',results.hess_inv.todense())
print('Log-likelihood: ', log_like_sick(sick, *results.x))


'''
--------------------------------------------------------------------
Question 2: b
--------------------------------------------------------------------
'''

log_lik_h0 = log_like_sick(sick, 1, 0, 0, 0, 0.1)
log_lik_mle = log_like_sick(sick, *results.x)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)

print("Excercise 2b")
print('chi squared of H0 with 5 degrees of freedom p-value = {:.4}'.format(pval_h0))

