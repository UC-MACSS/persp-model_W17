#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 00:14:28 2017

@author: luxihan
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import pandas as pd
import os
np.set_printoptions(precision = 5)

#####Problem 1#####
##Part a)
#Plot the Histogram
income = np.loadtxt("incomes.txt", comments="#", delimiter=",", unpack=False)
weights = np.array([1 / len(income)] * len(income))
count, bins, patches = plt.hist(income, 30, weights = weights)
plt.xlabel("Income")
plt.ylabel("Proportion of People")
plt.title("Income for MACSS Students Histogram", fontsize = 20)
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "hist_income")
plt.savefig(output_path)
plt.show()


def log_norm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, =None if no cutoff is given, otherwise
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
    if cutoff == None:
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(np.log(cutoff), loc=mu, scale=sigma)
            
    pdf_vals    = ((1/(sigma * np.sqrt(2 * np.pi) * xvals) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) /
                    prob_notcut)
    
    return pdf_vals
    
def log_lik_norm(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = log_norm_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val
    
    
def crit(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the normally distributed random
             variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cutoff = args
    log_lik_val = log_lik_norm(xvals, mu, sigma, cutoff)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val

##Part b)
#Plot the hypothesized pdf    
dist_pts = np.linspace(0, 150000, 300000)
mu_1 = 9.0
sig_1 = 0.2
#plt.show()
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_1, sig_1, None),
         linewidth=2, color='r', label='$\mu$={},$\sigma$={}'.format(mu_1, sig_1))
plt.legend(loc='upper right')
plt.ylim([-1e-5, None])
plt.title('PDF of Hypothesized Distribution', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Density')
output_path = os.path.join(output_dir, "hypo_pdf")
plt.savefig(output_path)
plt.show()

##Part c)
count, bins, patches = plt.hist(income, 30, normed = True, label = "Income Bins")
plt.title('Income for MACSS Students', fontsize=20)
plt.xlabel('Income')
plt.ylabel('Density')
#plt.xlim([0, 550])
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_1, sig_1, None),
         linewidth=2, color='r', label='1: $\mu$={},$\sigma$={}'.format(mu_1, sig_1))
plt.legend(loc='upper left')

mu_init = 11.3  # mu_2
sig_init = 0.2  # sig_2
params_init = np.array([mu_init, sig_init])
mle_args = (income, None)

#####Minimization STEP
results = opt.minimize(crit, params_init, args=(mle_args))
mu_MLE, sig_MLE = results.x
print('Problem 1')
print("Part(b); log liklihood value is {}".format(log_lik_norm(income, mu_1, sig_1, None)))
print('Part(C); mu_MLE= {:.2f}'.format(mu_MLE), ' sig_MLE= {:.2f}'.format(sig_MLE))

plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_MLE, sig_MLE, None), 
         linewidth=2, color='y', label='2: $\mu$={:.2f},$\sigma$={:.2f}'.format(mu_MLE, sig_MLE))
plt.legend(loc='upper right')
output_path = os.path.join(output_dir, "dist_income")
plt.savefig(output_path)
plt.show()
print("Part(c); Value of the log likelihood function is {:.2f}:"\
      .format(log_lik_norm(income, mu_MLE, sig_MLE, None)))
print("Part(c); The precise Variance Covariance Matrix is:")
print(results.hess_inv)
np.set_printoptions(suppress=True)
print("Part(c); Variance Covariance Matrix is:")
print(results.hess_inv)

log_lik_h0 = log_lik_norm(income, mu_1, sig_1, None)
log_lik_mle = log_lik_norm(income, mu_MLE, sig_MLE, None)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('Part (d): chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)

print("Part (e); The possibility of having an income higher than $100000 is {:.3f}"\
      .format(1- sts.norm.cdf(math.log(100000), loc=mu_MLE, scale=sig_MLE)))
print("Part (e); The possibility of having an income lower than $75000 is {:.3f}"\
      .format(sts.norm.cdf(math.log(75000), loc=mu_MLE, scale=sig_MLE)))
print()


#####Problem2######

def norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
            
    pdf_vals    = (1/(sigma * np.sqrt(2 * np.pi))) *\
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2)) 
    
    return pdf_vals
    
def regression_log_lik_norm(df, mu, sigma, betas):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    beta1, beta2, beta3, beta0 = betas
    xvals = df.iloc[:, 0] - beta1 * df.iloc[:, 1] - beta2 * df.iloc[:, 2] - beta3 * df.iloc[:, 3] - beta0
#    xvals = df.iloc[:, 0] - (betas * df.iloc[:, 1:].assign(cons = 1)).sum(axis = 1)
    pdf_vals = norm_pdf(xvals, mu, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val
    
def regression_crit(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the normally distributed random
             variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        regression_log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    sigma, beta1, beta2, beta3, beta0 = params
    betas = np.array([beta1, beta2, beta3, beta0])
    df = args[0]
    log_lik_val = regression_log_lik_norm(df, 0, sigma, betas)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val

######Initialization
df_sick = pd.read_csv("sick.txt")    
sig_init = 0.05  # sig_2
sig_init = 0.00302231160097
beta1, beta2, beta3, beta0 = (0.012,  0.4 , -0.009999, 0.25)
beta1, beta2, beta3, beta0 = (0.012936460342377384, 0.40048451899646853, -0.0099934707081930866, 0.25164799569325869)
#beta1, beta2, beta3, beta0 = (1,  0 , 0, 0)
params_init = np.array([sig_init, beta1, beta2, beta3, beta0])
mle_args = df_sick

#####Minimization STEP
results_regression = opt.minimize(regression_crit, params_init, args=(mle_args), tol = 1e-2,\
                                  method = 'Nelder-Mead', options = {'maxiter': 10000})
params_init = results_regression.x
results_regression = opt.minimize(regression_crit, params_init, args=(mle_args), tol = 1e-2,\
                                  method = 'L-BFGS-B', options = {'maxiter': 10000})
sig_MLE, beta0_MLE, beta1_MLE,beta2_MLE, beta3_MLE  = results_regression.x
betas = np.array([beta0_MLE, beta1_MLE, beta2_MLE, beta3_MLE])


print("Problem 2")
print('regression_sig_MLE=', sig_MLE, ' betas_MLE=', (beta0_MLE, \
                                        beta1_MLE,beta2_MLE, beta3_MLE))
print("The log liklihood value of the function is {}".format(regression_log_lik_norm(df_sick, 0, sig_MLE, betas)))
np.set_printoptions(suppress=False)
print('Varaiance Covaraince Matrix is as following:')
print(results_regression.hess_inv.todense())

log_lik_h0_regression = regression_log_lik_norm(df_sick, 0, 0.01, np.array([0, 0, 0, 0]))
log_lik_mle_regression = regression_log_lik_norm(df_sick, 0, sig_MLE, betas)
LR_val = 2 * (log_lik_mle_regression - log_lik_h0_regression)
pval_h0_regression = 1.0 - sts.chi2.cdf(LR_val, 5)
print('chi squared of H0 with 5 degrees of freedom p-value = ', pval_h0_regression)