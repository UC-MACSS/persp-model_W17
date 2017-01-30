# MACS PS2
#
# Shen Han    shenhan@uchicago.edu
#
# Jan 23, 2017

import numpy as np
import pandas as pd
import scipy.stats as sts
from scipy.stats import lognorm
import scipy.optimize as opt

import os
import matplotlib.pyplot as plt
import seaborn as sns


# Q1

incomes_array = np.loadtxt('incomes.txt')

plot = True
# if True, save output plots to current directory


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
    pdf_vals    = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2)))
    
    return pdf_vals


def log_lik_lognorm(xvals, mu, sigma):
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
    
    pdf_vals = lognorm(s =sigma, scale =np.exp(mu)).pdf(xvals)
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
    log_lik_val = log_lik_lognorm(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val


# 1a

if plot:
    fig, ax = plt.subplots()
    plt.hist(incomes_array, 30, normed = True)

    plt.title("Annual Incomes of MACSS '18, '19, '20")
    plt.xlabel(r'Annual Income ($)')
    plt.ylabel('Percent of Observations')

    plt.savefig('fig_1a')
    plt.close()

    
# 1b

mu_1b, sigma_1b = 9.0, 0.3
x_1b_array = np.linspace(0, 150000, 1000)

print('1b: {}\n'.format(log_lik_lognorm(incomes_array, mu_1b, sigma_1b)))

if plot:
    fig, ax = plt.subplots()
    plt.plot(x_1b_array, lognorm(scale = np.exp(mu_1b), s = sigma_1b
                                 ).pdf(x_1b_array),
             label = r'$\mu=9.0,\sigma=0.3$')

    plt.title(r'$Lognormal\ PDF\ of\ f(x|\mu=9.0,\sigma=0.3)$')
    plt.xlabel(r'$x$')
    plt.xlim(x_1b_array[0], x_1b_array[-1])
    plt.legend(loc = 'best')

    plt.savefig('fig_1b')
    plt.close()
    

# 1c

mu_1c = 11.5
sigma_1c = 0.2
params_init = np.array([mu_1c, sigma_1c])
mle_args = (incomes_array, None)

results = opt.minimize(crit, params_init, args = mle_args)
mu_mle, sigma_mle = results.x

print('1c: mu_mle = {:.2f}, sigma_mle = {:.2f}'.format(mu_mle, sigma_mle))
print('1c: The value of the likelihood function: {:.2f}'.format(
       log_lik_lognorm(incomes_array, mu_mle, sigma_mle)))
print('1c: VCV matrix: \n', results.hess_inv, '\n')

if plot:
    fig, ax = plt.subplots()
    plt.hist(incomes_array, 30, normed = True)
    plt.plot(x_1b_array, lognorm(scale = np.exp(mu_1b), s = sigma_1b
                                 ).pdf(x_1b_array),
             label = r'$\mu={},\sigma={}$'.format(mu_1b, sigma_1b))
    plt.plot(x_1b_array, lognorm(s = sigma_mle, scale = np.exp(mu_mle)).pdf(
             x_1b_array), label =
             r'$\mu_{{mle}} = {:.2f}, \sigma_{{mle}} = {:.2f}$'.format(mu_mle,
                                                                    sigma_mle))

    plt.legend(loc = 'best')             
    plt.title('Histogram, Lognormal PDF and ML Estimate')
    plt.xlabel(r'Income ($)')
    plt.ylabel('Density')

    plt.savefig('fig_1c')
    plt.close()
    

# 1d

log_lik_h0 = log_lik_lognorm(incomes_array, mu_1b, sigma_1b)
log_lik_mle = log_lik_lognorm(incomes_array, mu_mle, sigma_mle)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('1d: p-value from the chi-square test: {:.3f}\n'.format(pval_h0))


# 1e

print('1e: Probability of earning more than $100,000: {:.4f}'.format(
    lognorm.sf(100000, sigma_mle, loc = mu_mle, scale = np.exp(mu_mle))))

print('    Probability of earning less than $75,000: {:.4f}\n'.format(
    lognorm.cdf(75000, sigma_mle, loc = mu_mle, scale = np.exp(mu_mle))))



# Q2

sick_df = pd.read_csv('sick.txt')
sick_df.rename(columns = {'\ufeffsick': 'sick'}, inplace = True)

def log_like_sick(sick_df, b0, b1, b2, b3, sigma):
    error = sick_df.sick - b0 - b1*sick_df.age - b2*sick_df.children - b3*\
            sick_df.avgtemp_winter
    pdf_vals = norm_pdf(error, 0, sigma)
    log_lik_val = np.log(pdf_vals).sum()

    return log_lik_val


# 2a

def crit_sick(params, *args):
    b0, b1, b2, b3, sigma = params
    data = args[0]
    log_lik_val = log_like_sick(data, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

b0_init, b1_init, b2_init, b3_init, sig_init = 0, 0, 0, 0, .1

params_init = np.array([b0_init, b1_init, b2_init, b3_init, sig_init])
bounds = ((None, None),(None, None),(None, None),(None, None),(0.001, None))

results = opt.minimize(crit_sick, params_init, args=sick_df, method = "SLSQP", bounds=bounds)
b0, b1, b2, b3, sigma = results.x
results = opt.minimize(crit_sick,  results.x, args=sick_df, method = "L-BFGS-B", bounds=bounds)
b0, b1, b2, b3, sigma = results.x
print('2a: beta_0 = {:.4f}, beta_1 = {:.4f}, beta_2 = {:.4f}, beta_3 = {:.4f}, sigma = {:.4f}'.format(b0, b1, b2, b3, sigma))
print('2a: The value of the likelihood function: {:.2f}'.format(
       log_like_sick(sick_df, *results.x)))
print('2a: VCV matrix: \n', results.hess_inv.todense(), '\n')


# 2b

log_lik_h0 = log_like_sick(sick_df, 1, 0, 0, 0, 0.1)
log_lik_mle = log_like_sick(sick_df, *results.x)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print('2b: p-value from the chi-square test: {:.3f}\n'.format(pval_h0))
