# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------
Python version: Python 3.5
------------------------------------------------------------------------
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
import scipy.stats as sts

'''
------------------------------------------------------------------------
Exercise 1: Income data, lognormal distribution, and hypothesis testing
------------------------------------------------------------------------
(a) = Plot a histogram of percentages of the income.txt data with 30 bins.
(b) = Plot the lognormal PDF
(c) = Estimate the parameters of the lognormal distribution by maximum 
      likelihood and plot its PDF against the PDF from part (b) 
      and the histogram from part (a).
(d) = Perform a likelihood ratio test to determine the probability that 
      the data in incomes.txt came from the distribution in part (b).
(e) = What is the probability that you will earn more than $100,000? 
      What is the probability that you will earn less than $75,000?
------------------------------------------------------------------------
'''


df = np.loadtxt('incomes.txt')

'''
------------------------------------------------------------------------
(a) = Plot a histogram of percentages of the income.txt data with 30 bins.
------------------------------------------------------------------------
'''
num_bins = 30
weights = (1.0 / df.shape[0]) * np.ones_like(df)

plot_1a = True

if plot_1a:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath('__file__'))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    # create 2D histogram plot
    fig = plt.figure()
    n, bin_cuts, patches = plt.hist(df, num_bins, weights=weights)
    plt.title('Histogram of incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1a')
    plt.savefig(output_path)
    plt.close()

'''
------------------------------------------------------------------------
(b) = Plot the lognormal PDF. What is the value of the log likelihood 
      value for this parameterization of the distribution and 
      given this data?
------------------------------------------------------------------------
'''
# Define function that generates values of a lognormal pdf
def lognorm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sts.lognorm.pdf(x, s = sig, scale = np.exp(mu)).
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
    pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.lognorm.cdf(cutoff, s=sigma, loc=mu, scale=np.exp(mu))
            
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut

    return pdf_vals
    

# Define log likelihood function for the lognormal distribution
def log_lik_lognorm(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given lognormal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the lognormal distributed random
             variable
    mu     = scalar, mean of the lognormal distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormal distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        lognorm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, lognormal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals =lognorm_pdf(xvals, mu, sigma, 'None')
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val
    
plot_1b = True

if plot_1b:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath('__file__'))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    dist_pts = np.linspace(0, 150000, 500)
    mu_1 = 9.0
    sig_1 = 0.3
    fig = plt.figure()
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_1, sig_1, 'None'),
             linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
    plt.legend(loc='upper right')
    plt.title('Lognormal pdf with $\mu$=9.0 and $\sigma$=0.3', fontsize=17)
    plt.xlim(0, 150000)
    plt.xlabel('x')
    plt.ylabel('$f(x|\mu=9,\sigma=.3)$')
    output_path = os.path.join(output_dir, 'Fig_1b')
    plt.savefig(output_path)
    plt.close()
print('Log-likelihood 1: ', log_lik_lognorm(df, mu_1, sig_1))
    

'''
------------------------------------------------------------------------
(c) = Estimate the parameters of the lognormal distribution by maximum 
      likelihood and plot its PDF against the PDF from part (b) 
      and the histogram from part (a).
------------------------------------------------------------------------
'''

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

mu_init = 9  # mu_1
sig_init = 0.3  # sig_1
params_init = np.array([mu_init, sig_init])
mle_args = (df, df.max())

results = opt.minimize(crit, params_init, args=(mle_args), method = "SLSQP", bounds=((None, None), (1e-10, None)))
params_init = results.x

# In order to get variance-covariance matrix
results = opt.minimize(crit, params_init, args=(mle_args),method = "BFGS")
mu_MLE, sig_MLE = results.x
print('mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)
vcv_mle = results.hess_inv
print('VCV(MLE) = ',vcv_mle)
print('Log-likelihood of MLE estimate: ', log_lik_lognorm(df, mu_MLE, sig_MLE))

plot_1c = True

if plot_1c:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath('__file__'))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    fig = plt.figure()
    count, bins, ignored = plt.hist(df, 30, normed=True)
# Plot one test distributions from before
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_1, sig_1, 'None'),
         linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
    plt.legend(loc='upper right')
# Plot the MLE estimated distribution
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_MLE, sig_MLE, 'None'),
         linewidth=2, color='k', label='2: $\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')
    
    plt.title('Histogram of incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1c')
    plt.savefig(output_path)
    plt.close()
    
    
'''
------------------------------------------------------------------------
(d) = Perform a likelihood ratio test to determine the probability that 
      the data in incomes.txt came from the distribution in part (b).
------------------------------------------------------------------------
'''

log_lik_h0 = log_lik_lognorm(df, mu_1, sig_1)
log_lik_mle = log_lik_lognorm(df, mu_MLE, sig_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
print('chi squared of H0 with 2 degrees of freedom p-value = {0:.4f}'.format(pval_h0))



'''
------------------------------------------------------------------------
(e) = What is the probability that you will earn more than $100,000? 
      What is the probability that you will earn less than $75,000?
------------------------------------------------------------------------
'''

Prob_1 = 1- sts.lognorm.cdf(100000, sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
print('Probability that you will earn more than $100,000 is : {0:.3f} .'.format(Prob_1))

Prob_2 = sts.lognorm.cdf(75000, sig_MLE, loc= mu_MLE, scale = np.exp(mu_MLE))
print('Probability that you will earn less than $75,000 is : {0:.3f} .'.format(Prob_2))





'''
------------------------------------------------------------------------
Exercise 2: Linear regression and MLE 
------------------------------------------------------------------------
(a) = Estimate the parameters of the model by MLE. Report your estimates, 
      the value of the log likelihood function, and the estimated 
      variance covariance matrix of the estimates.
(b) = what is the likelihood that age, number of children, and average 
      winter temperature have no effect on the number of sick days?
------------------------------------------------------------------------
'''

df_sick= pd.read_csv('sick.txt', header = 0)


#Define function that generates values of normal pdf
def norm_pdf(xvals, mu, sigma):        
    pdf_vals    = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (xvals - mu)**2 / (2 * sigma**2))) 
    
    return pdf_vals

# Define log likelihood function
def log_like_sick(data, beta_0, beta_1, beta_2, beta_3, sigma):
    error = data['﻿sick']- (beta_0+ beta_1*data['age'] + beta_2*data['children']+ beta_3*data['avgtemp_winter'])
    pdf_vals = norm_pdf(error, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val


# Set up criterion function
def crit2(params, *args):
    beta_0, beta_1, beta_2, beta_3, sigma = params
    data = args[0]
    log_lik_val = log_like_sick(data, beta_0, beta_1, beta_2, beta_3, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val

'''
------------------------------------------------------------------------
(a) = Estimate the parameters of the model by MLE.
------------------------------------------------------------------------
'''
beta_0_init = 0
beta_1_init = 0
beta_2_init = 0
beta_3_init = 0
sig_init = .1

params_init = np.array([beta_0_init, beta_1_init, beta_2_init, beta_3_init, sig_init])
bnds = ((None, None),(None, None),(None, None),(None, None),(1e-10, None))

results = opt.minimize(crit2, params_init, args=df_sick, method = "SLSQP", bounds=bnds)
beta_0, beta_1, beta_2, beta_3, sigma = results.x

results = opt.minimize(crit2,  results.x, args=df_sick, method = "L-BFGS-B", bounds=bnds)
beta_0, beta_1, beta_2, beta_3, sigma = results.x

print('beta_0 =', beta_0, ' beta_1 =', beta_1, 'beta_2 =', beta_2, 'beta_3 =', beta_3, 'sigma =', sigma)
print('The variance-covariance matrix is: ',results.hess_inv.todense())
print('Log-likelihood: ', log_like_sick(df_sick, *results.x))



'''
------------------------------------------------------------------------
(b) = what is the likelihood that age, number of children, and average 
      winter temperature have no effect on the number of sick days?
------------------------------------------------------------------------
'''
log_lik_h0 = log_like_sick(df_sick, 1, 0, 0, 0, 0.1)
log_lik_mle = log_like_sick(df_sick, *results.x)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print('chi squared of H0 with 5 degrees of freedom p-value = {:.4f}'.format(pval_h0))





