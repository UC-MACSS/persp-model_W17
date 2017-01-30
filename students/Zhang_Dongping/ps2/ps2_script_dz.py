# MACS 30100: PS2
# Name: Dongping Zhang
# Python Version: 3.5
# Seed: None

import pandas as pd
import numpy as np
import scipy.stats as sts
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter as fstr
import scipy.special as spc
import scipy.stats as sts
import scipy.optimize as opt
import scipy.integrate as integration
import math

'''
------------------------------------------------------------------------------
Exercise 1/a.: Plot a density histogram with incomes data
------------------------------------------------------------------------------
incomes         = a cleaned dataframe of 200 data points of annual incomes 
                  of MACS students who graduated in 2018, 2019, and 2020 in 
                  1000 USD.
incomes_barplot = Boolean, =True if make a plot of incomes distribution
------------------------------------------------------------------------------
'''
# load the incomes.txt raw data
incomes = np.loadtxt('incomes.txt')

incomes_barplot = True
if incomes_barplot:
    '''
    -------------------------------------------------------------------- 
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved 
    weights     = weights used for the density histogram
    pdf_vals = ( 1 / (np.sqrt(var * 2 * np.pi)) ) * \
               np.exp( - (sick.sick - beta0 - (beta1 * sick.age) - \
                         (beta2 * sick.children) - \
                         (beta3 * sick.temp))**2 / (2 * var) )
    bin_num     = number of bins wanted
    -------------------------------------------------------------------- 
    ''' 
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
  
    # create a subplot
    fig, ax = plt.subplots()

    # determine the number of bins needed
    bin_num = 30
    # create weighsts for the probability density histograms
    weights = (1 / incomes.shape[0]) * np.ones_like(incomes)
    # create histogram
    n, bin_cuts, ignored = plt.hist(incomes, bin_num, \
                                    weights = weights, \
                                    color = 'maroon')

    # set xlimit
    plt.xlim([0, 150000])
    # create plot title and xlabel and ylabel
    plt.title('Histogram of MACS Student Annual Income')
    plt.xlabel(r'Annual Income ($\$$)')
    plt.ylabel(r'Percent of Observations in bin')    

    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_1')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/b.: Plot a lognormal PDF and compute loglikelihood value
------------------------------------------------------------------------------
dist_pts        = simulated points from 0 to 150000 of a length 200 with equal
                  intervals
lognorm_density = Boolean, =True if make a probability density histogram  of 
                  lognorm(mu = 9, sigma = 0.3)
log_lik_norm    = a function that computes the log-liklihood values of a 
                  lognormal distribution
------------------------------------------------------------------------------
'''
# plot the lognormal PDF f(x|mu=9, sigma=0.3) for 0<x<150,000
# by first generating 200 simulated points from 0 to 150000
dist_pts = np.linspace(0.01, 150000, 200)

lognorm_density = True
if lognorm_density:
    # create a subplot
    fig, ax = plt.subplots()
    # rescale the y-axis to scientific notation
    ax.yaxis.set_major_formatter(fstr('%.1e'))

    # place the layer of lognormal cdf onto the density histogram
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, 0.3, 0, np.exp(9)), \
             linewidth=2, color='r', label='lognormal: $\mu$=9, $\sigma$=0.3')
    plt.legend(loc='upper right')

    # create plot title and xlabel and ylabel
    plt.title(r'Lognormal PDF: f(x|$\mu$ = 9.0, $\alpha$ = 0.3)')
    plt.xlabel(r'X')
    plt.ylabel(r'Probability Density')     
    # set xlimit
    plt.xlim([0, 150000])
    # re-adjusting plot spaces
    plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1)

    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_2')
    plt.savefig(output_path) 
    plt.close() 

# compute the log-likelihood value for this parameterization
def log_lik_lognorm(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given lognormal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the log-normally distributed random
             variable
    mu     = scalar, mean of the log-normally distributed random variable
    sigma  = scalar > 0, standard deviation of the log-normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, log-normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of log-normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = sts.lognorm.pdf(xvals, sigma, 0, np.exp(mu))  
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

print('The log-likelihood value for incomes data under a lognormal(9, 0.3) is:\
 ', log_lik_lognorm(incomes, 9, 0.3))


'''
------------------------------------------------------------------------------
Exercise 1/c.: Plot a lognormal PDF and compute loglikelihood value
------------------------------------------------------------------------------
crit_lognorm     = the objective function of the log-normal distribution
mu_init          = an initial guess of parameter mu
sig_init         = an initial guess of parameter sig
params_init      = an array of the parameters of the log-normal distribution
mle_args         = the MLE arguments or the data points
lognorm_results  = the optimization result for the log-normal distribution
mu_MLE           = the MLE estimator of parameter mu
sig_MLE          = the MLE estimator of parameter sigma
mu_init2         = a second guess, based on MLE estimator mu, of the parameter
                   mu (the purpose of this is to use a different minimization
                   method to obtain the variance-covariance matrix)
sig_init2        = a second guess, based on MLE estimator sigma, of the 
                   parameter sigma (the purpose of this is to use a different 
                   minimization method to obtain the variance-covariance 
                   matrix)
params_init2     = an array of the second-guess parameters of the log-normal 
                   distribution
lognorm_results2 = the optimization result for the log-normal distribution
                   using method 'BFGS' in order to obtain variance-covariance
                   matrix
vcv_MLE          = the variance-covariance matrix of the MLE estimators
stderr_mu_MLE    = the standard deviation of MLE estimator mu
stderr_sig_MLE   = the standard deviation of MLE estimator sigma
three_elements   = Boolean, =True if make a probability density histogram  of 
                   lognorm(mu = 9, sigma = 0.3), 
                   lognorm(mu = mu_MLE, sigma = sig_MLE), and
                   density histogram of income data.
------------------------------------------------------------------------------
'''
# construct the objective function
def crit_lognorm(params, *args): 
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the log-normally distributed random variable
    sigma  = scalar > 0, standard deviation of the log-normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the log-normally distributed random
             variable
    
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
    xvals = args
    log_lik_val = log_lik_lognorm(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val


# initial guess of the lognormal parameters
mu_init = 11
sig_init = 0.2
params_init = np.array([mu_init, sig_init])
mle_args = (incomes)
# methods for constrained optimization
# 'SLSQP', 'TNC', 'L-BFGS-B'
lognorm_results = opt.minimize(crit_lognorm, params_init, args = (mle_args), \
                               method = 'L-BFGS-B', \
                               bounds = ((None, None), (1e-10, None)), \
                               tol = 1e-12)

# variable assignment of MLE parameters
mu_MLE, sig_MLE = lognorm_results.x

# variance-covariance assignemnt
vcv_MLE = lognorm_results.hess_inv.todense()
# standard deviation of MLE estimators obtained from variance-covariance matrix
stderr_mu_MLE = np.sqrt(vcv_MLE[0,0])
stderr_sig_MLE = np.sqrt(vcv_MLE[1,1])

# print out the MLE parameters
print('mu_MLE = ', mu_MLE)
print('sig_MLE = ', sig_MLE)
# print out the new log-likelihood value using MLE estimators
print('log-likelihood value = ', log_lik_lognorm(incomes, mu_MLE, sig_MLE))   
# print out the variance-covariance matrix
print('variance-covariance matrix: ', vcv_MLE) 
print('Standard error for mu estimate = ', stderr_mu_MLE)
print('Standard error for sigma estimate = ', stderr_sig_MLE)


three_elements = True
if three_elements: 
    # create a subplot
    fig, ax = plt.subplots()
    # rescale the y-axis to scientific notation
    ax.yaxis.set_major_formatter(fstr('%.1e'))

    # determine the number of bins needed
    bin_num = 30
    # create histogram
    n, bin_cuts, ignored = plt.hist(incomes, bin_num, \
                                    normed = True, \
                                    color = 'maroon')

    # place the layer of lognormal cdf onto the density histogram
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, 0.3, 0, np.exp(9)), \
             linewidth=2, color='r', \
             label='Lognormal Random Guess: $\mu$=9, $\sigma$=0.3')
    plt.legend(loc='upper right')

    # place the layer of lognormal cdf onto the density histogram
    plt.plot(dist_pts, sts.lognorm.pdf(dist_pts, sig_MLE, 0, np.exp(mu_MLE)), \
             linewidth=2, color='g', \
             label='Lognormal MLE: $\mu$=11.33, $\sigma$=0.21')
    plt.legend(loc='upper right')
    # set xlimit
    plt.xlim([0, 150000])
    # re-adjusting plot spaces
    plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1)

    # create plot title and xlabel and ylabel
    plt.title('Histogram of MACS Student Annual Income')
    plt.xlabel(r'Annual Income ($\$$)')
    plt.ylabel(r'Percent of Observing bin value')    
    
    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_3')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/d.: performing a likelihood ratio test
------------------------------------------------------------------------------
log_lik_lognorm_MLE = the log-likelihood value of the log-normal distribution
                      using MLE estimators obtained
log_lik_lognorm_H0  = the log-likelohood value of the log-normal distribution
                      using hypothesized parameters
LR_val_lognorm      = the likelihood-ratio-test test statistics
pval_h0             = the p-value for the likelihood ratio test
------------------------------------------------------------------------------
'''
# compute the log-likelihood values
log_lik_lognorm_MLE = log_lik_lognorm(incomes, mu_MLE, sig_MLE)
log_lik_lognorm_H0 = log_lik_lognorm(incomes, 9, 0.3)

# compute the likelihood ratio test statistics
LR_val_lognorm = 2 * (log_lik_lognorm_MLE - log_lik_lognorm_H0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val_lognorm, 2)
print('likelihood ratio test statistics is: ', LR_val_lognorm)
print('chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)
print('thus we would reject the null hypothesis!')


'''
------------------------------------------------------------------------------
Exercise 1/c.: compute some probabilities 
------------------------------------------------------------------------------
lognorm_prob_g100000 = the estimated probability of observing a data point
                       greater than 100000 under a log-normal distribution 
                       using MLE estimators
lognorm_prob_l75000  = the estimated probability of observing a data point less
                       than 75000 under a log-normal distribution using MLE
                       estimators 
------------------------------------------------------------------------------
'''
# estimate the probability of observing an income > $100,000
lognorm_prob_g100000 = 1 - sts.lognorm.cdf(100000, sig_MLE, 0, np.exp(mu_MLE)) 
print('The estimated probability of observing a UChicago MACSS graduate to \
earn an income of more than $100,000 after graduation under a lognormal \
distribution using MLE parameters is: ', lognorm_prob_g100000)

# estimate the probability of observing an income < $75,000 
lognorm_prob_l75000 = sts.lognorm.cdf(75000, sig_MLE, 0, np.exp(mu_MLE)) 
print('The estimated probability of observing a UChicago MACSS graduate to \
earn an income of less than $75,000 after graduation under a lognormal \
distribution using MLE parameters is: ', lognorm_prob_l75000)


'''
------------------------------------------------------------------------------
Exercise 2/a.: Linear regression and MLE
------------------------------------------------------------------------------
sick                    = a cleaned 200x4 dataframe of the sick data and the
                          four columns are 'sick', 'age', 'children', 'temp' 
log_lik_norm_ols        = a function used to compute the log-likelihood values
crit_norm_ols           = the objective function that would be used for 
                          optimization 
MLE_estimate_ols_params = a function that would take crit_norm_ols and would 
                          implemente a MLE estimation
bounds_ols_params       = an argument for MLE_estimate_ols_params if any of the
                          target estimated parameters have bounds
results_ols             = the MLE optimation result object
beta0_MLE               = the MLE estimator of parameter beta0
beta1_MLE               = the MLE estimator of parameter beta1
beta2_MLE               = the MLE estimator of parameter beta2
beta3_MLE               = the MLE estimator of parameter beta3
sig_MLE                 = the MLE estimator fo parameter sigma
results2_ols            = the optimization results using MLE estimated 
                          parameters and 'L-BFGS-B' method
vcv_ols1                = the variance-covariance matrix generated by results2
results3_ols            = the optimization results using initial values and 
                          'BFGS' method
vcv_ols                 = the variance-covariance matrix generated by results3
------------------------------------------------------------------------------
'''
sick = pd.read_table('sick.txt', sep = ',', header = 0, \
                     names = ['sick', 'age', 'children', 'temp'])


def log_lik_norm_ols(df, *params):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for the sick data given normal
    distribution parameters beta0-beta3 and sigma
    --------------------------------------------------------------------
    INPUTS:
    df     = a dataframe contains the cleaned sick data    
    params = (5,) vector, ([beta0, beta1, beta2, beta3, sig])
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    errors      = (N,) vector, normall distributed error term for the
                  based on the assumption 
    pdf_vals    = (N,) vector, normal PDF values for beta0, beta1, beta2
                  beta3, beta4, sigma based on normally distributed errors
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for orresponding to errors based on beta0, beta1, beta2
                  beta3, beta4, and sigma
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    beta0, beta1, beta2, beta3, sigma = params
    errors = df.sick - beta0 - beta1 * df.age - beta2 * df.children \
             - beta3 * df.temp
    pdf_vals = sts.norm.pdf(x = errors, loc = 0, scale = sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val 


def crit_norm_ols(params, args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (5,) vector, ([beta0, beta1, beta2, beta3, sig])
    beta0  = constant in the regression model
    beta1  = coefficient for age variable in the regression model
    beta2  = coefficient for children variable in the regression model
    beta3  = coefficient for temp variable in the regression model
    sig    = the standard deviation of the error term of the regression model
    args   = the error term of the regression model
    errors = (N,) vector, values of the normally distributed random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    log_lik_norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    beta0, beta1, beta2, beta3, sigma = params
    errors = args
    log_lik_val = log_lik_norm_ols(errors, *params)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val
                     

def MLE_estimate_ols_params(*params, df, opt_method, params_bounds = None): 
    '''
    --------------------------------------------------------------------
    This function is used to conduct the MLE based on the objective 
    function given. If the optimization scheme selected is 'L-BFGS-B',
    'TNC', or 'SLSQP', then the fucntion would require to have params_bounds
    as an input argument indicating the bounds of all parameters
    --------------------------------------------------------------------
    INPUTS:
    params        = (5,) vector, ([beta0, beta1, beta2, beta3, sig])
    beta0         = constant in the regression model
    beta1         = coefficient for age variable in the regression model
    beta2         = coefficient for children variable in the regression model
    beta3         = coefficient for temp variable in the regression model
    df            = a dataframe which is the dataset of the given 
                    parametrization
    opt_method    = a string of which the metho of optimization would be used
    params_bounds = a tuple indicating the bounds of estimated parameters 
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
    sts.minimize()
 
    OBJECTS CREATED WITHIN FUNCTION:
    params_init   = a numpy array of all estimated parameters
    mle_args      = the dataframe that would be used during the optimization
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: results
    --------------------------------------------------------------------
    '''
    params_init = np.array(list(params))
    mle_args = df
    if opt_method in ['L-BFGS-B', 'TNC', 'SLSQP']:
        results = opt.minimize(crit_norm_ols, params_init, \
                               args=(mle_args), \
                               bounds = params_bounds, \
                               method = opt_method)
    else:
        results = opt.minimize(crit_norm_ols, params_init, \
                               args = (mle_args), \
                               method = opt_method)
    return results


# set up parameter bounds and implement MLE procedures to obtain MLE estimators
bounds_ols_params = ((None, None), (None, None), (None, None), (None, None), \
                     (1e-10, None))
results_ols = MLE_estimate_ols_params(*(0, 0, 0, 0, 1), df = sick, \
                                      opt_method = 'SLSQP', \
                                      params_bounds = bounds_ols_params)

# assignment of MLE estimators
beta0_MLE, beta1_MLE, beta2_MLE, beta3_MLE, sig_MLE = results_ols.x
print('MLE estimator for OLS beta0 is: ', beta0_MLE)
print('MLE estimator for OLS beta1 is: ', beta1_MLE)
print('MLE estimator for OLS beta2 is: ', beta2_MLE)
print('MLE estimator for OLS beta3 is: ', beta3_MLE)
print('MLE estimator for OLS sigma is: ', sig_MLE)

# report the log-likelihood values
log_lik_MLE_OLS = log_lik_norm_ols(sick, beta0_MLE, beta1_MLE, \
                                   beta2_MLE, beta3_MLE, sig_MLE)
print('The log-likehihood value using those MLE estimators is: ', \
      log_lik_MLE_OLS)
     
# in order to get variance-covariance matrix, implement the same procedure
# by using MLE estimators and apply 'BFGS' method
results2_ols = MLE_estimate_ols_params(np.array([results_ols.x]), \
                                       df = sick, opt_method = 'L-BFGS-B', \
                                       params_bounds = bounds_ols_params) 
vcv_ols1 = results2_ols.hess_inv.todense()
print('Var-Covar Matrix for MLE estimators is:', vcv_ols1)

results3_ols = MLE_estimate_ols_params(*(0, 0, 0, 0, 1), \
                                       df = sick, opt_method = 'BFGS')
vcv_ols2 = results3_ols.hess_inv
print('Var-Covar Matrix for MLE estimators is:', vcv_ols2) 


'''
------------------------------------------------------------------------------
Exercise 2/b.: Use a likelihood ratio test to determine the probability
------------------------------------------------------------------------------
log_lik_OLS_MLE = the log-likelihood value of the normal distribution using the
                  MLE estimators obtained
log_lik_OLS_H0  = the log-likelihood value of the normal distribution using 
                  hypothesized parameters
LR_val_norm_ols = the likelihood-ratio-test test statistics
pval_ols_h0     = the p-value for the likelihood ratio test
------------------------------------------------------------------------------
'''
# compute the log-likelihood values
log_lik_OLS_MLE = log_lik_norm_ols(sick, beta0_MLE, beta1_MLE, beta2_MLE,
                                   beta3_MLE, sig_MLE)
log_lik_OLS_H0 = log_lik_norm_ols(sick, 1, 0, 0, 0, np.sqrt(0.01))

# compute the likelihood ratio test statistics
LR_val_norm_ols = 2 * (log_lik_OLS_MLE - log_lik_OLS_H0)
pval_ols_h0 = 1.0 - sts.chi2.cdf(LR_val_norm_ols, 5)
print('likelihood ratio test statistics is: ', LR_val_norm_ols)
print('chi squared of H0 with 5 degrees of freedom p-value = ', pval_ols_h0)
print('thus we would reject the null hypothesis!')
