# MACS 30100: PS3
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
import scipy.integrate as intgr
import numpy.linalg as lin


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
    # rescale the y-axis to scientific notation
    ax.yaxis.set_major_formatter(fstr('%.1e'))

    # determine the number of bins needed
    bin_num = 30

    # create histogram
    n, bin_cuts, ignored = plt.hist(incomes, bin_num, \
                                    normed = True, \
                                    color = 'maroon')

    # set xlimit
    plt.xlim([0, 150000])
    # re-adjusting plot spaces
    plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1)
    # create plot title and xlabel and ylabel
    plt.title('Probability Histogram of MACS Student Annual Income')
    plt.xlabel(r'Annual Income ($\$$)')
    plt.ylabel(r'Probability of observing a bin value')    

    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_1')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/b.: Esstimate the parameters of the lognormal distribution by MOM
------------------------------------------------------------------------------
lognorm_pdf     = lognormal probability distribution function
data_moments    = a function that computes data moments
model_moments   = a function that computes model moments
err_vec         = a function that computes the moment error vectors
criterion       = the criterion function for minimization
mu_init         = the initial guess of parameter mu
sig_init        = the initial guess of parameter sigma
W_hat           = the weighting matrix, which is an identity matrix
gmm_args        = the arguments needed to initialize optimation alogrithm
results1        = the optimization results object
mu_GMM1         = the GMM estimator of mu
sig_GMM1        = the GMM estimator of sigma
val_crit1       = the value of GMM criterion function at the estimated 
                  parameter values
data_moments1   = a tuple of two data moments
model_moments1  = a tuple of two model moments 
dist_pts        = simulated points from 0 to 150000 of a length 200 with equal
                  intervals
GMM1            = Boolean, =True if make a plot of incomes distribution and 
                  the probability density function of lognormal using MOM
                  parameters
------------------------------------------------------------------------------
'''
# first compose a function that can compute the probability 
def lognorm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Construct a pdf for a lognormal distribution with parameter mu
    and sigma
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
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    pdf_vals = sts.lognorm.pdf(xvals, sigma, 0, np.exp(mu)) 
    return pdf_vals


def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), variance(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of test scores data
    var_data  = scalar > 0, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    sigma_data = xvals.std()

    return mean_data, sigma_data


def model_moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), std(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        lognorm_pdf()
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    var_model  = scalar > 0, variance of test scores from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    std_model  = scalar > 0, standard deviation of test scores from model

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, std_model
    --------------------------------------------------------------------
    '''
    # compute the 1st moment -- mean
    x1fx = lambda x: x * lognorm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(x1fx, 0, 150000)
                                      # epsabs = 1e-10, epsrel = 1e-10)
    
    # compute the 2nd moment -- variance
    x2fx = lambda x: ( (x - mean_model) ** 2 ) * lognorm_pdf(x, mu, sigma)
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
                                      # epsabs = 1e-10, epsrel = 1e-10)
    # estimator is sigma, standard deviation, thus take a sqrt of variance
    sigma_model = np.sqrt(var_model)

    return mean_model, sigma_model


def err_vec(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        model_moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data    = scalar, mean value of data
    sigma_data   = scalar > 0, standard deviation of data
    moms_data    = (2, 1) matrix, column vector of two data moments
    mean_model   = scalar, mean value from model
    sigma_model  = scalar > 0, standard deviation from model
    moms_model   = (2, 1) matrix, column vector of two model moments
    err_vec      = (2, 1) matrix, column vector of two moment error
                   functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # compute data moments
    mean_data, sigma_data = data_moments(xvals)
    moms_data = np.array( [ [mean_data], [sigma_data] ] )

    # compute model moments
    mean_model, sigma_model = model_moments(mu, sigma)
    moms_model = np.array( [ [mean_model], [sigma_model] ] )

    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    args   = length 2 tuple, (xvals, W_hat)
    xvals  = (N,) vector, values of the lognormally distributed
             random variable
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, W = args
    err = err_vec(xvals, mu, sigma, simple = False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val


# initial guess of the lognormal parameters
mu_init = 11
sig_init = 0.2
params_init = np.array( [mu_init, sig_init] )
# define weighting matrix to be an identity matrix as required
W_hat = np.eye(2)
gmm_args = (incomes, W_hat)
# implement optimization algorithm
results1 = opt.minimize(criterion, params_init, args = (gmm_args), \
                        method = 'L-BFGS-B', \
                        bounds = ((None, None), (1e-10, None)))
# variable assignment of MLE parameters
mu_GMM1, sig_GMM1 = results1.x
# compute value of GMM criterion function using MOM estimators
val_crit1 = criterion([mu_GMM1, sig_GMM1], incomes, W_hat)
# compute data moments
data_moments1 = data_moments(incomes)
# compute model moments using estimated parameter values
model_moments1 = model_moments(mu_GMM1, sig_GMM1)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/b')
print('mu_GMM = ', mu_GMM1)
print('sigma_GMM = ', sig_GMM1)
print('val of GMM criterion function using GMM estimators: ', val_crit1)
print('-----------------------------------')
print('data moments are: ', data_moments1)
print('model moments using GMM estimators are: ', model_moments1) 
print('-----------------------------------')
print(results1)
print('----------------------------------------------------------------------')
print()


# plot the lognormal PDF f(x|mu=mu_GMM1, sigma=sig_GMM1) for 0<x<150,000
# by first generating 200 simulated points from 0 to 150000
dist_pts = np.linspace(0.01, 150000, 200)

GMM1 = True
if GMM1:
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
    plt.plot(dist_pts, 
             sts.lognorm.pdf( dist_pts, sig_GMM1, 0, np.exp(mu_GMM1) ), \
             linewidth=2, color='gold', \
             label='$GMM_{}$: $\mu$=11.33690998, $\sigma$=0.21302701')
    plt.legend(loc='upper left', prop={'size':10})

    # set xlimit
    plt.xlim([0, 150000])
    # re-adjusting plot spaces
    plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1)

    # create plot title and xlabel and ylabel
    plt.title('Probability Histogram of MACS Student Annual Income')
    plt.xlabel(r'Annual Income ($\$$)')
    plt.ylabel(r'Probability of observing a bin value')    
    
    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_2')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/c.: To perform a two-step GMM estimation 
------------------------------------------------------------------------------
err1           = moment error vector in percentage using GMM estimator
VCV2           = variance-covariance matrix based on moment error vector
W_hat2         = two-step weighting matrix
params_init2   = initial parameters using GMM estimators
gmm_args2      = new optimization arguments with two-step weighting matrix
results2       = an optimization result object
mu_GMM2        = GMM estimator for mu
sig_GMM2       = GMM estimator for sigma
val_crit2      = value of GMM criterion function using GMM estimators
data_moments2  = a tuple of two data moments
model_moments2 = a tuple of two model moments
GMM2           = Boolean, =True if make a plot of incomes distribution and 
                 the probability density function of lognormal using two
                 sets of GMM parameters
------------------------------------------------------------------------------
'''
# compute the vector of moment errors in percentage using GMM estimators
err1 = err_vec(incomes, mu_GMM1, sig_GMM1, False)
# computes the variance-covariance matrix based on moment errors
VCV2 = np.dot(err1, err1.T) / incomes.shape[0]
# take the inverse of the vcv matrix so as to get the two-step weighting matrix
W_hat2 = lin.pinv(VCV2)

# initial values now turned out to be GMM estimators
params_init2 = np.array( [mu_GMM1, sig_GMM1] )
gmm_args2 = (incomes, W_hat2)
results2 = opt.minimize(criterion, params_init2, args = (gmm_args2), \
                        method = 'L-BFGS-B', \
                        bounds = ((None, None), (1e-12, None)))

# variable assignment of MLE parameters
mu_GMM2, sig_GMM2 = results2.x
# compute value of GMM criterion function using MOM estimators
val_crit2 = criterion([mu_GMM2, sig_GMM2], incomes, W_hat2)
# compute data moments
data_moments2 = data_moments(incomes)
# compute model moments using estimated parameter values
model_moments2 = model_moments(mu_GMM2, sig_GMM2)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/c')
print('mu_GMM2 = ', mu_GMM1)
print('sigma_GMM2 = ', sig_GMM1)
print(r'val of GMM criterion function using GMM estimators and optimal vcov' \
      + ' matrix: ', val_crit2)
print('-----------------------------------')
print('data moments are: ', data_moments2)
print('model moments using GMM estimators and optimal vcov are: ', \
      model_moments2) 
print('-----------------------------------')
print(results2)
print('----------------------------------------------------------------------')
print()

# start plotting
GMM2 = True
if GMM2:
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
    plt.plot(dist_pts, 
             sts.lognorm.pdf( dist_pts, sig_GMM1, 0, np.exp(mu_GMM1) ), \
             linewidth=2, color='gold', \
             label='$GMM^{2M}$: $\mu$=11.33690998, $\sigma$=0.21302701')
    plt.plot(dist_pts, 
             sts.lognorm.pdf( dist_pts, sig_GMM2, 0, np.exp(mu_GMM2) ), \
             linewidth=2, color='green', linestyle=':',\
            label='$GMM^{2M}_{2step}$: $\mu$=11.33691003, $\sigma$=0.21302717')
    plt.legend(loc='upper left', prop={'size':10})

    # set xlimit
    plt.xlim([0, 150000])
    # re-adjusting plot spaces
    plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1)

    # create plot title and xlabel and ylabel
    plt.title('Probability Histogram of MACS Student Annual Income')
    plt.xlabel(r'Annual Income ($\$$)')
    plt.ylabel(r'Probability of observing a bin value')    
    
    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_3')
    plt.savefig(output_path) 
    plt.close() 
    

'''
------------------------------------------------------------------------------
Exercise 1/d.: To perform a GMM estimation using different moments 
------------------------------------------------------------------------------
data_3moments    = a function that computes data moments
model_3moments   = a function that computes model moments
err_vec3         = a function that computes the moment error vectors
criterion3       = the criterion function for minimization
mu_init 3        = the initial guess of parameter mu
sig_init3        = the initial guess of parameter sigma
W_hat3           = the weighting matrix, which is an identity matrix
gmm_args3        = the arguments needed to initialize optimation alogrithm
results3         = the optimization results object
mu_GMM3          = the GMM estimator of mu
sig_GMM3         = the GMM estimator of sigma
val_crit3        = the value of GMM criterion function at the estimated 
                   parameter values
data_moments3    = a tuple of two data moments
model_moment3    = a tuple of two model moments 
GMM3             = Boolean, =True if make a plot of incomes distribution and 
                   the probability density function of lognormal using MOM
                   parameters
------------------------------------------------------------------------------
'''
def data_3moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_dat = scalar in [0, 1], percent of observations
                 0 <= x < 75000
    bpct_2_dat = scalar in [0, 1], percent of observations
                 75000 <= x < 100000
    bpct_3_dat = scalar in [0, 1], percent of observations
                 x >= 100000

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >= 75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = xvals[xvals >= 100000].shape[0] / xvals.shape[0]

    return bpct_1_dat, bpct_2_dat, bpct_3_dat


def model_3moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the four model moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        sts.lognorm.pdf()
        xfx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_mod = scalar in [0, 1], percent of model observations in
                 bin 1
    bp_1_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_1_mod
    bpct_2_mod = scalar in [0, 1], percent of model observations in
                 bin 2
    bp_2_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_2_mod
    bpct_3_mod = scalar in [0, 1], percent of model observations in
                 bin 3
    bp_3_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_3_mod
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: bpct_1_mod, bpct_2_mod, bpct_3_mod
    --------------------------------------------------------------------
    '''
    xfx = lambda x: sts.lognorm.pdf(x, sigma, 0, np.exp(mu))
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 140000)
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod


def err_vec3(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, income data
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments3()
        model_moments3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    moms_data  = (3, 1) matrix, column vector of three data moments
    moms_model = (3, 1) matrix, column vector of three model moments
    err_vec    = (3, 1) matrix, column vector of three moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # compute data moments
    bpct_1_dat, bpct_2_dat, bpct_3_dat = data_3moments(xvals)
    moms_data = np.array( [ [bpct_1_dat], [bpct_2_dat], [bpct_3_dat] ] )

    # compute model moments
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_3moments(mu, sigma)
    moms_model = np.array( [ [bpct_1_mod], [bpct_2_mod], [bpct_3_mod] ] )
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ( (moms_model - moms_data) / moms_data )
    
    return err_vec


def criterion3(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    args   = length 2 tuple, (xvals, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (3, 1) matrix, column vector of four moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, W = args
    err = err_vec3(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


# initial guess of the lognormal parameters
mu_init_3M = 11
sig_init_3M = 0.2
params_init_3M = np.array( [mu_init_3M, sig_init_3M] )
# define weighting matrix to be an identity matrix as required
W_hat_3M = np.eye(3)
gmm_args_3M = (incomes, W_hat_3M)
# implement optimization algorithm
results_3M = opt.minimize(criterion3, params_init_3M, args = (gmm_args_3M), \
                          method = 'L-BFGS-B', \
                          bounds = ((None, None), (1e-10, None)))
# variable assignment of MLE parameters
mu_GMM_3M, sig_GMM_3M = results_3M.x
# compute value of GMM criterion function using MOM estimators
val_crit_3M = criterion3([mu_GMM_3M, sig_GMM_3M], incomes, W_hat_3M)
# compute data moments
data_moments_3M = data_3moments(incomes)
# compute model moments using estimated parameter values
model_moments_3M = model_3moments(mu_GMM_3M, sig_GMM_3M)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/d')
print('mu_GMM_3M = ', mu_GMM_3M)
print('sigma_GMM_3M = ', sig_GMM_3M)
print('val of GMM criterion function using GMM3 estimators: ', val_crit_3M)
print('-----------------------------------')
print('3 data moments are: ', data_moments_3M)
print('3 model moments using MOM3 estimators are: ', model_moments_3M) 
print('-----------------------------------')
print(results_3M)
print('----------------------------------------------------------------------')
print()


GMM_3M = True
if GMM_3M:
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
    plt.plot(dist_pts, 
             sts.lognorm.pdf( dist_pts, sig_GMM_3M, 0, np.exp(mu_GMM_3M) ), \
             linewidth=2, color='green', \
             label='$GMM^{3M}$: $\mu$=11.33672665, $\sigma$=0.21174642')
    plt.legend(loc='upper left', prop={'size':10})

    # set xlimit
    plt.xlim([0, 150000])
    # re-adjusting plot spaces
    plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1)

    # create plot title and xlabel and ylabel
    plt.title('Probability Histogram of MACS Student Annual Income')
    plt.xlabel(r'Annual Income ($\$$)')
    plt.ylabel(r'Probability of observing a bin value')    
    
    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_4')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 1/e.: To perform a two-step GMM estimation 
------------------------------------------------------------------------------
err_3M             = moment error vector in percentage using GMM estimator
VCV2_3M            = variance-covariance matrix based on moment error vector
W_hat_3M2S         = two-step weighting matrix
params_init_3M2S   = initial parameters using GMM estimators
gmm_args_3M2S      = new optimization arguments with two-step weighting matrix
results_3M2S       = an optimization result object
mu_GMM_3M2S        = GMM estimator for mu
sig_GMM_3M2S       = GMM estimator for sigma
val_crit_3M2S      = value of GMM criterion function using GMM estimators
data_moments_3M2S  = a tuple of two data moments
model_moments_3M2S = a tuple of two model moments
GMM_3M2S           = Boolean, =True if make a plot of incomes distribution and 
                     the probability density function of lognormal using two
                     sets of GMM parameters
------------------------------------------------------------------------------
'''
# compute the vector of moment errors in percentage using GMM estimators
err_3M = err_vec3(incomes, mu_GMM_3M, sig_GMM_3M, False)
# computes the variance-covariance matrix based on moment errors
VCV_3M = np.dot(err_3M, err_3M.T) / incomes.shape[0]
# take the inverse of the vcv matrix so as to get the two-step weighting matrix
W_hat_3M2S = lin.pinv(VCV_3M)

# initial values now turned out to be GMM estimators
params_init_3M2S = np.array( [mu_GMM_3M, sig_GMM_3M] )
gmm_args_3M2S = (incomes, W_hat_3M2S)
results_3M2S = opt.minimize(criterion3, \
                            params_init_3M2S, \
                            args = (gmm_args_3M2S), \
                            method = 'L-BFGS-B', \
                            bounds = ((None, None), (1e-12, None)))

# variable assignment of MLE parameters
mu_GMM_3M2S, sig_GMM_3M2S = results_3M2S.x
# compute value of GMM criterion function using MOM estimators
val_crit_3M2S = criterion3([mu_GMM_3M2S, sig_GMM_3M2S], incomes, W_hat_3M2S)
# compute data moments
data_moments_3M2S = data_3moments(incomes)
# compute model moments using estimated parameter values
model_moments_3M2S = model_3moments(mu_GMM_3M2S, sig_GMM_3M2S)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/e')
print('mu_GMM_3M2S = ', mu_GMM_3M2S)
print('sigma_GMM_3M2S = ', sig_GMM_3M2S)
print(r'val of GMM criterion function using GMM estimators and optimal vcov' \
      + ' matrix: ', val_crit_3M2S)
print('-----------------------------------')
print('data moments are: ', data_moments_3M2S)
print('model moments using MOM estimators and optimal vcov are: ', \
      model_moments_3M2S) 
print('-----------------------------------')
print(results_3M2S)
print('----------------------------------------------------------------------')
print()


# start plotting
GMM_3M2S = True
if GMM_3M2S:
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
    plt.plot(dist_pts, 
             sts.lognorm.pdf( dist_pts, sig_GMM_3M, 0, np.exp(mu_GMM_3M) ), \
             linewidth=2, color='green', \
             label='$GMM^{3M}$: $\mu$=11.33862142, $\sigma$=0.21374888')
    plt.plot(dist_pts, 
             sts.lognorm.pdf( dist_pts, sig_GMM_3M2S,\
                              0, np.exp(mu_GMM_3M2S) ), \
             linewidth=2, color='gold', linestyle=':',\
            label='$GMM^{3M}_{2step}$: $\mu$=11.33862147, $\sigma$=0.21374903')
    plt.legend(loc='upper left', prop={'size':10})

    # set xlimit
    plt.xlim([0, 150000])
    # re-adjusting plot spaces
    plt.subplots_adjust(left = 0.15, right = 0.9, top = 0.9, bottom = 0.1)

    # create plot title and xlabel and ylabel
    plt.title('Probability Histogram of MACS Student Annual Income')
    plt.xlabel(r'Annual Income ($\$$)')
    plt.ylabel(r'Probability of observing a bin value')    
    
    # output of the plot into the image directory
    output_path = os.path.join(output_dir, 'Fig_5')
    plt.savefig(output_path) 
    plt.close() 


'''
------------------------------------------------------------------------------
Exercise 2/a.: Estimate the parameters of a linear regression using GMM
------------------------------------------------------------------------------
------------------------------------------------------------------------------
'''
sick = pd.read_table('sick.txt', sep = ',', header = 0, \
                     names = ['sick', 'age', 'children', 'temp'])


def sick_data200moments(df):
    '''
    --------------------------------------------------------------------
    This function returns the 200 data moments for GMM
    --------------------------------------------------------------------
    INPUTS:
    df = sick dataframe
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    data_moments = tuple of length 200, 200 values of the sick varaible
                   is the data moments    

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: data_moments
    --------------------------------------------------------------------
    '''
    data_moments = ()
    for sick_val in df.sick:
        data_moments = data_moments + (sick_val,)
    return data_moments


def sick_model200moments(df, *betas):
    '''
    --------------------------------------------------------------------
    This function computes the 200 model moments for GMM
    --------------------------------------------------------------------
    INPUTS:
    df     = sick dataframes
    *betas = 3 betas, which are the parameters of the model needs to estiamte
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    beta0         = a parameter of the model
    beta1         = a parameter of the model
    beta2         = a parameter of the model
    beta3         = a parameter of the model
    model_moments = 200 model moments

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: (model_moments)
    --------------------------------------------------------------------
    '''
    beta0, beta1, beta2, beta3 = betas
    
    model_moments = beta0 + beta1 * df.age + beta2 * df.children + \
                    beta3 * df.temp
    return (model_moments)


def sick_err_vec(df, *betas):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    df     = sick dataframes
    *betas = 3 betas, which are the parameters of the model needs to estiamte
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        sick_data200moments()
        sick_model200moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    data_moments200  = 200 data moments
    moms_data        = a numpy array of 200 data moments
    model_moments200 = 200 model moments
    moms_model       = a numpy array of 200 data moments
    err_vec          = the difference between data moments and model moments

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # define parameters
    beta0, beta1, beta2, beta3 = betas

    # compute data moments
    data_moments200 = sick_data200moments(df)
    moms_data = np.array(data_moments200)

    # compute model moments
    model_moments200 = sick_model200moments(df, beta0, beta1, beta2, beta3)
    moms_model = np.array(model_moments200)

    err_vec = moms_data - moms_model

    return err_vec


def criterion_sick(betas, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    betas  = length of 4 tuples consisting of beta0, beta1, beta2, beta3
    args   = length 2 tuple, (df, W)
    df     = the sick dataframe
    W      = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
 
    OBJECTS CREATED WITHIN FUNCTION:
    err        = column vector of 200 moment error function
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    beta0, beta1, beta2, beta3 = betas
    df, W = args
    err = sick_err_vec(df, beta0, beta1, beta2, beta3)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val


# initial guess of betas
beta0_init = 0
beta1_init = 0
beta2_init = 0
beta3_init = 0
params_init_sick = np.array( [beta0_init, beta1_init, beta2_init, beta3_init] )
# define weighting matrix to be an identity matrix as required
W_hat_sick = np.eye(200)
gmm_args_sick = (sick, W_hat_sick)
# implement optimization algorithm
results_sick = opt.minimize(criterion_sick, \
                            params_init_sick, \
                            args = (gmm_args_sick), \
                            method = 'L-BFGS-B', \
                            bounds = ((None, None), (None, None), \
                                      (None, None), (None, None)))
# variable assignment of MLE parameters
beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results_sick.x
# compute value of GMM criterion function using MOM estimators
val_crit_sick = criterion_sick((beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM), \
                               sick, W_hat_sick)
# compute data moments
data_moments_sick = sick_data200moments(sick)
# compute model moments using estimated parameter values
model_moments_sick = sick_model200moments(sick, \
                                    beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM)

# print out the results
print('----------------------------------------------------------------------')
print('Problem 2')
print('beta0_GMM = ', beta0_GMM)
print('beta1_GMM = ', beta1_GMM)
print('beta2_GMM = ', beta2_GMM)
print('beta3_GMM = ', beta3_GMM)
print('val of GMM criterion function using GMM estimators: ', val_crit_sick)
print('-----------------------------------')
print(results_sick)
print('----------------------------------------------------------------------')
print()
