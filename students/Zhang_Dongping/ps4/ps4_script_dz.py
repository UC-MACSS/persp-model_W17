# MACS 30100: PS4
# Name: Dongping Zhang
# Python Version: 3.5
# Seed: 1234


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
    n, bin_cuts, ignored = plt.hist(incomes, bin_num, normed = True, \
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
Exercise 1/b.: Compose a lognormal pdf
------------------------------------------------------------------------------
LN_pdf     = a lognormal pdf function
test_array = an (2, 2) numpy array used to test if LN_pdf works 
test_pdf   = an (2, 2) numpy array of probability density corresponding to 
             test_array
------------------------------------------------------------------------------
'''
def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    The probability density function for a lognormal distribution with 
    parameter with mu is the mean of the normal distribution on which
    the lognormal is based, and sigma is the standard deviation of the
    normal distribution on which the lognormal is based
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector or an (N x S) matrix values of the log-normally 
             distributed random variable
    mu     = scalar, mean of the normally distributed random variable
             on which the log-normal is based
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable on which the log-normal is based
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    lognorm_pdf = the log-normal probability density function
    pdf_vals    = (N,) vector or an (N x S) matrix values of the  
                  log-normal PDF values for mu and sigma corresponding 
                  to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    lognorm_pdf = lambda xvals: \
                  ( 1 / ( xvals * sigma * np.sqrt(2 * np.pi) ) ) * \
                  np.exp( - (np.log(xvals) - mu)**2 / ( 2 * sigma**2) )

    if xvals.ndim == 1:
        pdf_vals = lognorm_pdf(xvals)  
    elif xvals.ndim == 2:     
        pdf_vals = np.array([ [lognorm_pdf(xvals.T[col])] \
                              for col in range(xvals.shape[1]) ]).T
        
    return pdf_vals


# testing of function constructed
test_array = np.array([ [200.0, 270.0], [180.0, 195.5] ])
test_pdf = LN_pdf(test_array, 5.0, 1.0)


# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/b')
print('Testing array is: ')
print(test_array)
print('Returning array of probability density is: ')
print(test_pdf)
print('----------------------------------------------------------------------')
print()


'''
------------------------------------------------------------------------------
Exercise 1/c.: Estimate the parameters of the lognormal distribution by SMM
------------------------------------------------------------------------------
LN_draws          = a function takes fixed uniform values to draw normal valus 
                    using an inverse normal pdf and exponentiating those normal
                    vals to get lognormal alues
data_moments      = a function that computes the data_moments
err_vec           = the error funcction
criterion         = the criterion function for optimization
mu_init           = initial guess of mu parameter
sigma_init        = initial guess of sigma parameter
params_init       = a (2, 0) numpy array of initial guesses
W_hat_I           = a (2, 0) identity matrix 
N                 = the number of observations within one simulation
S                 = the number of simulations
unif_vals         = a (200,) array of randomly drams uniform values from (0, 1)
sim_vals_SMM      = a (N, S) matrix of simulated lognormal values using uniform
                    values 
SMM_args          = the arguments needed to implement the optimization 
                    algorithms
results_SMM       = the optimization results
mu_SMM            = the mu parameter estimated by SMM
sigma_SMM         = the sigma parameter estimated by SMM
val_crit_SMM      = the value of SMM criterion function using the estimated SMM
                    parameters
data_moments_SMM  = a tuple of data moments
model_moments_SMM = a tuple of model moments
dist_pts          = simulated and equally spaced 200 points from 0 to 150000 
                    for plotting
SMM               =  Boolean, =True if make a plot of incomes distribution and
                     a layer of lognormal density function using SMM parameters 
------------------------------------------------------------------------------
'''    
def LN_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a lognormal distribution 
    based on values drawn from a uniform(0, 1). Based on those uniform
    values, draw the same (N x S) matrix of values from a normal 
    distribution with mean mu and standard deviation sigma. Finally, 
    take an exponential of those normal values to get the lognormal draws.
   --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the normally distributed random variable
                on which the log-normal is based
    sigma     = scalar > 0, standard deviation of the normally distributed
                random variable on which the log-normal is based
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.ppf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    norm_draws = normal draws through inverse normal cdf based on uniform 
                 values input into the function
    LN_draws   = lognormal values after exponentiating norm_draws
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: LN_draws
    --------------------------------------------------------------------
    ''' 
    norm_draws = sts.norm.ppf(unif_vals, mu, sigma)
    LN_draws = np.exp(norm_draws)

    return LN_draws


def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mu_data     = scalar or (S,) vector, mean value of test scores data
    sigma_data  = scalar > 0 or (S,) vector, variance of test scores data
    mus_data    = scalar or (S,) vector, mean values of simulated data
    sigmas_data = scalar > 0 or (S,) vector, variances of simulated data

    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, sigma_data or means_data, sigmas_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mu_data = xvals.mean()
        sigma_data = xvals.std() 
        return mu_data, sigma_data

    elif xvals.ndim == 2:
        mus_data = xvals.mean(axis = 0)
        sigmas_data = xvals.std(axis = 0)

        return mus_data, sigmas_data


def err_vec(data_vals, sim_vals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, incomes data
    sim_vals  = (N, S) matrix, S simulations of incomes data
    mu        = scalar, mean of the normally distributed random variable
                on which the log-normal is based
    sigma     = scalar > 0, standard deviation of the normally distributed
                random variable on which the log-normal is based
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        
    OBJECTS CREATED WITHIN FUNCTION:
    mu_data      = scalar, mean value of data
    sigma_data   = scalar > 0, variance of data
    moms_data    = (2, 1) matrix, column vector of two data moments
    mus_sim      = (N, 1) matrix, column vector of all simulated means
    sigmas_sim   = (N, 1) matrix, column vector of all simulated stds 
    mu_model     = scalar, estimated mean value from model
    var_model    = scalar > 0, estimated variance from model
    moms_model   = (2, 1) matrix, column vector of two model moments
    err_vec      = (2, 1) matrix, column vector of two moment error
                   functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    # construct data moments
    mu_data, sigma_data = data_moments(data_vals)
    moms_data = np.array([ [mu_data], [sigma_data] ])
    
    # construct model moments based on simulated method of moments
    mus_sim, sigmas_sim = data_moments(sim_vals)
    mu_model = mus_sim.mean()
    sigma_model = sigmas_sim.mean()
    moms_model = np.array([ [mu_model], [sigma_model] ])

    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params    = (2,) vector, ([mu, sigma])
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    args      = length 3 tuple,
                (xvals, unif_vals, W_hat)
    xvals     = (N,) vector, values of the normally distributed random 
                variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        LN_draws()
        err_vec
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, W_hat, = args
    sim_vals = LN_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma, simple = False)    
    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val


# making initial guess of the two parameters
mu_init = 11
sigma_init = 0.2
params_init = np.array( [mu_init, sigma_init] )
# use an identity matrix as the weighting matrix
W_hat_I = np.eye(2)
# simulate (N, S) matrix using uniform distribution to draw normal and exp
# those normals to lognormal
N = 300
S = 200
np.random.seed(seed = 1234)
unif_vals = sts.uniform.rvs(0, 1, size = (N, S))
# complie all necessary arguments for SMM estimation
SMM_args = (incomes, unif_vals, W_hat_I)
# implement SMM optimization
results_SMM = opt.minimize(criterion, params_init, args = (SMM_args), \
                           method = 'L-BFGS-B', \
                           bounds = ( (None, None), (1e-10, None) ), \
                           options={'eps': 1e-15})
# variable assignment of MLE parameters
mu_SMM, sigma_SMM = results_SMM.x
# compute the value of SMM
val_crit_SMM = criterion([mu_SMM, sigma_SMM], incomes, unif_vals, W_hat_I)
 
# compute the data moments and model moments
# simulated values using SMM estimators
sim_vals_SMM = LN_draws(unif_vals, mu_SMM, sigma_SMM)
# compute data moments
data_moments_SMM = data_moments(incomes)
# compute model moments by first
mean_sim_SMM, sigma_sim_SMM = data_moments(sim_vals_SMM)
mean_mod_SMM = mean_sim_SMM.mean()
sigma_mod_SMM = sigma_sim_SMM.mean()
model_moments_SMM = (mean_mod_SMM, sigma_mod_SMM)


# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/c')
print('mu_SMM = ', mu_SMM)
print('sigma_SMM = ', sigma_SMM)
print('val of GMM criterion function using GSMM estimators: ', val_crit_SMM)
print('-----------------------------------')
print('data moments are: ')
print(data_moments_SMM)
print('model moments using SMM estimators and identity matrix are: ')
print( model_moments_SMM) 
print('-----------------------------------')
print(results_SMM)
print('----------------------------------------------------------------------')
print()


# simulate equally spaced points of 200 from 0 to 150000 for plotting
dist_pts = np.linspace(0.01, 150000, 200)
# plotting
SMM = True
if SMM: 
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
    plt.plot(dist_pts, \
             sts.lognorm.pdf( dist_pts, sigma_SMM, 0, np.exp(mu_SMM) ), \
             linewidth=2, color='blue', \
             label='$SMM^{}$')
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
Exercise 1/d.: Compose a lognormal pdf
------------------------------------------------------------------------------
err                  = moment error vector in percentage using SMM estimator
VCV                  = variance-covariance matrix based on moment error vector
W_hat_2S             = two-step weighting matrix by taking the inverse of VCV
params_init_2S       = initial parameters using GMM estimators
SMM_args_2S          = new optimization arguments with two-step weighting 
                       matrix
results_SMM_2S       = an optimization result object
mu_SMM_2S            = 2step SMM estimator for mu
sigma_SMM_2S         = 2step GMM estimator for sigma
val_crit_SMM_2S      = value of GMM criterion function using GMM estimators
data_moments_SMM_2S  = a tuple of two data moments
model_moments_SMM_2S = a tuple of two model moments
SMM_2S               = Boolean, =True if make a plot of incomes distribution 
                       and the probability density function of lognormal using i
                       two sets of SMM parameters
------------------------------------------------------------------------------
'''
# compute the vector of moment errors in percentage using SMM estimators
err = err_vec(incomes, sim_vals_SMM, mu_SMM, sigma_SMM, False)
# computes the variance-covariance matrix based on moment errors
VCV = np.dot(err, err.T) / incomes.shape[0]
# take the inverse of the vcv matrix so as to get the two-step weighting matrix
W_hat_2S = lin.pinv(VCV)

# initial values now turned out to be GMM estimators
params_init_2S = np.array( [mu_SMM, sigma_SMM] )
SMM_args_2S = (incomes, unif_vals, W_hat_2S)
results_SMM_2S = opt.minimize(criterion, params_init_2S, args = (SMM_args_2S),\
                              method = 'L-BFGS-B', \
                              bounds = ( (None, None), (1e-12, None) ), \
                              options={'eps': 1e-15})


# variable assignment of parameters
mu_SMM_2S, sigma_SMM_2S = results_SMM_2S.x
# compute value of SMM criterion function using SMM estimators
val_crit_SMM_2S = criterion([mu_SMM_2S, sigma_SMM_2S], \
                            incomes,  unif_vals, W_hat_2S)


# compute the data moments and model moments
# simulate vals using 2step SMM estimators
sim_vals_SMM_2S = LN_draws(unif_vals, mu_SMM_2S, sigma_SMM_2S)
# compute data moments
data_moments_SMM_2S = data_moments(incomes)
# compute model moments by first
mean_sim_SMM_2S, sigma_sim_SMM_2S = data_moments(sim_vals_SMM_2S)
mean_mod_SMM_2S = mean_sim_SMM_2S.mean()
sigma_mod_SMM_2S = sigma_sim_SMM_2S.mean()
model_moments_SMM_2S = (mean_mod_SMM_2S, sigma_mod_SMM_2S)


# print out the results
print('----------------------------------------------------------------------')
print('Problem 1/d')
print('mu_SMM_2S = ', mu_SMM_2S)
print('sigma_GMM_2S = ', sigma_SMM_2S)
print(r'val of SMM criterion function using SMM estimators and optimal vcov' \
      + ' matrix: ', val_crit_SMM_2S)
print('-----------------------------------')
print('data moments are: ')
print(data_moments_SMM_2S)
print('model moments using SMM estimators and identity matrix are: ') 
print(model_moments_SMM_2S)
print('-----------------------------------')
print(results_SMM_2S)
print('----------------------------------------------------------------------')
print()

SMM_2S = True
if SMM_2S: 
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
    plt.plot(dist_pts, \
             sts.lognorm.pdf( dist_pts, sigma_SMM, 0, np.exp(mu_SMM) ), \
             linewidth=3, color='blue', \
             label='$SMM^{}$')
    plt.plot(dist_pts, \
             sts.lognorm.pdf( dist_pts, sigma_SMM_2S, 0, np.exp(mu_SMM_2S) ), \
             linewidth=0.5, color='gold', marker = '*', ls = ':',  \
             label='$SMM_{2Step}$')
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
