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
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
------------------------------------------------------------------------
Exercise 1: Income data, lognormal distribution, and SMM
------------------------------------------------------------------------
(a) = Plot a histogram of percentages of the income.txt data with 30 bins.
(b) = Write your own function for the lognormal PDF above called LN_pdf().
(c) = Estimate the parameters of the lognormal distribution by simulated 
	  method of moments (SMM).
(d) = Perform the two-step SMM estimator by using your estimates from 
	  part (c) with two moments
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
    fig = plt.figure(figsize = (10,10))
    n, bin_cuts, patches = plt.hist(df, num_bins, weights=weights,normed = True)
    plt.title('Histogram of incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1a')
    plt.savefig(output_path)
    plt.close()


'''
------------------------------------------------------------------------
(b) = Write your own function for the lognormal PDF above called LN_pdf().
------------------------------------------------------------------------
'''

# Define function that generates values of a lognormal pdf
def LN_pdf(xvals, mu, sigma):
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
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''  
    pdf_vals    = 1/(xvals * sigma * np.sqrt(2 * np.pi)) *np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))
    return pdf_vals

 
xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
mu_test = 5.0
sigma_test = 1.0
test_value = LN_pdf(xvals,mu_test,sigma_test)
print('Part(1.b)')
print('The lognormal pdf is', test_value)


'''
------------------------------------------------------------------------
(c) = Estimate the parameters of the lognormal distribution by simulated 
	  method of moments (SMM).
------------------------------------------------------------------------
'''


# Define function that draws N x S test score values from a 
# lognormal distribution
def LN_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a lognormal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma. These draws correspond to an (N x S) matrix 
    of randomly generated draws from a uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    norm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                 values drawn from truncated normal PDF with base
                 normal distribution N(mu, sigma) and cutoffs
                 (cut_lb, cut_ub)
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: lognorm_draws
    --------------------------------------------------------------------
    '''

    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    lognormal_draws = np.exp(norm_draws)
    
    return lognormal_draws

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
    mean_data = scalar or (S,) vector, mean value of test scores data
    var_data  = scalar > 0 or (S,) vector, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        var_data = xvals.var()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        var_data = xvals.var(axis=0)
    
    return mean_data, var_data


def err_vec(data_vals, sim_vals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    sim_vals  = (N, S) matrix, S simulations of test scores data
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    var_model  = scalar > 0, estimated variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, var_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_sim, var_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    var_model = var_sim.mean()
    moms_model = np.array([[mean_model], [var_model]])
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
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, W_hat = args
    sim_vals = LN_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val

#initialization

N = 200
S = 300 
np.random.seed(1234)
mu_init = 9.0
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
unif_vals= sts.uniform.rvs(0, 1, size=(N, S))

smm_args = (df, unif_vals, W_hat)
results = opt.minimize(criterion, params_init, args=(smm_args),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))

mu_SMM, sig_SMM = results.x

print('Part(1.c)')
print('mu_SMM=', mu_SMM, ' sig_SMM=', sig_SMM)

mean_data, var_data = data_moments(df)
print("mean_data = ", mean_data, ", var_data = ", var_data, ", std_data = ", np.sqrt(var_data))

sim_vals = LN_draws(unif_vals, mu_SMM, sig_SMM)
mean_sim, var_sim = data_moments(sim_vals)
mean_mod = mean_sim.mean()
var_mod = var_sim.mean()
print('Mean of one step SMM model =', mean_mod, ', Variance of one step SMM model =', var_mod,'Standard deviation of one step SMM model=', np.sqrt(var_mod))

# criterion function value
params_SMM = np.array([mu_SMM, sig_SMM])
crit_val = criterion(params_SMM, *smm_args)[0][0]
print('The value of SMM criterion function at the estimated parameter values =', crit_val)



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
    fig = plt.figure(figsize = (10,10))
    dist_pts = np.linspace(0, 150000, 500)  
    count, bins, ignored = plt.hist(df, 30, normed=True)
# Plot the GMM estimated distribution
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM, sig_SMM),\
			linewidth=2, color='k', label='$\mu$ = {:.3f}, $\sigma$ = {:.3f}'\
                                                .format(mu_SMM, sig_SMM))    
    plt.legend(loc='upper right')
    plt.title('Incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1C')
    plt.savefig(output_path)
    plt.close()

'''
------------------------------------------------------------------------
(d) = Perform the two-step SMM estimator by using your estimates from 
	  part (c) with two moments
------------------------------------------------------------------------
'''

err2_1 = err_vec(df, sim_vals, mu_SMM, sig_SMM,False)
VCV2_1 = np.dot(err2_1, err2_1.T) / df.shape[0]
# print(VCV2_1)
W_hat2_1 = lin.pinv(VCV2_1)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
# print(W_hat2_1)

params_init2_1 = np.array([mu_SMM, sig_SMM])
# W_hat3 = np.array([[1. / VCV2[0, 0], 0.], [0., 1. / VCV2[1, 1]]])
smm_args2_1 = (df, unif_vals, W_hat2_1)
results2_1 = opt.minimize(criterion, params_init2_1, args=(smm_args2_1),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM2, sig_SMM2 = results2_1.x

print('Part(1.d)')
print('mu_SMM2=', mu_SMM2, ' sig_SMM2=', sig_SMM2)


mean_data, var_data = data_moments(df)
print("mean_data = ", mean_data, ", var_data = ", var_data, ", std_data = ", np.sqrt(var_data))
sim_vals_2 = LN_draws(unif_vals, mu_SMM2, sig_SMM2)
mean_sim_2, var_sim_2 = data_moments(sim_vals_2)
mean_mod_2 = mean_sim_2.mean()
var_mod_2 = var_sim_2.mean()

print('Mean of two step SMM model =', mean_mod_2, ', Variance of two step SMM model =', var_mod_2,'Standard deviation of two step SMM model=', np.sqrt(var_mod_2))

# get value of criterion function
params_SMM_2 = np.array([mu_SMM2, sig_SMM2])
crit_val_2= criterion(params_SMM_2, *smm_args2_1)[0][0]
print('The value of SMM criterion function at the estimated parameter values =', crit_val_2)

plot_1d = True

if plot_1d:
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
    fig = plt.figure(figsize = (10,10))
    count, bins, ignored = plt.hist(df, 30, normed=True)
# Plot the SMM estimated distribution
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM, sig_SMM),\
    	linewidth=2, color='k', label='1: $\mu$ = {:.3f}, $\sigma$ = {:.3f}'\
                                                .format(mu_SMM, sig_SMM))
    plt.legend(loc='upper right')
# Plot the two-step SMM estimated distribution
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM2, sig_SMM2),\
    	linewidth=2, color='r', linestyle = '--',label='2: $\mu$ = {:.3f}, $\sigma$ = {:.3f}'\
                                                .format(mu_SMM2, sig_SMM2))    
    plt.legend(loc='upper right')
    plt.xlim(0, 150000)
    plt.title('Incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1d')
    plt.savefig(output_path)
    plt.close()

















