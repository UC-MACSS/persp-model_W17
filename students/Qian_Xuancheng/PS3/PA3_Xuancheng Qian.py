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
Exercise 1: Income data, lognormal distribution, and GMM
------------------------------------------------------------------------
(a) = Plot a histogram of percentages of the income.txt data with 30 bins.
(b) = Plot your estimated lognormal PDF against the histogram from part (a).
(c) = Perform the two-step GMM estimator by using your estimates from part (b) 
      and Plot your estimated lognormal PDF against the histogram from part (a) 
      and the estimated PDF from part (b). 
(d) = Now estimate the lognormal PDF to fit the data by GMM using different 
      moments.
(e) = Perform the two-step GMM estimator by using your estimates from part (d) 
      and plot your estimated log- normal PDF against the histogram from 
      part (a) and the estimated PDF from part (d).
(f) = Which of the four estimations from parts (b), (c), (d), and (e) 
      fits the data best? Justify your answer.
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
    n, bin_cuts, patches = plt.hist(df, num_bins, weights=weights,normed = True)
    plt.title('Histogram of incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1a')
    plt.savefig(output_path)
    plt.close()


# Define function that generates values of a lognormal pdf
def lognorm_pdf(xvals, mu, sigma):
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
    # pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))
    pdf_vals    = 1/(xvals * sigma * np.sqrt(2 * np.pi)) *np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))
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
    var_data = xvals.var()
    
    return mean_data, var_data


def model_moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), variance(model data)).
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
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, var_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * lognorm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 10000000, limit = 100)
    x2fx = lambda x: ((x - mean_model) ** 2) * lognorm_pdf(x, mu, sigma)
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 10000000, limit = 100)
    return mean_model, var_model

def err_vec(xvals, mu, sigma,simple):
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
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, var_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_model, var_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [var_model]])
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
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the lognormally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        lognorm_pdf()
    
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
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    return crit_val


'''
------------------------------------------------------------------------
(b) = Plot your estimated lognormal PDF against the histogram from part (a).
      Use the average income and standard deviation of income as your two moments.
      Use the identity matrix as your weighting matrix 
------------------------------------------------------------------------
'''


mu_init = 9
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
gmm_args = (df, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x

print('Part(1.b)')
print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)
params_gmm = np.array([mu_GMM1,sig_GMM1])
print('The value of GMM criterion function at the estimated parameter values =', criterion(params_gmm,*gmm_args))
mean_data, var_data = data_moments(df)
mean_model, var_model = model_moments(mu_GMM1, sig_GMM1)
# err1 = err_vec(df, mu_GMM1, sig_GMM1, cutoff, False).reshape(2,)
print('Mean of points =', mean_data, ', Variance of points =', var_data, ', Standard deviation of points=', np.sqrt(var_data))
print('Mean of model =', mean_model, ', Variance of model =', var_model,', Standard deviation of model=', np.sqrt(var_model))
# print('Error vector=', err1)


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
    fig = plt.figure()
    count, bins, ignored = plt.hist(df, 30, normed=True)
# Plot the GMM estimated distribution
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1),
         linewidth=2, color='k', label='$\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')
    plt.title('Incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1b')
    plt.savefig(output_path)
    plt.close()


'''
------------------------------------------------------------------------
(c) = Perform the two-step GMM estimator by using your estimates from part (b) 
      and Plot your estimated lognormal PDF against the histogram from part (a) 
      and the estimated PDF from part (b). 
------------------------------------------------------------------------
'''
err1 = err_vec(df, mu_GMM1, sig_GMM1, False)
VCV2 = np.dot(err1, err1.T) / df.shape[0]
# print(VCV2)
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
# print(W_hat2)

params_init = np.array([mu_GMM1, sig_GMM1])
# W_hat3 = np.array([[1. / VCV2[0, 0], 0.], [0., 1. / VCV2[1, 1]]])
gmm_args = (df, W_hat2)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results.x


print('Part(1.c)')
print('mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2)
params_gmm = np.array([mu_GMM2,sig_GMM2])
print('The value of GMM criterion function at the estimated parameter values =', criterion(params_gmm,*gmm_args))

mean_data, var_data = data_moments(df)
mean_model, var_model = model_moments(mu_GMM2, sig_GMM2)
# err1 = err_vec(df, mu_GMM2, sig_GMM2, False).reshape(2,)
print('Mean of points =', mean_data, ', Variance of points =', var_data, 'Standard deviation of points=', np.sqrt(var_data))
print('Mean of 2 step GMM model =', mean_model, ', Variance of 2 step GMM model =', var_model,'Standard deviation of 2 step GMM model=', np.sqrt(var_model))
# print('Error vector=', err1)

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
    count, bins, ignored = plt.hist(df, 30, normed=True)
# Plot the GMM estimated distribution
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1, sig_GMM1),
         linewidth=2, color='k', label='1: $\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')
# Plot the two-step GMM estimated distribution
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM2, sig_GMM2),
         linewidth=2, color='r', label='2: $\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')
    plt.xlim(0, 150000)
    plt.title('Incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1c')
    plt.savefig(output_path)
    plt.close()




'''
------------------------------------------------------------------------
(d) = Now estimate the lognormal PDF to fit the data by GMM using different 
      moments.
------------------------------------------------------------------------
'''
def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the four data moments for GMM
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
                 100000<= x < max(x)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = xvals[xvals >= 100000].shape[0] / xvals.shape[0]

    return bpct_1_dat, bpct_2_dat, bpct_3_dat

def model_moments3(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the four model moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        lognorm_pdf()
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
    xfx = lambda x: lognorm_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000,limit = 100)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000,limit =100)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 10000000,limit = 100)
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod


def err_vec3(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments3()
        model_moments3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    
    return err_vec

#
def criterion3(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
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
    xvals,W = args
    err = err_vec3(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

# print(data_moments3(pts))
# print(model_moments3(9, 0.3, 'None'))
# params = np.array([9, 0.3])
# print(criterion3(params, df, 'None', np.eye(3)))

mu_init = 11
sig_init = 0.2
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(3)
# W_hat[1, 1] = 2.0
# W_hat[2, 2] = 2.0
gmm_args = (df, W_hat)
results_3 = opt.minimize(criterion3, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM1_3, sig_GMM1_3 = results_3.x

print('Part(1.d)')
print('mu_GMM1_3=', mu_GMM1_3, ' sig_GMM1_3=', sig_GMM1_3)
params = np.array([mu_GMM1_3, sig_GMM1_3])
print('The value of GMM criterion function at the estimated parameter values =', criterion3(params, df, np.eye(3)))

print(data_moments3(df))
print(model_moments3(mu_GMM1_3, sig_GMM1_3))



# params_gmm = np.array([mu_GMM2,sig_GMM2])
# print('The value of GMM criterion function at the estimated parameter values =', criterion(params_gmm,*gmm_args))

# mean_data, var_data = data_moments(df)
# mean_model, var_model = model_moments(mu_GMM2, sig_GMM2, cutoff)
# err1 = err_vec(df, mu_GMM2, sig_GMM2, cutoff, False).reshape(2,)
# print('Mean of points =', mean_data, ', Variance of points =', var_data, 'Standard deviation of points=', np.sqrt(var_data))
# print('Mean of 2 step GMM model =', mean_model, ', Variance of 2 step GMM model =', var_model,'Standard deviation of 2 step GMM model=', np.sqrt(var_model))
# print('Error vector=', err1)


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
    dist_pts = np.linspace(0, 150000, 500)
    fig = plt.figure()
    count, bins, ignored = plt.hist(df, 30, normed=True)
# Plot the GMM estimated distribution
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1_3, sig_GMM1_3),
         linewidth=2, color='k', label='$\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')
    plt.title('Incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1d')
    plt.savefig(output_path)
    plt.close()


'''
------------------------------------------------------------------------
(e) = Perform the two-step GMM estimator by using your estimates from part (d) 
      and plot your estimated log- normal PDF against the histogram from 
      part (a) and the estimated PDF from part (d).
------------------------------------------------------------------------
'''

err1_3 = err_vec3(df, mu_GMM1_3, sig_GMM1_3, False)
VCV2_3 = np.dot(err1_3, err1_3.T) / df.shape[0]
# print(VCV2_3)
W_hat2_3= lin.pinv(VCV2_3)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
# print(W_hat2_3)


# Note that this takes a little time because the intgr.quad() commands
# are a little slow
mu_init = 11
sig_init = 0.3
params_init = np.array([mu_GMM1_3, sig_GMM1_3])
gmm_args = (df, W_hat2_3)
results2_3 = opt.minimize(criterion3, params_init, args=(gmm_args),
                       method='TNC', bounds=((1e-10, None), (1e-10, None)))
mu_GMM2_3, sig_GMM2_3 = results2_3.x

print('Part(1.e)')
print('mu_GMM2_3=', mu_GMM2_3, ' sig_GMM2_3=', sig_GMM2_3)
params = np.array([mu_GMM2_3, sig_GMM2_3])
print('The value of GMM criterion function at the estimated parameter values =', criterion3(params, df, np.eye(3)))
print(data_moments3(df))
print(model_moments3(mu_GMM2_3, sig_GMM2_3))



plot_1e = True

if plot_1e:
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
    count, bins, ignored = plt.hist(df, 30, normed=True)
# Plot the GMM estimated distribution
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM1_3, sig_GMM1_3),
         linewidth=2, color='k', label='1: $\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')
# Plot the two-step GMM estimated distribution
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_GMM2_3, sig_GMM2_3),
         linewidth=2, color='r', ls = 'dashed',label='2: $\mu$=11.33,$\sigma$=0.21')
    plt.legend(loc='upper right')
    plt.xlim(0, 150000)
    plt.title('Incomes of MACSS graduates: 2018-2020', fontsize=17)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations in bin')
    output_path = os.path.join('images', 'Fig_1e')
    plt.savefig(output_path)
    plt.close()


'''
------------------------------------------------------------------------
(f) = Which of the four estimations from parts (b), (c), (d), and (e) 
      fits the data best? Justify your answer.
------------------------------------------------------------------------
'''



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

    






'''
------------------------------------------------------------------------
Exercise 2: Linear regression and GMM 
------------------------------------------------------------------------
(a) = Estimate the parameters of the model by GMM. Use the identity matrix 
      as the estimator for the optimal weighting matrix
------------------------------------------------------------------------
'''
df_sick = pd.read_csv('sick.txt', header= 0, names = ['sick', 'age', 'children', 'avgtemp_winter'])


#define data moments
def data_moments_sick(data):
    data_moments = data['sick']
    return data_moments


#define model moments
def model_moments_sick(data, beta_0, beta_1, beta_2, beta_3):
    model_moments = beta_0+ beta_1*data['age'] + beta_2*data['children']+ beta_3*data['avgtemp_winter']
    return model_moments

#define error term
def err_vec_sick(data, beta_0, beta_1, beta_2, beta_3,simple):
    data_moments = data['sick']
    model_moments = beta_0+ beta_1*data['age'] + beta_2*data['children']+ beta_3*data['avgtemp_winter']
    if simple:
        err_vec = model_moments - data_moments
    else:
        err_vec = (model_moments- data_moments) / model_moments
    return err_vec

# define criterion
def criterion_sick(params, *args):
    beta_0, beta_1, beta_2, beta_3 = params
    data, W= args
    err = err_vec_sick(data, beta_0, beta_1, beta_2, beta_3,simple = True)
    crit_val = np.dot(np.dot(err.T, W), err)
    return crit_val


beta_0_init = 0
beta_1_init = 0
beta_2_init = 0
beta_3_init = 0

params_init = np.array([beta_0_init, beta_1_init, beta_2_init, beta_3_init])

W_hat = np.eye(200)
gmm_args = (df_sick,W_hat)
results_sick = opt.minimize(criterion_sick, params_init, args=(gmm_args),method='L-BFGS-B')
beta_0_GMM, beta_1_GMM, beta_2_GMM, beta_3_GMM = results_sick.x
print('Part(2.a)')
print('beta_0 =', beta_0_GMM, ' beta_1 =', beta_1_GMM, 'beta_2 =', beta_2_GMM, 'beta_3 =', beta_3_GMM)
params_gmm = np.array([beta_0_GMM, beta_1_GMM, beta_2_GMM, beta_3_GMM])
print('The value of GMM criterion function at the estimated parameter values =', criterion_sick(params_gmm,*gmm_args))




