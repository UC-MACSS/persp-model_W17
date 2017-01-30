# Perspective on Computational Modeling
# Problem Set #3
# Xingyun Wu


import os
import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt
import scipy.integrate as intgr
import matplotlib
import matplotlib.pyplot as plt

np.seterr(all='ignore')


# Create directory if images directory does not already exist
'''
--------------------------------------------------------------------
cur_path    = string, path name of current directory
output_fldr = string, folder in current path to save files
output_dir  = string, total path of images folder
output_path = string, path of file name of figure to be saved
--------------------------------------------------------------------
'''
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


# PROBLEM 1: SOME INCOME DATA, LOGNORMAL DISTRIBUTION AND GMM
# read in the data
income = np.loadtxt('incomes.txt')

# 1(a)
plot_1a = True
if plot_1a:
    num_bins = 30
    weights = (1/len(income))*np.ones_like(income)
    count, bins, ignored = plt.hist(income, num_bins, weights = weights, \
                                    color='pink', normed=True)
    # set title and labels
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    plt.show()
    plt.close()

# 1(b)
print('Problem 1(b):')
# Define function that generates values of a truncated normal pdf
def trunc_lognorm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the truncated normal pdf with mean mu and
    standard deviation sigma. If the cutoff is finite, then the PDF
    values are inflated upward to reflect the zero probability on values
    above the cutoff. If there is no cutoff given or if it is given as
    infinity, this function does the same thing as
    sp.stats.norm.pdf(x, loc=mu, scale=sigma).
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
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) 
    return pdf_vals

# Define function that generate data moments
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


def model_moments(mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), variance(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
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
    xfx = lambda x: x * trunc_lognorm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, cutoff)
    x2fx = lambda x: ((x - mean_model) ** 2) * trunc_lognorm_pdf(x, mu, sigma) 
    (var_model, v_m_err) = intgr.quad(x2fx, 0, cutoff)
    
    return mean_model, var_model


def err_vec(xvals, mu, sigma, cutoff, simple):
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
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
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
    mean_model, var_model = model_moments(mu, sigma, cutoff)
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
    xvals, cutoff, W = args
    err = err_vec(xvals, mu, sigma, cutoff, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

# set basic parameters
cutoff = 150000
mu_init = 11.0
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
# calculation
# step 1
gmm_args_1b = (income, cutoff, W_hat)
results_1b = opt.minimize(criterion, params_init, args=(gmm_args_1b),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM_1b, sig_GMM_1b = results_1b.x
print('mu_GMM1=', mu_GMM_1b, ' sig_GMM1=', sig_GMM_1b)
# step 2
mean_data, var_data = data_moments(income)
mean_model_1b, var_model_1b = model_moments(mu_GMM_1b, sig_GMM_1b, cutoff)
err1 = err_vec(income, mu_GMM_1b, sig_GMM_1b, cutoff, False).reshape(2,)
params_GMM_1b = np.array([mu_GMM_1b, sig_GMM_1b])
value_1b = criterion(params_GMM_1b, *gmm_args_1b)[0][0]
print('Mean of points =', mean_data, ', standard deviation of points =', np.sqrt(var_data))
print('Mean of model =', mean_model_1b, ', standard deviation of model =', np.sqrt(var_model_1b))
print('Error vector=', err1)
print('The value of GMM criterion function: ', value_1b)
print()

# plot 1(b)
plot_1b = True
if plot_1b:
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(income, 30, normed=True, color='pink')
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Plot the estimated GMM PDF
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM_1b, sig_GMM_1b),\
         linewidth=2, color='k', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'\
                                                .format(mu_GMM_1b, sig_GMM_1b))
    plt.legend(loc='upper left')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1b')
    plt.savefig(output_path)
    plt.show()
    plt.close()

# Problem 1(c)
print('Problem 1(c):')
err2 = err_vec(income, mu_GMM_1b, sig_GMM_1b, cutoff, False)
VCV2 = np.dot(err2, err2.T) / income.shape[0]
print(VCV2)
W_hat_1c = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
print(W_hat_1c)
# calculation
# step 1
gmm_args_1c = (income, cutoff, W_hat_1c)
results_1c = opt.minimize(criterion, params_init, args=(gmm_args_1c),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM_1c, sig_GMM_1c = results_1c.x
print('mu_GMM2=', mu_GMM_1c, ' sig_GMM2=', sig_GMM_1c)
# step 2
mean_data, var_data = data_moments(income)
mean_model_1c, var_model_1c = model_moments(mu_GMM_1c, sig_GMM_1c, cutoff)
err_1c = err_vec(income, mu_GMM_1c, sig_GMM_1c, cutoff, False).reshape(2,)
params_GMM_1c = np.array([mu_GMM_1c, sig_GMM_1c])
value_1c = criterion(params_GMM_1c, *gmm_args_1c)[0][0]
print('Mean of points =', mean_data, ', standard deviation of points =', np.sqrt(var_data))
print('Mean of model =', mean_model_1c, ', standard deviation of model =', np.sqrt(var_model_1c))
print('Error vector=', err_1c)
print('The value of GMM criterion function: ', value_1c)

# plot 1(c)
plot_1c = True
if plot_1c:
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(income, 30, normed=True, color='pink')
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Plot the estimated GMM PDF
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM_1b, sig_GMM_1b),\
         linewidth=2, color='k', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.\
                                                format(mu_GMM_1b, sig_GMM_1b))
    plt.legend(loc='upper left')
    # Plot the estimated GMM PDF2
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM_1c, sig_GMM_1c),\
         linewidth=2, color='b', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.\
                                                format(mu_GMM_1c, sig_GMM_1c))
    plt.legend(loc='upper left')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1c')
    plt.savefig(output_path)
    plt.show()
    plt.close()

print()

# Problem 1(d)
print('Problem 1(d):')

# Define the necessary functions
def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the four data moments for GMM
    (binpct_1, binpct_2, binpct_3, binpct_4).
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
                 100000 <= x 

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >= 75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = xvals[(xvals >= 100000)].shape[0] / xvals.shape[0]

    return bpct_1_dat, bpct_2_dat, bpct_3_dat


def model_moments3(mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    This function computes the four model moments for GMM
    (binpct_1, binpct_2, binpct_3, binpct_4).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
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
    xfx = lambda x: trunc_lognorm_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, cutoff)
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod


def err_vec3(xvals, mu, sigma, cutoff, simple):
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
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments4()
        model_moments4()
    
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
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma, cutoff)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    
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
        err_vec4()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (3, 1) matrix, column vector of three moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cutoff, W = args
    err = err_vec3(xvals, mu, sigma, cutoff, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

# set basic parameters
cutoff = 150000
mu_init = 11.0
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
W_hat_1d = np.eye(3)
gmm_args_1d = (income, cutoff, W_hat_1d)
results_1d = opt.minimize(criterion3, params_init, args=(gmm_args_1d),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM_1d, sig_GMM_1d = results_1d.x
params_GMM_1d = np.array([mu_GMM_1d, sig_GMM_1d])
value_1d = criterion3(params_GMM_1d, *gmm_args_1d)[0][0]
print('mu_GMM_1d=', mu_GMM_1d, ' sig_GMM_1d=', sig_GMM_1d)
print('data moments: ', data_moments3(income))
print('model moments: ', model_moments3(mu_GMM_1d, sig_GMM_1d, cutoff))
print('The value of GMM criterion function: ', value_1d)

# plot 1(d)
plot_1d = True
if plot_1d:
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(income, 30, normed=True, color='pink')
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Plot the estimated GMM PDF
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM_1d, sig_GMM_1d),\
         linewidth=2, color='k', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.\
                                                format(mu_GMM_1d, sig_GMM_1d))
    plt.legend(loc='upper left')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1d')
    plt.savefig(output_path)
    plt.show()
    plt.close()

print()

# Problem 1(e)
print('Problem 1(e):')
# set the weighting matrix
err_1e = err_vec3(income, mu_GMM_1d, sig_GMM_1d, cutoff, False)
VCV_1e = np.dot(err_1e, err_1e.T) / income.shape[0]
print(VCV_1e)
W_hat_1e = lin.pinv(VCV_1e)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
print(W_hat_1e)
# calculation
params_1e = np.array([mu_GMM_1d, sig_GMM_1d])
gmm_args_1e = (income, cutoff, W_hat_1e)
results_1e = opt.minimize(criterion3, params_1e, args=(gmm_args_1e),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM_1e, sig_GMM_1e = results_1e.x
results_1e
params_GMM_1e = np.array([mu_GMM_1e, sig_GMM_1e])
value_1e = criterion3(params_GMM_1e, *gmm_args_1e)[0][0]
# print
print('mu_GMM_1e=', mu_GMM_1e, ' sig_GMM_1e=', sig_GMM_1e)
print('data moments: ', data_moments3(income))
print('model moments: ', model_moments3(mu_GMM_1e, sig_GMM_1e, cutoff))
print('The value of GMM criterion function: ', value_1e)

# plot 1(e)

plot_1e = True
if plot_1e:
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(income, 30, normed=True, color='pink')
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Plot the estimated GMM PDF of 1(d)
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM_1d, sig_GMM_1d),\
         linewidth=2, color='k', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.\
                                                format(mu_GMM_1d, sig_GMM_1d))
    plt.legend(loc='upper left')
    # Plot the estimated GMM PDF of 1(e)
    dist_inc = np.linspace(0, cutoff, cutoff-0)
    plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM_1e, sig_GMM_1e),\
         linewidth=2, color='b', label='$\mu$ = {:.4f}, $\sigma$ = {:.4f}'.\
                                                format(mu_GMM_1e, sig_GMM_1e))
    plt.legend(loc='upper left')    
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1e')
    plt.savefig(output_path)
    plt.show()
    plt.close()

print()

# Problem 1(f)
print('Problem 1(f): discussion is shown in the pdf file')

print()
print()

# PROBLEM 2: LINEAR REGRESSION AND GMM
print('Problem 2: Linear Regression and GMM')
sick = pd.read_csv('sick.txt', encoding = 'utf-8-sig')
#sick_i = sick.iloc[:, 0]
#age_i = sick.iloc[:, 1]
#children_i = sick.iloc[:, 2]
#temp_winter_i = sick.iloc[:, 3]

def err_vec(data, b0, b1, b2, b3, simple):
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
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
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
    moms_data = np.array([[i] for i in data['sick']])
    moms_model = b0 + b1*data['age'] + b2*data['children'] + b3*data['avgtemp_winter']
    moms_model = np.array([[j] for j in moms_model])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec

def criterion2(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
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
    if len(params) == 1: # somehow at the end the optimization function would 
						 # pack parameters into an array, so I have to unpack
        b0, b1, b2, b3 = params[0]
    else:
        b0, b1, b2, b3 = params
    data, W = args
    err = err_vec(data, b0, b1, b2, b3, simple=True)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

# 2(a)
# setup initial guess
sig_init2 = 1 # standard normal distribution
b0 = 1
b1 = 0.2
b2 = 0.3
b3 = 0.4
W2 = np.eye(200)

#params_init2 = np.array([sick_i, age_i, children_i, temp_winter_i])

#bnds2 = ((None, None), (None, None), (None, None), (None, None))

params_init2 = np.array([b0, b1, b2, b3])
gmm_args2 = (sick, W2)
results2 = opt.minimize(criterion2, params_init2, args=(gmm_args2),
                       method='L-BFGS-B')
b0_GMM, b1_GMM, b2_GMM, b3_GMM = results2.x
print('b0_GMM=', b0_GMM, ' b1_GMM=', b1_GMM, ' b2_GMM=', b2_GMM, ' b3_GMM=', b3_GMM)
params_GMM = np.array([b0_GMM, b1_GMM, b2_GMM, b3_GMM])
print('The value of GMM criterion function value is: ', criterion2(params_GMM, *gmm_args2)[0][0])