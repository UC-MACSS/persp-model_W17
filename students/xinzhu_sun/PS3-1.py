import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math
import os
np.random.seed(seed=1234)

# Problem 1: some income data, lognormal distribution and GMM
incomes = np.loadtxt('incomes.txt')

#(a) Plot a histogram of percentages of the data with 30 bins.
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    Fig_1a  = Boolean, = True if make a histogram of percentages of the
                         annual incomes of MACSS graduats
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    count, bins, patches = plt.hist(incomes, 30, normed = True)
    plt.title('Annual incomes of MACSS graduates:2018-2020', fontsize=20)
    plt.xlabel('Annual incomes')
    plt.ylabel('Percent of Graduates')
    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    plt.show()
    plt.close()

#(b) Perform GMM with 2 moments
def trunc_lognorm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the truncated normal pdf with mean mu and
    standard deviation sigma. 
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
    
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) 
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
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_lognorm_pdf()
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
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * trunc_lognorm_pdf(x, mu, sigma) 
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
    
    return mean_model, var_model

def err_vec(xvals, mu, sigma, simple):
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
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, W_hat)
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
    mu, sigma = params
    xvals, W = args
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

mu_init = 11.0
sig_init = 0.3
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
gmm_args = (incomes, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM, sig_GMM = results.x

# Plot the estimated lognormal PDF against the histogram from (a)
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    Fig_1b  = Boolean, = True if plot the estimated lognormal PDF against
                         the income distribution histogram.
    --------------------------------------------------------------------
    '''
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(incomes, 30, normed = True)
    # Plot the estimated lognormal PDF
    dist_incomes = np.linspace(0, 160000, 15000)
    plt.plot(dist_incomes, trunc_lognorm_pdf(dist_incomes, mu_GMM, sig_GMM),\
             linewidth=2, color='r', label='The lognormal PDF of GMM with 2 moments')
    plt.title('Income Distribution of MACSS graduates:2018-2020', fontsize=20)
    plt.legend(loc='upper right', prop={'size':8})
    plt.xlabel('Annual incomes')
    plt.ylabel('Density')
    output_path = os.path.join(output_dir, 'Fig_1b')
    plt.savefig(output_path)
    plt.show()

# Report the value of GMM criterion function with 2 moments
crit_val = criterion(results.x, *gmm_args)
print('1(b). The GMM estimator mu is ', mu_GMM, ', sigma is ', sig_GMM, '.')
print('      The value of the criterion function is ', crit_val, '.')
# Report the two data moments and two model moments
mean_data, var_data = data_moments(incomes)
mean_model, var_model = model_moments(mu_GMM, sig_GMM)
print('      Mean of points is ', mean_data, ', standard deviation of points is ', np.sqrt(var_data), '.')
print('      Mean of model is ', mean_model, ', standard deviation of model is ', np.sqrt(var_model), '.')

#(c) Perform the two-step GMM with 2 moments
err1 = err_vec(incomes, mu_GMM, sig_GMM, False)
VCV2 = np.dot(err1, err1.T) / incomes.shape[0]
# Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
W_hat2 = lin.pinv(VCV2)
params_init_two_step = np.array([mu_GMM, sig_GMM])
gmm_args_two_step = (incomes, W_hat2)
results_two_step = opt.minimize(criterion, params_init_two_step, args=(gmm_args_two_step),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
# Report the estimates as well as the criterion function value
mu_GMM_two_step, sig_GMM_two_step = results_two_step.x
crit_val_two_step= criterion(results_two_step.x, *gmm_args_two_step)
print('1(c). The two-step GMM estimator mu is ', mu_GMM_two_step, ', sigma is ', sig_GMM_two_step, '.')
print('      The value of the two-step GMM criterion function is ', crit_val_two_step, '.')

# Plot the estimated lognormal PDF against the histogram from (a) and the
# estimated PDF from (b)
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    Fig_1c  = Boolean, = True if plot the estimated lognormal PDF of two-
                         step GMM against the income distribution histog-
                         ram and the estimated lognormal PDF from (b).
    --------------------------------------------------------------------
    '''
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(incomes, 30, normed = True)
    # Plot the estimated lognormal PDF from (b)
    dist_incomes = np.linspace(0, 160000, 15000)
    plt.plot(dist_incomes, trunc_lognorm_pdf(dist_incomes, mu_GMM, sig_GMM),\
             linewidth=2, color='r', label='The lognormal PDF of GMM with 2 moments')
    # Plot the estimated lognormal PDF of two-step GMM
    plt.plot(dist_incomes, trunc_lognorm_pdf(dist_incomes, mu_GMM_two_step, sig_GMM_two_step),\
             linewidth=2, color='g', label='The lognormal PDF of two-step GMM with 2 moments')
    plt.title('Income Distribution of MACSS graduates:2018-2020', fontsize=20)
    plt.legend(loc='upper right', prop={'size':8})
    plt.xlabel('Annual incomes')
    plt.ylabel('Density')
    output_path = os.path.join(output_dir, 'Fig_1c')
    plt.savefig(output_path)
    plt.show()

# Report the two data moments and two model moments of two-step GMM
mean_data_two_step, var_data_two_step = data_moments(incomes)
mean_model_two_step, var_model_two_step = model_moments(mu_GMM_two_step, sig_GMM_two_step)
print('      Mean of points is ', mean_data_two_step, ', standard deviation of points is ', np.sqrt(var_data_two_step), '.')
print('      Mean of model is ', mean_model_two_step, ', standard deviation of model is ', np.sqrt(var_model_two_step), '.')

#(d) Perform GMM with 3 moments
def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    (mean(data), variance(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    np.array([per1_model, per2_model, per3_model]):
        A numpy array that contains the three data moment value
    
    FILES CREATED BY THIS FUNCTION: None
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >= 75000) & (xvals < 100000)].shape[0] / xvals.shape[0])
    bpct_3_dat = xvals[xvals >= 100000].shape[0] / xvals.shape[0]
    
    return bpct_1_dat, bpct_2_dat, bpct_3_dat
    
def model_moments3(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the three model moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_lognorm_pdf()
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
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)    
    
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
    args   = length 2 tuple, (xvals, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (3, 1) matrix, column vector of three moment error
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

W_hat_3 = np.eye(3)
gmm_args_3 = (incomes, W_hat_3)
results_3 = opt.minimize(criterion3, params_init, args=(gmm_args_3),
                         method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM_3, sig_GMM_3 = results_3.x

# Plot the estimated GMM lognormal PDF with 3 moments against the histogram 
# from (a)
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    Fig_1d  = Boolean, = True if plot the estimated GMM lognormal PDF 
                         with 3 moments the income distribution histogram.
    --------------------------------------------------------------------
    '''
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(incomes, 30, normed = True)
    # Plot the estimated GMM lognormal PDF with 3 moments 
    dist_incomes = np.linspace(0, 160000, 15000)
    plt.plot(dist_incomes, trunc_lognorm_pdf(dist_incomes, mu_GMM_3, sig_GMM_3),\
             linewidth=2, color='r', label='The lognormal PDF of GMM with 3 moments')
    plt.title('Income Distribution of MACSS graduates:2018-2020', fontsize=20)
    plt.legend(loc='upper right', prop={'size':8})
    plt.xlabel('Annual incomes')
    plt.ylabel('Density')
    output_path = os.path.join(output_dir, 'Fig_1d')
    plt.savefig(output_path)
    plt.show()

# Report the value of GMM criterion function with 3 moments
crit_val_3 = criterion3(results_3.x, *gmm_args_3)
print('1(d). The GMM estimator mu is ', mu_GMM_3, ', sigma is ', sig_GMM_3, '.')
print('      The value of the criterion function is ', crit_val_3, '.')
# Report the two data moments and two model moments of GMM with 3 moments
bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(incomes)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM_3, sig_GMM_3)
print('      The 3 data moments are ', bpct_1_dat, ', ', bpct_2_dat, ', ', bpct_3_dat, '.')
print('      The 3 model moments are ', bpct_1_mod, ', ', bpct_2_mod, ', ', bpct_3_mod, '.')

#(e) Perform the two-step GMM with 3 moments
err1_3 = err_vec3(incomes, mu_GMM_3, sig_GMM_3, False)
VCV2_3 = np.dot(err1_3, err1_3.T) / incomes.shape[0]
# Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
W_hat2_3 = lin.pinv(VCV2_3)
params_init_two_step_3 = np.array([mu_GMM_3, sig_GMM_3])
gmm_args_two_step_3 = (incomes, W_hat2_3)
results_two_step_3 = opt.minimize(criterion3, params_init_two_step_3, args=(gmm_args_two_step_3),
                                  method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
# Report the estimates as well as the criterion function value
mu_GMM_two_step_3, sig_GMM_two_step_3 = results_two_step_3.x
crit_val_two_step_3= criterion3(results_two_step_3.x, *gmm_args_two_step_3)
print('1(e). The two-step GMM with 3 moments estimator mu is ', mu_GMM_two_step_3, ', sigma is ', sig_GMM_two_step_3, '.')
print('      The value of the two-step GMM criterion function with 3 moments is ', crit_val_two_step_3, '.')

# Plot the estimated lognormal PDF against the histogram from (a) and the
# estimated PDF from (d)
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    Fig_1e  = Boolean, = True if plot the estimated lognormal PDF of two-
                         step GMM against the income distribution histog-
                         ram and the estimated lognormal PDF from (d).
    --------------------------------------------------------------------
    '''
    # Plot the histogram of the data
    count, bins, ignored = plt.hist(incomes, 30, normed = True)
    # Plot the estimated GMM lognormal PDF with 3 moments 
    dist_incomes = np.linspace(0, 160000, 15000)
    plt.plot(dist_incomes, trunc_lognorm_pdf(dist_incomes, mu_GMM_3, sig_GMM_3),\
             linewidth=2, color='r', label='The lognormal PDF of GMM with 3 moments')
    # Plot the estimated lognormal PDF of two-step GMM with 3 moments
    plt.plot(dist_incomes, trunc_lognorm_pdf(dist_incomes, mu_GMM_two_step_3, sig_GMM_two_step_3),\
             linewidth=2, color='g', label='The lognormal PDF of two-step GMM with 3 moments')
    plt.title('Income Distribution of MACSS graduates:2018-2020', fontsize=20)
    plt.legend(loc='upper right', prop={'size':8})
    plt.xlabel('Annual incomes')
    plt.ylabel('Density')
    output_path = os.path.join(output_dir, 'Fig_1e')
    plt.savefig(output_path)
    plt.show()

# Report the two data moments and two model moments of two-step GMM
bpct_1_dat_two_step, bpct_2_dat_two_step, bpct_3_dat_two_step = data_moments3(incomes)
bpct_1_mod_two_step, bpct_2_mod_two_step, bpct_3_mod_two_step = model_moments3(mu_GMM_two_step_3, sig_GMM_two_step_3)
print('      The 3 data moments are ', bpct_1_dat_two_step, ', ', bpct_2_dat_two_step, ', ', bpct_3_dat_two_step, '.')
print('      The 3 model moments are ', bpct_1_mod_two_step, ', ', bpct_2_mod_two_step, ', ', bpct_3_mod_two_step, '.')
















