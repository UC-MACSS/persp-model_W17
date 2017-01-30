# Problem set 3 script
# Yiqing Zhu

# Import packages
import os
import numpy as np
import pandas as pd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# load the data for problem 1
incomes_data = np.loadtxt('incomes.txt')
# load the data for problem 2
sick_data = pd.read_csv("sick.txt", encoding = 'utf-8-sig')
#sick_data = np.loadtxt('sick.txt', skiprows=1, delimiter=',')

'''
-------------------------------------------------------------------------------
Problem 1a:
Plot a histogram of percentages of the income.txt data with 30 bins. 
Make sure that the bins are weighted using the normed=True option.
Make sure your plot has correct x-axis and y-axis labels as well as a plot title.
-------------------------------------------------------------------------------
'''
plot_1a = True

if plot_1a:
    print('Problem 1a:')
    
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram
    fig, ax = plt.subplots()
    plt.hist(incomes_data, bins = 30, normed = True)
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1a')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()

    
'''
-------------------------------------------------------------------------------
Problem 1b:
Estimate the parameters of the lognormal distribution by generalized method of 
moments. Use the average income and standard deviation of income as your 
two moments. Use the identity matrix as your weighting matrix W. 
Plot your estimated lognormal PDF against the histogram from part (a). 
Report the value of your GMM criterion function at the estimated parameter values. 
Report and compare your two data moments against your two model moments at 
the estimated parameter values.
-------------------------------------------------------------------------------
'''
def log_norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the log normal pdf with mean mu and standard
    deviation sigma. This function does the same thing as 
    sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:

    log_pdf_vals = (N,) vector, log normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_pdf_vals
    --------------------------------------------------------------------
    '''
            
    log_pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi))) *\
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))

    return log_pdf_vals

    
def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), std(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of data
    std_data  = scalar > 0, standard deviation of data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, std_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    std_data = xvals.std()
    
    return mean_data, std_data


def model_moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), std(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_norm_pdf()
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    var_model  = scalar > 0, variance of data from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    std_model  = scalar > 0, standard deviation of data from model
                 
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, std_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * log_norm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma) 
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
    std_model = np.sqrt(var_model)
    
    return mean_model, std_model


def err_vec(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, data
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
    std_data   = scalar > 0, standard deviation of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    std_model  = scalar > 0, standard deviation from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [std_model]])
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
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_norm_pdf()
    
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

# Perform the GMM estimation, report the estimated parameter values
mu_init = 9.0
std_init = 0.3
params_init = np.array([mu_init, std_init])
W_hat = np.eye(2)
gmm_args = (incomes_data, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x
print('\nProblem 1b:')
print('mu_GMM1 =', mu_GMM1, ' sig_GMM1 =', sig_GMM1)

# Report the value of GMM criterion function at the estimated parameter values
params_GMM = np.array([mu_GMM1, sig_GMM1])
value = criterion(params_GMM, *gmm_args)[0][0]
print('The value of GMM criterion function at the estimated parameter values is: {}'
		  .format(value))

# Report data moments and model moments at the estimated parameter values
mean_data, std_data = data_moments(incomes_data)
mean_model, std_model = model_moments(mu_GMM1, sig_GMM1)
print('Mean of points =', mean_data, ', Standard deviation of points =', std_data)
print('Mean of model =', mean_model, ', Standard deviation of model =', std_model)


plot_1b = True

if plot_1b:
		
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram from part(a) and estimated lognormal PDF
    fig, ax = plt.subplots()
    plt.hist(incomes_data, bins = 30, normed = True)
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')
    dist_pts = np.linspace(0, 150000, 10000)
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1),
		 linewidth=2, color='k', label='1: $\mu_{GMM1}$,$\sigma_{GMM1}$')
    plt.legend(loc='upper right')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1b')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()

    
'''
-------------------------------------------------------------------------------
Problem 1c:
Perform the two-step GMM estimator by using your estimates from part (b) 
with two moments to generate an estimator for the variance covariance 
matrix Ωˆ2step, which you then use to get the two-step estimator for 
the optimal weighting matrix Wˆ 2step. 
Report your estimates as well as the criterion function value at these estimates.
Plot your estimated lognormal PDF against the histogram from part(a) 
and the estimated PDF from part (b). 
Report and compare your two data moments against your two model moments 
at the estimated parameter values.
-------------------------------------------------------------------------------
'''
# Find the two-step weighting matrix
err1 = err_vec(incomes_data, mu_GMM1, sig_GMM1, False)
VCV2 = np.dot(err1, err1.T) / incomes_data.shape[0]
W_hat2 = lin.pinv(VCV2)

# Perform the two-step GMM estimation, report the estimated parameter values
params_init = np.array([mu_GMM1, sig_GMM1])
gmm_args = (incomes_data, W_hat2)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results.x
print('\nProblem 1c:')
print('mu_GMM2 =', mu_GMM2, ' sig_GMM2 =', sig_GMM2)

# Report the value of GMM criterion function at the estimated parameter values
params_GMM = np.array([mu_GMM2, sig_GMM2])
value = criterion(params_GMM, *gmm_args)[0][0]
print('The value of two-step GMM criterion function at the estimated parameter values is: {}'
		  .format(value))

# Report data moments and model moments at the estimated parameter values
mean_model2, std_model2 = model_moments(mu_GMM2, sig_GMM2)
print('Mean of points =', mean_data, ', Standard deviation of points =', std_data)
print('Mean of model =', mean_model2, ', Standard deviation of model =', std_model2)


plot_1c = True

if plot_1c:
		
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram from part(a), estimated lognormal PDF from part(b),
    # and the two-step estimated lognormal PDF
    fig, ax = plt.subplots()
    plt.hist(incomes_data, bins = 30, normed = True)
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')
    dist_pts = np.linspace(0, 150000, 10000)
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1),
		 linewidth=2, color='k', label='1: $\mu_{GMM1}$,$\sigma_{GMM1}$')
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM2, sig_GMM2),
		 linewidth=2, color='r', label='2: $\mu_{GMM2}$,$\sigma_{GMM2}$')
    plt.legend(loc='upper right')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1c')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()


'''
-------------------------------------------------------------------------------
Problem 1d:
Now estimate the lognormal PDF to fit the data by GMM using different moments. 
Use percent of individuals who earn less than $75,000, percent of individuals 
who earn between $75,000 and $100,000, and percent of individuals who earn 
more than $100,000 as your three moments. Use the identity matrix as your 
estimator for the optimal weighting matrix. 
Plot your estimated lognormal PDF against the histogram from part (a). 
Report the value of your GMM criterion function at the estimated parameter values.
Report and compare your three data moments against your three model moments at 
the estimated parameter values.
-------------------------------------------------------------------------------
'''
def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, income data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_dat = scalar in [0, 1], percent of observations
                 0 <= x < 75000
    bpct_2_dat = scalar in [0, 1], percent of observations
                 75000 <= x <= 100000
    bpct_3_dat = scalar in [0, 1], percent of observations
                 100000 < x

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >= 75000) & (xvals <= 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = (xvals[xvals > 100000].shape[0] / xvals.shape[0])

    return (bpct_1_dat, bpct_2_dat, bpct_3_dat)


def model_moments3(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the three model moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_norm_pdf()
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
    xfx = lambda x: log_norm_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0.0000001, 75000)
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
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments3()
        model_moments3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    moms_data  = (2, 1) matrix, column vector of three data moments
    moms_model = (2, 1) matrix, column vector of three model moments
    err_vec    = (2, 1) matrix, column vector of three moment error
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

    
# Perform the three-moment GMM estimation, report the estimated parameter values
mu_init = 11
std_init = 0.2
params_init = np.array([mu_init, std_init])
W_hat3 = np.eye(3)
gmm_args = (incomes_data, W_hat3)
results = opt.minimize(criterion3, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM3, sig_GMM3 = results.x
print('\nProblem 1d:')
print('mu_GMM3 =', mu_GMM3, ' sig_GMM3 =', sig_GMM3)

# Report the value of three-moment GMM criterion function at the estimated parameter values
params_GMM = np.array([mu_GMM3, sig_GMM3])
value = criterion3(params_GMM, *gmm_args)[0][0]
print('The value of three-moment GMM criterion function at the estimated parameter values is: {}'
		  .format(value))

# Report data moments and model moments at the estimated parameter values
bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(incomes_data)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM3, sig_GMM3)
print('Data moments are: {}, {}, {}'.format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
print('Model moments are: {}, {}, {}'.format(bpct_1_mod, bpct_2_mod, bpct_3_mod))


plot_1d = True

if plot_1d:
		
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram from part(a) and the three-moment estimated lognormal PDF
    fig, ax = plt.subplots()
    plt.hist(incomes_data, bins = 30, normed = True)
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')
    dist_pts = np.linspace(0, 150000, 10000)
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM3, sig_GMM3),
		 linewidth=2, color='g', label='3: $\mu_{GMM3}$,$\sigma_{GMM3}$')
    plt.legend(loc='upper right')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1d')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()

'''
-------------------------------------------------------------------------------
Problem 1e:
Perform the two-step GMM estimator by using your estimates from part (d) 
with three moments to generate an estimator for the variance covariance matrix Ωˆ2step,
which you then use to get the two-step estimator for the optimal weighting matrix Wˆ 2step.
Report your estimates as well as the criterion function value at these estimates.
Plot your estimated lognormal PDF against the histogram from part (a) and 
the estimated PDF from part (d). 
Report and compare your three data moments against your three model moments 
at the estimated parameter values.
-------------------------------------------------------------------------------
'''
# Perform the two-step GMM estimation, report the estimated parameter values
err2 = err_vec3(incomes_data, mu_GMM3, sig_GMM3, False)
VCV2 = np.dot(err2, err2.T) / incomes_data.shape[0]
W_hat4 = lin.pinv(VCV2)
params_init = np.array([mu_GMM3, sig_GMM3])
gmm_args = (incomes_data, W_hat4)
results = opt.minimize(criterion3, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM4, sig_GMM4 = results.x
print('\nProblem 1e:')
print('mu_GMM4 =', mu_GMM4, ' sig_GMM4 =', sig_GMM4)

# Report the value of two-step GMM criterion function at the estimated parameter values
params_GMM = np.array([mu_GMM4, sig_GMM4])
value = criterion3(params_GMM, *gmm_args)[0][0]
print('The value of two-step GMM criterion function at the estimated parameter values is: {}'
		  .format(value))

# Report data moments and model moments at the estimated parameter values
bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(incomes_data)
bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM4, sig_GMM4)
print('Data moments are: {}, {}, {}'.format(bpct_1_dat, bpct_2_dat, bpct_3_dat))
print('Model moments are: {}, {}, {}'.format(bpct_1_mod, bpct_2_mod, bpct_3_mod))


plot_1e = True

if plot_1e:
		
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram from part(a), the three-moment estimated lognormal PDF
    # from part(b), and the two-step estimated lognormal PDF
    fig, ax = plt.subplots()
    plt.hist(incomes_data, bins = 30, normed = True)
    plt.title('PDF values for 2018-2020 MACSS Graduates annual income', fontsize = 12)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'PDF value')
    dist_pts = np.linspace(0, 150000, 10000)
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM3, sig_GMM3),
		 linewidth=2, color='g', label='3: $\mu_{GMM3}$,$\sigma_{GMM3}$')
    plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM4, sig_GMM4),
		 linewidth=2, color='y', label='4: $\mu_{GMM4}$,$\sigma_{GMM4}$')
    plt.legend(loc='upper right')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1e')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()


'''
-------------------------------------------------------------------------------
Problem 2a:
Estimate the parameters of the model (β0, β1, β2, β3) by GMM by solving the 
minimization problem of the GMM criterion function. Use the identity matrix 
as the estimator for the optimal weighting matrix. Treat each of the 200 values
of the variable sicki as your data moments (200 data moments). Treat the 
predicted or expected sick values from your model as your model moments 
(200 model moments). Let the error function of the moments be the simple 
difference (not percent difference) of the data moments from the model moments.
Use these error functions in your criterion function to estimate the model 
parameters (β0, β1, β2, β3) by GMM. 
Report your estimates and report the value of your GMM criterion function. 
-------------------------------------------------------------------------------
'''
def sick_data_moments(data):
    '''
    --------------------------------------------------------------------
    This function generates each of the 200 values of the variable sicki
    as the data moment for GMM.
    --------------------------------------------------------------------
    INPUTS:
    data = (N,) vector. sick data
    
    RETURNS: data_moments
    --------------------------------------------------------------------
    '''
    data_moments = []
    for i in range(data.shape[0]):
        data_value = data.iloc[i,0]
        data_moments.append(data_value)
    
    return np.array(data_moments)

    
def sick_model_moments(data, b0, b1, b2, b3):
    '''
    --------------------------------------------------------------------
    This function generates the predicted or expected sick values from
    the model as the model moments for GMM. 
    --------------------------------------------------------------------
    INPUTS:
    data = (N,) vector. sick data
    b0, b1, b2, b3 = given parameters 
    
    RETURNS: model_moments
    --------------------------------------------------------------------
    '''
    model_moments = []
    for i in range(data.shape[0]):
        age_i = data.iloc[i,1]
        children_i= data.iloc[i,2]
        avgtemp_winter_i = data.iloc[i,3]
        model_value = b0 + b1 * age_i + b2 * children_i + b3 * avgtemp_winter_i
        model_moments.append(model_value)
    
    return np.array(model_moments)
    
    
def sick_err_vec(data, b0, b1, b2, b3):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors for GMM.
    --------------------------------------------------------------------
    INPUTS:
    data = (N,) vector. sick data
    b0, b1, b2, b3 = given parameters 
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    moms_data = sick_data_moments(data)
    moms_model = sick_model_moments(data, b0, b1, b2, b3)
    err_vec = moms_model - moms_data
    
    return err_vec
    

def sick_criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (4,) vector, ([b0,b1,b2,b3])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 2 tuple, (xvals, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    b0, b1, b2, b3 = params
    data, W = args
    err = sick_err_vec(data, b0, b1, b2, b3)
    crit_val = np.dot(np.dot(err.T, W), err) 

    return crit_val


b0_init, b1_init, b2_init, b3_init = (0.1, 0.1, 0.2, 0.2)

# Perform the GMM estimation, report the estimated parameter values
sick_params_init = np.array([b0_init, b1_init, b2_init, b3_init])
sick_bnds = ((None, None),(None, None), (None, None), (None, None))
sick_W_hat = np.eye(200)
sick_args = (sick_data, sick_W_hat) 
sick_results = opt.minimize(sick_criterion, sick_params_init, args=(sick_args),
                       method='L-BFGS-B', bounds=sick_bnds)

b0_GMM, b1_GMM, b2_GMM, b3_GMM = sick_results.x
print('\nProblem 2a:')
print('beta_0 = ', b0_GMM, 'beta_1 = ', b1_GMM,\
      'beta_2 = ', b2_GMM, 'beta_3 = ', b3_GMM)

# Report the value of GMM criterion function at the estimated parameter values
sick_params_init = np.array([b0_GMM, b1_GMM, b2_GMM, b3_GMM])
value_sick = sick_criterion(sick_params_init, *sick_args)

print('The value of GMM criterion function at the estimated parameter values is: {}'
		  .format(value_sick))
