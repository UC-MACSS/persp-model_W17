# Import packages and load the data
import numpy as np
import pandas as pd
import os
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


pts_income = np.loadtxt('incomes.txt') # load txt file
mydf = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'avgtemp_winter']) #load txt file as a pd df


#--------------------------------------------------------------------
#Problem 1a
#--------------------------------------------------------------------
def f1a():
	'''
	Problem 1a
	--------------------------------------------------------------------
	This function plots a histogram of percentages of the incomes.txt data
	with 30 bins. The bins are weighted using the normed=True option.
	--------------------------------------------------------------------
	INPUT: none
	RETURN: none
	--------------------------------------------------------------------
	'''
	fig, ax = plt.subplots(figsize=(10,6)) # Set the size of the window of the figure to be plotted.
	count, bins, ignored = plt.hist(pts_income, 30, normed=True)
	plt.title('Annual Incomes of MACSS graduates: 2018-2020', fontsize=20)
	plt.xlabel('income')
	plt.ylabel('percent of income')
	# plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted.


#--------------------------------------------------------------------
#Problem 1b
#--------------------------------------------------------------------
# Define a function that generates values of a lognormal pdf
def log_norm_pdf(xvals, mu, sigma, cutoff):
	'''
	--------------------------------------------------------------------
	This function generates pdf values from the lognormal pdf with mean mu 
	and standard deviation sigma. If the cutoff is given, then the PDF 
	values are inflated upward to reflect the zero probability on values
	above the cutoff. If there is no cutoff given, this function does 
	the same thing as sp.stats.lognorm.pdf(x, loc=mu, scale=sigma).
	--------------------------------------------------------------------
	INPUTS:
	xvals  = (N,) vector, values of the lonormally distributed random
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
	pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
			   corresponding to xvals data
	
	FILES CREATED BY THIS FUNCTION: None
	
	RETURNS: pdf_vals
	--------------------------------------------------------------------
	'''
	if cutoff == None:
		prob_notcut = 1.0
	else:
		prob_notcut = sts.lognorm.pdf(cutoff, loc=mu, scale=sigma)
			
	pdf_vals   = ((1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut)

	return pdf_vals


def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), standard deviation(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, incomes data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of incomes data
    std_data  = scalar > 0, standard deviation of incomes data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, std_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    std_data = xvals.std()
    
    return mean_data, std_data


def model_moments(mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), standard deviation(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_norm_pdf()
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of incomes from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    var_model  = scalar > 0, variance of incomes from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution

    std_model  = scalar > 0, standatd deviation of incomes from model
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, std_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * log_norm_pdf(x, mu, sigma, cutoff)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000) # 150000:upper bound of incomes
    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma, cutoff) 
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
    std_model = np.sqrt(var_model) # calculate the std.
    
    return mean_model, std_model


def err_vec(xvals, mu, sigma, cutoff, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, incomes data
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
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
    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = model_moments(mu, sigma, cutoff)
    moms_model = np.array([[mean_model], [std_model]])
    if simple: 
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec


# Define a criterion function that takes as inputs the parameters and the estimator for the weighting matrix Ŵ
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
    xvals, cutoff, W = args
    err = err_vec(xvals, mu, sigma, cutoff, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


def f1b():
	'''
	Problem 1b
	--------------------------------------------------------------------
	This function estimates mu & std of the lognormal distribution 
	by generalized method of moments. The identity matrix will be used as
	the weighting matrix W hat.
	--------------------------------------------------------------------
	INPUT: none
	RETURN: mu_GMM1, sig_GMM1
	--------------------------------------------------------------------
	'''
	mu_init = 9.0  # get this guess value from PS2
	sig_init = 0.3 # get this guess value from PS2
	params_init = np.array([mu_init, sig_init])
	W_hat = np.eye(2)
	gmm_args = (pts_income, None, W_hat)
	results_b = opt.minimize(criterion, params_init, args=(gmm_args), method='Nelder-Mead', bounds=((None, None), (1e-10, None)))
	mu_GMM1, sig_GMM1 = results_b.x
	# Report the estimated parameter values:
	print('1b. mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)

	# Calculate and report the value of the GMM criterion function at the estimated parameter values:
	params_GMM = np.array([mu_GMM1, sig_GMM1])
	value = criterion(params_GMM, *gmm_args)[0][0]
	print('The value of the GMM criterion function at the estimated parameter values is', value)

	# Calculate and report and compare your two data moments against your two model moments at the estimated parameter values.
	mean_data, std_data = data_moments(pts_income)
	mean_model, std_model = model_moments(mu_GMM1, sig_GMM1, None)
	err1 = err_vec(pts_income, mu_GMM1, sig_GMM1, None, False).reshape(2,)
	print('Mean of incomes =', mean_data, ', Standard deviation of incomes =', std_data)
	print('Mean of model =', mean_model, ', Standard deviation of model =', std_model)
	print('Error vector=', err1)
	#print(results_b)

	# Plot a histogram of percentages of the income.txt data with 30 bins (part a):
	fig, ax = plt.subplots(figsize=(10,6)) # Set the size of the window of the figure to be plotted.
	count, bins, ignored = plt.hist(pts_income, 30, normed=True)
	plt.title('Annual Incomes of MACSS graduates: 2018-2020', fontsize=20)
	plt.xlabel('income')
	plt.ylabel('percent of income')

	# Plot the estimated GMM PDF:
	dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
	plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1, None), linewidth=2, color='k', label='$\mu_{b}$: 11.337,$\sigma_{b}$: 0.213')
	plt.legend(loc='upper right')
	bins

	return mu_GMM1, sig_GMM1


#--------------------------------------------------------------------
#Problem 1c
#--------------------------------------------------------------------
def f1c(mu_GMM1, sig_GMM1):
	'''
	Problem 1c
	--------------------------------------------------------------------
	This function uses the estimates from problem (b) with two moments to
	generate an estimator for the variance covariance matrix omega hat 
	(2step), which it then uses to get the two-step estimator for the 
	optimal weighting matrix ^W (2step).
	--------------------------------------------------------------------
	INPUT: mu_GMM1 from problem b
		   sig_GMM1 from problem b
	RETURN: mu_GMM2, sig_GMM2
	--------------------------------------------------------------------
	'''
	err1 = err_vec(pts_income, mu_GMM1, sig_GMM1, None, False)
	VCV2 = np.dot(err1, err1.T) / pts_income.shape[0]
	# print(VCV2)
	W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
	# print(W_hat2)
	params_init = np.array([mu_GMM1, sig_GMM1])
	gmm_args = (pts_income, None, W_hat2)
	results_c = opt.minimize(criterion, params_init, args=(gmm_args), method='Nelder-Mead', bounds=((None, None), (1e-10, None)))
	mu_GMM2, sig_GMM2 = results_c.x
	# Report the estimated parameter values:
	print('1c. mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2)

	# Calculate and report the value of the GMM criterion function at the estimated parameter values:
	params_GMM = np.array([mu_GMM2, sig_GMM2])
	value = criterion(params_GMM, *gmm_args)[0][0]
	print('The value of the GMM criterion function at the estimated parameter values is', value)

	# Calculate and report and compare your two data moments against your two model moments at the estimated parameter values.
	mean_data, std_data = data_moments(pts_income)
	mean_model, std_model = model_moments(mu_GMM2, sig_GMM2, None)
	err1 = err_vec(pts_income, mu_GMM2, sig_GMM2, None, False).reshape(2,)
	print('Mean of incomes =', mean_data, ', Standard deviation of incomes =', std_data)
	print('Mean of model =', mean_model, ', Standard deviation of model =', std_model)
	print('Error vector=', err1)
	#print(results_c)

	# Plot a histogram of percentages of the income.txt data with 30 bins (part a):
	fig, ax = plt.subplots(figsize=(10,6)) # Set the size of the window of the figure to be plotted.
	count, bins, ignored = plt.hist(pts_income, 30, normed=True)
	plt.title('Annual Incomes of MACSS graduates: 2018-2020', fontsize=20)
	plt.xlabel('income')
	plt.ylabel('percent of income')

	# Plot the estimated GMM PDF (part b):
	dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
	plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1, None), linewidth=2, color='k', label='$\mu_{b}$: 11.337,$\sigma_{b}$: 0.213')
	plt.legend(loc='upper right')
	bins

	# Plot the estimated GMM PDF (part c):
	dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
	plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM2, sig_GMM2, None), '--', linewidth=2, color='red', label='$\mu_{c}$: 11.330,$\sigma_{c}$: 0.214')
	plt.legend(loc='upper right')
	bins

	return mu_GMM2, sig_GMM2


#--------------------------------------------------------------------
#Problem 1d
#--------------------------------------------------------------------
def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, incomes data

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
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = (xvals[(xvals >=100000)].shape[0] /
                  xvals.shape[0])

    return bpct_1_dat, bpct_2_dat, bpct_3_dat


def model_moments3(mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    This function computes the three model moments for GMM
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
    xfx = lambda x: log_norm_pdf(x, mu, sigma, cutoff)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 150000)
    
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
        data_moments3()
        model_moments3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    moms_data  = (3, 1) matrix, column vector of two data moments
    moms_model = (3, 1) matrix, column vector of two model moments
    err_vec    = (3, 1) matrix, column vector of two moment error
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
    xvals, cutoff, W = args
    err = err_vec3(xvals, mu, sigma, cutoff, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


def f1d(mu_GMM2, sig_GMM2):
	'''
	Problem 1d
	--------------------------------------------------------------------
	This function estimates the lognormal PDF to fit the data by GMM 
	using three moments: 1) percent of individuals who earn less than 
	$75,000, 2) percent of individuals who earn between $75,000 and 
	$100,000, and 3) percent of individuals who earn more than $100,000.
	The identity matrix is used as the estimator for the optimal weighting
	matrix.
	--------------------------------------------------------------------
	INPUT: mu_GMM2, sig_GMM2
	RETURN: mu_GMM1_3, sig_GMM1_3
	--------------------------------------------------------------------
	'''
	mu_init = 11.0 # initial guess from the results of previous parts
	sig_init = 0.2 # initial guess from the results of previous parts
	params_init = np.array([mu_init, sig_init])
	W_hat = np.eye(3)
	gmm_args = (pts_income, None, W_hat)
	results_d = opt.minimize(criterion3, params_init, args=(gmm_args), method='Nelder-Mead', bounds=((None, None), (1e-10, None)))
	mu_GMM1_3, sig_GMM1_3 = results_d.x
	# Report the estimated parameter values:
	print('1d. mu_GMM1_3=', mu_GMM1_3, ' sig_GMM1_3=', sig_GMM1_3)

	# Calculate and report the value of the GMM criterion function at the estimated parameter values:
	params_GMM = np.array([mu_GMM1_3, sig_GMM1_3])
	value = criterion3(params_GMM, *gmm_args)[0][0]
	print('The value of the GMM criterion function at the estimated parameter values is', value)

	# Calculate and report and compare the three data moments against the three model moments at the estimated parameter values:
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(pts_income)
	bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM1_3, sig_GMM1_3, None)
	print('Data moment1(<75,000) = ', bpct_1_dat, 'Data moment2(75,000~100,000) = ', bpct_2_dat, 'Data moment3(>100,000) = ', bpct_3_dat)
	print('Model moment1(<75,000) = ', bpct_1_mod, 'Model moment2(75,000~100,000) = ', bpct_2_mod, 'Model moment3(>100,000) = ', bpct_3_mod)
	#print(results_d)

	# Plot a histogram of percentages of the income.txt data with 30 bins (part a):
	f1a()

	# Plot the estimated GMM PDF (part d):
	dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
	plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1_3, sig_GMM1_3, None), linewidth=2, color='green', label='$\mu_{d}$: 11.337,$\sigma_{d}$: 0.212')
	plt.legend(loc='upper right')

	return mu_GMM1_3, sig_GMM1_3


#--------------------------------------------------------------------
#Problem 1e
#--------------------------------------------------------------------
def f1e(mu_GMM1_3, sig_GMM1_3):
	'''
	Problem 1e
	--------------------------------------------------------------------
	This function uses the estimates from part (d) with three moments to
	generate an estimator for the variance covariance matrix omega hat 
	(2step), which it then uses to get the two-step estimator for the 
	optimal weighting matrix ^W (2step).
	--------------------------------------------------------------------
	INPUT: mu_GMM1_3 from part d
		   sig_GMM1_3 from part d
	RETURN: none
	--------------------------------------------------------------------
	'''
	err1 = err_vec(pts_income, mu_GMM1_3, sig_GMM1_3, None, False)
	VCV2 = np.dot(err1, err1.T) / pts_income.shape[0]
	# print(VCV2)
	W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
	# print(W_hat2)
	params_init = np.array([mu_GMM1_3, sig_GMM1_3])
	gmm_args = (pts_income, None, W_hat2)
	results_e = opt.minimize(criterion, params_init, args=(gmm_args), method='Nelder-Mead', bounds=((None, None), (1e-10, None)))
	mu_GMM2_3, sig_GMM2_3 = results_e.x
	# Report the estimated parameter values:
	print('1e. mu_GMM2_3=', mu_GMM2_3, ' sig_GMM2_3=', sig_GMM2_3)

	# Calculate and report the value of the GMM criterion function at the estimated parameter values:
	params_GMM = np.array([mu_GMM2_3, sig_GMM2_3])
	value = criterion(params_GMM, *gmm_args)[0][0]
	print('The value of the GMM criterion function at the estimated parameter values is', value)

	# Calculate and report and compare the three data moments against the three model moments at the estimated parameter values.
	bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(pts_income)
	bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu_GMM2_3, sig_GMM2_3, None)
	print('Data moment1(<75,000) = ', bpct_1_dat, 'Data moment2(75,000~100,000) = ', bpct_2_dat, 'Data moment3(>100,000) = ', bpct_3_dat)
	print('Model moment1(<75,000) = ', bpct_1_mod, 'Model moment2(75,000~100,000) = ', bpct_2_mod, 'Model moment3(>100,000) = ', bpct_3_mod)
	#print(results_e)

	# Plot a histogram of percentages of the income.txt data with 30 bins (part a):
	f1a()

	# Plot the estimated GMM PDF (part d):
	dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
	plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1_3, sig_GMM1_3, None), linewidth=3, color='green', label='$\mu_{d}$: 11.337,$\sigma_{d}$: 0.212')
	plt.legend(loc='upper right')

	# Plot the estimated GMM PDF (part e):
	dist_pts = np.linspace(0, 150000, 10000) # 150000 is the upper bound of incomes
	plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM2_3, sig_GMM2_3, None), '--', linewidth=3, color='pink', label='$\mu_{e}$: 11.337,$\sigma_{e}$: 0.213')
	plt.legend(loc='upper right')


#--------------------------------------------------------------------
#Problem 2a
#--------------------------------------------------------------------
def err_vec_problem2(mydf, b0, b1, b2, b3):
	'''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    '''
	moms_data = np.array([[i] for i in mydf['sick']])
	moms_model = b0 + b1 * mydf['age'] + \
				 b2 * mydf['children'] + b3 * mydf['avgtemp_winter']
	moms_model = np.array([[j] for j in moms_model])
	err_vec = moms_model - moms_data

	return err_vec


def criterion_problem2(params, *args):
	'''
	--------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    '''
	mydf, W = args
	if len(params) == 1: # unpack parameters from the resulted array
		b0, b1, b2, b3 = params[0]
	else:
		b0, b1, b2, b3 = params
	err = err_vec_problem2(mydf, b0, b1, b2, b3)
	crit_val = np.dot(np.dot(err.T, W), err)
	return crit_val


def f2a():
	'''
	Problem 2a
	--------------------------------------------------------------------
	This function estimates the parameters of the model (b0, b1, b2, b3)
	by GMM by solving the minimization problem of the GMM criterion 
	function. The identity matrix is used as the estimator for the optimal
	weighting matrix.
	--------------------------------------------------------------------
	INPUT: none
	RETURN: none
	--------------------------------------------------------------------
	'''
	b0 = 0.0
	b1 = 0.0
	b2 = 0.0
	b3 = 0.0
	params_init = np.array([b0, b1, b2, b3])
	W = np.eye(200)
	gmm_args = (mydf, W)
	results_2a = opt.minimize(criterion_problem2, params_init, args = (gmm_args), method = 'Nelder-Mead')
	b0_GMM, b1_GMM, b2_GMM, b3_GMM = results_2a.x
	# Report the estimated parameter values:
	print('2a. Beta_0 = ', b0_GMM, 'Beta_1 = ', b1_GMM, 'Beta_2 = ', b2_GMM, 'Beta_3 = ', b3_GMM)
	# print(results_2a) # succeeded

	# Calculate and report the value of the GMM criterion function at the estimated parameter values:
	params_GMM = np.array([b0_GMM, b1_GMM, b2_GMM, b3_GMM])
	value = criterion_problem2(params_GMM, *gmm_args)[0][0]
	print('The value of GMM criterion function at the estimated parameter values is: ', value)


def save_figure(filename):
	'''
	--------------------------------------------------------------------
	This function saves figures into the images/ directory. 
	--------------------------------------------------------------------
	INPUT: a string represents the filename of a figure to be saved
	RETURN: none
	--------------------------------------------------------------------
	'''
	fn = os.path.split(filename)[1]
	if not os.path.exists('images'):
		os.mkdir('images')
	fn = os.path.join('images', fn)
	plt.savefig(fn)
	print(fn + ' saved.')


# Plots produced in problem 1a, 1b, 1c, 1d, 1e are saved in the "images" folder.
if '__main__'==__name__:
	# plt.ion()
	f1a(); save_figure('Fig_1a');  plt.close()
	mu_GMM1, sig_GMM1 = f1b(); save_figure('Fig_1b');  plt.close()
	mu_GMM2, sig_GMM2 = f1c(mu_GMM1, sig_GMM1); save_figure('Fig_1c');  plt.close()
	mu_GMM1_3, sig_GMM1_3 = f1d(mu_GMM2, sig_GMM2); save_figure('Fig_1d');  plt.close()
	f1e(mu_GMM1_3, sig_GMM1_3); save_figure('Fig_1e');  plt.close()
	f2a(); 
	pass