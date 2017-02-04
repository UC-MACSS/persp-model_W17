'''
This is Bobae's Python script for Perspectives on Computational Modeling PS3.
'''
# Import packages and load the data
import numpy as np
import pandas as pd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn
import os

# set the directory for saving images
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

'''
--------------------------------------------------------------------
1(a) Plot a histogram of percentages of the income.txt data with 30 bins.
Make sure that the bins are weighted using the normed=True option. Make sure
your plot has correct x-axis and y-axis labels as well as a plot title.
--------------------------------------------------------------------
'''
# read data in
incomes = np.loadtxt('incomes.txt')

if True:
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())

    # save the plot
    output_path_1a = os.path.join(output_dir, 'fig_1a')
    plt.savefig(output_path_1a, bbox_inches = 'tight')

    plt.show()
    plt.close()
'''
--------------------------------------------------------------------
1(b) Estimate the parameters of the lognormal distribution by generalized
method of moments. Use the average income and standard deviation of income as
your two moments. Use the identity matrix as your weighting matrix.

Plot your estimated lognormal PDF against the histogram from part (a).
Report the value of your GMM criterion function at the estimated param-
eter values. Report and compare your two data moments against your two
model moments at the estimated parameter values.
--------------------------------------------------------------------
'''
# define necessary functions
def inc_lognorm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma. This function does the same thing as
    scipy.stats.lognorm.pdf(x, s = sigma, scale= np.exp(mu).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the lognormally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
               corresponding to xvals data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    # mu = np.log(mean) - ((.5) * np.log(std/mean)**2 + 1)
    # sigma = np.sqrt(np.log((std/mean)**2 + 1))
    pdf_vals = sts.lognorm.pdf(xvals, s = sigma, scale = np.exp(mu))

    return pdf_vals

def inc_data_moments(xvals):
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
    std_data  = scalar > 0, standard deviation of test scores data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mean_data, std_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    std_data = xvals.std()

    return mean_data, std_data

def inc_model_moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), standard_deviation(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        inc_lognorm_pdf()
        xfx()
        x2fx()

    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    std_model  = scalar > 0, standard deviation of test scores from model
    s_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the stadnard deviation of the distribution

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mean_model, std_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * inc_lognorm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * inc_lognorm_pdf(x, mu, sigma)
    (std_model, s_m_err) = np.sqrt(intgr.quad(x2fx, 0, 150000))

    return mean_model, std_model

def inc_err_vec(xvals, mu, sigma, simple):
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
        inc_data_moments()
        inc_model_moments()

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
    mean_data, std_data = inc_data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = inc_model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [std_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec

def inc_criterion(params, *args):
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
    W      = (R, R) matrix, estimate of optimal weighting matrix

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        inc_err_vec()

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
    err = inc_err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

# setup for optimizaiton
inc_pts = np.linspace(0, 150000, 150000)
inc_mu_0 = 11
inc_sig_0 = 0.2
params_0 = np.array([inc_mu_0, inc_sig_0])
W_hat = np.eye(2)
inc_gmm_args0 = (incomes, W_hat)
inc_bounds = ((None, None), (1e-10, None))

# optimization
results_1b = opt.minimize(inc_criterion, params_0, args=(inc_gmm_args0),
                       method='L-BFGS-B', bounds=inc_bounds)
# results
mu_GMM_1b, sig_GMM_1b = results_1b.x
inc_mean_data, inc_std_data = inc_data_moments(incomes)
mean_model_1b, std_model_1b = inc_model_moments(mu_GMM_1b, sig_GMM_1b)

if True:
    # plot the results
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM_1b, sig_GMM_1b),
          linewidth=2, color='r',
          label='$\mu$ GMM_1b = {}, $\sigma$ GMM_1b = {}'.format(mu_GMM_1b, sig_GMM_1b))
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())
    plt.legend(loc='upper right')

    # save the plot
    output_path_1b = os.path.join(output_dir, 'fig_1b')
    plt.savefig(output_path_1b, bbox_inches = 'tight')

    plt.show()
    plt.close()

if True:
    # report the results
    print('Mean of data =', inc_mean_data, ', Standard Deviation of data =', inc_std_data)
    print('Mean of model =', mean_model_1b, ', Standard Deviation of model =', std_model_1b)
    print(mu_GMM_1b, sig_GMM_1b)
    print(results_1b.fun)


'''
--------------------------------------------------------------------
1(c) Perform the two-step GMM estimator by using your estimates from part
(b) with two moments to generate an estimator for the variance covariance
matrix, which you then use to get the two-step estimator for the optimal
weighting matrix.

Report your estimates as well as the criterion function value at these
estimates. Plot your estimated lognormal PDF against the histogram from
part (a) and the estimated PDF from part (b). Report and compare your
two data moments against your two model moments at the estimated parameter
values.
--------------------------------------------------------------------
'''
# setup for optimizaiton
inc_err = inc_err_vec(incomes, mu_GMM_1b, sig_GMM_1b, False)
VCV_2 = np.dot(inc_err, inc_err.T) / incomes.shape[0]
W_hat_2 = lin.pinv(VCV_2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
inc_gmm_args_2step = (incomes, W_hat_2)

# optimization, with original guess
results_1c = opt.minimize(inc_criterion, params_0, args=(inc_gmm_args_2step),
                       method='L-BFGS-B', bounds=inc_bounds)
# # optimization, with previous GMM results
# results_1c2 = opt.minimize(inc_criterion, results_1b.x, args=(inc_gmm_args_2step),
#                        method='L-BFGS-B', bounds=inc_bounds)
# results
mu_GMM_1c, sig_GMM_1c = results_1c.x
mean_model_1c, std_model_1c = inc_model_moments(mu_GMM_1c, sig_GMM_1c)
# mu_GMM_1c2, sig_GMM_1c2 = results_1c2.x
# mean_model_1c2, std_model_1c2 = inc_model_moments(mu_GMM_1c2, sig_GMM_1c2)

if True:
    # plot the results
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM_1b, sig_GMM_1b),
          linewidth=2, color='r',
          label='$\mu$ GMM_1b = {}, $\sigma$ GMM_1b = {}'.format(mu_GMM_1b, sig_GMM_1b))
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM_1c, sig_GMM_1c),
          linewidth=2, color='g',
          label='$\mu$ GMM_1c = {}, $\sigma$ GMM_1c = {}'.format(mu_GMM_1c, sig_GMM_1c))
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())
    plt.legend(loc='upper right')

    # save the plot
    output_path_1c = os.path.join(output_dir, 'fig_1c')
    plt.savefig(output_path_1c, bbox_inches = 'tight')

    plt.show()
    plt.close()

if False:
    # plot the results
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM_1b, sig_GMM_1b),
          linewidth=2, color='r',
          label='$\mu$ GMM_1b = {}, $\sigma$ GMM_1b = {}'.format(mu_GMM_1b, sig_GMM_1b))
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM_1c2, sig_GMM_1c2),
          linewidth=2, color='b',
          label='$\mu$ GMM_1c = {}, $\sigma$ GMM_1c = {}'.format(mu_GMM_1c2, sig_GMM_1c2))
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())
    plt.legend(loc='upper right')

    # save the plot
    output_path_1c2 = os.path.join(output_dir, 'fig_1c2')
    plt.savefig(output_path_1c2, bbox_inches = 'tight')

    plt.show()
    plt.close()

if True:
    # report the results
    print('Mean of points =', inc_mean_data, ', Standard Deviation of points =', inc_std_data)
    print('Mean of model1 =', mean_model_1c, ', Standard Deviation of model1 =', std_model_1c)
    print('Estimatied parameters1 =', mu_GMM_1c, sig_GMM_1c)
    print('Criterion function value1 =', results_1c.fun)

    # print('Mean of model2 =', mean_model_1c2, ', Standard Deviation of model2 =', std_model_1c2)
    # print('Estimatied parameters2 =', mu_GMM_1c2, sig_GMM_1c2)
    # print('Criterion function value2 =', results_1c2.fun)

'''
--------------------------------------------------------------------
1(d) Now estimate the lognormal PDF to t the data by GMM using different
moments. Use percent of individuals who earn less than $75,000, percent
of individuals who earn between $75,000 and $100,000, and percent of
individuals who earn more than $100,000 as your three moments. Use the
identity matrix as your estimator for the optimal weighting matrix.

Plot your estimated lognormal PDF against the histogram from part (a).
Report the value of your GMM criterion function at the estimated parameter
values. Report and compare your three data moments against your three
model moments at the estimated parameter values.
--------------------------------------------------------------------
'''
# define necessary functions
def inc_data_moments3(xvals):
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
				 75000 <= x < 100000
	bpct_3_dat = scalar in [0, 1], percent of observations
				 100000 <= x

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: bpct_1, bpct_2, bpct_3
	--------------------------------------------------------------------
	'''
    bpct_1_dat = (xvals[xvals < 75000].shape[0] / xvals.shape[0])
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] / xvals.shape[0])
    bpct_3_dat = (xvals[xvals >=100000].shape[0] / xvals.shape[0])

    return bpct_1_dat, bpct_2_dat, bpct_3_dat

def inc_model_moments3(mu, sigma):
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
		inc_lognorm_pdf()
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
    xfx = lambda x: inc_lognorm_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 150000)
    # bpct_3_mod = 1 - bpct_1_mod - bpct_2_mod

    return bpct_1_mod, bpct_2_mod, bpct_3_mod

def inc_err_vec3(xvals, mu, sigma, simple):
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
		inc_data_moments3()
		inc_model_moments3()

	OBJECTS CREATED WITHIN FUNCTION:
	moms_data  = (3, 1) matrix, column vector of three data moments
	moms_model = (3, 1) matrix, column vector of three model moments
	err_vec    = (3, 1) matrix, column vector of three moment error
				 functions

	FILES CREATED BY THIS FUNCTION: None

	RETURNS: err_vec
	--------------------------------------------------------------------
	'''
    bpct_1_dat, bpct_2_dat, bpct_3_dat = inc_data_moments3(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod = inc_model_moments3(mu, sigma)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)

    return err_vec

def inc_criterion3(params, *args):
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
	W  = (R, R) matrix, estimate of optimal weighting matrix

	OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
		inc_err_vec3()

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
	err = inc_err_vec3(xvals, mu, sigma, simple=False)
	crit_val = np.dot(np.dot(err.T, W), err)

	return crit_val

# setup for optimizaiton
W_hat3 = np.eye(3)
inc_gmm_args3 = (incomes, W_hat3)

# optimization
results_1d = opt.minimize(inc_criterion3, params_0, args=(inc_gmm_args3),
                        method='L-BFGS-B', bounds=inc_bounds)
# results
mu_GMM3_1d, sig_GMM3_1d = results_1d.x
bpct_1_dat3, bpct_2_dat3, bpct_3_dat3 = inc_data_moments3(incomes)
bpct_1_mod3_1d, bpct_2_mod3_1d, bpct_3_mod3_1d = inc_model_moments3(mu_GMM3_1d, sig_GMM3_1d)

if True:
    # plot the results
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM3_1d, sig_GMM3_1d),
          linewidth=2, color='r',
          label='$\mu$ GMM3_1d = {}, $\sigma$ GMM3_1d = {}'.format(mu_GMM3_1d, sig_GMM3_1d))
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())
    plt.legend(loc='upper right')

    # save the plot
    output_path_1d = os.path.join(output_dir, 'fig_1d')
    plt.savefig(output_path_1d, bbox_inches = 'tight')

    plt.show()
    plt.close()

if True:
    # report the results
    print('Moments of data = ', bpct_1_dat3, ',', bpct_2_dat3, ',', bpct_3_dat3)
    print('Moments of model = ', bpct_1_mod3_1d, ',', bpct_2_mod3_1d, ',', bpct_3_mod3_1d)
    print(mu_GMM3_1d, sig_GMM3_1d)
    print(results_1d.fun)
'''
--------------------------------------------------------------------
1(e) Perform the two-step GMM estimator by using your estimates from
part (d) with three moments to generate an estimator for the variance
covariance matrix, which you then use to get the two-step estimator for
the optimal weighting matrix.

Report your estimates as well as the criterion function value at these
estimates. Plot your estimated lognormal PDF against the histogram from
part (a) and the estimated PDF from part (d). Report and compare your
three data moments against your three model moments at the estimated
parameter values.
--------------------------------------------------------------------
'''
# setup for optimizaiton
inc_err3 = inc_err_vec3(incomes, mu_GMM3_1d, sig_GMM3_1d, False)
VCV3_2 = np.dot(inc_err3, inc_err3.T) / incomes.shape[0]
W_hat3_2 = lin.pinv(VCV3_2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
inc_gmm_args3_2step = (incomes, W_hat3_2)

# optimization, with original guess
results_1e = opt.minimize(inc_criterion3, params_0, args=(inc_gmm_args3_2step),
                        method='L-BFGS-B', bounds=inc_bounds)
# # optimization, with previous GMM results
# results_1e2 = opt.minimize(inc_criterion3, results_1d.x, args=(inc_gmm_args3_2step),
#                         method='L-BFGS-B', bounds=inc_bounds)

# results
mu_GMM3_1e, sig_GMM3_1e = results_1e.x
bpct_1_mod3_1e, bpct_2_mod3_1e, bpct_3_mod3_1e = inc_model_moments3(mu_GMM3_1e, sig_GMM3_1e)
# mu_GMM3_1e2, sig_GMM3_1e2 = results_1e2.x
# bpct_1_mod3_1e2, bpct_2_mod3_1e2, bpct_3_mod3_1e2 = inc_model_moments3(mu_GMM3_1e2, sig_GMM3_1e2)

if True:
    # plot the results, with original guess
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM3_1d, sig_GMM3_1d),
          linewidth=2, color='r',
          label='$\mu$ GMM3_1d = {}, $\sigma$ GMM3_1d = {}'.format(mu_GMM3_1d, sig_GMM3_1d))
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM3_1e, sig_GMM3_1e),
          linewidth=2, color='g',
          label='$\mu$ GMM3_1e = {}, $\sigma$ GMM3_1e = {}'.format(mu_GMM3_1e, sig_GMM3_1e))
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())
    plt.legend(loc='upper right')

    # save the plot
    output_path_1e = os.path.join(output_dir, 'fig_1e')
    plt.savefig(output_path_1e, bbox_inches = 'tight')

    plt.show()
    plt.close()

if False:
    # plot the results, with previous GMM results
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM3_1d, sig_GMM3_1d),
          linewidth=2, color='r',
          label='$\mu$ GMM3_1d = {}, $\sigma$ GMM3_1d = {}'.format(mu_GMM3_1d, sig_GMM3_1d))
    plt.plot(inc_pts, inc_lognorm_pdf(inc_pts, mu_GMM3_1e2, sig_GMM3_1e2),
          linewidth=2, color='b',
          label='$\mu$ GMM3_1e = {}, $\sigma$ GMM3_1e = {}'.format(mu_GMM3_1e2, sig_GMM3_1e2))
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())
    plt.legend(loc='upper right')

    # save the plot
    output_path_1e2 = os.path.join(output_dir, 'fig_1e2')
    plt.savefig(output_path_1e2, bbox_inches = 'tight')

    plt.show()
    plt.close()

if True:
    # report the results
    print('Moments of data = ', bpct_1_dat3, ',', bpct_2_dat3, ',', bpct_3_dat3)
    print('Moments of model1 = ', bpct_1_mod3_1e, ',', bpct_2_mod3_1e, ',', bpct_3_mod3_1e)
    print('Estimated parameters1 = ', mu_GMM3_1e, sig_GMM3_1e)
    print('Criterion function value1 =', results_1e.fun)

    # print('Moments of model2 = ', bpct_1_mod3_1e2, ',', bpct_2_mod3_1e2, ',', bpct_3_mod3_1e2)
    # print('Estimated parameters2 = ', mu_GMM3_1e2, sig_GMM3_1e2)
    # print('Criterion function value2 =', results_1e2.fun)

'''
--------------------------------------------------------------------
1(f) Which of the four estimations from parts (b), (c), (d), and (e)
fits the data best? Justify your answer.
--------------------------------------------------------------------
'''
# refer to my PDf file for the answer to this question

'''
--------------------------------------------------------------------
2(a) Estimate the parameters of the model (beta0, beta1, beta2, beta3)
by GMM by solving the minimization problem of the GMM criterion function.
Use the identity matrix as the estimator for the optimal weighting matrix.
Treat each of the 200 values of the variable sicki as your data moments
m(xi) (200 data moments). Treat the predicted or expected sick values from
your model as your model moments (200 model moments). Let the error function
of the moments be the simple dierence (not percent dierence) of the data
moments from the model moments. Use these error functions in your criterion
function to estimate the model parameters (beta0, beta1, beta2, beta3) by GMM.

Report your estimates and report the value of your GMM criterion function.
--------------------------------------------------------------------
'''
# read data in
sick = pd.read_csv('sick.txt', encoding = 'utf-8-sig')
# print(sick)

# define necessary functions
def sic_err_vec(data, *params):
    b0, b1, b2, b3 = params
    moms_data = np.array(data.iloc[:,0])
    moms_model = b0 + b1 * data.iloc[:,1] + b2 * data.iloc[:,2] + b3 * data.iloc[:,3]
    err_vec = moms_model - moms_data

    return err_vec

def sic_criterion(params, *args):
    data, W = args
    b0, b1, b2, b3 = params
    err = sic_err_vec(data, b0, b1, b2, b3)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

# setup for optimizaiton
sic_b0, sic_b1, sic_b2, sic_b3 = 1.0, 0.0, 0.0, 0.01 # initial guess for betas
sic_params = (sic_b0, sic_b1, sic_b2, sic_b3)
sic_W_hat = np.eye(len(sick)) # weight matrix
sic_gmm_args = (sick, sic_W_hat)
sic_bounds = ((None, None), (None, None), (None, None), (None, None))

# optimization
sic_results = opt.minimize(sic_criterion, sic_params, args = (sic_gmm_args),
                            method='L-BFGS-B', bounds=sic_bounds)
# results
sic_b0_GMM, sic_b1_GMM, sic_b2_GMM, sic_b3_GMM = sic_results.x
sic_crit_val = sic_results.fun

if True:
    # report the results
    print(sic_b0_GMM, sic_b1_GMM, sic_b2_GMM, sic_b3_GMM)
    print(sic_results.fun)
