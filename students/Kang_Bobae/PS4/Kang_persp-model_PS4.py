'''
This is Bobae's Python script for Perspectives on Computational Modeling PS4.
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

# read data in
incomes = np.loadtxt('incomes.txt')

# define necessary functions
def lognorm_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a lognormal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma and cutoffs (cut_lb, cut_ub). These draws
    correspond to an (N x S) matrix of randomly generated draws from a
    uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the lognormal distribution
    sigma     = scalar > 0, standard deviation of the lognormal distribution

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.lognorm.ppf()

    OBJECTS CREATED WITHIN FUNCTION:
    lognorm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from lognormal PDF

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: lognorm_draws
    --------------------------------------------------------------------
    '''
    # lognorm_draws = sts.lognorm.ppf(unif_vals, scale=np.exp(mu), s=sigma)
    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    lognorm_draws = np.exp(norm_draws)
    return lognorm_draws

def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar,
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

def err_vec(data_vals, sim_vals, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    sim_vals  = (N, S) matrix, S simulations of test scores data
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
    mu        = scalar, mean of the lognormally distributed random variable
    sigma     = scalar > 0, standard deviation of the lognormally
                distributed random variable
    args      = length 3 tuple,
                (xvals, unif_vals, W_hat)
    xvals     = (N,) vector, values of the lognormally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        LN_pdf()

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
    sim_vals = lognorm_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val

'''
--------------------------------------------------------------------
1(a) Plot a histogram of percentages of the income.txt data with 30 bins.
Make sure that the bins are weighted using the normed=True option. Make sure
your plot has correct x-axis and y-axis labels as well as a plot title.
--------------------------------------------------------------------
'''
if False:
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.2*n.max())

    # save the plot
    # output_path_1a = os.path.join(output_dir, 'fig_1a')
    # plt.savefig(output_path_1a, bbox_inches = 'tight')

    plt.show()
    plt.close()

'''
--------------------------------------------------------------------
1(b) Write your own function for the lognormal PDF above called LN pdf().
--------------------------------------------------------------------
'''
def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    This function takes lognormally distributed data, mu, sigma and
    returns the pdf values.
    --------------------------------------------------------------------
    INPUTS:
    xvals     = (N, S) matrix, (N,) vector, or scalar,
                test scores data, either real world or simulated. Real world
                data will come in the form (N,). Simulated data comes in the
                form (N,) or (N, S).
    mu        = scalar, mean of the lognormal distribution
    sigma     = scalar > 0, standard deviation of the lognormal distribution

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        numpy.log()
        numpy.exp()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  lognormal pdf values

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    # m = mu
    # v = sigma**2
    # logm = np.log(m/(np.sqrt(1+(v/m**2))))
    # logs = np.sqrt(np.log(1 + (v/m**2)))
    # pdf_vals = (np.exp(-1*(np.log(xvals)-logm)**2/(2*logs**2)) /(xvals*logs*np.sqrt(2*np.pi)))
    pdf_vals = (np.exp(-1*(np.log(xvals)-mu)**2/(2*sigma**2)) / (xvals*sigma*np.sqrt(2*np.pi)))

    return pdf_vals

xvals = np.array([[200.0, 270.0], [180.0, 195.5]])
mu_test, sig_test = 5.0, 1.0
test_pdf_vals = LN_pdf(xvals, mu_test, sig_test)

if True:
    print('1b result: ', test_pdf_vals)
'''
--------------------------------------------------------------------
1(c) Estimate the parameters of the lognormal distribution by simulated method
of moments (SMM). Plot your estimated lognormal PDF against the histogram
from part (a). Report the value of your SMM criterion function at the estimated
parameter values. Report and compare your two data moments against your two
model moments at the estimated parameter values.
--------------------------------------------------------------------
'''
# setup for optimization
S, N = 300, 200
W_hat0 = np.eye(2)
np.random.seed(seed=1234)
unif_vals = sts.uniform.rvs(0, 1, size=(N, S))
mean_dat, var_dat = incomes.mean(), incomes.var()
bounds = ((0, None), (1e-10, None))

# mu_dat = np.log(mean_dat/(np.sqrt(1+(var_dat/mean_dat**2))))
# sig_dat = np.sqrt(np.log(1 + (var_dat/mean_dat**2)))
# sim_vals =lognorm_draws(unif_vals, mu_dat, sig_dat)
sim_vals =lognorm_draws(unif_vals, 11., .2)
mean_sim, var_sim = data_moments(sim_vals)
mean_mod, var_mod = mean_sim.mean(), var_sim.mean()

# smm_params_1c = np.array([mu_dat, sig_dat]) # using mu and sig from data as initial values
smm_params_1c = np.array([11., .2]) # using arbitrary numbers as initial values
smm_args_1c = (incomes, unif_vals, W_hat0)

# optimize
results_1c = opt.minimize(criterion, smm_params_1c, args=(smm_args_1c),
                          method='L-BFGS-B',
                          bounds=bounds) #options={'eps': 1.0}
# report the results
mu_SMM_1c, sig_SMM_1c = results_1c.x
sim_vals_1c =lognorm_draws(unif_vals, mu_SMM_1c, sig_SMM_1c)
mean_sim_1c, var_sim_1c = data_moments(sim_vals_1c)
mean_mod_1c, var_mod_1c = mean_sim_1c.mean(), var_sim_1c.mean()
mean_diff_1c, var_diff_1c = (mean_mod_1c - mean_dat)/mean_dat, (var_mod_1c - var_dat)/var_dat
if True:
    print('mu_SMM_1c=', mu_SMM_1c, ' sig_SMM_1c=', sig_SMM_1c)
    print('Criterion function value 1c: ', results_1c.fun)
    print('data mean:', mean_dat, 'data variance: ', var_dat)
    print('sim mean:', mean_mod_1c, 'sim variance', var_mod_1c)
    print('mean % diff: ', mean_diff_1c*100, 'variance % diff: ', var_diff_1c*100)

# plot the results
dist_pts = np.linspace(1e-10, 150000, 150000)
if True:
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM_1c, sig_SMM_1c),
            linewidth=2, color='g',
            label='1: $\mu$ = {},$\sigma$ = {}'.format(mu_SMM_1c, sig_SMM_1c))
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
'''
--------------------------------------------------------------------
1(d) Perform the two-step SMM estimator by using your estimates from part
(c) with two moments to generate an estimator for the variance covariance
matrix. Report your estimates as well as the criterion function value at
these estimates. Plot your estimated lognormal PDF against the histogram
from part (a) and the estimated PDF from part (c). Report and compare your
two data moments against your two model moments at the estimated parameter
values.
--------------------------------------------------------------------
'''
# setup for optimization
err = err_vec(incomes, sim_vals_1c, False)
VCV2 = np.dot(err, err.T) / incomes.shape[0]
print(VCV2)
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
print(W_hat2)
# smm_params_1d = np.array([mu_dat, sig_dat]) # using mu and sig from data as initial values
# smm_params_1d = np.array([11., .2]) # using arbitrary numbers as initial values
smm_params_1d = np.array([mu_SMM_1c, sig_SMM_1c]) # using the optimized values from 1(c)
smm_args_1d = (incomes, unif_vals, W_hat2)

# optimize
results_1d = opt.minimize(criterion, smm_params_1d, args=(smm_args_1d),
                          method='TNC',
                          bounds=bounds) #options={'eps': 1.0
# report the results
mu_SMM_1d, sig_SMM_1d = results_1d.x
sim_vals_1d =lognorm_draws(unif_vals, mu_SMM_1d, sig_SMM_1d)
mean_sim_1d, var_sim_1d = data_moments(sim_vals_1d)
mean_mod_1d, var_mod_1d = mean_sim_1d.mean(), var_sim_1d.mean()
mean_diff_1d, var_diff_1d = (mean_mod_1d - mean_dat)/mean_dat, (var_mod_1d - var_dat)/var_dat
if True:
    print('mu_SMM_1d=', mu_SMM_1d, ' sig_SMM_1d=', sig_SMM_1d)
    print('Criterion function value 1d: ', results_1d.fun)
    print('data mean:', mean_dat, 'data variance: ', var_dat)
    print('model mean:', mean_mod_1d, 'model variance', var_mod_1d)
    print('mean % diff: ', mean_diff_1d*100, 'variance % diff: ', var_diff_1d*100)

# plot the results
dist_pts = np.linspace(1e-10, 150000, 150000)
if True:
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM_1d, sig_SMM_1d),
            linewidth=2, color='r', #linestyle='--',
            label='2: $\mu$ = {},$\sigma$ = {}'.format(mu_SMM_1d, sig_SMM_1d))
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

# checking the full results
if True:
    print('results 1c: \n', results_1c)
    print('results 1d: \n', results_1d)
