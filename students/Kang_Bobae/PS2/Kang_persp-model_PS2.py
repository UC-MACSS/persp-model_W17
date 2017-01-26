'''
------------------------------------------------------------------------
This is Bobae's PS2 for MACS 30100: Perspectives on Computational Modeling
------------------------------------------------------------------------
'''
# import pacakges
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import scipy as sp
import scipy.stats as sts
import scipy.special as spc
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

# read data for exercise 1
incomes = np.loadtxt('incomes.txt')

'''
1.(a) Plot a histogram of percentages of the income.txt data with 30 bins.
Make sure that the bins are weighted such that the height of each bin
represents the percent of the income observations in that bin. In other
words, all the bin heights should sum to 1. Make sure your plot
has correct x-axis and y-axis labels as well as a plot title.
'''
if True:
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, weights = np.ones_like(incomes)/len(incomes))
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Percent of observations in bin')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.1*n.max())

    # save the plot
    output_path_1a = os.path.join(output_dir, 'fig_1a')
    plt.savefig(output_path_1a, bbox_inches = 'tight')

    plt.show()
    plt.close()

'''
1.(b) Plot the lognormal PDF with mu = 9.0 and sigma = 0.3 for
0 <= x <= 150000. What is the value of the log likelihood value for this
parameterization of the distribution and given this data?
'''
def incomes_pdf(xvals, mu, sigma):
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
    pdf_vals    = sts.lognorm.pdf(xvals, s = sigma, scale = np.exp(mu))

    return pdf_vals

def log_lik_incomes(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the lognormally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        incomes_pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = incomes_pdf(xvals, mu, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

inc_pts = np.linspace(0, 150000, 1500000)
inc_mu_1 = 9.0
inc_sig_1 = 0.3

if True:
    plt.plot(inc_pts, incomes_pdf(inc_pts, inc_mu_1, inc_sig_1),
          linewidth=2, color='r', label='1: $\mu$=9.0,$\sigma$=0.3')
    plt.xlim(0, 150000)
    plt.title('Plot of pdf with the initial guess for parameters', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.legend(loc='upper right')

    # save the plot
    output_path_1b = os.path.join(output_dir, 'fig_1b')
    plt.savefig(output_path_1b, bbox_inches = 'tight')

    plt.show()
    plt.close()

if True:
    print('the log likelihood value = ', log_lik_incomes(incomes, inc_mu_1, inc_sig_1))

'''
1.(c) Estimate the parameters of the lognormal distribution by maximum likeli-
hood and plot its PDF against the PDF from part (b) and the histogram
from part (a). Report the ML estimates for mu and sigma, the value of
the likelihood function, and the variance-covariance matrix.
'''
def crit_incomes(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    args   = length 1 tuple, (xvals)
    xvals  = (N,) vector, values of the lognormally distributed random
             variable

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_lik_incomes()

    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals = args
    log_lik_val = log_lik_incomes(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

# initial attempt
inc_mle_args = (incomes)
inc_bounds = ((None, None), (1e-10, None))

inc_mu_0 = 9.0  # mu_1
inc_sig_0 = 0.3  # sig_1
inc_params_1 = np.array([inc_mu_0, inc_sig_0])
inc_results1 = opt.minimize(crit_incomes, inc_params_1, args=(inc_mle_args),
                            bounds = inc_bounds, method = 'L-BFGS-B') # other methods, 'L-BFGS-B', 'SLSQS'
inc_mu_MLE1, inc_sig_MLE1 = inc_results1.x

# second attempt (first ML estimates)
inc_params_2 = np.array([inc_mu_MLE1, inc_sig_MLE1])
inc_results2 = opt.minimize(crit_incomes, inc_params_2, args=(inc_mle_args),
                            bounds = inc_bounds, method = 'L-BFGS-B') # other methods, 'L-BFGS-B', 'SLSQS'
inc_mu_MLE2, inc_sig_MLE2 = inc_results2.x

# third attempt (second ML estimates)
inc_params_3 = np.array([inc_mu_MLE2, inc_sig_MLE2])
inc_results3 = opt.minimize(crit_incomes, inc_params_3, args=(inc_mle_args),
                            bounds = inc_bounds, method = 'L-BFGS-B') # other methods, 'L-BFGS-B', 'SLSQS'
inc_mu_MLE3, inc_sig_MLE3 = inc_results3.x
print(inc_results3)

# plot the pdf against the data
if True:
    # income data histogram
    n, bin_cuts, patches = plt.hist(incomes, bins = 30, normed = True)
    # a pdf graph with the initial guess
    plt.plot(inc_pts, incomes_pdf(inc_pts, inc_mu_0, inc_sig_0),
            linewidth=2, color='r', label='initial guess: $\mu$=9.0,$\sigma$=0.3')
    # plt.plot(inc_pts, incomes_pdf(inc_pts, inc_mu_MLE1, inc_sig_MLE1),
    #         linewidth=2, color='b', label='first ML estimates')
    # plt.plot(inc_pts, incomes_pdf(inc_pts, inc_mu_MLE2, inc_sig_MLE2),
    #         linewidth=2, color='k', label='second ML estimates')
    # plt.plot(inc_pts, incomes_pdf(inc_pts, inc_mu_MLE3, inc_sig_MLE3),
    #         linewidth=2, color='w', label='thrid ML estimates')
    # a pdf graph with the final MLE values
    plt.plot(inc_pts, incomes_pdf(inc_pts, inc_mu_MLE3, inc_sig_MLE3),
            linewidth=2, color='g', label='final MLE: $\mu$=11.33,$\sigma$=0.21')
    plt.xlim(0, 150000)
    plt.ylim(0, 1.1*n.max())
    plt.title('Histogram of annual incomes of MACSS graudates', fontsize = 15)
    plt.xlabel('Annual income (USD)')
    plt.ylabel('Normalized ratio')
    plt.legend(loc='upper right')

    # save the plot
    output_path_1c = os.path.join(output_dir, 'fig_1c')
    plt.savefig(output_path_1c, bbox_inches = 'tight')

    plt.show()
    plt.close()

# present the result
if True:
    print('mu_MLE = ', inc_mu_MLE3, ' sig_MLE = ', inc_sig_MLE3)
    print('the value of the likelihood function = ', -1 * inc_results3.fun)
    print('VCV = ', inc_results3.hess_inv.todense())

'''
1.(d) Perform a likelihood ratio test to determine the probability that
the data in incomes.txt came from the distribution in part (b).
'''
inc_log_lik_h0 = log_lik_incomes(incomes, inc_mu_0, inc_sig_0)
inc_log_lik_mle = log_lik_incomes(incomes, inc_mu_MLE3, inc_sig_MLE3)
inc_LR_val = 2 * (inc_log_lik_mle - inc_log_lik_h0)
inc_pval_h0 = 1.0 - sts.chi2.cdf(inc_LR_val, 2)
if True:
    print('chi squared of H0 with 2 degrees of freedom p-value = ', inc_pval_h0)

'''
1.(e) Using that estimated model from part (c),
What is the probability that you will earn more than $100,000?
What is the probability that you will earn less than $75,000?
'''
inc_100000 = 1 - sp.integrate.quad(lambda x: incomes_pdf(x, inc_mu_MLE3, inc_sig_MLE3), 0, 100000)[0]
inc_75000 = sp.integrate.quad(lambda x: incomes_pdf(x, inc_mu_MLE3, inc_sig_MLE3), 0, 75000)[0]
if True: # predictions
    print('The probability that you will earn more than $100,000:' , inc_100000)
    print('The probability that you will earn less than $75,000:' , inc_75000)

# read data for exercise 2
sick = pd.read_csv('sick.txt')
# print(sick)
'''
2.(a) Estimate the parameters to maximize the likelihood of seeing the data in
sick.txt. Report your estimates, the value of the log likelihood function,
and the estimated variance covariance matrix of the estimates.
'''
def sick_pdf(data, b0, b1, b2, b3, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    data    = (N, M), DataFrame, with 1 response (N,) vector and
                M-1 predictors (N,) vector
    params  = (5,) vector, ([beta0, beta1, beta2, beta3, sigma])

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    sigma   = scalar > 0, standard deviation of the normally distributed
                    error
    err     = (N,), vector, values of the normally distributed error
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    err         = (data[0] - b0 - b1*data[1] - b2*data[2] - b3*data[3])
    pdf_vals    = sts.norm.pdf(err, loc = 0, scale = sigma)
    # ((1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (err - 0)**2 / (2 * sigma**2))))

    return pdf_vals

def log_lik_sick(data, params):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    data    = (N, M), DataFrame, with M-1 predictors
    params  = (5,) vector, ([beta0, beta1, beta2, beta3, sigma])

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        sick_pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    b0, b1, b2, b3, sigma = params[0], params[1], params[2], params[3], params[4]
    pdf_vals = sick_pdf(data, b0, b1, b2, b3, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val

def crit_sick(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    data    = (N, M), DataFrame, with M-1 predictors
    params  = (5,) vector, (beta0, beta1, beta2, beta3, sigma)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_lik_sick()

    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative of log_lik_val

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    data = args
    log_lik_val = log_lik_sick(data, params)
    neg_log_lik_val = -log_lik_val

    return neg_log_lik_val

yvals = sick[sick.columns[0]]
xvals1, xvals2, xvals3 = sick[sick.columns[1]], sick[sick.columns[2]], sick[sick.columns[3]]
sic_data = (yvals, xvals1, xvals2, xvals3)
sic_bounds = ((None, None), (None, None), (None, None), (None, None), (1e-10, None))

# first attemp with method = 'SLSQP'
sic_mle_args = (sic_data)
sic_sig = np.sqrt(0.01)
sic_beta0, sic_beta1, sic_beta2, sic_beta3 = 1.0, 0.0, 0.0, 0.01
sic_params_1 = (sic_beta0, sic_beta1, sic_beta2, sic_beta3, sic_sig)
sic_results1 = opt.minimize(crit_sick, sic_params_1, args = (sic_mle_args),
                            bounds = sic_bounds, method = 'SLSQP') # other methods, 'L-BFGS-B', 'SLSQP'
sic_beta0_MLE1, sic_beta1_MLE1, sic_beta2_MLE1, sic_beta3_MLE1, sic_sig_MLE1 = sic_results1.x

# second attemp with method = 'L-BFGS-B' for hess_inv matrix
sic_params_2 = (sic_beta0_MLE1, sic_beta1_MLE1, sic_beta2_MLE1, sic_beta3_MLE1, sic_sig_MLE1)
sic_results2 = opt.minimize(crit_sick, sic_params_2, args = (sic_mle_args),
                            bounds = sic_bounds, method = 'L-BFGS-B') # other methods, 'L-BFGS-B', 'SLSQP'
sic_beta0_MLE2, sic_beta1_MLE2, sic_beta2_MLE2, sic_beta3_MLE2, sic_sig_MLE2 = sic_results2.x
print(sic_results2)

if True:
    print('beta0_MLE = ', sic_beta0_MLE2,
        'beta1_MLE = ', sic_beta1_MLE2,
        'beta2_MLE = ', sic_beta2_MLE2,
        'beta3_MLE = ', sic_beta3_MLE2,
        'sig**2_MLE = ', sic_sig_MLE2**2)
    print('the value of the likelihood function = ', -1 * sic_results2.fun)
    print('VCV = ', sic_results2.hess_inv.todense())

'''
2.(b) Use a likelihood ratio test to determine the probability that beta0 = 1.0,
sigma**2 = 0.01 and beta1, beta2, beta3 = 0. That is, what is the likelihood
that age, number of children, and average winter temperature have no effect on the
number of sick days?
'''
sic_params_mle = (sic_beta0_MLE2, sic_beta1_MLE2, sic_beta2_MLE2, sic_beta3_MLE2, sic_sig_MLE2)

sic_log_lik_h0 = log_lik_sick(sic_data, sic_params_1)
sic_log_lik_mle = log_lik_sick(sic_data, sic_params_mle)
sic_LR_val = 2 * (sic_log_lik_mle - sic_log_lik_h0)
sic_pval_h0 = 1.0 - sts.chi2.cdf(sic_LR_val, 2)
if True:
    print('chi squared of H0 with 5 degrees of freedom p-value = ', sic_pval_h0)
