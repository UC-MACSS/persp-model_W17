# Python script for Problem Set 2
# MACS 301
# Julian McClellan

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt
import pandas as pd

COVERAGE = 25 #%
MU_INIT = 9
SIG_INIT = .3

# methods: TNC, SLSQP

# Utility fuctions
def make_output_dir():
    '''
    Create directory if images directory does not already exist
    '''
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    return(output_dir)

# Unused
def man_lognorm_pdf(x, mu, sigma):
    '''
    '''
    rv = 1 / (sigma*x*np.sqrt(2*np.pi)) * np.exp(-1/2*((np.log(x) - mu) / sigma)**2)
    return(rv)


# Exercise 1a
def plot_income_hist(incomes, part_c = False):
    '''
    This function takes incomes array and plots an appropriate percentage 
    histogram for Exercise 1a.
    '''
    # Plot percentages histogram with 30 bins
    num_incomes = len(incomes)
    num_bins = 30

    if part_c:
        hist_wgts = None
    else:
        hist_wgts = (100 / num_incomes) * np.ones(num_incomes)

    fig, ax = plt.subplots()
    plt.hist(incomes, num_bins, weights = hist_wgts, normed = part_c)
    plt.ylabel(r'Percentage of incomes (%)')
    plt.xlabel(r'Annual Incomes (\$s)')
    
    # plt.show()
    
    # If need to add the PDF from 1c, do not save, return some x_vals and ax
    if part_c: 
        max_income = incomes.max()
        return(np.linspace(0.001, max_income, COVERAGE * max_income), fig, ax)
    else: # Else, if this is being used for just 1a, then save the figure.
        plt.title('Annual Incomes of MACSS Students')
        output_path = os.path.join(make_output_dir(), 'Fig_1a')
        plt.savefig(output_path)
        plt.close()

    return(None)


# Exercise 1b | Plotting lognormal pdf
def plot_lognorm_pdf(bounds, coverage, mu, sigma, fig_name = 'Fig_1b'):
    '''
    '''
    assert (0 < coverage < 100), 'Coverage must be between 0% and 100%'

    # X-axis values creation
    start, stop = bounds 
    x_vals = np.linspace(start, stop, coverage * (stop - start))  

    # Y-axis values (lognorm pdf) creation
    lognorm_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
    lognorm_pdf = lognorm_dist.pdf(x_vals)

    # Plotting and saving
    fig, ax = plt.subplots()
    plt.xlim(0, 150000)
    plt.plot(x_vals, lognorm_pdf, label = '$f(x|\mu={:.3f}, \sigma={:.3f}$'
            .format(mu, sigma))
    plt.xlabel('x')
    plt.ylabel('$f(x|\mu=9,\sigma=.3)$')
    plt.title('Log normal pdf. $\mu={:.3f}$, $\sigma={:.3f}$'
              .format(mu, sigma))
    plt.tight_layout()
    output_path = os.path.join(make_output_dir(), fig_name)
    plt.savefig(output_path)
    plt.close()
    return(None)


# Exercise 1b | Calculating log likelihood value
def calc_lognorm_likelihood(values, mu, sigma):
    '''
    Given an array of values, a mu, and a sigma, this function calculates the
    log likelihood from the appropriate lognormal distribution.
    '''
    lognorm_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
    pdf = lognorm_dist.pdf(values)
    ln_pdf_vals = np.log(pdf)
    log_lik_val = ln_pdf_vals.sum()

    return(log_lik_val) 


# Exercise 1c | Obtain maximum likelihood estimators of the lognormal dist.
def crit(params, *args):
    '''
    This function is a criterion function for optimization

    Inputs:
        params: (tuple, length 2)
            mu: mean of lognormal distribution
            sigma: standard deviation of lognormal distribution
        args: (tuple, length 1)
            xvals: list of xvalues to calculate log likelihood for

    Ouputs:
        neg_log_lik_val: the negative log likelihood value so we can optimize a
                         minimization problem
    '''
    mu, sigma = params
    xvals = args
    log_lik_val = calc_lognorm_likelihood(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    
    return(neg_log_lik_val)


def estimate_params1(mu_init, sig_init, values):
    '''
    This function takes intial values of mu and sigma, as well as an array of 
    real data with which to conduct the MLE, and returns the estimated parameters
    mu_MLE, and sig_MLE.
    '''
    params_init = np.array([mu_init, sig_init])
    mle_args = (values)
    bnds = ((None, None), (0.00001, None)) # Want standard deviation to be positive

    # While this minimize with the SLSQP method and the above bounds works, it
    # does not return an inverse Hessian (variance-covariance) matrix
    results = opt.minimize(crit, params_init, method='L-BFGS-B', args=(mle_args), 
                           bounds = bnds)

    # For the lognormal function the below does not work.
    # results = opt.minimize(crit, params_init, args=(mle_args))
    return(results)


# Exercise 1c | Plot the PDF against the pdf from part (b) and the histogram 
# from part (a). Plot the estimated pdf for 0 <= x <= 150,000. Report the ML 
# estimates for mu and sigma, the value of the likelihood function, and the 
# variance-covariance matrix.
def plot_alot(mu_init, sig_init, values):
    '''
    This function takes an initial value of mu and sigma, as well as the array
    of student incomes and estimates the parameters of the lognormal distribution
    by maximum likelihood and makes several required plots.
    '''
    results1 = estimate_params1(mu_init, sig_init, incomes)
    mu_mle, sig_mle = results1.x
    mle_label = '$f(x|\mu={:.3f}, \sigma={:.3f})$'.format(mu_mle, sig_mle)
    init_label = '$f(x|\mu={}, \sigma={})$'.format(MU_INIT, SIG_INIT)

    mle_lognorm = sts.lognorm(scale = np.exp(mu_mle), s = sig_mle)
    init_lognorm = sts.lognorm(scale = np.exp(MU_INIT), s = SIG_INIT)

    # Plot the PDF against the histogram from part (a)  and the pdf from part (b).
    # Get new x vals (max is incomes.max())
    x_vals, fig, ax = plot_income_hist(incomes, part_c = True) 
    trunc_mle_ln_pdf = mle_lognorm.pdf(x_vals)
    trunc_init_ln_pdf = init_lognorm.pdf(x_vals)
    plt.plot(x_vals, trunc_mle_ln_pdf, label = mle_label, color = 'red')
    plt.plot(x_vals, trunc_init_ln_pdf, label = init_label, color = 'cyan')

    plt.ylabel('$f(x|\mu,\sigma)$')
    plt.title('Incomes of MACSS Students Histogram (normed) with Lognormal PDFs', 
              y = 1.02) # Move title up slightly
    plt.legend(loc = 1) # Legend in upper right
    plt.tight_layout() # I want to see the labels

    # Save figure
    output_path = os.path.join(make_output_dir(), 'Fig_1c')
    plt.savefig(output_path)
    plt.close()

    # Report ML estimates for mu and sigma
    print('mu_MLE = {:.3f} | sig_MLE = {:.3f}\n'.format(mu_mle, sig_mle))

    # Report value of the likelihood function
    log_lik_val = calc_lognorm_likelihood(incomes, mu = mu_mle, sigma = sig_mle)
    print('The log likelihood value of the data given mu = {:.3f} and sigma = {:.3f} is {:.3f}\n'
          .format(mu_mle, sig_mle, log_lik_val))

    # Report the variance covariance matrix (Empirical Hessian Estimator)
    # print('The variance-covariance matrix is {}'.format(results.hess_inv))
    # !!! The only method that terminates successfully is 'SLSQP' and its 
    # results object does not have the inverse Hessian Matrix...

    # Return the parameters and the MLE lognorm distribution, they will be 
    # useful for 1d and 1e, respectively
    return(mu_mle, sig_mle, mle_lognorm, results1) 

# Exercise 1d | LRT test to determine if the data in incomes.txt came from the
# distribution in part (b)
def lrt_lognorm(values, *mle_test_params):
    '''
    '''
    mu_mle, sig_mle, mu_tst, sig_tst = mle_test_params

    log_lhood_mle = calc_lognorm_likelihood(values, mu_mle, sig_mle)
    log_lhood_tst = calc_lognorm_likelihood(values, mu_tst, sig_tst)

    lr_val = 2 * (log_lhood_mle - log_lhood_tst)

    p_val = 1.0 - sts.chi2.cdf(lr_val, 2)

    return(p_val)

# Exercise 1e | See below in the if __name__ == '__main__': block

# Exercise 2a | MLE parameter estimation
def calc_norm_likelihood(df, *params):
    '''
    Given a dataframe of values, and parameters (beta_0, beta_1, beta_2, sigma)
    this function calculates the log likelihood from the appropriate normal 
    distribution.
    '''
    sigma, b0, b1, b2, b3 = params
    values = df['sick'] - b0 + b1 * df['age'] + b2 * df['children']\
            + b3 * df['avgtemp_winter']
    norm_dist = sts.norm(scale = sigma)
    pdf = norm_dist.pdf(values)
    ln_pdf_vals = np.log(pdf)
    log_lik_val = ln_pdf_vals.sum()

    return(log_lik_val) 


def crit2(params, *args):
    '''
    This function is a criterion function for optimization

    Inputs:
        params: (tuple, length 2)
            mu: mean of lognormal distribution
            sigma: standard deviation of lognormal distribution
        args: (tuple, length 1)
            xvals: list of xvalues to calculate log likelihood for

    Ouputs:
        neg_log_lik_val: the negative log likelihood value so we can optimize a
                         minimization problem
    '''
    sigma, b0, b1, b2, b3 = params
    xvals = args[0]
    log_lik_val = calc_norm_likelihood(xvals, *params)
    neg_log_lik_val = -log_lik_val
    
    return(neg_log_lik_val)
                     

def estimate_params2(*params, values):
    '''
    This function takes intial values of mu and sigma, as well as an array of 
    real data with which to conduct the MLE, and returns the estimated parameters
    mu_MLE, and sig_MLE.
    '''
    params_init = np.array(list(params))
    mle_args = (values)
    # bnds = ((0, None), (None, None), (None, None), (None, None), (None, None)) # Want standard deviation to be positive

    results = opt.minimize(crit2, params_init, args=(mle_args)) # method = 'SLSQP', 
                           # bounds = bnds)
    return(results)

# Exercise 2b | Likelihood ratio test
def lrt_norm(df, **mle_test_params):
    '''
    '''
    log_lhood_mle = calc_norm_likelihood(df, *mle_test_params['mle'])
    log_lhood_tst = calc_norm_likelihood(df, *mle_test_params['tst'])

    lr_val = 2 * (log_lhood_mle - log_lhood_tst)

    p_val = 1.0 - sts.chi2.cdf(lr_val, 2)

    return(p_val)


if __name__ == '__main__':
    np.seterr(all = 'ignore') # Ignore numpy warnings
    incomes = np.loadtxt('incomes.txt') # Load incomes data
    # Exercise 1a
    plot_income_hist(incomes)

    # Exercise 1b
    print('Exercise 1b')
    plot_lognorm_pdf([0.001, 150000], coverage = COVERAGE, 
                                       mu = MU_INIT, sigma = SIG_INIT)
    log_lik_val = calc_lognorm_likelihood(incomes, mu = MU_INIT, sigma = SIG_INIT)
    print('The log likelihood value of the data given mu = 9 and sigma = .3 is {:.3f}\n'
          .format(log_lik_val))

    # Exercise 1c
    print('Exercise 1c')
    # I cheated by calculating 11.331 and .212 with the SLSQP method first
    # and then I am using these numbers as a starting point for the L-BFGS-B method
    # since that method returns a useable inverse hessian matrix.
    mu_mle, sig_mle, mle_lognorm, results1 = plot_alot(11.331, .212, incomes)
    print('Variance covariance matrix: {}\n'.format(results1.hess_inv.sk))

    # Exercise 1d
    print('Exercise 1d')
    p_val1 = lrt_lognorm(incomes, *(mu_mle, sig_mle, MU_INIT, SIG_INIT))
    print('Likelihood Ratio Test p-value is: {}\n'
         .format(p_val1))
    print('This number is really low (< .05), so it is unlikely that the data'
          ' came from the distribution in part (b)\n')

    # Exercise 1e
    print('Exercise 1e')
    print('The probability that I will earn more than $100,000 is {:.3f}\n'
          .format(1 - mle_lognorm.cdf(100000)))
    print('The probability that I will earn less than $75,000 is {:.3f}\n'
          .format(mle_lognorm.cdf(75000)))

    # Exercise 2a
    print('Exercise 2a')
    sick_dat = pd.read_csv('sick.txt', encoding = 'utf-8-sig') # Read in CSV
    results = estimate_params2(*(1, 0, 0, 0, 0), values = sick_dat)
    sig, b0, b1, b2, b3 = results.x
    print('The estimates for beta_0, beta_1, beta_2, beta_3, and sigma^2 are:'
          '{:.3f}, {:.3f}, {:.3f}, {:.6f}, and {:.8f}, respectively.\n'
          .format(b0, b1, b2, b3, sig**2))
    print('The value of the log likelihood function is: {}\n'.format(
            calc_norm_likelihood(sick_dat, *(sig, b0, b1, b2, b3))))
    print('The estimated variance covariance matrix of the estimates is: {}\n'
           .format(results.hess_inv))

    # Exercise 2b
    print('Exercise 2b')
    p_val2 = lrt_norm(sick_dat, **{'mle': (sig, b0, b1, b2, b3), 
                                 'tst': (np.sqrt(.01), 1, 0, 0, 0)})
    print('Likelihood Ratio Test p-value is: {}\n'
         .format(p_val1))
    print('This number is really low (< .05), so it is unlikely that age'
          ' number of children, and average winter temperature have no effect on'
          'the number of sick days.\n')

    
