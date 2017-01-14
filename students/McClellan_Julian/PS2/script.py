# Python script for Problem Set 2
# MACS 301
# Julian McClellan

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt


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

def man_lognorm_pdf(x, mu, sigma):
    '''
    '''
    rv = 1 / (sigma*x*np.sqrt(2*np.pi)) * np.exp(-1/2*((np.log(x) - mu) / sigma)**2)
    return(rv)


# Exercise 1a
def plot_income_hist(incomes):
    '''
    This function takes incomes array and plots an appropriate percentage 
    histogram for Exercise 1a.
    '''
    # Plot percentages histogram with 30 bins
    num_incomes = len(incomes)
    num_bins = 30
    hist_wgts = (100 / num_incomes) * np.ones(num_incomes)

    fig, ax = plt.subplots()
    plt.hist(incomes, num_bins, weights = hist_wgts)
    plt.ylabel(r'Percentage of incomes (%)')
    plt.xlabel(r'Annual Incomes (\$s)')
    plt.title('Annual Incomes of MACSS Students')

    # plt.show()
    # Save figure
    output_path = os.path.join(make_output_dir(), 'Fig_1a')
    plt.savefig(output_path)
    plt.close()

    return(None)


# Exercise 1b | Plotting lognormal pdf
def plot_lognorm_pdf(bounds, coverage, mu, sigma):
    '''
    '''
    assert (0 < coverage < 100), 'Coverage must be between 0% and 100%'

    # X-axis values creation
    start, stop = bounds 
    x_vals = np.linspace(start, stop, coverage * (stop - start))  

    # Y-axis values (lognorm pdf) creation
    lognorm_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
    lognorm_pdf = lognorm_dist.pdf(x_vals)

    # Plotting
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)
    plt.xlim(0, 150000)
    plt.plot(x_vals, lognorm_pdf)
    plt.xlabel('x')
    plt.ylabel('$f(x|\mu=9,\sigma=.3)$')
    plt.title('Log normal pdf. $\mu={}$, $\sigma={}$'.format(mu, sigma))
    # plt.show()

    # Save figure
    output_path = os.path.join(make_output_dir(), 'Fig_1b')
    plt.savefig(output_path)
    plt.close()


# Exercise 1b | Calculating log likelihood value
def calc_lognorm_likelihood(values, mu, sigma, manual = False):
    '''
    Given an array of values, a mu, and a sigma, this function calculates the
    log likelihood from the appropriate lognormal distribution.
    '''
    lognorm_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
    pdf = lognorm_dist.pdf(values)
    ln_pdf_vals = np.log(pdf)
    log_lik_val = ln_pdf_vals.sum()\

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


def estimate_params(mu_init, sig_init, values):
    '''
    This function takes intial values of mu and sigma, as well as an array of 
    real data with which to conduct the MLE, and returns the estimated parameters
    mu_MLE, and sig_MLE.
    '''
    params_init = np.array([mu_init, sig_init])
    mle_args = (values)
    bnds = ((None, None), (0, None)) # Want standard deviation to be positive

    results = opt.minimize(crit, params_init, method='SLSQP', args=(mle_args), 
                           bounds = bnds)
    mu_MLE, sig_MLE = results.x
    print('mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)





if __name__ == '__main__':
    incomes = np.loadtxt('incomes.txt') # Load incomes data
    # Exercise 1a
    plot_income_hist(incomes)

    # Exercise 1b
    plot_lognorm_pdf([0.001, 150000], coverage = 25, mu = 9, sigma = .3)
    log_lik_val = calc_lognorm_likelihood(incomes, mu = 9, sigma = .3)
    print('The log likelihood value of the data given mu = 9 and sigma = .3 is {}'.format(log_lik_val))

    # Exercise 1c
