# Python script for Problem Set 2
# MACS 301
# Julian McClellan

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt

COVERAGE = 25

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
def plot_income_hist(incomes, part_c = False):
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
def plot_lognorm_pdf(bounds, coverage, mu, sigma, part_c = False, 
                     fig_name = 'Fig_1b'):
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
    plt.xlim(0, 150000)
    plt.plot(x_vals, lognorm_pdf, label = '$f(x|\mu={}, \sigma={}$'
            .format(round(mu, 3), round(sigma, 3)))
    plt.xlabel('x')

    if part_c: # If need to add the PDF from 1c, do not save, return the x_vals
        return(x_vals)
    else:
        plt.ylabel('$f(x|\mu=9,\sigma=.3)$')
        plt.title('Log normal pdf. $\mu={}$, $\sigma={}$'
                  .format(round(mu, 3), round(sigma, 3)))
        plt.tight_layout()
        output_path = os.path.join(make_output_dir(), fig_name)
        plt.savefig(output_path)
        plt.close()
    return(None)


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
    results = estimate_params(mu_init, sig_init, incomes)
    mu_mle, sig_mle = results.x
    mle_label = '$f(x|\mu={}, \sigma={})$'.format(round(mu_mle, 3), 
                                                  round(sig_mle, 3))

    # Plot the PDF against the PDF from part b
    # Plot part b, retrieve x_vals
    x_vals = plot_lognorm_pdf([0.001, 150000], coverage = COVERAGE, mu = 9, 
                              sigma = .3, part_c = True) 

    # Y-axis values (lognorm pdf) creation
    mle_lognorm_dist = sts.lognorm(scale = np.exp(mu_mle), s = sig_mle)
    mle_lognorm_pdf = mle_lognorm_dist.pdf(x_vals)
    # Plot on top of figure from part b
    plt.plot(x_vals, mle_lognorm_pdf, label = mle_label)
    plt.ylabel('$f(x|\mu,\sigma)$')
    plt.title('Log normal pdfs with parameters $\mu,\sigma$')
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(make_output_dir(), 'Fig_1c_with_b')
    plt.savefig(output_path)
    plt.close()

    # Plot the estimated pdf for ) <= x <= 150,000.
    plot_lognorm_pdf([0.001, 150000], COVERAGE, mu_mle, sig_mle, part_c = False, 
                     fig_name = 'Fig_1c')

    # Plot the PDF against the histogram from part a (twin y-label)
    # Get new x vals (max is incomes.max())
    x_vals, fig, ax = plot_income_hist(incomes, part_c = True) 
    trunc_mle_ln_pdf = mle_lognorm_dist.pdf(x_vals)
    ax2 = ax.twinx()
    ax2.plot(x_vals, trunc_mle_ln_pdf, label = mle_label, color = 'red')
    ax2.set_ylabel(mle_label)
    plt.title('Annual Incomes of MACSS Students Histogram with Lognormal PDF', 
              y = 1.02) # Move title up slightly
    plt.legend(loc = 2) # Legend in upper left
    plt.tight_layout() # I want to see the labels

    # Save figure
    output_path = os.path.join(make_output_dir(), 'Fig_1c_with_a')
    plt.savefig(output_path)
    plt.close()


if __name__ == '__main__':
    incomes = np.loadtxt('incomes.txt') # Load incomes data
    # Exercise 1a
    plot_income_hist(incomes)

    # Exercise 1b
    plot_lognorm_pdf([0.001, 150000], coverage = COVERAGE, mu = 9, sigma = .3)
    log_lik_val = calc_lognorm_likelihood(incomes, mu = 9, sigma = .3)
    print('The log likelihood value of the data given mu = 9 and sigma = .3 is {}'.format(log_lik_val))

    # Exercise 1c
    plot_alot(9, .3, incomes)
