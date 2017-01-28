# Python script for Problem Set 2
# MACS 301
# Julian McClellan

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as sts
import scipy.optimize as opt
import pandas as pd
import scipy.integrate as intgr
import numpy.linalg as lin

COVERAGE = 25 #%
MU_INIT = 9
SIG_INIT = .3

LBS = [0, 75000, 100000]
UBS = [75000, 100000, np.inf]

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


# Exercise 1a
def plot_income_hist(incomes, norm = True, add_on = False):
    '''
    This function takes incomes array and plots an appropriate histogram for 
    Exercise 1a.

    Also, it can return the fig, ax, and space of x for plotting additional 
    elements on top of the histogram.
    '''
    # Plot percentages histogram with 30 bins
    num_incomes = len(incomes)
    num_bins = 30

    if norm:
        hist_wgts = None
    else:
        hist_wgts = (100 / num_incomes) * np.ones(num_incomes)

    fig, ax = plt.subplots()
    plt.hist(incomes, num_bins, weights = hist_wgts, normed = norm)
    plt.ylabel(r'PDF value of Incomes')
    plt.xlabel(r'Annual Incomes (\$s)')
    
    if add_on: 
        max_income = incomes.max()
        return(np.linspace(0.001, max_income, COVERAGE * max_income), fig, ax)
    else: # Else, if this is being used for just 1a, then save the figure.
        plt.title('PDF Values for Annual Incomes of MACSS Students')
        plt.tight_layout()
        output_path = os.path.join(make_output_dir(), 'Fig_1a')
        plt.savefig(output_path)
        plt.close()

    return(None)

# Exercise 1b / 1d | Data moments
def data_moments(xvals, percent = False):
    '''
    This function returns the appropriate data moments; either mean and sd or
    some moments for percentages of observations that fall within certain 
    values.
    '''
    if percent:
        N = xvals.shape[0]
        data_moms = []
        for lb, ub in zip(LBS, UBS):
            proportion = xvals[(xvals <= ub) & (xvals > lb)] / N
            data_moms.append(proportion)   
        return(tuple(data_moms))  

    else: # If not using percents as moments, using mean and sd
        mean_data = xvals.mean()
        sd_data = xvals.std()

        return mean_data, sd_data

# Exercise 1b | model moments
def model_moments(mu, sigma, percent = False):
    '''
    '''
    lognorm_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
    if percent:
        pass
    else:
        lognorm_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
        xfx = lambda x: x * lognorm_dist.pdf(x)

        # Choice of upper bound of integration is arbitrary since np.inf the real
        # upper bound of the lognormal distribution, infinity (np.inf) does not
        # seem to work.
        mean_model, m_m_err = intgr.quad(xfx, 0, 150000)
        x2fx = lambda x: ((x - mean_model) ** 2) * lognorm_dist.pdf(x)
        var_model, v_m_err = intgr.quad(x2fx, 0, 150000)

                           # sd = sqrt(var)
        return(mean_model, np.sqrt(var_model))
    

# Exercise 1b | err_vec
def err_vec(xvals, mu, sigma):
    '''
    '''
    mean_data, sd_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [sd_data]])
    mean_model, sd_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [sd_model]])

    err_vec = (moms_model - moms_data) / moms_data

    return(err_vec)

# Exercise 1b | criterion function
def criterion(params, *args):
    '''
    '''
    mu, sigma = params
    xvals, W = args
    err = err_vec(xvals, mu, sigma)
    crit_val = np.dot(np.dot(err.T, W), err)

    return(crit_val)

# Exercise 1b/1c/1d | overlay plot
def pdf_on_hist(mus, sigmas, xvals, figlab = '1b'):
    '''
    '''
    # Get the domain space, figure, and axes from histogram in part a
    domain, fig, ax = plot_income_hist(xvals, norm = True, add_on = True) 

    # Jank way to allow plot multiple pdfs
    if not isinstance(mus, list):
        mus = [mus]
    if not isinstance(sigmas, list):
        sigmas = [sigmas]

    for mu, sigma in zip(mus, sigmas):
        lognorm = sts.lognorm(scale = np.exp(mu), s = sigma)

        # Plot the PDF against the histogram from part (a)
        ln_pdf = lognorm.pdf(domain)
        plt.plot(domain, ln_pdf, 
            label = '$f(x|\mu={:.3f},\sigma={:.3f}$'.format(mu, sigma))

    plt.ylabel('PDF Values')
    plt.title('Incomes of MACSS Students Histogram (normed) with Lognormal PDF', 
              y = 1.02) # Move title up slightly
    plt.legend(loc = 1) # Legend in upper right
    plt.tight_layout() # I want to see the labels

    # Save figure
    output_path = os.path.join(make_output_dir(), 'Fig_{}'.format(figlab))
    plt.savefig(output_path)
    plt.close()


# Exercise 1d |


if __name__ == '__main__':
    np.seterr(all = 'ignore') # Ignore numpy warnings
    incomes = np.loadtxt('incomes.txt') # Load incomes data

    # Exercise 1a
    plot_income_hist(incomes)

    # Exercise 1b
    print('Exercise 1b')
    params_init = np.array([MU_INIT, SIG_INIT])
    W_hat = np.eye(2)
    gmm_args = (incomes, W_hat)
    results = opt.minimize(criterion, params_init, args=(gmm_args),
                            method = 'L-BFGS-B', bounds=((None, None), (1e-10, None)))
    mu_GMM1, sig_GMM1 = results.x
    print('mu_GMM1={:.3f}'.format(mu_GMM1), ' sig_GMM1={:.3f}\n'.format(sig_GMM1))

    # Plot the GMM_ln_pdf against normed histogram
    pdf_on_hist(mu_GMM1, sig_GMM1, incomes)

    print('Criterion function value with GMM1 estimates = {}\n'.format(
        criterion((mu_GMM1, sig_GMM1), *(incomes, W_hat))))

    data_mu, data_sig = data_moments(incomes)
    mod1_mu, mod1_sig = model_moments(mu_GMM1, sig_GMM1)
    print('The data moment mu = {:.3f}, the model moment mu = {:.3f}\n'.format(
        data_mu, mod1_mu))
    print('The model moment sigma = {:.3f}, the model moment sigma = {:.3f}\n'.format(
        data_sig, mod1_sig))

    # Exercise 1c
    print('Exercise 1c')
    params_init = np.array([round(mu_GMM1, 3), round(sig_GMM1, 3)])
    errvec1c = err_vec(incomes, mu_GMM1, sig_GMM1)
    var_covar = np.dot(errvec1c, errvec1c.T) / incomes.shape[0]
    W_hat_2s = lin.pinv(var_covar)
    results2 = opt.minimize(criterion, params_init, args=((incomes, W_hat_2s)),
                            method = 'L-BFGS-B', bounds=((None, None), (1e-10, None)))
    mu_GMM2, sig_GMM2 = results2.x
    print('mu_GMM2={:.3f}'.format(mu_GMM2), ' sig_GMM2={:.3f}\n'.format(sig_GMM2))

    # Plot both the GMM_ln_pds against the normed histogram
    pdf_on_hist([mu_GMM1, mu_GMM2], [sig_GMM1, sig_GMM2], incomes, figlab = '1c')

    print('Criterion function value with GMM2 estimates = {}\n'.format(
        criterion((mu_GMM2, sig_GMM2), *(incomes, W_hat_2s))))

    mod2_mu, mod2_sig = model_moments(mu_GMM2, sig_GMM2)
    print('The data moment mu = {:.3f}, the model moment mu = {:.3f}\n'.format(
        data_mu, mod2_mu))
    print('The model moment sigma = {:.3f}, the model moment sigma = {:.3f}\n'.format(
        data_sig, mod2_sig))
