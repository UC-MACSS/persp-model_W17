# Python script for Problem Set 4
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
import pdb

COVERAGE = 25 #%
MU_INIT = 9
SIG_INIT = .3
UPPER_INCOME = 150000

LBS = [0, 75000, 100000]
UBS = [75000, 100000, UPPER_INCOME]


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


def man_lognorm_pdf(x, mu, sigma): # This is unused
    '''
    '''
    rv = 1 / (sigma*x*np.sqrt(2*np.pi)) * np.exp(-1/2*((np.log(x) - mu) / sigma)**2)
    return(rv)


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
        # lognorm = sts.lognorm(scale = np.exp(mu), s = sigma)

        # Plot the PDF against the histogram from part (a)
        # ln_pdf = lognorm.pdf(domain)
        ln_pdf = man_lognorm_pdf(domain, mu, sigma)
        plt.plot(domain, ln_pdf, 
            label = '$f(x|\mu={:.3f},\sigma={:.3f}$'.format(mu, sigma))

    plt.ylabel('PDF Values')
    plt.title('Incomes of MACSS Students Histogram (normed) with Lognormal PDF(s)', 
              y = 1.02) # Move title up slightly
    plt.legend(loc = 1) # Legend in upper right
    plt.tight_layout() # I want to see the labels

    # Save figure
    output_path = os.path.join(make_output_dir(), 'Fig_{}'.format(figlab))
    plt.savefig(output_path)
    plt.close()


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

# Exercise 1b | LN_pdf()

def LN_pdf(xvals, mu, sigma):
    '''
    '''
    ln_dist = sts.lognorm(scale = np.exp(mu), s = sigma)

    rv = ln_dist.pdf(xvals)
    return(rv)


# Exercise 1c | Estimate mu and sigma with SMM
def moments(vals):
    '''
    '''
    if vals.ndim == 2:
        mean_vec = vals.mean(axis = 0)
        std_vec = vals.std(axis = 0)

        # Let's just take the average hereq
        mean = mean_vec.mean()
        std = std_vec.mean()
    elif vals.ndim == 1:
        mean = vals.mean()
        std = vals.std()

    return(mean, std)


def err_vec(data_vals, sim_vals, mu, sigma):
    '''
    '''
    mean_data, var_data = moments(data_vals)
    moms_data = np.array([[mean_data], [var_data]])

    mean_model, var_model = moments(sim_vals)
    moms_model = np.array([[mean_model], [var_model]])

    err_vec = (moms_model - moms_data) / moms_data
    
    return(err_vec)


def LN_draws(num_obs, num_sims, mu, sigma):
    '''
    '''
    np.random.seed(1234)

    norm_dist = sts.norm(mu, sigma)
    norm_draws = norm_dist.rvs((num_obs, num_sims))

    ln_draws = np.exp(norm_draws)
    return(ln_draws)



def criterion(params, *args):
    '''
    '''
    mu, sigma = params
    xvals, W_hat, num_sims = args
    sim_vals = LN_draws(len(xvals), num_sims, mu, sigma)

    err = err_vec(xvals, sim_vals, mu, sigma)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    return(crit_val)


def estimate_params(xvals, mu_init, sig_init, num_sims = 300, two_step = False):
    '''
    '''
    params_init1 = np.array([mu_init, sig_init])

    weights = np.eye(2)

    smm_args = (xvals, weights, num_sims)
    results1 = opt.minimize(criterion, params_init1, args=(smm_args),
                              method='L-BFGS-B',
                              bounds=BNDS)

    mu_SMM1, sig_SMM1 = results1.x

    sim_vals_SMM1 = LN_draws(len(xvals), num_sims, mu_SMM1, sig_SMM1)
    data_moments = moments(xvals)
    if two_step:
        # Define new initial parameters (results of our previous SMM)
        params_init2 = np.array([11, .2])

        # Calculate two step weighting matrix
        errvec = err_vec(xvals, sim_vals_SMM1, mu_SMM1, sig_SMM1)
        weights_twostep = lin.pinv(np.dot(errvec, errvec.T) / xvals.shape[0])

        smm_args2 = (xvals, weights_twostep, num_sims)
        results2 = opt.minimize(criterion, params_init2, args = (smm_args2),
                                method = 'TNC', bounds = BNDS)

        mu_SMM2, sig_SMM2 = results2.x
        print('mu_SMM2=', mu_SMM2, ' sig_SMM2=', sig_SMM2)
        print('Critical value is: {}'.format(criterion((mu_SMM2, sig_SMM2),
                                            *(xvals, weights_twostep, num_sims))))

        

        sim_vals_SMM2 = LN_draws(len(xvals), num_sims, mu_SMM2, sig_SMM2)
        model_moments2 = moments(sim_vals_SMM2)
        print('Data moment mu = {}, sigma = {}.\n Model Moments mu = {}, sigma = {}'
              .format(data_moments[0], data_moments[1], model_moments2[0], model_moments2[1]))
        pdf_on_hist([mu_SMM1, mu_SMM2], [sig_SMM1, sig_SMM2], xvals, figlab = '1d')

    else:


        print('mu_SMM1=', mu_SMM1, ' sig_SMM1=', sig_SMM1)
        print('Critical value is: {}'.format(criterion((mu_SMM1, sig_SMM1), 
                                            *(xvals, weights, num_sims))))

        model_moments1 = moments(sim_vals_SMM1)
        print('Data moment mu = {}, sigma = {}.\n Model Moments mu = {}, sigma = {}'
              .format(data_moments[0], data_moments[1], model_moments1[0], model_moments1[1]))
        pdf_on_hist(mu_SMM1, sig_SMM1, xvals, figlab = '1c')





if __name__ == '__main__':
    np.seterr(all = 'ignore') # Ignore numpy warnings
    incomes = np.loadtxt('incomes.txt') # Load incomes data
    BNDS = ((None, None), (1e-10, None))

    # Exercise 1a
    print('Exercise 1a')
    plot_income_hist(incomes)

    # Exercise 1b
    test_vals = np.array([[200, 270], [180, 195]])
    mu_test = 5
    sig_test = 1

    print('Exercise 1b')
    print(LN_pdf(test_vals, mu_test, sig_test), '\n')

    print('Exercise 1c')
    estimate_params(incomes, MU_INIT, SIG_INIT)
    print('\n')

    print('Exercise 1d')
    estimate_params(incomes, MU_INIT, SIG_INIT, 300, True)