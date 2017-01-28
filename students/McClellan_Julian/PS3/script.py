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


def man_lognorm_pdf(x, mu, sigma):
    '''
    '''
    rv = 1 / (sigma*x*np.sqrt(2*np.pi)) * np.exp(-1/2*((np.log(x) - mu) / sigma)**2)
    return(rv)


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
            proportion = xvals[(xvals <= ub) & (xvals > lb)].shape[0] / N
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
        # fx = lambda x: lognorm_dist.pdf(x)
        fx = lambda x: man_lognorm_pdf(x, mu, sigma)
        mod_moms = []
        for lb, ub in zip(LBS, UBS):
            prop, error = intgr.quad(fx, lb, ub, limit = 100)
            mod_moms.append(prop)
        return(tuple(mod_moms))
    else:
        # xfx = lambda x: x * lognorm_dist.pdf(x)
        xfx = lambda x: x * man_lognorm_pdf(x, mu, sigma)
        # Choice of upper bound of integration is arbitrary since np.inf the real
        # upper bound of the lognormal distribution, infinity (np.inf) does not
        # seem to work.
        mean_model, m_m_err = intgr.quad(xfx, 0, UPPER_INCOME, limit = 100)
        # x2fx = lambda x: ((x - mean_model) ** 2) * lognorm_dist.pdf(x)
        x2fx = lambda x: ((x - mean_model) ** 2) * man_lognorm_pdf(x, mu, sigma)
        var_model, v_m_err = intgr.quad(x2fx, 0, UPPER_INCOME, limit = 100)

                           # sd = sqrt(var)
        return(mean_model, np.sqrt(var_model))
    

# Exercise 1b | err_vec
def err_vec(xvals, mu, sigma, percent = False):
    '''
    '''
    moms_data = np.array([[rv] for rv in data_moments(xvals, percent)])
    moms_model = np.array([[rv] for rv in model_moments(mu, sigma, percent)])

    err_vec = (moms_model - moms_data) / moms_data

    return(err_vec)


# Exercise 1b | criterion function
def criterion(params, *args):
    '''
    '''
    mu, sigma = params
    xvals, W, percent = args

    err = err_vec(xvals, mu, sigma, percent)
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


# Exercise 2a
def err_vec_ols(data, *betas):
    b0, b1, b2, b3 = betas
    sick_data = np.array(data['sick'])
    sick_model = b0 + b1 * data['age'] + b2 * data['children'] + b3 * data['avgtemp_winter']

    err_vec = sick_data - sick_model
    return(err_vec)


def criterion_ols(params, *args):
    data = args[0] # See 'WTF'
    W = np.eye(len(data))
    err = err_vec_ols(data, *params)
    crit_val = np.dot(np.dot(err.T, W), err)

    return(crit_val)


if __name__ == '__main__':
    np.seterr(all = 'ignore') # Ignore numpy warnings
    incomes = np.loadtxt('incomes.txt') # Load incomes data
    bnds = ((None, None), (1e-10, None))

    # Exercise 1a
    plot_income_hist(incomes)


    # Exercise 1b
    print('Exercise 1b')
    params_init1b = np.array([MU_INIT, SIG_INIT])
    W_hat_2id = np.eye(2)
    gmm_args1b = (incomes, W_hat_2id, False)
    results1b = opt.minimize(criterion, params_init1b, args=(gmm_args1b),
                            method = 'L-BFGS-B', bounds=bnds)
    mu_GMM1b, sig_GMM1b = results1b.x
    print('mu_GMM1b={:.3f}'.format(mu_GMM1b), ' sig_GMM1b={:.3f}\n'.format(sig_GMM1b))

    # Plot the GMM_ln_pdf against normed histogram
    pdf_on_hist(mu_GMM1b, sig_GMM1b, incomes)

    print('Criterion function value with GMM1b estimates = {}\n'.format(
        criterion((mu_GMM1b, sig_GMM1b), *(incomes, W_hat_2id, False))))

    data_mu, data_sig = data_moments(incomes)
    mod1b_mu, mod1b_sig = model_moments(mu_GMM1b, sig_GMM1b)
    print('The data moment mu = {:.3f}, the model moment mu = {:.3f}\n'.format(
        data_mu, mod1b_mu))
    print('The model moment sigma = {:.3f}, the model moment sigma = {:.3f}\n'.format(
        data_sig, mod1b_sig))


    # Exercise 1c
    print('Exercise 1c')
    errvec1c = err_vec(incomes, mu_GMM1b, sig_GMM1b)
    var_covar1c = np.dot(errvec1c, errvec1c.T) / incomes.shape[0]
    W_hat_2s1c = lin.pinv(var_covar1c)
    gmm_args1c = (incomes, W_hat_2s1c, False)
                                        # Reuse 1b params
    results1c = opt.minimize(criterion, np.array([11, .75]), args=(gmm_args1c),
                            method = 'L-BFGS-B', bounds=bnds)
    mu_GMM1c, sig_GMM1c = results1c.x
    print('mu_GMM1c={:.3f}'.format(mu_GMM1c), ' sig_GMM1c={:.3f}\n'.format(sig_GMM1c))

    # Plot both the GMM1c/1b LN pdfs against the normed histogram
    pdf_on_hist([mu_GMM1b, mu_GMM1c], [sig_GMM1b, sig_GMM1c], incomes, figlab = '1c')

    print('Criterion function value with GMM1c estimates = {}\n'.format(
        criterion((mu_GMM1c, sig_GMM1c), *(incomes, W_hat_2s1c, False))))

    mod1c_mu, mod1c_sig = model_moments(mu_GMM1c, sig_GMM1c)
    print('The data moment mu = {:.3f}, the model moment mu = {:.3f}\n'.format(
        data_mu, mod1c_mu))
    print('The model moment sigma = {:.3f}, the model moment sigma = {:.3f}\n'.format(
        data_sig, mod1c_sig))


    # Exercise 1d
    params_init1d = np.array([11, .2])
    print('Exercise 1d')
    W_hat_3id = np.eye(3)
    gmm_args1d = (incomes, W_hat_3id, True)    # Reuse 1b params
    results1d = opt.minimize(criterion, params_init1d, args=(gmm_args1d),
                                method = 'L-BFGS-B', bounds=bnds
                             )
    mu_GMM1d, sig_GMM1d = results1d.x
    print('mu_GMM1d={:.3f}'.format(mu_GMM1d), ' sig_GMM1d={:.3f}\n'.format(sig_GMM1d))

    # Plot the GMM1d LN pdf against the normed histogram
    pdf_on_hist(mu_GMM1d, sig_GMM1d, incomes, figlab = '1d')


    print('Criterion function value with GMM1d estimates = {}\n'.format(
        criterion((mu_GMM1d, sig_GMM1d), *(incomes, W_hat_3id, True))))

    dm_percent = data_moments(incomes, True)
    mm_percent1d = model_moments(mu_GMM1d, sig_GMM1d, True)
    print('The proportion of incomes in the data between $0 and $75,000 = {:.3f}'.format(dm_percent[0]))
    print('The proportion of incomes in the data between $75,000 and $100,000 = {:.3f}'.format(dm_percent[1]))
    print('The proportion of incomes in the data between $100,000 and ${} = {:.3f}\n'.format(UPPER_INCOME, dm_percent[2]))

    print('The proportion of incomes in the model between $0 and $75,000 = {:.3f}'.format(mm_percent1d[0]))
    print('The proportion of incomes in the model between $75,000 and $100,000 = {:.3f}'.format(mm_percent1d[1]))
    print('The proportion of incomes in the model between $100,000 and ${} = {:.3f}\n'.format(UPPER_INCOME, mm_percent1d[2]))


    # Exercise 1e
    print('Exercise 1e')
    errvec1e = err_vec(incomes, mu_GMM1d, mu_GMM1d, True)
    var_covar1e = np.dot(errvec1e, errvec1e.T) / incomes.shape[0]
    W_hat_2s1e = lin.pinv(var_covar1e)
    gmm_args1e = (incomes, W_hat_2s1e, True)
    results1e = opt.minimize(criterion, params_init1d, args=(gmm_args1e),
                method = 'TNC', bounds=bnds
                            )
    mu_GMM1e, sig_GMM1e = results1e.x
    print('mu_GMM1e={:.3f}'.format(mu_GMM1e), ' sig_GMM1e={:.3f}\n'.format(sig_GMM1e))

    # Plot the GMM1d/1e LN pdfs against the normed histogram
    pdf_on_hist([mu_GMM1d, mu_GMM1e], [sig_GMM1d, sig_GMM1e], incomes, figlab = '1e')

    print('Criterion function value with GMM1e estimates = {}\n'.format(
        criterion((mu_GMM1e, sig_GMM1e), *(incomes, W_hat_2s1e, True))))

    mm_percent1e = model_moments(mu_GMM1e, sig_GMM1e, True)
    print('The proportion of incomes in the data between $0 and $75,000 = {:.3f}'.format(dm_percent[0]))
    print('The proportion of incomes in the data between $75,000 and $100,000 = {:.3f}'.format(dm_percent[1]))
    print('The proportion of incomes in the data between $100,000 and ${} = {:.3f}\n'.format(UPPER_INCOME, dm_percent[2]))

    print('The proportion of incomes in the model between $0 and $75,000 = {:.3f}'.format(mm_percent1e[0]))
    print('The proportion of incomes in the model between $75,000 and $100,000 = {:.3f}'.format(mm_percent1e[1]))
    print('The proportion of incomes in the model between $100,000 and ${} = {:.3f}\n'.format(UPPER_INCOME, mm_percent1e[2]))


    # Exercise 2a
    print('Exercise 2a')
    sick_dat = pd.read_csv('sick.txt', encoding = 'utf-8-sig') # Read in CSV

    params_init2a = np.array([0, 0, 0, 0])
    results2a = opt.minimize(criterion_ols, params_init2a, args=sick_dat,
                            method = 'L-BFGS-B')
    beta_vec = results2a.x
    print('Beta estimates (0, 1, 2, and 3) are: {}'.format(beta_vec))

    print('Criterion function value with GMM2a estimates = {}\n'.format(
        criterion_ols(np.array(beta_vec), *(sick_dat, 0))))
                                       #          ^^^
                                       # WTF If I don't have a 2 element
                                       # tuple thrown in here, criterion_ols
                                       # turns my dataframe into a string 'sick'.
                                       #
                                       # Don't believe me?
                                       # try running lines 321-322 with 
                                       # *(sick_dat).
                                       #
                                       # Think the issue has something to do with
                                       # Python's *args unpacking.