import pandas as pd
import numpy as np
import math
from numpy import log as ln
from numpy import exp as exp
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

# Problem 1
graph = True

if graph:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    Fig_1  = Boolean, = True if make a histogram of percentages of the
                         income
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    # (a) : Plot a histogram
    df = pd.read_table('incomes.txt', names = ['incomes'])
    num_obs = len(df['incomes'])
    fig, ax = plt.subplots()
    hist_wgts = (1 / num_obs) * np.ones(num_obs)
    num_bins = 30
    plt.hist(df['incomes'], num_bins, weights = hist_wgts)
    plt.title('Histogram of Annual Income of MACSS Students', fontsize = 17)
    plt.xlabel('Annual income')
    plt.ylabel('Percent of students')
    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    plt.show()
    plt.close()


    # (b): Plot the lognormal PDF
    def lognorm_pdf(xvals, mu, sigma, cutoff):
        '''
        --------------------------------------------------------------------
        Generate pdf values from the lognormal pdf with mean mu and standard
        deviation sigma. If the cutoff is given, then the PDF values are
        inflated upward to reflect the zero probability on values above the
        cutoff. If there is no cutoff given, this function does the same
        thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
        --------------------------------------------------------------------
        INPUTS:
        xvals  = (N,) vector, values of the lognormally distributed random
                 variable
        mu     = scalar, mean of the lognormally distributed random variable
        sigma  = scalar > 0, standard deviation of the lognormally distributed
                 random variable
        cutoff = scalar or string, ='None' if no cutoff is given, otherwise
                 is scalar upper bound value of distribution. Values above
                 this value have zero probability
        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
        OBJECTS CREATED WITHIN FUNCTION:
        prob_notcut = scalar
        pdf_vals = (N,) vector, normal PDF values for mu and sigma
                   corresponding to xvals data
        FILES CREATED BY THIS FUNCTION: None
        RETURNS: pdf_vals
        --------------------------------------------------------------------
        '''
        if cutoff == 'None':
            prob_notcut = 1.0
        else:
            prob_notcut = sts.norm.cdf(np.log(cutoff), loc=mu, scale=sigma)
        pdf_vals = ((1/(sigma * np.sqrt(2 * np.pi) * xvals) * np.exp( - \
                   (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut) 
        return pdf_vals

    def log_lik_lognorm(xvals, mu, sigma, cutoff):
        '''
        --------------------------------------------------------------------
        Compute the log likelihood function for data xvals given lognormal
        distribution parameters mu and sigma.
        --------------------------------------------------------------------
        INPUTS:
        xvals  = (N,) vector, values of the lognormally distributed random
                 variable
        mu     = scalar, mean of the lognormally distributed random variable
        sigma  = scalar > 0, standard deviation of the lognormally distributed
                 random variable
        cutoff = scalar or string, ='None' if no cutoff is given, otherwise
                 is scalar upper bound value of distribution. Values above
                 this value have zero probability
        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            norm_pdf()
        OBJECTS CREATED WITHIN FUNCTION:
        pdf_vals    = (N,) vector, lognormal PDF values for mu and sigma
                      corresponding to xvals data
        ln_pdf_vals = (N,) vector, natural logarithm of lognormal PDF values
                      for mu and sigma corresponding to xvals data
        log_lik_val = scalar, value of the log likelihood function
        FILES CREATED BY THIS FUNCTION: None
        RETURNS: log_lik_val
        --------------------------------------------------------------------
        '''
        pdf_vals = lognorm_pdf(xvals, mu, sigma, cutoff)
        ln_pdf_vals = np.log(pdf_vals)
        log_lik_val = ln_pdf_vals.sum()

        return log_lik_val

    dist_pts = np.linspace(0, 150000, 500)
    mu_0 = 9
    sig_0 = 0.3
    # Plot the lognormal PDF
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_0, sig_0, 'None'),
             linewidth = 2, color = 'r', label = '1: $\mu$=9.0, $\sigma$=0.3')
    plt.legend(loc = 'upper right')
    plt.title('Lognormal PDF', fontsize = 17)
    plt.xlabel('Income')
    plt.ylabel('PDF')
    output_path = os.path.join(output_dir, 'Fig_1b')
    plt.savefig(output_path)
    plt.show()
    plt.close()

    print('1b. ', 'Log-likelihood : ', log_lik_lognorm(df['incomes'], mu_0, sig_0, 'None'))

    # (c): Estimate the parameters of the lognormal distibution by ML, plot
    #      the PDF and histogram. Plot the estimated PDF and report MLLE, etc.
    def crit(params, *args):
        '''
        --------------------------------------------------------------------
        This function computes the negative of the log likelihood function
        given parameters and data. This is the minimization problem version
        of the maximum likelihood optimization problem
        --------------------------------------------------------------------
        INPUTS:
        params = (2,) vector, ([mu, sigma])
        mu     = scalar, mean of the normally distributed random variable
        sigma  = scalar > 0, standard deviation of the normally distributed
                 random variable
        args   = length 2 tuple, (xvals, cutoff)
        xvals  = (N,) vector, values of the normally distributed random
                 variable
        cutoff = scalar or string, ='None' if no cutoff is given, otherwise
                 is scalar upper bound value of distribution. Values above
                 this value have zero probability
        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            log_lik_norm()
        OBJECTS CREATED WITHIN FUNCTION:
        log_lik_val = scalar, value of the log likelihood function
        neg_log_lik_val = scalar, negative of log_lik_val
        FILES CREATED BY THIS FUNCTION: None
        RETURNS: neg_log_lik_val
        --------------------------------------------------------------------
        '''
        mu, sigma = params
        xvals, cutoff = args
        log_lik_val = log_lik_lognorm(xvals, mu, sigma, cutoff)
        neg_log_lik_val = -log_lik_val
        return neg_log_lik_val

    def mini(mu_init, sig_init, mle_args):
        mu_init = mu_init
        sig_init = sig_init
        params_init = np.array([mu_init, sig_init])
        mle_args = mle_args
        bnds = ((None, None), (.0001, None))
        results = opt.minimize(crit, params_init, args=(mle_args), bounds=bnds)
        mu_MLE, sig_MLE = results.x
        return (results, mu_MLE, sig_MLE)

    mu_init = 11.3
    sig_init = 0.2
    results, mu_MLE, sig_MLE = mini(mu_init, sig_init, (df['incomes'], 'None'))
    print('1c. ', 'mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)

    num_obs = len(df['incomes'])
    fig, ax = plt.subplots()
    hist_wgts = (1 / num_obs) * np.ones(num_obs)
    num_bins = 30
    plt.hist(df['incomes'], num_bins, normed=True)
    plt.title('Histogram of Annual Income of MACSS Students', fontsize = 17)
    plt.xlabel('Annual income')
    plt.ylabel('Percent of students') 
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_0, sig_0, 'None'), linewidth=2, color='r', label='1: $\mu$=9.00, $\sigma$=0.30')
    plt.plot(dist_pts, lognorm_pdf(dist_pts, mu_MLE, sig_MLE, 'None'), linewidth=2, color='b', label='2: $\mu$=11.33, $\sigma$=0.21')
    plt.legend(loc='upper right')
    output_path = os.path.join(output_dir, 'Fig_1c')
    plt.savefig(output_path)
    plt.show()
    plt.close()
    print('    ', 'Maximized log-likelihood : ', log_lik_lognorm(df['incomes'], mu_MLE, sig_MLE, 'None'))

    vcv_mle = results.hess_inv.todense()
    print('    ', 'VCV(MLE) = ', vcv_mle)


    # (d): Perform a likelihood ratio test
    log_lik_h0 = log_lik_lognorm(df['incomes'], mu_0, sig_0, 'None')
    log_lik_mle = log_lik_lognorm(df['incomes'], mu_MLE, sig_MLE, 'None')
    LR_val = 2 * (log_lik_mle - log_lik_h0)
    pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
    print('1d. ', 'chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)

    # (e): Use the estimated model to determine the probabilities
    ansKey1b_1 = 100 - 100 * np.round(sts.lognorm.cdf(100000, s = sig_MLE, loc = mu_MLE, scale = exp(mu_MLE)), 4)
    ansKey1b_2 = 100 * np.round(sts.lognorm.cdf(75000, s = sig_MLE, loc = mu_MLE, scale = exp(mu_MLE)), 4)
    print('1e. ' + 'The possibility of having an income higher than $100000 is' + str(ansKey1b_1) + '.')
    print('    ' + 'The possibility of having an income lower than $75000 is' +  str(ansKey1b_2) + '.')
    

# Problem 2
    # (a) 
    df = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'avgtemp_winter'])
    def norm_pdf(xvals, mu, sigma, cutoff):
        '''
        --------------------------------------------------------------------
        Generate pdf values from the normal pdf with mean mu and standard
        deviation sigma. If the cutoff is given, then the PDF values are
        inflated upward to reflect the zero probability on values above the
        cutoff. If there is no cutoff given, this function does the same
        thing as sp.stats.norm.pdf(x, loc=mu, scale=sigma).
        --------------------------------------------------------------------
        INPUTS:
        xvals  = (N,) vector, values of the normally distributed random
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
        pdf_vals = (N,) vector, normal PDF values for mu and sigma
                   corresponding to xvals data
        
        FILES CREATED BY THIS FUNCTION: None
    
        RETURNS: pdf_vals
        --------------------------------------------------------------------
        '''
        if cutoff == 'None':
            prob_notcut = 1.0
        else:
            prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)
        pdf_vals = ((1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - mu)**2 / (2 * sigma**2))) / prob_notcut)
        return pdf_vals

    def log_lik_sick(xdf, b0, b1, b2, b3, sigma, cutoff):
        '''
        --------------------------------------------------------------------
        Compute the log likelihood function for data xvals given normal
        distribution parameters mu and sigma.
        --------------------------------------------------------------------
        INPUTS:
        xvals  = (N,) vector, values of the normally distributed random
                 variable
        bs     = scalar, parameters
        sigma  = scalar > 0, standard deviation of the normally distributed
                  random variable
        cutoff = scalar or string, ='None' if no cutoff is given, otherwise
                 is scalar upper bound value of distribution. Values above
                 this value have zero probability
        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            norm_pdf()
    
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
        xvals = xdf['sick'] - b0 - b1 * xdf['age'] - b2 * xdf['children'] - b3 * xdf['avgtemp_winter']
        pdf_vals = norm_pdf(xvals, 0, sigma, cutoff)
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
        params = (2,) vector, ([mu, sigma])
        mu     = scalar, mean of the normally distributed random variable
        sigma  = scalar > 0, standard deviation of the normally distributed
                 random variable
        args   = length 2 tuple, (xvals, cutoff)
        xvals  = (N,) vector, values of the normally distributed random
                 variable
    
        OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
            regression_log_lik_norm()
    
        OBJECTS CREATED WITHIN FUNCTION:
        log_lik_val = scalar, value of the log likelihood function
        neg_log_lik_val = scalar, negative of log_lik_val
    
        FILES CREATED BY THIS FUNCTION: None
     
        RETURNS: neg_log_lik_val
        --------------------------------------------------------------------
        '''
        b0, b1, b2, b3, sigma = params
        xdf, cutoff, method = args
        log_lik_val = log_lik_sick(xdf, b0, b1, b2, b3, sigma, cutoff)
        neg_log_lik_val = -log_lik_val
        return neg_log_lik_val

    def mini_sick(b0_init, b1_init, b2_init, b3_init, sig_init, mle_args):
        params_init = np.array([b0_init, b1_init, b2_init, b3_init, sig_init])
        mle_args = mle_args
        bnds = ((None, None), (None, None), (None, None), (None, None), (.000001, None))
        results = opt.minimize(crit_sick, params_init, args=(mle_args), bounds=bnds, method=mle_args[2])
        b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig_MLE = results.x
        return (results, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig_MLE)

    b0_init, b1_init, b2_init, b3_init, sig_init = (1, 0, 0, 0, 1)
    results, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig_MLE =  mini_sick(b0_init, b1_init, b2_init, b3_init, sig_init, (df, 'None', 'SLSQP'))
    b0_init, b1_init, b2_init, b3_init, sig_init = (b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig_MLE)
    results, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig_MLE =  mini_sick(b0_init, b1_init, b2_init, b3_init, sig_init, (df, 'None', 'L-BFGS-B'))
    print('2a. ', 'b0_MLE=', b0_MLE, ' b1_MLE=', b1_MLE, ' b2_MLE=', b2_MLE, ' b3_MLE=', b3_MLE, ' sig_MLE=', sig_MLE)
    print('    ', 'Maximized log-likelihood : ', log_lik_sick(df, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig_MLE, 'None'))

    vcv_mle = results.hess_inv.todense()
    print('    ', 'VCV(MLE) = ', vcv_mle)


    #(b): Use a likelihood ratio test to determine the probability
    b0_test, b1_test, b2_test, b3_test, sig_test = (1, 0, 0, 0, 0.01)
    log_lik_h0 = log_lik_sick(df, b0_test, b1_test, b2_test, b3_test, sig_test, 'None')
    log_lik_mle = log_lik_sick(df, b0_MLE, b1_MLE, b2_MLE, b3_MLE, sig_MLE, 'None')
    LR_val = 2 * (log_lik_mle - log_lik_h0)
    pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)

    print('2b. ', 'chi squared of H0 with 5 degrees of freedom p-value = ', pval_h0, '. That is, the likelihood that age, number of children, and average winter temperature have no effect on the sick days is very low.')
