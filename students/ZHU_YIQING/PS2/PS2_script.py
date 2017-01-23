# Problem set 2 script
# Yiqing Zhu

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy.optimize as opt


'''
-------------------------------------------------------------------------------
Problem 1a:
Plot a histogram of percentages of the income.txt data with 30 bins. Make sure
that the bins are weighted such that the height of each bin represents the 
percent of the income observations in that bin. In other words, all the bin 
heights should sum to 1.
-------------------------------------------------------------------------------
'''
# Load income data
incomes = np.loadtxt('incomes.txt') 

plot_1a = True

if plot_1a:
    print('Problem 1a:')
    
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the histogram
    num_incomes = len(incomes)
    fig, ax = plt.subplots()
    hist_wgts = (1 / num_incomes) * np.ones(num_incomes)
    plt.hist(incomes, bins = 30, weights = hist_wgts)
    plt.title('Annual Income of 2018-2020 MACSS Graduates', fontsize = 14)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1a')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()

    
'''
-------------------------------------------------------------------------------
Problem 1b:
Plot the lognormal PDF f(x|μ = 9.0, σ = 0.3) for 0 ≤ x ≤ 150,000 and find the 
log likelihood value for this parameterization of the distribution.
-------------------------------------------------------------------------------
'''
# Define function that generates values of a log normal pdf
def log_norm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the log normal pdf with mean mu and standard
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

    prob_notcut  = scalar 
    log_pdf_vals = (N,) vector, log normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)
            
    log_pdf_vals    = ((1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) /
                    prob_notcut)

    return log_pdf_vals


plot_1b = True

if plot_1b:
    print('\nProblem 1b:')
    
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    # Plot the distribution
    dist_incomes = np.linspace(0, 150000, 500)
    mu_1b = 9.0
    sig_1b = 0.3
    fig, ax = plt.subplots()
    plt.plot(dist_incomes, log_norm_pdf(dist_incomes, mu_1b, sig_1b, 'None'),
         linewidth=2, color='r', label='1: $\mu$=9.0, $\sigma$=0.3')
    plt.title('Log normal PDF', fontsize = 14)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations')
    plt.legend(loc='upper right')

    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1b')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()
    
    
# Define log likelihood function for the log normal distribution
def log_lik_norm(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given log normal
    distribution parameters mu and sigma.
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
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        log_norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    log_pdf_vals    = (N,) vector, log normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_log_pdf_vals = (N,) vector, natural logarithm of log normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val     = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    log_pdf_vals = log_norm_pdf(xvals, mu, sigma, cutoff)
    ln_log_pdf_vals = np.log(log_pdf_vals)
    log_lik_val = ln_log_pdf_vals.sum()
    
    return log_lik_val

    
# Compute the log likelihood value for this parameterization of the distribution.
print('Log-likelihood 1b: ', log_lik_norm(incomes, mu_1b, sig_1b, 'None'))

    
'''
-------------------------------------------------------------------------------
Problem 1c:
Estimate the parameters of the lognormal distribution by maximum likelihood
and plot its PDF against the PDF from part (b) and the histogram from part (a).
Plot the estimated PDF for 0 ≤ x ≤ 150, 000. Report the ML estimates for μ and 
σ, the value of the likelihood function, and the variance-covariance matrix.
-------------------------------------------------------------------------------
'''
# The criterion function
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
    log_lik_val = log_lik_norm(xvals, mu, sigma, cutoff)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val

    
# The minimize() function
mu_init = 11
sig_init = 0.2
params_init = np.array([mu_init, sig_init])
mle_args = (incomes, 'None')
results = opt.minimize(crit, params_init, args=(mle_args), method = "SLSQP",
                       bounds=((None, None), (0.00001, None)))

params_init = results.x
results = opt.minimize(crit, params_init, args=(mle_args), method = "BFGS")

# Estimate the parameters of the lognormal distribution by maximum likelihood
mu_MLE, sig_MLE = results.x
print('\nProblem 1c:')
print('mu_MLE =', mu_MLE, ' sig_MLE =', sig_MLE)

# Compute the value of the likelihood function
log_like_val_MLE = log_lik_norm(incomes, mu_MLE, sig_MLE, 'None')
print('Log-likelihood MLE: ', log_like_val_MLE)

# Report the variance-covariance matrix
#stderr_mu_mle = np.sqrt(vcv_mle[0,0])
#stderr_sig_mle = np.sqrt(vcv_mle[1,1])
vcv_mle = results.hess_inv
print('Variance-Covariance matrix: \n', vcv_mle)

# Plot the MLE estimated distribution
plot_1c = True

if plot_1c:
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)
    
    fig, ax = plt.subplots()
    plt.title('Annual Income of 2018-2020 MACSS Graduates', fontsize = 14)
    plt.xlabel(r'Annual income($)')
    plt.ylabel(r'Percent of observations')
    
    # Plot the histogram of the data
    plt.hist(incomes, bins = 30, normed=True)
    
    # Plot the 1b distribution
    dist_incomes = np.linspace(0, 150000, 500)
    mu_1b = 9.0
    sig_1b = 0.3
    plt.plot(dist_incomes, log_norm_pdf(dist_incomes, mu_1b, sig_1b, 'None'),
         linewidth=2, color='r', label='1: $\mu$=9.0, $\sigma$=0.3')

    # Plot the MLE distribution
    plt.plot(dist_incomes, log_norm_pdf(dist_incomes, mu_MLE, sig_MLE, 'None'),
         linewidth=2, color='k', label='3: $\mu$=11.3, $\sigma$=0.2')
    plt.legend(loc='upper right')
    
    # Save the plot
    output_path = os.path.join(output_dir, 'fig_1c')
    plt.savefig(output_path)
    
    plt.show()
    plt.close()


'''
-------------------------------------------------------------------------------
Problem 1d:
Perform a likelihood ratio test to determine the probability that the data in 
incomes.txt came from the distribution in part (b).
-------------------------------------------------------------------------------
'''
log_lik_h0 = log_lik_norm(incomes, mu_1b, sig_1b, 'None')
log_lik_mle = log_lik_norm(incomes, mu_MLE, sig_MLE, 'None')
LR_val = 2 * (log_lik_mle - log_lik_h0)
chi2_ho = sts.chi2.pdf(LR_val, 2)
print('\nProblem 1d:')
print('Chi squared of H0 with 2 degrees of freedom = ', chi2_ho)
print("It is very unlikely that the data in incomes.txt came from the distribution in part (b).")


'''
-------------------------------------------------------------------------------
Problem 1e:
Using MLE model to compute the probability that you will earn more than 
$100,000 and that you will earn less than $75,000.
-------------------------------------------------------------------------------
'''
p1 = (1 - sts.lognorm.cdf(100000, s=sig_MLE, loc=mu_MLE, scale=np.exp(mu_MLE))) * 100
p2 = sts.lognorm.cdf(75000, s=sig_MLE, loc=mu_MLE, scale=np.exp(mu_MLE)) * 100
print('\nProblem 1e:')
print('The probability that a MACSS graduate earn more than $100,000: ', p1, '%')
print('The probability that a MACSS graduate earn less than $75,000: ', p2, '%')


'''
-------------------------------------------------------------------------------
Problem 2a:
Estimate the parameters by maximum likelihood using the fact that each error 
term is distributed normally N(0, σ2). Report estimates, the value of the log 
likelihood function, and the estimated variance covariance matrix.
-------------------------------------------------------------------------------
'''
# Define function that generates values of a normal pdf
def norm_pdf(xvals, mu, sigma):
    pdf_vals = (1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - mu)**2 / (2 * sigma**2)))
    return pdf_vals

    
# Define log likelihood function for the normal distribution
def log_like_norm_2(data, b0, b1, b2, b3, sigma):
    error = data['\ufeffsick'] - (b0+ b1*data['age'] + b2*data['children']+
            b3*data['avgtemp_winter'])
    pdf_vals = norm_pdf(error, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

    
# The criterion function
def crit_2(params, *args):
    b0, b1, b2, b3, sigma = params
    sick_data = args[0]
    log_lik_val = log_like_norm_2(sick_data, b0, b1, b2, b3, sigma)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val
    

sick_data = pd.read_csv('sick.txt', header = 0)

# Find MLE
b0_0 = 0.2
b1_0 = 0.1
b2_0 = 0.4
b3_0 = 0.0
sigma_0 = 1

params_init = np.array([b0_0, b1_0, b2_0, b3_0, sigma_0])
bnds = ((None, None),(None, None),(None, None),(None, None),(0.00001, None))

results = opt.minimize(crit_2, params_init, args=sick_data, method="SLSQP", bounds=bnds)
b0, b1, b2, b3, sigma = results.x
results = opt.minimize(crit_2,  results.x, args=sick_data, method="L-BFGS-B", bounds=bnds)
b0, b1, b2, b3, sigma = results.x

print("\nProblem 2a:")
print('Beta0 =', b0)
print('Beta1 =', b1)
print('Beta2 =', b2)
print('Beta3 =', b3)
print('Sigma =', sigma)

# log likelihood
log_like_val_2 = log_like_norm_2(sick_data, *results.x)
print('Log-likelihood: ', log_like_val_2)

# the variance-covariance matrix
vcv_mle_2 = results.hess_inv.todense()
print('Variance-Covariance matrix: \n', vcv_mle_2)


'''
-------------------------------------------------------------------------------
Problem 2b:
Use a likelihood ratio test to determine the probability that β0 = 1.0, 
σ2 = 0.01 and β1,β2,β3 = 0. That is, what is the likelihood that age, number 
of children, and average winter temperature have no effect on the number of 
sick days?
-------------------------------------------------------------------------------
'''
log_lik_h0 = log_like_norm_2(sick_data, 1, 0, 0, 0, 0.1)
log_lik_mle = log_like_norm_2(sick_data, *results.x)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print("\nProblem 2b")
print('Chi squared of H0 with 5 degrees of freedom p-value: ', pval_h0)
print("It is very unlikely that age, number of children, and average winter temperature have no effect on sick days.")