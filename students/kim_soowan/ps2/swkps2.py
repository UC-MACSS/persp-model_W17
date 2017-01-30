#Soo Wan Kim
#MACS30100
#Problem Set 2

#Used Python version 3.5.2

#import packages
import os
import seaborn
import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.stats import norm
import scipy.optimize as opt

#create images folder in current directory
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

##### Problem 1 #####

#load data
inc = np.loadtxt('incomes.txt') #read in the data

### (a)

#plot the data and save the image in images folder
plot = True

if plot:  
    fig1 = plt.figure()
    num_bins = 30 #include 30 bins
    #make histogram of percentages, not count
    weights = (1 / inc.shape[0]) * np.ones_like(inc)
    n, bin_cuts, patches = plt.hist(inc, num_bins, weights=weights)

    plt.title('Annual incomes of 2018-2020 MACSS graduates')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Percent of income amounts')
    plt.xlim([0, 150000])

    output_path = os.path.join(output_dir, '1a') #save image
    plt.savefig(output_path)
    plt.close()

### (b)
#generate evenly spaced samples
inc_pts = np.linspace(0, 150000, 150000)

#set parameters
mu = 9.0
sigma = 0.3

#plot the data and save the image in images folder
plot = True

if plot:  
    fig2 = plt.figure()
    #lognormal pdf
    plt.plot(inc_pts, lognorm.pdf(inc_pts, s = sigma, scale = np.exp(mu)), linewidth=1, color='r', 
        label='1: $\mu$=9,$\sigma$=0.3')
    plt.legend(loc='upper right')
    plt.title('Lognormal PDF ($\mu$=9, $\sigma$=0.3, 0 $\leq$ x $\leq$ 150,000)')
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.xlim([0, 150000])

    output_path = os.path.join(output_dir, '1b') #save image
    plt.savefig(output_path)
    plt.close()

# Define log likelihood function for the lognormal distribution
def log_lik_lognorm(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the negative of the log likelihood function for data xvals given lognormal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the lognormally distributed random
             variable
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, lognormal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of lognormal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    neg_log_lik_val = scalar, negative value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    
    #generate lognormal PDF values for mu = 9 and sigma = 0.3
    pdf_vals = lognorm.pdf(xvals, s = sigma, scale = np.exp(mu))
    #generate log of lognormal PDF values
    ln_pdf_vals = np.log(pdf_vals)
    #sum the log of lognormal PDF values to calculate the value of the log likelihood function
    log_lik_val = ln_pdf_vals.sum()
    neg_log_lik_val = -1*log_lik_val
    
    return neg_log_lik_val

print('1b')
print('Log-likelihood 1: ', -1*log_lik_lognorm(inc, mu, sigma))

### (c)

def crit(params, *args):
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
        log_lik_lognorm()
    OBJECTS CREATED WITHIN FUNCTION:
    log_lik_val = scalar, value of the log likelihood function
    FILES CREATED BY THIS FUNCTION: None
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals = args
    neg_log_lik_val = log_lik_lognorm(xvals, mu, sigma)

    return neg_log_lik_val

# first MLE attempt

#set initial values for mu and sigma
mu_init = 9.0 
sig_init = 0.3  
params_init = np.array([mu_init, sig_init])
mle_args = (inc)
bnds = ((None, None), (0.00001, None))

#the 'SLSQP' method produces good estimates at first try but does not produce the variance covariance matrix
results1 = opt.minimize(crit, params_init, args=(mle_args),
                            bounds = bnds, method = 'SLSQP')
mu_MLE1, sig_MLE1 = results1.x

# second MLE attempt, used the 'BFGS' method to produce the variance-covariance matrix. 
# estimates are almost identical to the ones produced in the first attempt using the other method
params_2 = np.array([mu_MLE1, sig_MLE1])
results2 = opt.minimize(crit, params_2, args=(mle_args),
                            bounds = bnds, method = 'BFGS')
mu_MLE2, sig_MLE2 = results2.x

#plot the data and save the image in images folder
plot = True

if plot:  
    fig3 = plt.figure()
    #normalize bins
    count, bins, ignored = plt.hist(inc, 30, normed=True)

    #PDF using the initial parameters with mu = 9 and sigma = 0.3
    plt.plot(inc_pts, lognorm.pdf(inc_pts, s = sig_init, scale = np.exp(mu_init)), linewidth=1, color='r', 
        label='initial parameters: $\mu$=9,$\sigma$=0.3')
    #PDF using MLE parameters
    plt.plot(inc_pts, lognorm.pdf(inc_pts, s = sig_MLE1, scale = np.exp(mu_MLE1)),
        linewidth=2, color='g', label='MLE parameters: $\mu$=11.33,$\sigma$=0.21')

    plt.legend(loc='upper right')

    plt.title('Lognormal distribution models of annual incomes of 2018-2020 MACSS graduates')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Normalized ratio')
    plt.xlim([0, 150000])

    output_path = os.path.join(output_dir, '1c') #save image
    plt.savefig(output_path)
    plt.close()

print('')
print('1c')
print('mu_MLE=', mu_MLE1, ' sig_MLE=', sig_MLE1)
print('Log-likelihood (MLE): ', -1*log_lik_lognorm(inc, mu_MLE1, sig_MLE1))
vcv_mle = results2.hess_inv
print('VCV(MLE) = ', vcv_mle)

### (d)
#likelihood ratio test to determine the probability that the data comes from the distribution in part b
log_lik_h0 = -1*log_lik_lognorm(inc, mu_init, sig_init)
log_lik_mle = -1*log_lik_lognorm(inc, mu_MLE1, sig_MLE1)
pval_h0 = 1.0 - sts.chi2.cdf(2 * (log_lik_mle - log_lik_h0), 2)

print('')
print('1d')
print('h0: ', pval_h0)

### (e)

print('')
print('1e')
print('Probability that I will earn more than $100,000: ', 1 - lognorm.cdf(100000, s = sig_MLE1, loc= mu_MLE1, scale = np.exp(mu_MLE1)))
print('Probability that I will earn less than $75,000: ', lognorm.cdf(75000, s = sig_MLE1, loc= mu_MLE1, scale = np.exp(mu_MLE1)))

##### Problem 2 #####

### (a)

sick = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'temp_winter']) #read in the data

def norm_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the normal pdf with mean mu and standard
    deviation sigma. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''            
    pdf_vals    = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - mu)**2 / (2 * sigma**2))
    
    return pdf_vals

#similar to log_lik_lognorm except distribution is normal instead of lognormal
def log_lik_norm(sick, age, children, temp_winter, b0, b1, b2, b3, sigma):

    #model solved for error terms
    x = sick - b0 - b1 * age - b2 * children - b3 * temp_winter
    pdf_vals = norm.pdf(x, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    neg_log_lik_val = -1*log_lik_val

    return neg_log_lik_val

def crit_sick(params, *args):

    b0, b1, b2, b3, sigma = params
    sick, age, children, temp_winter = args
    neg_log_lik_val = log_lik_norm(sick, age, children, temp_winter, b0, b1, b2, b3, sigma)

    return neg_log_lik_val

#first MLE attempt with hypothesis_0: age, number of children, and average winter temperature have no effect on number of sick days

#initialize parameters
b0_init = 1
b1_init = 0
b2_init = 0
b3_init = 0
sig_init = 0.1

params_init = np.array([b0_init, b1_init, b2_init, b3_init, sig_init])
mle_args = (sick["sick"], sick["age"], sick["children"], sick["temp_winter"])
bnds = ((None, None), (None, None), (None, None), (None, None), (.000001, None))

results1 = opt.minimize(crit_sick, params_init, method='SLSQP', args = mle_args, bounds = bnds)
b0_MLE1, b1_MLE1, b2_MLE1, b3_MLE1, sig_MLE1 = results1.x

#second MLE attempt with parameters taken from the first attempt

params_2 = np.array([b0_MLE1, b1_MLE1, b2_MLE1, b3_MLE1, sig_MLE1])
results2 = opt.minimize(crit_sick, params_2, method='L-BFGS-B', args = mle_args, bounds = bnds)
b0_MLE2, b1_MLE2, b2_MLE2, b3_MLE2, sig_MLE2 = results2.x

print("")
print('2a')
print('b0_MLE = ', b0_MLE2, ' b1_MLE = ', b1_MLE2, "b2_MLE = ", b2_MLE2, "b3_MLE = ", b3_MLE2, "sigma_MLE = ", sig_MLE2)
print('Log-likelihood (MLE): ', -1*log_lik_norm(sick["sick"], sick["age"], sick["children"], sick["temp_winter"], b0_MLE2, b1_MLE2, b2_MLE2, b3_MLE2, sig_MLE2))
print('VCV(MLE) = ', results2.hess_inv.todense())

### (b)

#likelihood ratio test to determine probability that b0 = 1, variance = 0.01, and b1, b2, b3 = 0
log_lik_h0 = -1*log_lik_norm(sick["sick"], sick["age"], sick["children"], sick["temp_winter"], b0_init, b1_init, b2_init, b3_init, sig_init)
log_lik_mle = -1*log_lik_norm(sick["sick"], sick["age"], sick["children"], sick["temp_winter"], b0_MLE2, b1_MLE2, b2_MLE2, b3_MLE2, sig_MLE2)
pval_h0 = 1.0 - sts.chi2.cdf(2 * (log_lik_mle - log_lik_h0), 2)

print("")
print("2b")
print('h0: ', pval_h0)