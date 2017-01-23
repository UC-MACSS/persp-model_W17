# Perspective on Computational Modeling: Problem Set #2
# Xingyun Wu


import numpy as np
import pandas as pd
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math
import os


# Create directory if images directory does not already exist
'''
--------------------------------------------------------------------
cur_path    = string, path name of current directory
output_fldr = string, folder in current path to save files
output_dir  = string, total path of images folder
output_path = string, path of file name of figure to be saved
--------------------------------------------------------------------
'''
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)


# PROBLEM 1: SOME INCOME DATA, LOGNORMAL DISTRIBUTION, AND HYPOTHESIS TESTING
# 1(a)
income = np.loadtxt('incomes.txt')
plot_1a = True
if plot_1a:
    num_bins = 30
    weights = (1/len(income))*np.ones_like(income)
    count, bins, ignored = plt.hist(income, num_bins, weights = weights)
    # set title and labels
    plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Percent of Students')
    # Save the plot    
    output_path = os.path.join(output_dir, 'Fig_1a')
    plt.savefig(output_path)
    plt.show()
    plt.close()

# 1(b)

# set initial parameters
sigma = 0.3
mu = 9.0
lower_bound = 0.00000000001
upper_bound = 150000
nums = upper_bound - lower_bound
x_vals = np.linspace(lower_bound, upper_bound, nums)
cutoff = upper_bound

# Define function that generates values of a normal pdf
def lognorm_pdf(x_vals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate log normal pdf values from the log normal pdf with mean mu 
    and standard deviation sigma. If the cutoff is given, then the PDF 
    values are inflated upward to reflect the zero probability on values 
    above the cutoff.
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

    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    lognorm = sts.lognorm(s = sigma, scale = np.exp(mu))
    pdf_vals = lognorm.pdf(x_vals)
    return pdf_vals

dist_inc = np.linspace(upper_bound, lower_bound, nums)

# plot the lognormal PDF
fig_1b = True
if fig_1b:
    plt.plot(dist_inc, lognorm_pdf(dist_inc, mu_1, sig_1, cutoff), \
             linewidth=2, color='r', label='1(b): $\mu$=9.0, $\sigma$=0.3')
    plt.legend(loc='upper right')
    # set title and labels
    plt.title("Lognormal PDF")
    plt.xlabel("Annual Income")
    plt.xlim([0, 150000])
    # save the plot    
    output_path = os.path.join(output_dir, 'Fig_1b')
    plt.savefig(output_path)
    plt.show()
    plt.close()
    
# Define log likelihood function for the normal distribution
def log_lik_norm(x_vals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable

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
    pdf_vals = lognorm_pdf(x_vals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val

# calculate the log likelihood value
print('Task 1(b): Log-likelihood value: ', log_lik_norm(income, mu_1, \
      sig_1, cutoff))

# 1(c)
# set the criterion function
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
    x_vals, cutoff = args
    log_lik_val = log_lik_norm(x_vals, mu, sigma, cutoff)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val

# the minimize function: test the given parameters
params_init = np.array([mu_init, sig_init])
mle_args = (income, cutoff)
bnds = ((None, None), (0.00000000001, None))

results = opt.minimize(crit, params_init, args=(mle_args), 
            bounds = bnds, method = 'SLSQP')
mu_MLE, sig_MLE = results.x # MLE for mu, sigma 
print("Task 1(c): mu_MLE = ", mu_MLE, ", sigma_MLE = ", sig_MLE, ".")
# find that mu_MLE = 11.33, sigma_MLE = 0.21
# to make plot look nicer, use the approximation
mu_MLE = 11.33
sig_MLE = 0.21
# calculate the log likelihood value
print('Task 1(c): Log-likelihood value: ', log_lik_norm(income, mu_MLE, \
      sig_MLE, cutoff))

# plot the histogram from part (a)
num_bins = 30
weights = (1/income.shape[0])*np.ones_like(income)
count, bins, ignored = plt.hist(income, num_bins, weights = weights, normed=True)
# plot the PDF from part(b)
plt.plot(dist_inc, lognorm_pdf(dist_inc, mu_1, sig_1, cutoff), linewidth=2, color='r', label='1(b): $\mu$=9.0, $\sigma$=0.3')
# plot MLE estimated distribution
plt.plot(dist_inc, lognorm_pdf(dist_inc, mu_MLE, sig_MLE, cutoff), linewidth=2, color='g', label='1(c): $\mu$='+str(mu_MLE)+', $\sigma$='+str(sig_MLE))
# set title, labels and legend
plt.title("Annual Income of U Chicago's MACSS Graduates", fontsize=15)
plt.xlabel('Annual Income')
plt.ylabel('Percent of Graduates')
plt.xlim([0, cutoff])
plt.legend(loc='upper right')
# save the plot    
output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
plt.show()
plt.close()

# variance-covariance matrix
# since the code in Jupyter Notebook could not be directly used, calculate minimizer again with another method
params_mle = np.array([mu_MLE, sig_MLE]) # the parameters here are approximation of real mu_MLE and sig_MLE
results_mle = opt.minimize(crit, params_mle, args=(mle_args), 
                bounds = bnds, method = 'L-BFGS-B')
vcv_mle = results_mle.hess_inv.todense()
print("Task 1(d): The Varianvce-covarianve matrix(VCV(MLE)) is \n", vcv_mle)

# 1(d) likelihood ratio test
log_lik_h0 = log_lik_norm(income, mu_1, sig_1, cutoff)
log_lik_mle = log_lik_norm(income, mu_MLE, sig_MLE, cutoff)
LR_val = 2 * (log_lik_mle - log_lik_h0)
chi2_ho = sts.chi2.pdf(LR_val, 2)
print("Task 1(d): chi squared of H0 with 2 degrees of freedom = ", chi2_ho)

# 1(e) probabilities
lognorm_mle = sts.lognorm(s = sig_MLE, scale = np.exp(mu_MLE))
cdf_mle_100000 = lognorm_mle.cdf(100000)
cdf_mle_75000 = lognorm_mle.cdf(75000)
# the probability of greater than 100,000 shoube be 1 - cdf(100000)
prob_more_than_100000 = 1 - cdf_mle_100000
print("Task 1(e)")
print("The probability of earning more than $100,000 is: ", prob_more_than_100000)
# the probability of less than 75,000 should be cdf(75000) itself
print("The probability of earning less than $75,000 is: ", cdf_mle_75000)


# PROBLEM 2: LINEAR REGRESSION AND MLE
# Tried to read-in 'sick.txt" with numpy.loadtxt, but failed becalse it contains askii. Turn to use pandas.
sick = pd.read_csv('sick.txt', skiprows=1)
sick_i = sick.iloc[:, 0]
age_i = sick.iloc[:, 1]
children_i = sick.iloc[:, 2]
temp_winter_i = sick.iloc[:, 3]

# Define function that generates values of a normal pdf
def norm_pdf(xvals, mu, sigma):
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
    pdf_vals = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (xvals - mu)**2 / (2 * sigma**2))
    return pdf_vals

# Define log likelihood function for the normal distribution
def log_lik_norm2(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
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
    pdf_vals = norm_pdf(xvals, mu, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()    
    return log_lik_val

def crit2(params, *args):
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
    sigma, beta0, beta1, beta2, beta3 = params
    sick_i, age_i, children_i, temp_winter_i = args
    xvals = sick_i - beta0 - beta1 * age_i - beta2 * children_i - beta3 * temp_winter_i
    log_lik_val = log_lik_norm2(xvals, 0, sigma)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val

# 2(a)
# setup initial guess
sig_init2 = 1 # standard normal distribution
beta0 = 1
beta1 = 0.2
beta2 = 0.3
beta3 = 0.4

# testing the initial guess
params_init2 = np.array([sig_init2, beta0, beta1, beta2, beta3])
mle_args2 = (sick_i, age_i, children_i, temp_winter_i)
bnds2 = ((0.00000000001, None), (None, None), (None, None), (None, None), (None, None))

results = opt.minimize(crit2, params_init2, args=(mle_args2), 
            bounds = bnds2, method = 'SLSQP')
sig2_MLE, b0_MLE, b1_MLE, b2_MLE, b3_MLE = results.x # MLE for mu, sigma 
print("Task 2(a): ")
print("sig_MLE = ", sig2_MLE)
print("b0_MLE = ", b0_MLE)
print("b1_MLE = ", b1_MLE)
print("b2_MLE = ", b2_MLE)
print("b3_MLE = ", b3_MLE)

# to get variance-covariance matrix
# get the Hessian_inv
params_mle2 = np.array([sig2_MLE, b0_MLE, b1_MLE, b2_MLE, b3_MLE])
results_mle2 = opt.minimize(crit2, params_mle2, 
                                args=(mle_args2), bounds = bnds2,
                                method = 'L-BFGS-B')
vcv_mle2 = results_mle2.hess_inv.todense()
print('Task 2(a): The Varianvce-covarianve matrix(VCV(MLE)) is \n', vcv_mle2)
# get the data using MLE to generate the value of likelihood function
data_MLE = sick_i - b0_MLE - (b1_MLE * age_i)- (b2_MLE * children_i) \
            - (b3_MLE * temp_winter_i)
print("The value of the likelihood function is: ")
print(log_lik_norm2(data_MLE, 0, sig2_MLE))

# 2(b)
sig_test = math.sqrt(0.01)
b0_test = 1.0
b1_test = 0
b2_test = 0
b3_test = 0
# likelihood ratio test
data_h0 = sick_i - b0_test - (b1_test * age_i) - (b2_test * children_i) \
                             - (b3_test * temp_winter_i)
log_lik_h02 = log_lik_norm2(data_h0, 0, sig_test)
log_lik_mle2 = log_lik_norm2(data_MLE, 0, sig_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
chi2_ho = sts.chi2.pdf(LR_val, 2)
print('chi squared of H0 with 2 degrees of freedom = ', chi2_ho)