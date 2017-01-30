'''
----------------------------------------------------------------------
PA2 - Perspectives for Computational Modeling
----------------------------------------------------------------------
Student: Rodrigo Valdes Ortiz
----------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import math
import random 
import scipy.stats as sts
import math
import os 
import scipy.optimize as opt
import pandas as pd


'''
------------------------------------------------------------------------
Ex. 1: Some income data, lognormal distribution, and hypothesis testing
------------------------------------------------------------------------
'''

def make_output_dir():
    '''
    Create directory if images directory does not already exist
    '''
    cur_path = os.path.split(os.path.abspath("__file__"))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

    return(output_dir)

output_dir = make_output_dir()   
pts = np.loadtxt('incomes.txt')

# Define function that generates values of a log-normal pdf
def log_norm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma. If the cutoff is given, then the PDF values are
    inflated upward to reflect the zero probability on values above the
    cutoff. If there is no cutoff given, this function does the same
    thing as sp.stats.lognorm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
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
    pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    if cutoff == 'None':
        prob_notcut = 1.0
    else:
        prob_notcut = sts.norm.cdf(cutoff, loc=mu, scale=sigma)
            
    pdf_vals = ((1/(xvals * sigma * np.sqrt(2 * np.pi)) * 
        np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut)
    
    return pdf_vals

def log_lik_norm(xvals, mu, sigma, cutoff):
    '''
    Computes the log likelihood function for data xvals given log-normal
    distribution parameters mu and sigma.

    Inputs:
       xvals: array
       mu: float
       sigma: float
       cutoff: float
    Outputs:
       log_lik_val: float
    '''
    pdf_vals = log_norm_pdf(xvals, mu, sigma, cutoff)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    return log_lik_val

# 1.a Histogram
weights = np.ones_like(pts)/len(pts)
count, bins, ignored = plt.hist(pts, 30, weights=weights)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('Percent of incomes')
plt.xlim([40000, 150000])  # This gives the xmin and xmax to be plotted"
#plt.show()
print("1. Part (A)")
output_path = os.path.join(output_dir, 'one')
plt.savefig(output_path)
plt.close()

# 1.b Plot the log-normal pdf
print("\n 1. Part (B)")
# Initial Histogram
weights = 100 * np.ones_like(pts)/len(pts)
count, bins, ignored = plt.hist(pts, 30, weights=weights, normed = True)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('Percent of incomes')
plt.xlim([0.1, 150000])  # This gives the xmin and xmax to be plotted"

# Declare parameters
dist_pts = np.linspace(0.1, 150000, 500)
mu_1 = 9
sig_1 = 0.3

plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_1, sig_1, 150000),
         linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3')
plt.xlim([0.1, 150000])  # This gives the xmin and xmax to be plotted"
plt.legend(loc='upper right')
# plt.show()
output_path = os.path.join(output_dir, 'two')
plt.savefig(output_path)
plt.close()

ll_value = log_lik_norm(pts, mu_1, sig_1, 150000)
print("Log likelihood = ", ll_value)

# 1.C 
print("\n 1. Part (C)")
# Define auxiliary functions for Part c
INCOME = np.loadtxt('incomes.txt')
# < Parameters >
MU = 9.0
SIGMA = 0.3

# < Defining Function >
# --------------------------------------------------
# FUNCTION_1: Defining a function that generates lognormal Probability Density Function
# INPUT_1: x values
# INPUT_2: mu
# INPUT_3: sigma
# OUTPUT: pdf values
# --------------------------------------------------
def FUNCTION_1(INPUT_1, INPUT_2, INPUT_3):
    OUTPUT = sts.lognorm.pdf(INPUT_1, scale=np.exp(INPUT_2), s=INPUT_3)
    return OUTPUT

# < Defining Function >
# --------------------------------------------------
# FUNCTION_2: Defining a function that computes log likelihood value
# INPUT_1: x values
# INPUT_2: mu
# INPUT_3: sigma
# OUTPUT: log likelihood value
# --------------------------------------------------
def FUNCTION_2(INPUT_1, INPUT_2, INPUT_3):
    OUTPUT = np.log(FUNCTION_1(INPUT_1, INPUT_2, INPUT_3)).sum()
    return OUTPUT

# < Defining Function >
# --------------------------------------------------
# FUNCTION_3: Defining a function that computes negative log likelihood value
# INPUT_1: (mu, sigma)
# INPUT_2: (x values)
# OUTPUT: negative log likelihood value
# --------------------------------------------------
def FUNCTION_3(INPUT_1, INPUT_2):
    INPUT_1_1, INPUT_1_2 = INPUT_1
    OUTPUT = -FUNCTION_2(INPUT_2, INPUT_1_1, INPUT_1_2)
    return OUTPUT


# < Defining Function >
# --------------------------------------------------
# FUNCTION_4: Defining a function that computes Maximum Likelihood Estimation
# INPUT_1: initial mu
# INPUT_2: initial sigma
# INPUT_3: values
# INPUT_4: method
# OUTPUT: MLE mu, MLE sigma
# --------------------------------------------------

def FUNCTION_4(INPUT_1, INPUT_2, INPUT_3, INPUT_4='L-BFGS-B'):
    PROCESS_1_1, PROCESS_2_1 = opt.minimize(FUNCTION_3, np.array([INPUT_1, INPUT_2]), args=(INPUT_3), bounds=((None, None),(1e-10, None)), method='SLSQP').x
    PROCESS_1_2, PROCESS_2_2 = opt.minimize(FUNCTION_3, np.array([PROCESS_1_1, PROCESS_2_1]), args=(INPUT_3), bounds=((None, None),(1e-10, None)), method=INPUT_4).x
    OUTPUT_1 = opt.minimize(FUNCTION_3, np.array([PROCESS_1_2, PROCESS_2_2]), args=(INPUT_3), bounds=((None, None),(1e-10, None)), method=INPUT_4)
    return OUTPUT_1
# print(FUNCTION_4(MU, SIGMA, INCOME))
master = FUNCTION_4(MU, SIGMA, INCOME)
a = master.x[0] # this is mu
b = master.x[1] # this is sigma


print("The ML estimate for mu is:", a)
print("The ML estimate for sigam is:", b)
print("The value of the likelihood function is:", master.fun)

plt.hist(INCOME, bins=30, normed=True)
plt.title('MACSS\'s Alumni Incomes and PDF. Class 2018, 2019, and 2020', fontsize=15)
plt.plot(np.linspace(0, 150000, 500), FUNCTION_1(np.linspace(0, 150000, 500), MU, SIGMA), label='1: $\mu$ =9, $\sigma$ = 0.3')
plt.plot(np.linspace(0, 150000, 500), FUNCTION_1(np.linspace(0, 150000, 500), a, b), label='1: $\mu$ =11.33, $\sigma$ = 0.21')
plt.xlim(0.1, 150000)
plt.xlabel('Incomes')
plt.ylabel('Incomes / PDF')
plt.legend(loc='upper right')
# plt.show()
output_path = os.path.join(output_dir, 'three')
plt.savefig(output_path)
plt.close()

# 1.d
print("\n 1. Part (D)")

log_lik_h0 = log_lik_norm(pts, mu_1, sig_1, 'None')
log_lik_mle = log_lik_norm(pts, a, b, 'None')
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print('chi squared of H0 with 5 degrees of freedom p-value = ', pval_h0)

# 1.e
print("\n 1. Part (E)")

exp_mu_MLE = np.exp(a)
bank = 100000
broke = 75000
bank_prob = 1 - sts.lognorm(s = b, scale = exp_mu_MLE).cdf(bank)
broke_prob = sts.lognorm(s = b, scale = exp_mu_MLE).cdf(broke)

print('The probability of a given MACSS graduate earning more than $100,000'\
      + ' is: {:.3f}.\n'.format(bank_prob))
print('The probability of a given MACSS graduate earning less than $75,000'\
      + ' is: {:.3f}.'.format(broke_prob))


'''
------------------------------------------------------------------------
Ex. 2: Estimate betas
------------------------------------------------------------------------
'''

# Problem 2
database = pd.read_csv('sick.txt', names = ["sick", "age", "children", "temp"], skiprows = [0])

# Define function
def pdf(database, list_guess):
    beta_0 = list_guess[0]
    beta_1 = list_guess[1]
    beta_2 = list_guess[2]
    beta_3 = list_guess[3]
    sigma = list_guess[4]
    error = (database["sick"] - beta_0 - beta_1 * database["age"] - beta_2 * database["children"] - beta_3 * database["temp"])
    pdf_vals = (1 / sigma * math.sqrt(2 * math.pi)) * np.exp((- (error) ** 2 ) / (2 * (sigma ** 2)))
    return pdf_vals

def log_lik(database, list_guess):
    pdf_vals = pdf(database, list_guess)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_sum = ln_pdf_vals.sum()
    return log_lik_sum

def crit(params, *args):
    beta_0, beta_1, beta_2, beta_3, sigma = params
    sick, age, children, temp = args
    database = pd.DataFrame(sick, columns = ['sick'])
    database['age'] = age
    database['children'] = children
    database['temp'] = temp
    log_lik_val = log_lik(database, params)
    neg_log_lik_val = -log_lik_val
    return neg_log_lik_val

sick = np.array(database["sick"])
age = np.array(database["age"])
children = np.array(database["children"])
temp = np.array(database["temp"])

list_guess = [0.0, 0.005, 0.05, 0.005, 0.3]
# 0, 0.005, 0.05, 0.005, 0.3
params_init = np.array(list_guess)
mle_args = (sick, age, children, temp)
results = opt.minimize(crit, params_init, args=mle_args, method = 'Powell')
beta_0_ml, beta_1_ml, beta_2_ml, beta_3_ml, sigma_ml = results.x

print("\n 2. Exercise")

print('beta_0 =', beta_0_ml, 'beta_1=', beta_1_ml, 'beta_2=', beta_2_ml, 'beta_3=', beta_3_ml, 'sigma2=', sigma_ml ** 2)
print("")
print("The value of the likelihood function is", results.fun)
print("The variace covariance matrix is:", results.direc)
# Log-likelihood

# Use a likelihood ratio test to determine the probability that β0 = 1.0, σ2 = 0.01 and β1,β2,β3 = 0. 
# That is, what is the likelihood that age, number of children, and average winter temperature have no effect 
# on the number of sick days?

list_h0 = [1, 0, 0, 0, 0.01]
list_results = [beta_0_ml, beta_1_ml, beta_2_ml, beta_3_ml, sigma_ml]

log_lik_h0 = log_lik(database, list_h0)
log_lik_mle = log_lik(database, list_results)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 5)
print('chi squared of H0 with 5 degrees of freedom p-value = ', pval_h0)











