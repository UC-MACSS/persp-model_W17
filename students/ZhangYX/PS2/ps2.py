#########################
# Problem Set 2
# MACS 30100, Dr. Evans
# Yinxian Zhang
#########################

import numpy as np
import scipy as sp
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
Create an 'images' folder, if not existed, to save picture outputs when running this script
'''
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

'''
Question 1
-----------------------------------------
Lognormal distribution of some income data
------------------------------------------
'''    

# load income data
inc = np.loadtxt('income.txt')
# len(inc)

# Q1(a): plot a histogram of the income data
#---------------------------------------------
plot = True   
if plot: 
    bins = 30 # 30 bins
    weights = 1/len(inc) * np.ones(len(inc)) # add weights to make all the bin heights sum to 1
    n, bins, ignored = plt.hist(inc, bins=bins, weights=weights) # plot histogram
    
    plt.xlim([40000, 150000]) # set the two ends of x axis 
    plt.title('Income Distribution of MACS graduates') 
    plt.xlabel('Income(\$)')
    plt.ylabel('Percentage of graduates')
    #n.sum() # check if all bin heights sum to 1
    
    output = os.path.join(output_dir, 'plot1')
    plt.savefig(output) 
    plt.show()
    plt.close() 

# Q1(b): plot the lognormal PDF for x in range (0, 150000]
#----------------------------------------------------------

def log_norm_pdf(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Generate pdf values for the lognormal distribution with mean mu and standard deviation sigma. If the cutoff is given, then the PDF values are inflated upward to reflect the zero probability on values above the
    cutoff. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the log normally distributed random
             variable
    mu     = scalar, mean of the log normally distributed random variable
    sigma  = scalar > 0, standard deviation of the log normally distributed
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
        prob_notcut = sts.lognorm.cdf(cutoff, s=sigma, loc=mu, scale=np.exp(mu)) # use lognormal CDF function
            
    pdf_vals = ((1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))) / prob_notcut) # log normal distribution function 
    
    return pdf_vals 


def log_lik_norm(xvals, mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given log normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the log normally distributed random
             variable
    mu     = scalar, mean of the log normally distributed random variable
    sigma  = scalar > 0, standard deviation of the log normally distributed
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
    pdf_vals = log_norm_pdf(xvals, mu, sigma, cutoff) # get pdf values
    ln_pdf_vals = np.log(pdf_vals) # take log of the pdf values
    log_lik_val = ln_pdf_vals.sum() # sum the logs
   
    return log_lik_val # get the log likelihood value

# plot the lognormal PDF    
inc_sim = np.linspace(0.01, 150000, 200) # simulate income data, x should be strictly positive in the PDF, therefore the lower bound larger than 0 
mu_1 = 9 
sig_1 = 0.3

if plot:
   
    plt.plot(inc_sim, log_norm_pdf(inc_sim, mu_1, sig_1, 150000), linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3') # plot the lognormal PDF with the cutoff being 150000
 
    plt.title('Lognormal Distribution of Income with $\mu$=9 and $\sigma$=0.3')
    plt.xlabel('Income(\$)')
    plt.ylabel('lognormal PDF')
    
    output = os.path.join(output_dir, 'plot2')
    plt.savefig(output)  
    plt.show()
    plt.close() 
    
q2b_log_val = log_lik_norm(inc, mu_1, sig_1, 150000) # get the log likelihood value for this parameterization and given the MACS income data. 
 
# Q1(c): estimate the parameters by MLE
#------------------------------------------

def crit(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. 
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the log normally distributed random variable
    sigma  = scalar > 0, standard deviation of the log normally distributed
             random variable
    args   = length 2 tuple, (xvals, cutoff)
    xvals  = (N,) vector, values of the log normally distributed income   data
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
    mu, sigma = params # unpack parameters
    xvals, cutoff = args # unpack args
    log_lik_val = log_lik_norm(xvals, mu, sigma, cutoff) # get log likehood value
    neg_log_lik_val = -log_lik_val # get negative log likelihood 
    
    return neg_log_lik_val

# set parameters for the minimizer     
mu_init = 9 # intial guess of mu
sig_init = 0.3 # intial guess of sigma
params_init = np.array([mu_init, sig_init])
mle_args = (inc, 150000) 

# run minimizer to get MLE estimates
results1 = opt.minimize(crit, params_init, args=(mle_args), bounds=((None, None),(0.1, None)), method= 'L-BFGS-B')# plug in initial paramers and args, set the lower bounds of sigma to be larger than zero as sigma should be strictly postive in PDF function.
mu_MLE1, sig_MLE1 = results1.x # get the estimated parameters
fun1 = - results1.fun # get the value of log likehood function
hessian1 = results1.hess_inv.todense() # get the inverse hessian matrix as variance-covariance matrix

# plot the PDF against PDF from (b) and the histogram from (a)
if plot:
    plt.hist(inc, 30, normed=True) # plot histogram
    plt.plot(inc_sim, log_norm_pdf(inc_sim, mu_1, sig_1, 150000), linewidth=2, color='r', label='1: $\mu$=9,$\sigma$=0.3') # plot PDF from (b)
    plt.plot(inc_sim, log_norm_pdf(inc_sim, mu_MLE1, sig_MLE1, 150000), linewidth=2, color='k', label='2: $\mu$=mu_MLE,$\sigma$=sig_MLE') # plot PDF with parameters estimated by MLE
    plt.legend(loc='upper right') # add legend
    
    plt.title('Actural and estimated distributions of MACS graduates income')
    plt.xlabel('Income(\$)')
    plt.ylabel('Lognormal PDF')
    
    output = os.path.join(output_dir, 'plot3')
    plt.savefig(output)
    plt.show()
    plt.close() 

# Q1(d): perform a likelihood ratio test
#------------------------------------------

log_lik_h0 = log_lik_norm(inc, mu_1, sig_1, 150000) # get log likelihood value with the distribution in part(b)
log_lik_mle = log_lik_norm(inc, mu_MLE1, sig_MLE1, 150000) # get log likelihood value of the actural distribution
LR_val = 2 * (log_lik_mle - log_lik_h0) # get LR value from the likelihood ratio test
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2) # the probability that H0 is true with two degrees of freedom

# Q1(e): the probability of earning more than 100000 and less than 75000
#------------------------------------------------------------------------
probility1 = sts.lognorm.cdf(75000, s=sig_MLE1, loc=mu_MLE1, scale=np.exp(mu_MLE1))#probility of earning less than 75,000

probility2 = 1 - sts.lognorm.cdf(100000, s=sig_MLE1, loc=mu_MLE1, scale=np.exp(mu_MLE1)) # probility of earning more than 100,000

 
 

 
'''
Question 2
---------------------------
Linear regression and MLE
---------------------------
'''    

# Q2(a): estimate the five parameters of the model

# load data
df = pd.read_csv('sick.txt', header=0)
age = df['age']
temp = df['avgtemp_winter']
child = df['children']
sick = df['sick']

def norm_pdf(xvals, sigma):
    '''
    --------------------------------------------------------------------
    Generate value of the log likelihood function of standard normal distribution with standard deviation sigma. 
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OBJECTS CREATED WITHIN FUNCTION:
    prob_notcut = scalar 
    pdf_vals = (N,) vector, PDF values for mu=0 and sigma=sigma
               corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of PDF values
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    --------------------------------------------------------------------
    '''
    pdf_vals = (1/(sigma * np.sqrt(2 * np.pi)) *
                    np.exp(-(xvals)**2 / (2 * sigma**2))) # PDF function of standard normal distribution with mu = 0 and sigma
    ln_pdf_vals = np.log(pdf_vals) # take log 
    log_lik_val = ln_pdf_vals.sum() # sum the logs
    
    return log_lik_val  # returns values of log likelihood function

def crit2(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. 
    --------------------------------------------------------------------
    INPUTS:
    params = (5,) vector, ([beta0, beta1, beta2, beta3, sigma])
    from beta0 to beta3 = scalar, parameters of the function y ~ beta0 +  beta1*x1 + beta2*x2 + beta3*x3 + error 
    sigma  = scalar > 0, standard deviation of the normally distributed
             error term. 
    args   = length 4 tuple, (y, x1, x2, x3)
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    xvals = scalar, value of the random error given the formula y ~ beta0 +  beta1*x1 + beta2*x2 + beta3*x3 + error 
    log_lik_val = scalar, value of log likelihood function given the parameters and the values of error
    neg_log_lik_val = scalar, negative of log_lik_val 
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: neg_log_lik_val
    --------------------------------------------------------------------
    '''
    beta0, beta1, beta2, beta3, sigma = params
    y, x1, x2, x3 = args
    xvals = y - beta0 - beta1*x1 - beta2*x2 - beta3*x3 # given the model: error ~ sick - beta0 - beta1*age - beta2*temp - beta3*child
    log_lik_val = norm_pdf(xvals, sigma) # get the value of log likelihood function
    neg_log_lik_val = -log_lik_val # get negative value for minimizer
    
    return neg_log_lik_val

# set parameters for the minimizer
beta0_init = 1 # initial guess for five parameters
beta1_init = 0
beta2_init = 0
beta3_init = 0
sig_init2 = 1  
params_init2 = np.array([beta0_init, beta1_init, beta2_init, beta3_init, sig_init2]) 
mle_args2 = (sick, age, child, temp) # plug in the sick data
bd = ((None, None),(None, None), (None, None), (None, None), (0.001, None)) # set bounds for the minimizer because sigma should be strickly positive in the PDF function.

results2 = opt.minimize(crit2, params_init2, args=(mle_args2), bounds=bd, method='SLSQP') # plug in initial paramers and data to the crit funtion to search for minimum
beta0_mle, beta1_mle, beta2_mle, beta3_mle, sig_mle = results2.x # unpack estimated parameters
fun2 = - results2.fun # get the value of the loglikehood function

# SLSQP method does not report hessian matrix, therefore we use the estimated parameters given by SLSQP to run again the minimizer with L-BFGS-B method, which will give us the hessian matrix
params_init3 = np.array([beta0_mle, beta1_mle, beta2_mle, beta3_mle, sig_mle]) # plug in the ML estimated parameters 
results3 = opt.minimize(crit2, params_init3, args=(mle_args2), bounds=bd, method='L-BFGS-B')
hessian2 = results3.hess_inv.todense()

# Q2(b): likelihood ratio test
# set parameters of null hypothesis
b0 = 1
sig0 = 0.1
b1 = 0
b2 = 0
b3 = 0
param_h0 = np.array([b0, b1, b2, b3, sig0])
log_lik_h02 = - crit2(param_h0, *(sick, age, child, temp)) # value of log likeihood function given the null hypothesis
log_lik_mle2 = fun2 # value of the log likelihood function given MLE estimation
LR_val2 = 2 * (log_lik_mle2 - log_lik_h02) # get LR value from the likelihood ratio test
pval_h02 = 1.0 - sts.chi2.cdf(LR_val2, 2) # the probability that H0 is true with two degrees of freedom

if __name__ == '__main__':   
    print()
    print ('BEGINNING OF PROBLEM SET 2')
    print()
    print ('MACS 30100, Dr. Evans')
    print ('Yinxian Zhang')
    print()
    print ('Question 1')
    print('plot1, plot2 and plot 3 are saved')
    print()
    print('Q1(b)')
    print('The log likelihood value for this parameterization and given the MACS income data is :', q2b_log_val)
    print('Q1(c)')
    print('The ML estimates for mu and sigma are:', mu_MLE1, sig_MLE1)
    print('The value of the likelihood function is:', fun1)
    print('The variance-covariance matrix is:', hessian1)
    print('Q1(d)')
    print('The probability that the data in incomes.txt came from the distribution in part (b) is:', pval_h0)
    print('Q1(e)')
    print('The probability that I earn less than 75,000 is:', probility1)
    print('The probability that I earn more than 100,000 is:', probility2)
    print()
    print ('Question 2')
    print('Q2(a)')
    print('The ML estimates for beta0, beta1, beta2, beta3 and sigma^2 are:', beta0_mle, beta1_mle, beta2_mle, beta3_mle, sig_mle**2)
    print('The value of the likelihood function is:', fun2)
    print('The variance-covariance matrix is:', hessian2)
    print('Q2(b)')
    print('The probability that age, number of children, and average winter temperature have no effect on the number of sick days is:', pval_h02)  
    print ()
    print ('END OF PROBLEM SET 2')