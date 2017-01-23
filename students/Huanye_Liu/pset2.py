#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 21:09:42 2017

@author: huanyeliu
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import scipy.stats as sts
import os

cur_path = os.path.split(os.path.abspath('pset2.py'))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
'''
---------------------------------------
the following code fragment is for 1(a)
---------------------------------------
'''
num_of_bin = 30
# read the data from income.txt file
incomes = np.loadtxt('incomes.txt')
plt.title('Histogram of annual incomes')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students' )
# adjust the left margin
plt.subplots_adjust(left=0.2)
plt.hist(incomes,bins=num_of_bin,normed=True)
output_path = os.path.join(output_dir, 'Fig_1a')
plt.savefig(output_path)
plt.clf()
#plt.plot()


'''
---------------------------------------
the following code fragment is for 1(b)
--------------------------------------- 
'''

def ln_pdf_value(x,mu,sigma):
    '''
    ---------------------------------------------------------------------------
    The function returns a numpy array of pdf values from lognormal distribution
    with parameter mu and sigma. this function does
    the same thing as sp.stats.lognorm.pdf(x,loc=mu,s=sigma,scale=exp(mu))
    ---------------------------------------------------------------------------
    Inputs:
       x: (N,)a numpy array of x values of the lognormally distributed variable. 
       mu: the mean of the lognormal distribution.
       sigma: the standard deviation of the lognormal distribution.
      
       OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
       OBJECTS CREATED WITHIN FUNCTION:
       prob_notcut = scalar 
       pdf_vals = (N,) vector, lognormal PDF values for mu and sigma
               corresponding to xvals data
    
       FILES CREATED BY THIS FUNCTION: None
    
       RETURNS: pdf_vals
    '''
    
    pdf_vals = (1/(x * sigma * np.sqrt(2 * np.pi)) * \
                np.exp( - (np.log(x) - mu)**2 / (2 * sigma**2))) 
    return pdf_vals


    
def log_lik_ln(x, mu, sigma):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given lognormal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the  lognormally distributed random
             variable
    mu     = scalar, mean of the lognormally distributed random variable
    sigma  = scalar > 0, standard deviation of the lognormally distributed
             random variable
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        ln_pdf_value()
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals    = (N,) vector, normal PDF values for mu and sigma
                  corresponding to xvals data
    ln_pdf_vals = (N,) vector, natural logarithm of normal PDF values
                  for mu and sigma corresponding to xvals data
    log_lik_val = scalar, value of the log likelihood function
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: log_lik_val
    '''
    #lognorm_dist = sts.lognorm(s = sigma,scale = np.exp(mu))
    #pdf = lognorm_dist.pdf(x)
    pdf_values = ln_pdf_value(x,mu,sigma)
    ln_pdf_vals = np.log(pdf_values)
    log_lik_val = ln_pdf_vals.sum()

    return log_lik_val 


   
    
    
# parameters   
step = 0.1
start = 0
end = 150000
mu = 9
sigma = 0.3 

# values of x from interval(0,150000)
x = np.arange(start,end,step)
# values of y (pdf value) based on lognormal distribution
ln_pdf = ln_pdf_value(x,mu,sigma)
# plot the pdf of the lognormal distribution
plt.title('lognormal pdf')
plt.xlim([start,end])
plt.plot(x, ln_pdf) 

output_path = os.path.join(output_dir, 'Fig_1b')
plt.savefig(output_path)
plt.clf()

# sum up all log likelihood values given the distribution
loglikelihood = log_lik_ln(incomes,mu,sigma)
print("1(b): The value of the log likelihood value given the data and distribution"\
      " is",loglikelihood)

'''
---------------------------------------
The following code segment is for 1(c)
---------------------------------------
'''



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
    xvals, cutoff= args
    log_lik_val = log_lik_ln(xvals, mu, sigma)
    neg_log_lik_val = -log_lik_val
    
    return(neg_log_lik_val)


mu_init = 9
sig_init = 0.5
    
params_init = np.array([mu_init, sig_init])
mle_args = (incomes, None)
# run the optimization using TNC method
results = opt.minimize(crit, params_init,method ="TNC", args=(mle_args))
                      

# temporaray estimates of mu and sigma
mu_next, sig_next = results.x
params_next = np.array([mu_next, sig_next])
# to get the variance-covrariance matrix, use default method again
results = opt.minimize(crit,params_next,args=(mle_args))
# final estimates of mu and sigma
mu_MLE, sig_MLE = results.x
# the lognormal pdf values using the maximum likelihood estimates
ln_pdf_MLE = ln_pdf_value(x,mu_MLE,sig_MLE)

# answers to 1(c) question
print('1(c): mu_MLE=', mu_MLE, ' sig_MLE=', sig_MLE)
print('1(c): The log likelihood value given MLEs and the data is',\
      log_lik_ln(incomes, mu_MLE, sig_MLE))
print('1(c): The variance-covariance matrix of MLE estimates is',\
      results.hess_inv)

# three overlaid plots
plt.hist(incomes,bins=num_of_bin,normed=True,label='Histogram of income data')
plt.plot(x, ln_pdf,label='The lognormal pdf given $\mu$=9,$\sigma$=0.3')
plt.plot(x,ln_pdf_MLE,label='The lognormal pdf using the maximum likelihood estimates')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students' ) 
plt.legend(loc='upper right',prop={'size':8})
output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
plt.clf()

'''
---------------------------------------
The following code segment is for 1(d)
---------------------------------------
'''
#  likelihood ratio test to see if the h0 model is significantly different 
#  from the MLE model,where h0 is the lognormal distribution specified in (b)
log_lik_h0 = log_lik_ln(incomes, mu, sigma)
log_lik_mle = log_lik_ln(incomes, mu_MLE, sig_MLE)
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
# answers to 1(d) questions
print('1(d): chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)

'''
---------------------------------------
The following code segment is for 1(e)
---------------------------------------
'''

# calculate the probabilities using cdf of the lognormal distribution with 
# maximum likelihood estimates as parameters 
print('The probability that I will earn more than $100,000 is ',\
      1-sts.lognorm.cdf(100000,s=sig_MLE,scale=np.exp(mu_MLE)))
print('The probability that I will earn more than $100,000 is ',\
      sts.lognorm.cdf(75000,s=sig_MLE,scale=np.exp(mu_MLE)))

'''
---------------------------------------
The following code segment is for 2(a)
---------------------------------------
'''
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
    prob_notcut = scalar 
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
   
            
    pdf_vals = (1/(sigma * np.sqrt(2 * np.pi))) *np.exp( - (xvals - mu)**2 / (2 * sigma**2))
    return pdf_vals

def log_lik_norm(df,param_list):
    '''
    --------------------------------------------------------------------
    Compute the log likelihood function for data xvals given normal
    distribution parameters mu and sigma.
    --------------------------------------------------------------------
    INPUTS:
    sick  =  a pandas dataframe storing all personal profile information
    param_list = a tuple of all paramters to be estimated
    
    
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
    sigma, beta0,beta1, beta2, beta3 = tuple(param_list)
    # calculate x values based on the linear regression equation
    xvals = df['sick']-beta0-beta1*df['age']-beta2*df['children']-beta3*df['avgtemp_winter']
    pdf_vals = norm_pdf(xvals, 0, sigma)
    ln_pdf_vals = np.log(pdf_vals)
    log_lik_val = ln_pdf_vals.sum()
    
    return log_lik_val

def crit1(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the negative of the log likelihood function
    given parameters and data. This is the minimization problem version
    of the maximum likelihood optimization problem
    --------------------------------------------------------------------
    INPUTS:
    params = (5,) vector, ([sigma, beta0,beta1, beta2, beta3])
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    beta0-3 = scalar, parameters of the linear regreassion model to be estimated
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
   
    df, cutoff = args
    log_lik_val = log_lik_norm(df, params)
    neg_log_lik_val = -log_lik_val
    
    return neg_log_lik_val
# read the sick data from text file 'sick.txt'    
sick = pd.read_csv('sick.txt',encoding = 'utf-8-sig')
# initial guess of the estimate values
params_init = np.array([0.100,1.000,0,0,0])
mle_args = (sick,None)
# make sure the value of the standard deviation is always above zero
bnd = ((0.001, None), (None, None), (None, None), (None, None), (None, None))
# optimization using SLSQP method
results = opt.minimize(crit1, params_init, args=(mle_args), bounds = bnd,method='SLSQP')
# the mimimize function above doesn't converge so run it again
results = opt.minimize(crit1, results.x, args=(mle_args), bounds = bnd,method='SLSQP')
# to get the variance-covrariance matrix, use default method again
results = opt.minimize(crit1,  results.x, args=(mle_args), bounds=bnd)
# extract estimates after results
sig_MLE,beta0_MLE,beta1_MLE,beta2_MLE,beta3_MLE = results.x

# answers to 2(a) questions
print("2(a): sig_MLE=",sig_MLE,'beta0_MLE=',beta0_MLE,'beta1_MLE=',beta1_MLE,\
'beta2_MLE=',beta2_MLE,'beta3_MLE=',beta3_MLE )
print('2(a): The log likehood function using the maximum likelihood estimates is ',\
      log_lik_norm(sick,[sig_MLE,beta0_MLE,beta1_MLE,beta2_MLE,beta3_MLE]))
print('2(a): The variance-covariance matrix of MLE estimates is',\
      results.hess_inv.todense())   

'''
---------------------------------------
The following code segment is for 2(b)
---------------------------------------
'''
#  likelihood ratio test to see if the h0 model is significantly different 
#  from the MLE model,where h0 is the lognormal distribution specified in 2(b)
log_lik_h0 = log_lik_norm(sick, [0.1,1,0,0,0])
log_lik_mle = log_lik_norm(sick, [sig_MLE,beta0_MLE,beta1_MLE,beta2_MLE,beta3_MLE])
LR_val = 2 * (log_lik_mle - log_lik_h0)
pval_h0 = 1.0 - sts.chi2.cdf(LR_val, 2)
# answers to 1(d) questions
print('2(b): chi squared of H0 with 2 degrees of freedom p-value = ', pval_h0)




