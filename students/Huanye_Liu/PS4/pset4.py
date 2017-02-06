#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:39:05 2017

@author: huanyeliu
"""

import numpy as np 
import numpy.linalg as lin
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os

cur_path = os.path.split(os.path.abspath('pset4.py'))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

'''
---------------------------------------
the following code fragment is for 1(a)
---------------------------------------
'''

incomes = np.loadtxt('incomes.txt')
num_of_bins =30
plt.title("Histogram of annual incomes")
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students')
# adjust the left margin
plt.subplots_adjust(left=0.2)
count, bins, ignored = plt.hist(incomes,bins=num_of_bins,normed=True)

output_path = os.path.join(output_dir, 'Fig_1a')
plt.savefig(output_path)
plt.clf()

'''
---------------------------------------
the following code fragment is for 1(b)
---------------------------------------
'''
def LN_pdf(xvals,mu,sigma):
    '''
    ---------------------------------------------------------------------------
    The function returns a numpy array of pdf values from lognormal distribution
    with parameter mu and sigma. 
    ---------------------------------------------------------------------------
    Inputs:
       xvals: (N,) or (N,S) a numpy array of x values of the lognormally 
              distributed variable. 
       mu: the mean of the lognormal distribution.
       sigma: the standard deviation of the lognormal distribution.
      
       OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
       OBJECTS CREATED WITHIN FUNCTION:
       prob_notcut = scalar 
       pdf_vals = (N,) or (N,S) numpy array, lognormal PDF values given mu and 
                  sigma corresponding to xvals data.
    
       FILES CREATED BY THIS FUNCTION: None
    
       RETURNS: pdf_vals
    '''
    
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * \
                np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))
    return pdf_vals
    
    
# a simple test of the LN_pdf function
xvals = np.array([[200,270],[180,195.5]])

mu_test = 5.0
sigma_test= 1.0
print("1b. The pdf values of test inputs as specified is ",\
      LN_pdf(xvals,mu_test,sigma_test))

'''
---------------------------------------
the following code fragment is for 1(c)
---------------------------------------
'''

def norm_draws(unif_vals,mu,sigma):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from normal distribution based 
    on a normal distribution with mean mu and standard deviation sigma. 
    These draws correspond to an (N x S) matrix of randomly generated 
    draws from a uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()
    
    OBJECTS CREATED WITHIN FUNCTION:
    
    tnorm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from truncated normal PDF with base
                  normal distribution N(mu, sigma)
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''
    tnorm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    return tnorm_draws
    
def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of test scores data
    var_data  = scalar > 0 or (S,) vector, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        var_data = xvals.var()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        var_data = xvals.var(axis=0)
    
    return mean_data, var_data
    
def err_vec(data_vals, sim_vals, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    sim_vals  = (N, S) matrix, S simulations of test scores data
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    var_model  = scalar > 0, estimated variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, var_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_sim, var_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    var_model = var_sim.mean()
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec
 
def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params    = (2,) vector, ([mu, sigma])
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals,unif_vals,W_hat = args
    # transform the uniformally simulated numbers to lognormally
    # simulated numbers by using the normal distribution as medium 
    sim_vals = np.exp(norm_draws(unif_vals, mu, sigma))
    err = err_vec(xvals, sim_vals,simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val
    
# randomly generate a uniformally distributed numbers
S=300
N=200
# set the seed
np.random.seed(seed=1234)
unif_vals = sts.uniform.rvs(0,1,size=(N,S))
#now perform the SMM estimation
mu_init = 11
sigma_init = 0.2
params_init = np.array([mu_init,sigma_init])
w_hat = np.eye(2)

result1 = opt.minimize(criterion,params_init,args=(incomes,unif_vals,w_hat),\
                       method = 'L-BFGS-B',
                       bounds=((1e-10, None), (1e-10, None)))
mu_smm,sigma_smm = result1.x
print('1c: mu_smm = ',mu_smm,'sigma_mu = ', sigma_smm)
# plot the lognormal pdf using smm estimates against the histogram
plt.hist(incomes,bins=30,normed=True,label="histogram of incomes")
plt.title(r'Histogram of incomes v.s. SMM pdf')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students')

x = np.linspace(0,150000,100000)
plt.plot(x,LN_pdf(x,mu_smm,sigma_smm),color='r',\
         label='mu_smm='+str(round(mu_smm,3))+', sigma_smm='+\
         str(round(sigma_smm,3)))
plt.legend(loc='upper left',prop={'size':7})

output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
plt.clf()

print("1c: the value of the SMM criterion function at the given smm"\
      " estimates is ",criterion((mu_smm,sigma_smm), incomes,unif_vals,w_hat)[0][0])

print("1c: data mean = ",incomes.mean(),' data variance = ',incomes.var())  
# to figure out the moments of the simulated data, first regenerate the
# lognormally distributed data

ln_vals = np.exp(norm_draws(unif_vals,mu_smm,sigma_smm))

mean_sim_vector, var_sim_vector = data_moments(ln_vals)
print("1c: Model mean = ",mean_sim_vector.mean(),'Model variance = ',\
      var_sim_vector.mean())


'''
---------------------------------------
the following code fragment is for 1(d)
---------------------------------------
'''

# the 2-step estimator for the optimal weighting matrix
sim_vals = np.exp(norm_draws(unif_vals, mu_smm, sigma_smm))
err = err_vec(incomes, sim_vals,False)
VCV = np.dot(err,err.T)/incomes.shape[0]
w_hat2 = lin.pinv(VCV)

# now run the optimization with the optimal weighting matrix
mu_init = 11
sigma_init = 0.2
params_init = np.array([mu_init,sigma_init])
result2 = opt.minimize(criterion, params_init, args=(incomes,unif_vals,w_hat2),\
                       method = 'Nelder-Mead',\
                       bounds=((None, None), (1e-10, None)))


mu_smm_w, sigma_smm_w = result2.x
print('1d: using optimal weighting matrix, mu_smm = ',mu_smm_w,\
      'mu_sigma = ', sigma_smm_w)

# plot the lognormal pdf using smm estimates from both (c) and (d)
#against the histogram
plt.hist(incomes,bins=30,normed=True,label="histogram of incomes")
plt.title(r'Histogram of incomes v.s. SMM pdf')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students')

x = np.linspace(0,150000,100000)
plt.plot(x,LN_pdf(x,mu_smm,sigma_smm),color='k',\
         label='mu_smm='+str(round(mu_smm,3))+', sigma_smm='+\
         str(round(sigma_smm,3)))
plt.plot(x,LN_pdf(x,mu_smm_w,sigma_smm_w),color='r',\
         label='mu_smm='+str(round(mu_smm_w,3))+', sigma_smm='+\
         str(round(sigma_smm_w,3)))
plt.legend(loc='upper left',prop={'size':7})

output_path = os.path.join(output_dir, 'Fig_1d')
plt.savefig(output_path)
plt.clf()

print("1d: the value of the SMM criterion function at the given smm"\
      " estimates is ",criterion((mu_smm_w,sigma_smm_w), incomes,unif_vals,w_hat2)[0][0])

print("1d: data mean = ",incomes.mean(),' data variance = ',incomes.var())  
# to figure out the moments of the simulated data, first regenerate the
# lognormally distributed data

ln_vals = np.exp(norm_draws(unif_vals,mu_smm_w,sigma_smm_w))

mean_sim_vector, var_sim_vector = data_moments(ln_vals)
print("1d: Model mean = ",mean_sim_vector.mean(),'Model variance = ',\
      var_sim_vector.mean())


