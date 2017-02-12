#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:32:02 2017

@author: luxihan
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
import scipy.stats as sts
import scipy.integrate as intgr
import pandas as pd
import os
#np.set_printoptions(precision = 5)

#####Problem 1#####
##Part a)
#Plot the Histogram
income = np.loadtxt("incomes.txt", comments="#", delimiter=",", unpack=False)
count, bins, patches = plt.hist(income, 30, normed = True)
plt.xlabel("Income")
plt.ylabel("Density")
plt.title("Income for MACSS Students Histogram", fontsize = 20)
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = "images"
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "hist_income")
plt.savefig(output_path, bbox_inches='tight')
plt.show()

##Part b)##
def LN_pdf(xvals, mu, sigma):
    '''
    Given the x values, the distribution mean and standard deviation
    return the pdf of the log normal pdf
    
    Input:
        xvals; numpy array or float.
        mu: the expectation of the log normal distribution
        sigma: the standard deviation of the log normal distribution
        
    Output:
        pdf_val: the pdf value of the x values
    '''
    ln_xvals = np.log(xvals)
    pdf_val = 1/(xvals * sigma * np.sqrt(2 * np.pi)) *\
                    np.exp( - (ln_xvals - mu)**2 / (2 * sigma**2))
    return pdf_val

def LN_normal_draw(unif_val, mu, sigma):
    '''
    Given the uniform distributed values, the distribution mean 
    and standard deviation, return the value of the log normal distribution
    draw
    
    Input:
        unif_val: numpy array. N by S matrix. A draw from the uniform distribution
        mu: the expectation of the log normal distribution
        sigma: the standard deviation of the log normal distribution
        
    Output:
        rv: the value of the log normal distribution
    '''
    rv = sts.norm.ppf(unif_val, mu, sigma)
    rv = np.exp(rv)
    return rv

def data_moment(xvals):
    '''
    Given the data, report the data moment
    
    Input:
        xvals: numpy array the data
        
    Output:
       np.array([mean_data, var_data]): the data moment 
    '''
    mean_data = xvals.mean()
    var_data = xvals.var()
    
    return np.array([mean_data, var_data])

def sim_moment(unif_val, mu, sigma):
    '''
    Given the uniform distributed values, the distribution mean 
    and standard deviation, return simulated moments
    draw
    
    Input:
        unif_val: numpy array. N by S matrix. A draw from the uniform distribution
        mu: the expectation of the log normal distribution
        sigma: the standard deviation of the log normal distribution
        
    Output:
        np.array([mean_sim, var_sim]): the simualted moments
    '''
    LN_norm_val = LN_normal_draw(unif_val, mu, sigma)
    mean_sim = LN_norm_val.mean()
    var_sim = LN_norm_val.var(axis = 0).mean()
    
    return np.array([mean_sim, var_sim])

def err_vec(xvals, unif_val, mu, sigma):
    '''
    Given the data, the uniform distributed values, the distribution mean 
    and standard deviation, return simulated moments
    draw
    
    Input:
        xvals: numpy array, the data
        unif_val: numpy array. N by S matrix. A draw from the uniform distribution
        mu: the expectation of the log normal distribution
        sigma: the standard deviation of the log normal distribution
        
    Output:
        err: numpy array, the difference between the data and simulated moments
    '''
    data_mom = data_moment(xvals)
    sim_mom = sim_moment(unif_val, mu, sigma)
    err = (data_mom - sim_mom) / data_mom
    err = err.reshape(2,1)
    return err

def criterion(params, *args):
    '''
    Given the parameter values, the data and the weighting matrix, return the
    criterion value to minimize over
    
    Input:
        params: mu and sigma
        *args: data, weighting matrix
        
    Output:
        crit_val: float, the criterion value to minimize over
    '''
    mu, sigma = params
    xvals, unif_val, W = args
    err = err_vec(xvals, unif_val, mu, sigma)
    crit_val = np.dot(np.dot(err.T, W), err)
    return crit_val

##Initialization
mu_init  = 9
sigma_init = 0.2
params_init = np.array([mu_init, sigma_init])
W_hat = np.eye(2)
np.random.seed(1234)
N = 200
S = 300
unif_val = sts.uniform.rvs(0, 1, size=(N, S))
smm_args = (income, unif_val, W_hat)
#result = opt.minimize(criterion, params_init, args=(smm_args), tol = 1e-16,
#                       method='Nelder-Mead')
result = opt.minimize(criterion, params_init, args=(smm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_smm, sigma_smm = result.x
err_one = err_vec(income, unif_val, mu_smm, sigma_smm)
crit_one = criterion(result.x, income, unif_val, W_hat)

##PLOT the PDF for one step estimator
dist_pers = np.linspace(0, 140000, 1000000)
count, bins, patches = plt.hist(income, 30, normed = True)
plt.plot(dist_pers, LN_pdf(dist_pers, mu_smm, sigma_smm), \
         label = 'Orginal, $\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_smm, sigma_smm))
plt.xlabel("Income")
plt.ylabel("Density")
plt.legend(loc = "upper left", prop={'size':8})
plt.title('Income PDF SMM', fontsize = 15)
output_path = os.path.join(output_dir, "1c")
plt.savefig(output_path, bbox_inches='tight')


W_smm = np.linalg.pinv(1/len(income) * err_one @ err_one.T)
params_init = np.array([mu_smm, sigma_smm])
smm_args = (income, unif_val, W_smm)
#result_two = opt.minimize(criterion, params_init, args=(smm_args), tol = 1e-16,
#                    method = 'L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
result_two = opt.minimize(criterion, params_init, args=(smm_args), tol = 1e-14, \
                          method = 'Nelder-Mead')
mu_two, sigma_two = result_two.x
err_two = err_vec(income, unif_val, mu_two, sigma_two)
crit_two = criterion(result_two.x, income, unif_val, W_smm)
##Plot the TWO STEP PDF
plt.plot(dist_pers, LN_pdf(dist_pers, mu_two, sigma_two), c = 'y',\
         label = 'TWO-STEP, $\mu$ = {:.3f}, $\sigma$ = {:.3f}'.format(mu_two, sigma_two))
plt.legend(loc = "upper left", prop={'size':8})
plt.title('Income PDF SMM(TWO STEP)', fontsize = 15)
output_path = os.path.join(output_dir, "1d")
plt.savefig(output_path, bbox_inches='tight')
plt.show()


##Print Statement
test_value = LN_pdf(np.array([[200.0, 270.0], [180.0,195.5]]), 5 ,1)
print("1b): the log normal pdf value for the test case is {}"
     .format(test_value))
print("1c): The one step SMM estimation is: mu = {:.3f}, sigma = {:.3f}"\
     .format(mu_smm, sigma_smm))
print("1c): the exponential version is: mu = {:.3f}, sigma = {:.3f}"\
     .format(math.e**(mu_smm + 1/2*sigma_smm**2), \
             (math.e ** (2 * mu_smm + sigma_smm ** 2)*(math.e ** (sigma_smm ** 2)-1))**0.5))
print("1d): the TWO step weighting matrix is: ")
print(W_smm)
print("1d): The TWO step SMM estimation is: mu = {:.3f}, sigma = {:.3f}"\
     .format(mu_two, sigma_two))
print("1c): the exponential version is: mu = {:.3f}, sigma = {:.3f}"\
     .format(math.e**(mu_two + 1/2*sigma_two**2), \
             (math.e ** (2 * mu_two + sigma_two ** 2)*(math.e ** (sigma_two ** 2)-1))**0.5))
