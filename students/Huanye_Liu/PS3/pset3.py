#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 19:38:45 2017

@author: huanyeliu
"""

# Import packages and load the data
import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib.pyplot as plt
import os

cur_path = os.path.split(os.path.abspath('pset3.py'))[0]
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
#plt.show()
output_path = os.path.join(output_dir, 'Fig_1a')
plt.savefig(output_path)
plt.clf()

'''
---------------------------------------
the following code fragment is for 1(b)
---------------------------------------
'''

def ln_pdf(x,mu,sigma):
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

    
def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), variance(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of test scores data
    var_data  = scalar > 0, variance of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    var_data = xvals.var()
    
    return mean_data, var_data


    
def model_moments(mu,sigma,cutoff):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), variance(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        ln_pdf(x,mu,sigma)
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    var_model  = scalar > 0, variance of test scores from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, var_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * ln_pdf(x, mu, sigma)
    # split the integral interval into 3 pieces to avoid failing to converge
    (mean_model,m_m_err) = np.add(intgr.quad(xfx,0,50000.0),intgr.quad(xfx,50000.0,100000.0))
    (mean_model, m_m_err) = np.add((mean_model,m_m_err),intgr.quad(xfx,100000.0,cutoff))
    x2fx = lambda x: ((x - mean_model) ** 2) * ln_pdf(x, mu, sigma)
    # split the integral interval into 3 pieces to avoid failing to converge
    (var_model, v_m_err) = np.add(intgr.quad(x2fx,50000.0,100000.0),intgr.quad(x2fx, 100000.0,cutoff))
    (var_model, v_m_err) = np.add(intgr.quad(x2fx,0,50000.0),(var_model,v_m_err))
    
    return mean_model, var_model
    
def err_vec(xvals, mu, sigma,cutoff,simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()
        model_moments()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, var_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_model, var_model = model_moments(mu, sigma,cutoff)
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec
    
def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
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
    xvals,cutoff,W = args
    err = err_vec(xvals, mu, sigma, cutoff, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val
    
# the initial value of mu and sigma
mu_init = 9
sig_init = 0.5

cutoff = 150000

params_init = np.array([mu_init, sig_init])
#set the identity matrix as the weighting matrix
W_hat = np.eye(2)
# the tuple of argments for the criterion function
gmm_args = (incomes,cutoff, W_hat)
# run the optimization using BFGS method
results = opt.minimize(criterion, params_init, args=(gmm_args),\
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))


mu_GMM1, sig_GMM1 = results.x

print('1(b): mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)
# Plot the histogram of the data
plt.title('Histogram of annual incomes')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students' )
plt.hist(incomes,bins=num_of_bin,normed=True,label='Histogram of income data')
plt.xlim([0, 150000])
# Plot the estimated GMM PDF
step = 0.1
start = 0
end = 150000
x = np.arange(start,end,step)

# using the GMM estimate
plt.plot(x,ln_pdf(x,mu_GMM1,sig_GMM1),linewidth=2, color='k', \
         label='The lognormal pdf using the GMM estimates')
plt.legend(loc='upper left',prop={'size':8})
#plt.show()
output_path = os.path.join(output_dir, 'Fig_1b')
plt.savefig(output_path)
plt.clf()
# report the value of your GMM criterion function at the GMM emstimated parameters
params = mu_GMM1, sig_GMM1

print('1(b): The value of the GMM criterion function given the GMM estimated parameters'\
      ' is',criterion(params, incomes,cutoff,W_hat)[0][0])

mean_data, var_data = data_moments(incomes)
mean_model, var_model = model_moments(mu_GMM1, sig_GMM1, cutoff)
print('1(b): Mean of incomes =', mean_data, ', Variance of incomes =', var_data)
print('1(b): Mean of model =', mean_model, ', Variance of model =', var_model)



'''
---------------------------------------
the following code fragment is for 1(c)
---------------------------------------
'''
# calculate the error vector given the GMM estimate using identity matrix as the 
# weighting matrix
err1 = err_vec(incomes, mu_GMM1, sig_GMM1, cutoff, False)
# construct an estimator for the variance covariance matrix
VCV2 = np.dot(err1, err1.T) / incomes.shape[0]

# the inverse of the variance covariance matrix is the optimal weighting matrix
W_hat2 = lin.pinv(VCV2)

params_init = np.array([mu_init, sig_init])
gmm_args = (incomes, cutoff, W_hat2)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='TNC', bounds=((1e-10, None), (1e-10, None)))


mu_GMM2, sig_GMM2 = results.x
print('mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2)

# Plot the histogram of the data
plt.title('Histogram of annual incomes')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students' )
plt.hist(incomes,bins=num_of_bin,normed=True,label='Histogram of income data')
plt.xlim([0, 150000])
# Plot the estimated GMM PDFs and compare 
step = 0.1
start = 0
end = 150000
x = np.arange(start,end,step)

# plot using the GMM estimates in (b) and (c)
plt.plot(x,ln_pdf(x,mu_GMM1,sig_GMM1),linewidth=2, color='k', \
         label='The lognormal pdf using the GMM estimates from (b)')
plt.plot(x,ln_pdf(x,mu_GMM2,sig_GMM2),linewidth=2, color='r', \
         label='The lognormal pdf using the GMM estimates from (c)')
plt.legend(loc='upper left',prop={'size':8})
#plt.show()
output_path = os.path.join(output_dir, 'Fig_1c')
plt.savefig(output_path)
plt.clf()
# report the value of your GMM criterion function at the GMM emstimated parameters
params = mu_GMM2, sig_GMM2

print('1(c): The value of the GMM criterion function given the GMM estimated parameters'\
      ' is',criterion(params, incomes,cutoff,W_hat)[0][0])

mean_data, var_data = data_moments(incomes)
mean_model, var_model = model_moments(mu_GMM2, sig_GMM2, cutoff)
print('1(c): Mean of incomes =', mean_data, ', Variance of incomes =', var_data)
print('1(c): Mean of model =', mean_model, ', Variance of model =', var_model)

'''
---------------------------------------
the following code fragment is for 1(d)
---------------------------------------
'''
def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, income data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_dat = scalar in [0, 1], percent of observations
                 0 <= x < 75000
    bpct_2_dat = scalar in [0, 1], percent of observations
                 75000 <= x < 100000
    bpct_3_dat = scalar in [0, 1], percent of observations
                 x>100000
    

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3, 
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = (xvals[(xvals >=100000)].shape[0] /
                  xvals.shape[0])
    

    return bpct_1_dat, bpct_2_dat, bpct_3_dat

def model_moments3(mu, sigma, cutoff):
    '''
    --------------------------------------------------------------------
    This function computes the three model moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        ln_pdf()
        xfx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_mod = scalar in [0, 1], percent of model observations in
                 bin 1
    bp_1_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_1_mod
    bpct_2_mod = scalar in [0, 1], percent of model observations in
                 bin 2
    bp_2_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_2_mod
    bpct_3_mod = scalar in [0, 1], percent of model observations in
                 bin 3
    bp_3_err   = scalar > 0, estimated error in the computation of the
                 integral for bpct_3_mod
    
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: bpct_1_mod, bpct_2_mod, bpct_3_mod
    --------------------------------------------------------------------
    '''
    xfx = lambda x: ln_pdf(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, cutoff)
    
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod

def err_vec3(xvals, mu, sigma, cutoff, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, test scores data
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments3()
        model_moments3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod =  model_moments3(mu, sigma, cutoff)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    
    return err_vec
  
def criterion3(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (4, 1) matrix, column vector of four moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, cutoff, W = args
    err = err_vec3(xvals, mu, sigma, cutoff, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val
    
mu_init = 9
sig_init = 0.5 
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(3)

gmm_args = (incomes, cutoff, W_hat)
results = opt.minimize(criterion3, params_init, args=(gmm_args),\
                       method='TNC', bounds=((None, None), (1e-10, None)))
mu_GMM3, sig_GMM3 = results.x
print('1(d) mu_GMM3=', mu_GMM3, ' sig_GMM3=', sig_GMM3)

plt.title('Histogram of annual incomes')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students' )
plt.hist(incomes,bins=num_of_bin,normed=True,label='Histogram of income data')
plt.xlim([0, 150000])
# Plot the estimated GMM PDF
step = 0.1
start = 0
end = 150000
x = np.arange(start,end,step)

# using the GMM estimate
plt.plot(x,ln_pdf(x,mu_GMM3,sig_GMM3),linewidth=2, color='k', \
         label='The lognormal pdf using the GMM estimates from (d)')
plt.legend(loc='upper left',prop={'size':8})
#plt.show()
output_path = os.path.join(output_dir, 'Fig_1d')
plt.savefig(output_path)
plt.clf()
# report the value of your GMM criterion function at the GMM emstimated parameters
params = mu_GMM3, sig_GMM3

print('1(d): The value of the GMM criterion function given the GMM estimated parameters'\
      ' is',criterion3(params, incomes,cutoff,W_hat)[0][0])

mean_data1, mean_data2, mean_data3 = data_moments3(incomes)
mean_model1, mean_model2, mean_model3 = model_moments3(mu_GMM3, sig_GMM3, cutoff)
print('Mean of incomes less than 75000 =', mean_data1, ', Mean of incomes in [75000,100000] =',\
      mean_data2, ', Mean of incomes greater than 100000 =', mean_data3)
print('Model mean of incomes than less than 75000 =', mean_model1, \
       ', Model mean of incomes [75000,100000] =',\
      mean_model2, \
      ', Model mean of incomes greater than 100000 =', mean_model3)

'''
---------------------------------------
the following code fragment is for 1(e)
---------------------------------------
'''
err3 = err_vec3(incomes, mu_GMM3, sig_GMM3,cutoff, False)
VCV3 = np.dot(err3, err3.T) / incomes.shape[0]
W_hat3 = lin.pinv(VCV3)
params_init = np.array([mu_GMM3, sig_GMM3])
gmm_args = (incomes,cutoff, W_hat3)
results = opt.minimize(criterion3, params_init, args=(gmm_args),\
                       method='TNC', bounds=((None, None), (1e-10, None)))
						  
mu_GMM4, sig_GMM4 = results.x

print('1(e): mu_GMM4=', mu_GMM4, ' sig_GMM4=', sig_GMM4)
 
# Plot the histogram of the data
plt.title('Histogram of annual incomes')
plt.xlabel(r'Annual income(\$s)')
plt.ylabel(r'Percent of students' )
plt.hist(incomes,bins=num_of_bin,normed=True,label='Histogram of income data')
plt.xlim([0, 150000])
# Plot the estimated GMM PDFs and compare 
step = 0.1
start = 0
end = 150000
x = np.arange(start,end,step)
 
# plot using the GMM estimates in (d) and (e)
plt.plot(x,ln_pdf(x,mu_GMM3,sig_GMM3),linewidth=2, color='k', \
          label='The lognormal pdf using the GMM estimates from (d)')
plt.plot(x,ln_pdf(x,mu_GMM4,sig_GMM4),linewidth=2, color='r', \
          label='The lognormal pdf using the GMM estimates from (e)')
plt.legend(loc='upper left',prop={'size':8})
#plt.show()
output_path = os.path.join(output_dir, 'Fig_1e')
plt.savefig(output_path)
plt.clf()
# report the value of your GMM criterion function at the GMM emstimated parameters
params = mu_GMM4, sig_GMM4
 
print('1(e): The value of the GMM criterion function given the GMM estimated parameters'\
       ' is',criterion3(params, incomes,cutoff,W_hat3)[0][0])
 
mean_data1, mean_data2, mean_data3 = data_moments3(incomes)
mean_model1, mean_model2, mean_model3 = model_moments3(mu_GMM4, sig_GMM4, cutoff)
print('Mean of incomes less than 75000 =', mean_data1, ', Mean of incomes in [75000,100000] =',\
       mean_data2, ', Mean of incomes greater than 100000 =', mean_data3)
print('Model mean of incomes than less than 75000 =', mean_model1, \
        ', Model mean of incomes [75000,100000] =',\
       mean_model2, \
       ', Model mean of incomes greater than 100000 =', mean_model3)

'''
---------------------------------------
the following code fragment is for 2(a)
---------------------------------------
'''

def data_moments4(sick):
    '''
    --------------------------------------------------------------------
    This function computes the 200 data moments each of which is one
    observational value of the variable sick
    --------------------------------------------------------------------
    INPUTS:
    sick = a dataframe of 200 observations

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    sick_data_l =  a numpy 1-d array of values of varaible sick
    
    

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: sick_data_l
    --------------------------------------------------------------------
    '''
    sick_data_l = sick['sick'].values
    

    return sick_data_l

def model_moments4(sick, betas):
    '''
    --------------------------------------------------------------------
    This function computes the 200 model moments each of which is predicted
    sick value of linear regression model.
    --------------------------------------------------------------------
    INPUTS:
    sick   = a dataframe of 200 observations
    betas  = a tuple of four parameters of the linear regression model.
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        None
    
    OBJECTS CREATED WITHIN FUNCTION:
    sick_mod_l = a numpy 1-d array of values of varaible sick
    
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: sick_mod_l
    --------------------------------------------------------------------
    '''
    (beta0,beta1, beta2, beta3) = betas
    sick_mod = beta0+beta1*sick['age']+beta2*sick['children']+beta3*sick['avgtemp_winter']
    sick_mod_l = sick_mod.values
     
    return sick_mod_l

def err_vec4(sick, betas, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    INPUTS:
    sick   = a dataframe of 200 observations
    betas  = a tuple of four parameters of the linear regression model.
    simple = boolean, =True if errors are simple difference, =False if
             errors are percent deviation from data moments
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments4()
        model_moments4()
    
    OBJECTS CREATED WITHIN FUNCTION:
    
    moms_data  = a numpy 1-d array of 200 values of varaible sick 
    
    moms_model = a numpy 1-d array of values of varaible sick
    err_vec    = (200, 1) matrix, column vector of 200 moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    sick_data_l = data_moments4(sick)
    moms_data = np.array([[i] for i in sick_data_l])
    sick_mod_l =  model_moments4(sick,betas)
    moms_model = np.array([[i] for i in sick_mod_l])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    
    return err_vec
  
def criterion4(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params = (2,) vector, ([mu, sigma])
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    args   = length 3 tuple, (xvals, cutoff, W_hat)
    xvals  = (N,) vector, values of the truncated normally distributed
             random variable
    cutoff = scalar or string, ='None' if no cutoff is given, otherwise
             is scalar upper bound value of distribution. Values above
             this value have zero probability
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec4()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (4, 1) matrix, column vector of four moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    beta0,beta1,beta2,beta3 = params
    sick, W = args
    # set simple = True, because the 200 moments are in the same range.
    err = err_vec4(sick,(beta0,beta1,beta2,beta3),simple=True)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val
    
# read the sick data from text file 'sick.txt'    
sick = pd.read_csv('sick.txt',encoding = 'utf-8-sig')
# run the optimization
params_init = np.array([1,0,0,0])
W = np.eye(200)
gmm_args = (sick,W)
results = opt.minimize(criterion4, params_init, args=(gmm_args),\
                       method='L-BFGS-B')
beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results.x
print('2(a) beta0_GMM=',beta0_GMM, 'beta1_GMM=',beta1_GMM, \
      'beta2_GMM=',beta2_GMM, 'beta3_GMM=',beta3_GMM) 
# the value of the GMM criterion function
params = beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM
print('2(a) the value of the GMM criterion function is',\
      criterion4(params, sick,W)[0][0])