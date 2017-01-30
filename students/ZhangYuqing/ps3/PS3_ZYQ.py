
#Yuqing Zhang
#Problem Sets 3

import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
np.seterr(all='ignore')
...
graph = True
...
if graph:
    '''
    --------------------------------------------------------------------
    cur_path    = string, path name of current directory
    output_fldr = string, folder in current path to save files
    output_dir  = string, total path of images folder
    output_path = string, path of file name of figure to be saved
    
    --------------------------------------------------------------------
    '''
    # Create directory if images directory does not already exist
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = 'images'
    output_dir = os.path.join(cur_path, output_fldr)
    if not os.access(output_dir, os.F_OK):
        os.makedirs(output_dir)

income = np.loadtxt('incomes.txt')
mu_init = 11.0
sig_init = 0.2
'''
------------------------------------------------
Exercise 1a: Plot a histogram of the income data
------------------------------------------------
'''
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
# fig1a, ax1a = plt.subplots()

plt.hist(income, num_bins, weights = weights,normed=True)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')
    
output_path = os.path.join(output_dir, '1a_inc_hist')
plt.savefig(output_path)
#plt.show()
plt.close

'''
------------------------------------------------------------
Exercise 1b: Estimate the parameters of the lognormal distribution 
by generalized method of moments.
------------------------------------------------------------
'''
def trunc_lognorm_pdf(xvals, mu_init, sigma_init):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the truncated normal pdf with mean mu and
    standard deviation sigma. If there is no cutoff given or if it is given as
    infinity, this function does the same thing as
    sp.stats.norm.pdf(x, loc=mu, scale=sigma).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N,) vector, values of the normally distributed random
             variable
    mu_init     = scalar, mean of the normally distributed random variable
    sigma_init  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    pdf_vals = (N,) vector, normal PDF values for mu and sigma
               corresponding to xvals data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
   
            
    log_pdf_vals    = ((1/(xvals*sigma_init * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu_init)**2 / (2 * sigma_init**2)))) 
    return log_pdf_vals


def data_moments(xvals): #take the data and calculate the mean and variance
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), standard deviation(data)).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar, mean value of test scores data
    std_data  = scalar > 0, standard deviation of test scores data
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_data, std_data
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    std_data = xvals.std()
    return mean_data, std_data


def model_moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), std(model data)).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_lognorm_pdf()
        xfx()
        x2fx()
    
    OBJECTS CREATED WITHIN FUNCTION:
    mean_model = scalar, mean value of test scores from model
    m_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the mean of the distribution
    std_model  = scalar > 0, standard deviation of test scores from model
    v_m_err    = scalar > 0, estimated error in the computation of the
                 integral for the variance of the distribution
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: mean_model, std_model
    --------------------------------------------------------------------
    '''
    xfx = lambda x: x * trunc_lognorm_pdf(x, mu, sigma)
   
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)

    x2fx = lambda x: ((x - mean_model) ** 2) * trunc_lognorm_pdf(x, mu, sigma) 
    
    (std_model, v_m_err) = np.sqrt(intgr.quad(x2fx, 0, 150000)) 
   
    return mean_model, std_model


def err_vec(xvals, mu, sigma, simple):
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
    std_data   = scalar > 0, standard deviation of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    std_model  = scalar > 0, standard deviation from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: err_vec
    --------------------------------------------------------------------

    '''
    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = model_moments(mu, sigma) 
    moms_model = np.array([[mean_model], [std_model]])
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
    xvals, W = args
    err = err_vec(xvals, mu, sigma,simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

#Estimate the parameters using identity matrix
params_init = np.array([mu_init, sig_init])
W_hat_1b = np.eye(2)
gmm_args_1b = (income, W_hat_1b)
results_1b = opt.minimize(criterion, params_init, args=(gmm_args_1b),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results_1b.x

print('Exercise 1b:')
#Report the estimates
print('The estimators mu_GMM1 is:',mu_GMM1)
print('The estimators sig_GMM1 is:',sig_GMM1)

#Report and compare two data moments against two model moments at the estimated parameter values
mean_data, std_data = data_moments(income)
mean_model, std_model = model_moments(mu_GMM1, sig_GMM1)
err1 = err_vec(income, mu_GMM1, sig_GMM1, False).reshape(2,)
print('Mean of points =', mean_data, ', Standard Deviation of points =', std_data)
print('Mean of model =', mean_model, ', Standard Deviation of model =', std_model)

#Report the value of your GMM criterion function
value_1b = criterion(([mu_GMM1,sig_GMM1]),*gmm_args_1b)
print('The GMM criterion value at mu_GMM1 and sig_GMM1 is:',value_1b)
print()
# Plot the estimated lognormal PDF against the histogram from part (a)
# 1. Plot the histogram of the data
fig1b, ax1b = plt.subplots()
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
plt.hist(income, num_bins, weights = weights,normed=True)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')

# 2. Plot the estimated GMM PDF

dist_inc = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM1, sig_GMM1),
         linewidth=2, color='k', label='1: $\mu_{GMM1}$,$\sigma_{GMM1}$')
plt.legend(loc='upper right')

'''
------------------------------------------------------------
Exercise 1c: Perform the two-step GMM estimator 
by using estimators mu_GMM1,sig_GMM1
------------------------------------------------------------
'''
err1 = err_vec(income, mu_GMM1, sig_GMM1, False) #use the GMM to generate the VCV
VCV2 = np.dot(err1, err1.T) / income.shape[0]
W_hat_1c = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

gmm_args_1c = (income, W_hat_1c)
results_1c = opt.minimize(criterion, params_init, args=(gmm_args_1c),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results_1c.x

print('Exercise 1c:')
#Report the estimates
print('The estimators mu_GMM2 is:',mu_GMM2)
print('The estimators sig_GMM2 is:',sig_GMM2)

#Report and compare two data moments against two model moments at the estimated parameter values
mean_model_1c, std_model_1c = model_moments(mu_GMM2, sig_GMM2)
err2 = err_vec(income, mu_GMM2, sig_GMM2, False).reshape(2,)
print('Mean of points =', mean_data, ', Standard Deviation of points =', std_data)
print('Mean of model =', mean_model_1c, ', Standard Deviation of model =', std_model_1c)

#Report the value of your GMM criterion function
value_1c = criterion(([mu_GMM2,sig_GMM2]),*gmm_args_1c)
print('The GMM criterion value at mu_GMM2 and sig_GMM2 is:',value_1c)
print()

#Plot your estimated lognormal PDF against the histogram from part (a) 
#and the estimated PDF from part (b)
# 1. Plot the histogram of the data
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
fig1c, ax1c = plt.subplots()
plt.hist(income, num_bins, weights = weights,normed=True)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')

#2. Plot the estimated GMM PDF

dist_inc = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM1, sig_GMM1),
         linewidth=2, color='k', label='1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM1, sig_GMM1))
plt.legend(loc='upper right')

#Plot the estimate GMM PDF using two step estimator
plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM2, sig_GMM2),
         linewidth=2, color='r', label='2: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM2, sig_GMM2))
plt.legend(loc='upper right')


'''
------------------------------------------------------------
Exercise 1d: estimate the lognormal PDF to fit the data 
by GMM using 3 moments.
------------------------------------------------------------
'''
def data_moments3(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N,) vector, test scores data

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1_dat = scalar in [0, 1], percent of observations
                 0 <= x < 75000
    bpct_2_dat = scalar in [0, 1], percent of observations
                 75000 <= x < 100000
    bpct_3_dat = scalar in [0, 1], percent of observations
                 100000 <= x 
 

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3
    --------------------------------------------------------------------
    '''
    bpct_1_dat = xvals[(xvals < 75000) & (xvals >= 0)].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = (xvals[(xvals >=100000) & (xvals < 150000)].shape[0] /
                  xvals.shape[0])

    return bpct_1_dat, bpct_2_dat, bpct_3_dat


def model_moments3(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the four model moments for GMM
    (binpct_1, binpct_2, binpct_3).
    --------------------------------------------------------------------
    INPUTS:
    mu     = scalar, mean of the normally distributed random variable
    sigma  = scalar > 0, standard deviation of the normally distributed
             random variable
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        trunc_norm_pdf()
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
    xfx = lambda x: trunc_lognorm_pdf(x, mu, sigma) 
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000) 
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 1500000)
    
    
    return bpct_1_mod, bpct_2_mod, bpct_3_mod


def err_vec3(xvals, mu, sigma, simple):
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
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
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
    
    W_hat  = (R, R) matrix, estimate of optimal weighting matrix
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        err_vec3()
    
    OBJECTS CREATED WITHIN FUNCTION:
    err        = (3, 1) matrix, column vector of four moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals,W = args
    err = err_vec3(xvals, mu, sigma,simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


print('Exercise 1d: ')
W_hat_1d = np.eye(3)

gmm_args_1d = (income, W_hat_1d)
results_1d = opt.minimize(criterion3, params_init, args=(gmm_args_1d),
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM3, sig_GMM3 = results_1d.x

#Report the estimates
print('mu_GMM3 using 3 moments is:',mu_GMM3)
print('sig_GMM3 using 3 moments is: ',sig_GMM3)

#Report the value of your GMM criterion function
value_1d = criterion3(([mu_GMM3,sig_GMM3]),*gmm_args_1d)[0][0]
print('The GMM criterion value at mu_GMM3 and sig_GMM3 is:',value_1d)

#Report and compare three data moments against 
#your three model moments at the estimated parameter values
inc_75,inc75_10,inc_10 = data_moments3(income)
inc_75_m,inc75_10_m,inc_10_m = model_moments3(mu_GMM3,sig_GMM3)
print('From data: ')
print('percent of individuals who earn less than $75,000 is: ', inc_75, 
    ', percent of individuals who earn between $75,000 and $100,000 is: ', inc75_10,
    ', percent of individuals who higher than $100,000 is: ', inc_10)
print('From model: ')
print('percent of individuals who earn less than $75,000 is: ', inc_75_m, 
    ', percent of individuals who earn between $75,000 and $100,000 is: ', inc75_10_m,
    ', percent of individuals who higher than $100,000 is: ', inc_10_m)
print()
#Plot your estimated lognormal PDF against the histogram from part (a) 
#1. Plot the histogram of the data
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
fig1d, ax1d = plt.subplots()
plt.hist(income, num_bins, weights = weights,normed=True)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')

#2. Plot the estimated GMM PDF using 3 moments

dist_inc = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM3, sig_GMM3),
         linewidth=2, color='y', label= '1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM3, sig_GMM3))
plt.legend(loc='upper right')

'''
------------------------------------------------------------
Exercise 1e: Perform the two-step GMM estimator 
by using your estimates from part (d) 
------------------------------------------------------------
'''
err3 = err_vec3(income, mu_GMM3, sig_GMM3, False) #use the GMM to generate the VCV
VCV3 = np.dot(err3, err3.T) / income.shape[0]
W_hat_1e = lin.pinv(VCV3)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
gmm_args_1e = (income, W_hat_1e)
results_1e = opt.minimize(criterion3, params_init, args=(gmm_args_1e),
                       method='TNC', bounds=((None, None), (1e-10, None)))
mu_GMM4, sig_GMM4 = results_1e.x

print('Exercise 1e: ')
#Report the estimates
print('mu_GMM4 using 2 step is:',mu_GMM4)
print('sig_GMM4 using 2 step is: ',sig_GMM4)

#Report the value of your GMM criterion function
params_GMM4 = np.array([mu_GMM4,sig_GMM4])
value_1e = criterion3(params_GMM4,*gmm_args_1e)[0][0]
print('The GMM criterion value at mu_GMM4 and sig_GMM4 is:',value_1e)

#Report and compare three data moments against 
#your three model moments at the estimated parameter values
inc_75_m_1e,inc75_10_m_1e,inc_10_m_1e = model_moments3(mu_GMM4,sig_GMM4)
print('From data: ')
print('percent of individuals who earn less than $75,000 is: ', inc_75, 
    ', percent of individuals who earn between $75,000 and $100,000 is: ', inc75_10,
    ', percent of individuals who higher than $100,000 is: ', inc_10)
print('From model: ')
print('percent of individuals who earn less than $75,000 is: ', inc_75_m_1e, 
    ', percent of individuals who earn between $75,000 and $100,000 is: ', inc75_10_m_1e,
    ', percent of individuals who higher than $100,000 is: ', inc_10_m_1e)
print()


# Plot the histogram of the data
weights = (1 / income.shape[0]) * np.ones_like(income) * 100
num_bins = 30
fig1e, ax1e = plt.subplots()
plt.hist(income, num_bins, weights = weights,normed=True)
plt.title('Income Distribution of MACSS Graduates', fontsize=20)
plt.xlabel('Income(\$)')
plt.ylabel('Percent of graduates ')

# Plot the estimated GMM PDF using 3 moments

dist_inc = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM3, sig_GMM3),
         linewidth=2, color='y', label= '1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM3, sig_GMM3))
plt.legend(loc='upper right')

#Plot the estimated GMM PDF by performing the two step estimator by using your estimates from part (d) with three moments 
dist_inc = np.linspace(0.000001, 150000, 150000)
plt.plot(dist_inc, trunc_lognorm_pdf(dist_inc, mu_GMM4, sig_GMM4),
         linewidth=2, color='g', label='1: $\mu$ = {:.4f}, $\sigma$ = {:.4f}'.format(mu_GMM4, sig_GMM4))
plt.legend(loc='upper right')

'''
------------------------------------------------------------
Exercise 1f: comparison between different method
------------------------------------------------------------
'''
print('Exercise 1f: ')
print('After plotting out the figures, I think the figures from part 1b, 1d, and 1e',
'fit with the histogram of the income data well. Among these three figure, (Fig 1e),the one',
'from 2-step GMM with three data moments fits the best. This pdf has the highest peak in the middle,',
'and slightly right skewed, which I think fits with the original data.')
print()
'''
------------------------------------------------------------
Exercise 2a: Estimate the parameters of the model ( 0,  1,  2,  3) 
by GMM by solving the minimization problem of the GMM criterion function
------------------------------------------------------------
'''
sick = np.loadtxt('sick.txt',skiprows=1,delimiter=',')
sick_i = sick[:,0]
age_i = sick[:,1]
children_i=sick[:,2]
temp_winter_i=sick[:,3]

def data_moments2(xvals):
    lst = []
    for i in range(len(xvals)):
        lst.append(sick_i[i])
    return lst
def model_moments2(xvals,beta0,beta1,beta2,beta3):
    lst = []
    for i in range(len(xvals)):
        model_moment = beta0 + beta1 *age_i[i] + beta2 * children_i[i] + beta3 * temp_winter_i[i]
        lst.append(model_moment)
    return lst
def err_vec2(xvals,beta0,beta1,beta2,beta3):
   
    moms_data = np.array(data_moments2(xvals))
    moms_model = np.array(model_moments2(xvals,beta0,beta1,beta2,beta3))
    err_vec2 = (moms_model - moms_data) 
    return err_vec2
def criterion_2(params, *args): 
   
    beta0,beta1,beta2,beta3 = params
    xvals, W = args
    err = err_vec2(xvals,beta0,beta1,beta2,beta3)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


beta0_init=0.1
beta1_init=0.1
beta2_init=0.1
beta3_init=0.1
params_init_2 = np.array([beta0_init,beta1_init,beta2_init,beta3_init])
W_hat = np.eye(200)
gmm_args = (income,W_hat)
results_2 = opt.minimize(criterion_2, params_init_2, args=(gmm_args),
                       method='L-BFGS-B')
beta0_GMM,beta1_GMM,beta2_GMM,beta3_GMM = results_2.x
print('Exercise 2a: ' )
#Report the estimates
print('beta0_GMM=', beta0_GMM, 'beta1_GMM=', beta1_GMM,'beta2_GMM=', beta2_GMM,'beta3_GMM=', beta3_GMM)

#Report value of GMM criterion function
params_GMM_2 = np.array([beta0_GMM,beta1_GMM,beta2_GMM,beta3_GMM])
value5 = criterion_2(params_GMM_2,*gmm_args)
print('The GMM criterion value for linear regression is:',value5)



