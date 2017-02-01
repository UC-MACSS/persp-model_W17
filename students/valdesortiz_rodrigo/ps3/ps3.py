'''
----------------------------------------------------------------------
PA3 - Perspectives for Computational Modeling
----------------------------------------------------------------------
Student: Rodrigo Valdes Ortiz
----------------------------------------------------------------------
'''


import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import math
import os 
import pandas as pd



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

'''
------------------------------------------------------------------------
Ex. 1: Some income data, lognormal distribution, and GMM
------------------------------------------------------------------------
'''

# Part (A)
# Plot a histogram of percentages of the income.txt data with 30 bins. 
# Make sure that the bins are weighted using the normed=True option. 
# Make sure your plot has correct x-axis and y-axis labels as well as 
# a plot title.

print("**********")
print("Question 1")
print("**********")

# 1.a Histogram
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('Percent of incomes')
plt.xlim([40000, 150000])  # This gives the xmin and xmax to be plotted"
#plt.show()
print("1. Part (A)")
output_path = os.path.join(output_dir, 'one_ps3')
plt.savefig(output_path)
plt.close()


# sts.lognorm.pdf(xvals, s = sigma, scale = np.exp(mu))
# sts.lognorm.pdf(pts, s = 17000, scale = np.exp(85000))

# Define function that generates values of a log-normal pdf
def log_norm_pdf(xvals, mu, sigma):
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
    pdf_vals = (1/(xvals * sigma * np.sqrt(2 * np.pi)) * 
        np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))

    # pdf_vals = sts.lognorm.pdf(xvals, s = sigma, scale = np.exp(mu))
    
    return pdf_vals


def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for GMM
    (mean(data), se(data)).
    --------------------------------------------------------------------
    '''
    mean_data = xvals.mean()
    se_data = xvals.std()
    
    return mean_data, se_data


def model_moments(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), se(model data)).
    --------------------------------------------------------------------
    '''
    # define limits
    inf = 0
    sup = 150000
    xfx = lambda x: x * log_norm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, inf, sup)

    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma) 
    (var_model, v_m_err) = intgr.quad(x2fx, inf, sup)
    
    return mean_model, np.sqrt(var_model)


def err_vec(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    '''
    mean_data, se_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [se_data]])
    mean_model, se_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [se_model]])
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
    '''
    mu, sigma = params
    xvals, W = args
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val

# Part 1.B
mu_init = 11
sig_init = 1
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
gmm_args = (pts, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x
# print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)

mean_data, std_data = data_moments(pts)
mean_model, std_model = model_moments(mu_GMM1, sig_GMM1)


print("1. Part (B)")
print("Te value of the criterion function is", results.fun[0][0])
print("The parameter values are:", "mu:", mu_GMM1, "sigma:", sig_GMM1)
print("The data moments are:", "mu:", mean_data, "sigma:", std_data)
print("The model moments are:", "mu:", mean_model, "sigma:", std_model)

# Plot the histogram of the data
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('Percent of incomes')
plt.xlim([40000, 150000])  # This gives the xmin and xmax to be plotted"

# Plot the estimated GMM PDF
dist_pts = np.linspace(0, 150000, 1000)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1),
         linewidth=2, color='k', label='1: $\mu_{GMM1}$,$\sigma_{GMM1}$')
plt.legend(loc='upper left')
# bins
output_path = os.path.join(output_dir, 'two_ps3')
plt.savefig(output_path)
#plt.show()
plt.close()

# Part 1.C
# First, I need to create the W matrix
err1 = err_vec(pts, mu_GMM1, sig_GMM1, False)
VCV2 = np.dot(err1, err1.T) / pts.shape[0]
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

# Then, optimize again
params_init_c = np.array([mu_GMM1, sig_GMM1])
gmm_args_c = (pts, W_hat2)
results_c = opt.minimize(criterion, params_init_c, args=(gmm_args_c),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM_c, sig_GMM_c = results_c.x
# print('mu_GMM_c=', mu_GMM_c, ' sig_GMM_c=', sig_GMM_c)

mean_model_c, std_model_c = model_moments(mu_GMM_c, sig_GMM_c)

print("1. Part (C)")
print("Mu_2step:", mu_GMM_c, "Sigma_2step:", sig_GMM_c)
print("Te value of the criterion function is", results_c.fun[0][0])
print("The data moments are:", "mu:", mean_data, "sigma:", std_data)
print("The model moments are:", "mu:", mean_model_c, "sigma:", std_model_c)

# Plot the histogram of the data
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('Percent of incomes')
plt.xlim([40000, 150000])  # This gives the xmin and xmax to be plotted"

# Plot the estimated GMM PDF
dist_pts = np.linspace(0, 150000, 1000)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM1, sig_GMM1),
         linewidth=2, color='k', label='1: $\mu_{GMM1}$,$\sigma_{GMM1}$')
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_GMM_c, sig_GMM_c),
         linewidth=2, color='r', label='2: $\mu_{2step}$,$\sigma_{2step}$')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, 'three_ps3')
plt.savefig(output_path)
# plt.show()
plt.close()


# Part 1.D

def data_moments_d(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the three data moments for GMM
    --------------------------------------------------------------------
    '''
    less_75 = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    center_75_100 = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    more_100 = (xvals[xvals >=100000].shape[0] /
                  xvals.shape[0])
    
    return (less_75, center_75_100, more_100)


def model_moments_d(mu, sigma):
    '''
    --------------------------------------------------------------------
    This function computes the two model moments for GMM
    (mean(model data), se(model data)).
    --------------------------------------------------------------------
    '''
    # define limits
    xfx = lambda x: log_norm_pdf(x, mu, sigma)
    (less_75_m, err_1) = intgr.quad(xfx, 0, 75000)
    (center_m, err_2) = intgr.quad(xfx, 75000, 100000)
    (more_100_m, err_3) = intgr.quad(xfx, 100000, 150000)
    
    return less_75_m, center_m, more_100_m

def err_vec_d(xvals, mu, sigma, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for GMM.
    --------------------------------------------------------------------
    '''
    less_75, center, more_100 = data_moments_d(xvals)
    moms_data = np.array([[less_75], [center], [more_100]])
    less_75_m, center_m, more_100_m = model_moments_d(mu, sigma)
    moms_model = np.array([[less_75_m], [center_m], [more_100_m]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)
    
    return err_vec


def criterion_d(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the GMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, W = args
    err = err_vec_d(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err) 
    
    return crit_val


# Part 1.D
mu_init = 11
sig_init = .3
params_init_d = np.array([mu_init, sig_init])
W_hat_d = np.eye(3)
gmm_args = (pts, W_hat_d)
results_d = opt.minimize(criterion_d, params_init_d, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_3m, sig_3m = results_d.x

m1_data, m2_data, m3_data = data_moments_d(pts)
m1_model, m2_model, m3_model = model_moments_d(mu_3m, sig_3m)

print("1. Part (D)")
print("The parameter values are:", "mu:", mu_3m, "sigma:", sig_3m)
print("Te value of the criterion function is", results_d.fun[0][0])
print("The data moments are:", m1_data, m2_data, m3_data)
print("The model moments are:", m1_model, m2_model, m3_model)

# Plot the histogram of the data
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('Percent of incomes')
plt.xlim([40000, 150000])  # This gives the xmin and xmax to be plotted"

# Plot the estimated PDF
dist_pts = np.linspace(0, 150000, 1000)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_3m, sig_3m),
         linewidth=2, color='k', label='1: $\mu_{3moments}$,$\sigma_{3moments}$')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, 'four_ps3')
plt.savefig(output_path)
# plt.show()
plt.close()

# Part 1.E
# First, I need to create the W matrix
err2 = err_vec(pts, mu_3m, sig_3m, False)
VCV2_e = np.dot(err2, err2.T) / pts.shape[0]
W_hat_e = lin.pinv(VCV2_e)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

# print("*********")
# print("Mu_3", mu_3m)
# print("Sigma_3", sig_3m)
# print(err2)
# print(VCV2_e)
# print(W_hat_e)
# print("*********")

# Optimize again
mu_init = 11
sig_init = .3
params_init_e = np.array([mu_3m, sig_3m])
gmm_args = (pts, W_hat_e)
results_e = opt.minimize(criterion, params_init_e, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_e, sig_e = results_e.x


m1_model_e, m2_model_e, m3_model_e = model_moments_d(mu_e, sig_e)

print("1. Part (E)")
print('mu_2step=', mu_e, ' sig_2step=', sig_e)
print("Te value of the criterion function is", results_e.fun[0][0])
print("The data moments are:", m1_data, m2_data, m3_data)
print("The model moments are:", m1_model_e, m2_model_e, m3_model_e)

# Plot the histogram of the data
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Incomes')
plt.ylabel('Percent of incomes')
plt.xlim([40000, 150000])  # This gives the xmin and xmax to be plotted"

# Plot the estimated PDF
dist_pts = np.linspace(0, 150000, 1000)
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_3m, sig_3m),
         linewidth=2, color='k', label='1: $\mu_{3moments}$,$\sigma_{3moments}$')
plt.plot(dist_pts, log_norm_pdf(dist_pts, mu_e, sig_e),
         linewidth=2, color='r', label='2: $\mu_{3mom_{2step}}$,$\sigma_{3mom_{2step}}$')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, 'five_ps3')
plt.savefig(output_path)
# plt.show()
plt.close()

'''
------------------------------------------------------------------------
Ex. 2: Linear regression
------------------------------------------------------------------------
'''
def err_vec_2(xvals, params, simple):
    beta0, beta1, beta2, beta3 = params
    moms_data = np.array(xvals['sick'])
    moms_model = beta0 + beta1 * xvals['age'] + beta2 * xvals['children'] + beta3 * xvals['avgtemp_winter']

    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec

def criterion_2(params, xvals, W):
    err = err_vec_2(xvals, params, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

# Load the data
print("**********")
print("Question 2")
print("**********")
print("2. Part (A)")
sick_data = pd.read_csv('sick.txt', header = 0, names = ['sick', 'age', 'children', 'avgtemp_winter'])

params_init_sick = np.array([0,0,0,0])

W_hat = np.eye(len(sick_data))
gmm_args = (sick_data, W_hat)
results_2 = opt.minimize(criterion_2, params_init_sick, args=((sick_data, W_hat)),
                   method='L-BFGS-B')
beta0_GMM, beta1_GMM, beta2_GMM, beta3_GMM = results_2.x
print("Beta0:", beta0_GMM, "\nBeta1:", beta1_GMM, "\nBeta2:", beta2_GMM, "\nBeta3:", beta3_GMM)
print("Value of GMM criterion function:", results_2.fun)



