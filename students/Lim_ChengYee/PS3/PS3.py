import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
cmap1 = matplotlib.cm.get_cmap('summer')
import pandas as pd
import os

#Question 1
print('Question 1')
#loading dataset
income = np.loadtxt('incomes.txt')

#part (a)
print('Part (a)')
plot = True

if plot:
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

    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.1, box.y0, box.width*0.9, box.height*0.9])

    #plotting histogram, normed = TRUE
    count, bins, patches = plt.hist(income, 30, normed = True)

    #labelling histogram
    plt.title('MACS Graduate Annual Incomes Normed Histogram', fontsize=12)
    plt.xlabel('Annual Income')
    plt.ylabel('PDF values of Income')

    output_path = os.path.join(output_dir, '1a')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

#defining function that generates values of a log normal pdf
def log_norm_pdf(xvals, mu, sigma):
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                   np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))

    return pdf_vals

#computing data moments i.e. mean and variance for GMM
def data_moments(xvals):

    mean_data = xvals.mean()
    std_data = xvals.std()

    return mean_data, std_data

#computing model moments for GMM
#model specified as lognormal pdf
def model_moments(mu, sigma):

    xfx = lambda x: x * log_norm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, 150000)
    x2fx = lambda x: ((x - mean_model) ** 2) * log_norm_pdf(x, mu, sigma)
    (var_model, v_m_err) = intgr.quad(x2fx, 0, 150000)
    std_model = np.sqrt(var_model)

    return mean_model, std_model

#computes the difference between the vector of moment errors for GMM
#simple = boolean, =True if errors are simple difference, =False if errors are percent deviation
def err_vec(xvals, mu, sigma, simple):

    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = model_moments(mu, sigma)
    moms_model = np.array([[mean_model], [std_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec

#compute the GMM weighted sum of squared moment errors criterion function given parameter values and weighting matrix
def criterion(params, *args):

    mu, sigma = params
    xvals, W = args
    err = err_vec(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

print('Part (b)')
#make initial guessses of mu and sigma
mu_init = 11
sig_init = 0.2
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2) #set weighting matrix to identity matrix
gmm_args = (income, W_hat) #gmm args specifies dataset, cutoff and weighting matrix
#optimise to obtain mu and sigma
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x
print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)
crit_1b = criterion(results.x, *gmm_args)
print('The GMM criterion function is', crit_1b[0][0])
mu_data_1b, sig_data_1b = data_moments(income)
print('Two data moments are mu: {:.3f} and sigma: {:.3f}'.format(mu_data_1b, sig_data_1b))
mu_model_1b, sig_model_1b = model_moments(mu_GMM1, sig_GMM1)
print('Two model moments are mu: {:.3f} and sigma: {:.3f}'.format(mu_model_1b, sig_model_1b))
print('The data and model moments are very similar, thus the GMM estimation is a good estimation.')

plot = True
if plot:
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
        
    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.1, box.y0, box.width*0.9, box.height*0.9])

    # Plot the estimated GMM PDF
    dist_income = np.linspace(0, 150000, 200)
    plt.plot(dist_income, log_norm_pdf(dist_income, mu_GMM1, sig_GMM1),
             linewidth=2, color='y', label='1: $\mu=11.3369$,$\sigma=0.2130$')
    plt.legend(loc='upper left', fontsize=10)
    #plotting histogram, normed = TRUE
    count, bins, patches = plt.hist(income, 30, normed = True)
    #labelling histogram
    plt.title('MACS Graduate Annual Incomes: 2018-2020', fontsize=12)
    plt.xlabel('Annual Income')
    plt.ylabel('PDF value of Income')

    output_path = os.path.join(output_dir, '1b')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

# In[417]:

#part (c) two step GMM estimator
print('Part (c)')
err1 = err_vec(income, mu_GMM1, sig_GMM1, simple = False)
VCV2 = np.dot(err1, err1.T) / income.shape[0]
#optimal weighting matrix
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

params_init = np.array([mu_GMM1, sig_GMM1])
gmm_args = (income, W_hat2)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((None, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results.x
print('mu_GMM2=', mu_GMM2, ' sig_GMM2=', sig_GMM2)

crit_1c = criterion(results.x, *gmm_args)
print('The GMM criterion function is', crit_1c[0][0])
mu_data_1b, sig_data_1b = data_moments(income)
print('Two data moments are mu: {:.3f} and sigma: {:.3f}'.format(mu_data_1b, sig_data_1b))
mu_model_1c, sig_model_1c = model_moments(mu_GMM2, sig_GMM2)
print('Two model moments are mu: {:.3f} and sigma: {:.3f}'.format(mu_model_1c, sig_model_1c))
print('The data and model moments are very similar, thus the GMM estimation is a good estimation.')

plot = True
if plot:
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
        
    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.1, box.y0, box.width*0.9, box.height*0.9])

    plt.plot(dist_income, log_norm_pdf(dist_income, mu_GMM2, sig_GMM2),
             linewidth=2, color='r', label='$\mu = 11.3369101  $,$\sigma=0.2130273$')
    plt.legend(loc='upper left', fontsize=8)
    #part (b) lognormal curve
    plt.plot(dist_income, log_norm_pdf(dist_income, mu_GMM1, sig_GMM1),
             linewidth=2, color='y', label='$\mu=11.3369100  $,$\sigma=0.2130271$')
    plt.legend(loc='upper left', fontsize=8)
    #plotting histogram, normed = TRUE
    count, bins, patches = plt.hist(income, 30, normed = True)
    #labelling histogram
    plt.title('MACS Graduate Annual Incomes: 2018-2020', fontsize=12)
    plt.xlabel('Annual Income')
    plt.ylabel('PDF values of Income')

    output_path = os.path.join(output_dir, '1c')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

#part(d)
print('Part (d)')
#defining function that generates values of a log normal pdf
def log_norm_pdf3(xvals, mu, sigma):
    pdf_vals    = (1/(xvals * sigma * np.sqrt(2 * np.pi)) *
                    np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2)))

    return pdf_vals

#computing three data moments 1. less than 75000, 2. between 75000 and 100000, 3. more than 100000
def data_moments3(xvals):

    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >=75000) & (xvals < 100000)].shape[0] /
                  xvals.shape[0])
    bpct_3_dat = (xvals[(xvals >=100000) & (xvals <150000)].shape[0] /
                  xvals.shape[0])

    return bpct_1_dat, bpct_2_dat, bpct_3_dat

# three model moments based on specified model: lognormal pdf
def model_moments3(mu, sigma):

    xfx = lambda x: log_norm_pdf3(x, mu, sigma)
    (bpct_1_mod, bp_1_err) = intgr.quad(xfx, 0, 75000)
    (bpct_2_mod, bp_2_err) = intgr.quad(xfx, 75000, 100000)
    (bpct_3_mod, bp_3_err) = intgr.quad(xfx, 100000, 150000)

    return bpct_1_mod, bpct_2_mod, bpct_3_mod

#computes percent deviation of vector of moment errors for GMM
def err_vec3(xvals, mu, sigma, simple):
    bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = 100 * ((moms_model - moms_data) / moms_data)

    return err_vec

#GMM weighted sum of squared errors for given parameters and weighting matrix
def criterion3(params, *args):
    mu, sigma = params
    xvals, W = args
    err = err_vec3(xvals, mu, sigma, simple=False)
    crit_val = np.dot(np.dot(err.T, W), err)

    return crit_val

mu_init3 = 11
sig_init3 = 0.2
params_init = np.array([mu_init3, sig_init3])
W_hat = np.eye(3)
gmm_args = (income, W_hat)
bounds =((None, None), (1e-10, None))

results_3 = opt.minimize(criterion3, params_init, args=(gmm_args), method='L-BFGS-B', bounds=bounds)
mu_GMM1_3, sig_GMM1_3 = results_3.x
print('mu_GMM1_3=', mu_GMM1_3, ' sig_GMM1_3=', sig_GMM1_3)
crit_1d = criterion3(results_3.x, *gmm_args)[0][0]
print('The value of the GMM criterion is', crit_1d)
print('The three data moments are:', data_moments3(income))
print('The three model moments are:', model_moments3(mu_GMM1_3, sig_GMM1_3))
print('Data and model moments match very closely again, this shows that the GMM estimation was a good one.')

plot = True
if plot:
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

    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.1, box.y0, box.width*0.9, box.height*0.9])

    # plot (a) and (d) together
    # part (a)
    count, bins, patches = plt.hist(income, 30, normed = True)
    plt.title('MACS Graduate Annual Incomes: 2018-2020', fontsize=12)
    plt.xlabel('Annual Income')
    plt.ylabel('PDF values of Income')

    dist_income = np.linspace(0, 150000, 200)
    # part (d)
    plt.plot(dist_income, log_norm_pdf3(dist_income, mu_GMM1_3, sig_GMM1_3),
             linewidth=2, color='y', label='$\mu=11.337$  ,$\sigma=0.212$')
    plt.legend(loc='upper left', fontsize = 10)

    output_path = os.path.join(output_dir, '1d')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

#part (e)
print('Part (e)')
#constructing the optimal weight matrix
err1_3 = err_vec3(income, mu_GMM1_3, sig_GMM1_3, True)
VCV2_3 = np.dot(err1_3, err1_3.T) / income.shape[0]
W_hat2_3 = lin.pinv(VCV2_3)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
params_init = np.array([mu_GMM1_3, sig_GMM1_3])
gmm_args = (income, W_hat2_3)
results2_3 = opt.minimize(criterion3, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds = bounds)
mu_GMM2_3, sig_GMM2_3 = results2_3.x
print('mu_GMM2_3=', mu_GMM2_3, ' sig_GMM2_3=', sig_GMM2_3)
crit_1e = criterion3(results2_3.x, *gmm_args)[0][0]
print('The value of the GMM criterion is', crit_1e)
print('The three data moments are:', data_moments3(income))
print('The three model moments are:', model_moments3(mu_GMM2_3, sig_GMM2_3))
print('Data and model moments match pretty closely again, this shows that the GMM estimation was a good one.')

plot = True
if plot:
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
    
    fig, ax = plt.subplots(1, 1)

    box = ax.get_position()
    ax.set_position([box.x0+0.1, box.y0, box.width*0.9, box.height*0.9])

    # Plot all graphs
    # part (a)
    count, bins, patches = plt.hist(income, 30, normed = True)
    plt.title('MACS Graduate Annual Incomes: 2018-2020', fontsize=12)
    plt.xlabel('Annual Income')
    plt.ylabel('PDF value of Income')

    dist_income = np.linspace(0, 150000, 200)
    # part (d)
    plt.plot(dist_income, log_norm_pdf3(dist_income, mu_GMM1_3, sig_GMM1_3),
             linewidth=2, color='y', label='$\mu=11.337$,$\sigma=0.212$')
    plt.legend(loc='upper left', fontsize=10)
    # part (e)
    plt.plot(dist_income, log_norm_pdf3(dist_income, mu_GMM2_3, sig_GMM2_3),
             linewidth=2, color='m', label='$\mu=11.327$,$\sigma=0.2105$')
    plt.legend(loc='upper left', fontsize=10)

    output_path = os.path.join(output_dir, '1e')
    plt.savefig(output_path)
    # plt.show()
    plt.close()

#Question 2
print('Question 2')
print('Part (a)')
#import data
sick = pd.read_csv('sick.txt', encoding = 'utf-8-sig') # Read in CSV

#define error vector based on the difference between actual y-values (sick) and estimated RHS
def err_vec_lm(xvals, beta0, beta1, beta2, beta3):
    moms_data = np.array(sick['sick'])
    moms_model = beta0 + beta1*xvals['age'] + beta2*xvals['children'] + beta3*xvals['avgtemp_winter']
    err_vec = moms_model - moms_data

    return err_vec

#sum of squared errors for given parameters and weighting matrix
def criterion_lm(params, *args):
    beta_0, beta_1, beta_2, beta_3 = params
    xvals, W = args
    err = err_vec_lm(xvals, *params)
    crit_val = np.dot(np.dot(err.T,W), err)

    return crit_val

beta_0_init = 0
beta_1_init = 0
beta_2_init = 0
beta_3_init = 0
params_init = np.array([beta_0_init, beta_1_init, beta_2_init, beta_3_init])
W_hat = np.eye(200)

mle_args = (sick, W_hat)

#getting GMM estimated parameters
results = opt.minimize(criterion_lm, params_init, args=mle_args, method = 'L-BFGS-B')
#ML Estimated parameters
b0_GMM, b1_GMM, b2_GMM, b3_GMM = results.x
results.x

criterion_sick = criterion_lm(results.x, *mle_args)

print('2a.')
print('beta_0 = {:.3f}, '.format(b0_GMM), 'beta_1 = {:.4f}, '.format(b1_GMM),
      'beta_2 = {:.3f}, '.format(b2_GMM), 'beta_3 = {:.3f}.'.format(b3_GMM))
print('The value of GMM criterion function at the estimated parameter values is: {:.5f}'.format(criterion_sick))
