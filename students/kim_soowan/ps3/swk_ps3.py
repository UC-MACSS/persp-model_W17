#Soo Wan Kim
#MACS30100
#Problem Set 3

#Used Python version 3.5.2

#import packages
import os
import seaborn
import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy.stats import norm
import scipy.optimize as opt
import scipy.integrate as intgr
from numpy import sqrt
import numpy.linalg as lin

#create images folder in current directory
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

##### Problem 1 #####

#load data
inc = np.loadtxt('incomes.txt') #read in the data

#define functions

#This function generates values of lognormal pdf
def lognorm_pdf(xvals, mu, sigma):
    pdf_vals = lognorm.pdf(xvals, s = sigma, scale = np.exp(mu))
    return pdf_vals

#This function generates values of lognormal cdf
def lognorm_cdf(xvals, mu, sigma):
    cdf_vals = lognorm.cdf(xvals, s = sigma, scale = np.exp(mu))
    return cdf_vals

#This function computes the two data moments for GMM (mean(data), std(data)) for 1(b)-1(c) 
def data_moments(xvals):
    mean_data = xvals.mean()
    std_data = xvals.std()
    return mean_data, std_data

#This function computes the two model moments for GMM (mean(model data), variance(model data)) for 1(b)-1(c)
def model_moments(mu, sigma, cutoff):
    xfx = lambda x: x * lognorm_pdf(x, mu, sigma)
    (mean_model, m_m_err) = intgr.quad(xfx, 0, cutoff)
    x2fx = lambda x: ((x - mean_model) ** 2) * lognorm_pdf(x, mu, sigma)
    (std_model, s_m_err) = sqrt(intgr.quad(x2fx, 0, cutoff))
    return mean_model, std_model

#This function computes the vector of moment errors (in percent deviation from the data moment vector) for GMM for 1(b)-1(c)
def err_vec(xvals, mu, sigma, cutoff):
    mean_data, std_data = data_moments(xvals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_model, std_model = model_moments(mu, sigma, cutoff)
    moms_model = np.array([[mean_model], [std_model]])
    err_vec = (moms_model - moms_data) / moms_data
    return err_vec

#This function computes the GMM weighted sum of squared moment errors 
#criterion function value given parameter values and an estimate of the weighting matrix.
def criterion(params, *args):
    mu, sigma = params
    xvals, cutoff, W = args
    err = err_vec(xvals, mu, sigma, cutoff)
    crit_val = np.dot(np.dot(err.T, W), err)
    return crit_val

#This function computes the three data moments for GMM for 1(d)-(e)
def data_moments3(xvals):
    bpct_1_dat = xvals[xvals < 75000].shape[0] / xvals.shape[0]
    bpct_2_dat = (xvals[(xvals >= 75000) & (xvals <= 100000)].shape[0] / xvals.shape[0])
    bpct_3_dat = xvals[xvals > 100000].shape[0] / xvals.shape[0]
    return bpct_1_dat, bpct_2_dat, bpct_3_dat

#This function computes the three model moments for GMM for 1(d)-(e)
def model_moments3(mu, sigma):
    bpct_1_mod = lognorm_cdf(75000, mu, sigma)
    bpct_3_mod = 1 - lognorm_cdf(100000, mu, sigma)
    bpct_2_mod = 1 - bpct_3_mod - bpct_1_mod
    return bpct_1_mod, bpct_2_mod, bpct_3_mod

#This function computes the vector of moment errors (in percent deviation from the data moment vector) for GMM
#for 1(d)-1(e)
def err_vec3(xvals, mu, sigma):
    bpct_1_dat, bpct_2_dat, bpct_3_dat = data_moments3(xvals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat]])
    bpct_1_mod, bpct_2_mod, bpct_3_mod = model_moments3(mu, sigma)
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod]])
    err_vec = (moms_model - moms_data) / moms_data
    return err_vec

#This function computes the GMM weighted sum of squared moment errors criterion function value for 1(d)-1(e)
def criterion3(params, *args):
    mu, sigma = params
    xvals, cutoff, W = args
    err = err_vec3(xvals, mu, sigma)
    crit_val = np.dot(np.dot(err.T, W), err) 
    return crit_val

### (a)

#plot histogram and save the image in images folder
plot = True

if plot:  
    fig1 = plt.figure()
    count, bins, ignored = plt.hist(inc, 30, normed=True)
    plt.title('Annual incomes of 2018-2020 MACSS graduates')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Ratio')
    plt.xlim([0, 150000])

    output_path = os.path.join(output_dir, '1a') #save image
    plt.savefig(output_path)
    plt.close()

### (b)

#set initial parameter guesses and arguments
mu_init = 11.3
sig_init = 0.212
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(2)
gmm_args = (inc, 150000, W_hat)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM1, sig_GMM1 = results.x
params_GMM1 = np.array([mu_GMM1, sig_GMM1])

#get criterion function value
GMMcrit1 = criterion(params_GMM1, *gmm_args)[0][0]

print('1b')
print('mu_GMM =', mu_GMM1, ' sig_GMM =', sig_GMM1)
print('Value of GMM criterion:', GMMcrit1)
mean_data, std_data = data_moments(inc)
mean_model, std_model = model_moments(mu_GMM1, sig_GMM1, 150000)
print('average income =', mean_data, 'standard deviation of income =', std_data)
print('mean of model =', mean_model, ', standard deviation of model =', std_model)

# Plot the estimated GMM PDF
plot = True

if plot:  
    fig2 = plt.figure()
    count, bins, ignored = plt.hist(inc, 30, normed=True)

    plt.suptitle('Lognormal distribution model of annual incomes of 2018-2020 MACSS graduates')
    plt.title('GMM with 2 moments', style = 'italic')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Ratio')
    plt.xlim([0, 150000])

    inc_pts = np.linspace(0, 150000, 150000)
    plt.plot(inc_pts, lognorm_pdf(inc_pts, mu_GMM1, sig_GMM1),
         linewidth=2, color='r', label='GMM parameters: $\mu$ = 11.3,$\sigma$ = 0.213')
    plt.legend(loc='upper left')

    output_path = os.path.join(output_dir, '1b') #save image
    plt.savefig(output_path)
    plt.close()

### (c)

#set parameters and arguments
err1 = err_vec(inc, mu_GMM1, sig_GMM1, 150000)
VCV2 = np.dot(err1, err1.T) / inc.shape[0]
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

params_init = np.array([mu_GMM1, sig_GMM1])
gmm_args = (inc, 150000, W_hat2)
results = opt.minimize(criterion, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM2, sig_GMM2 = results.x
params_GMM2= np.array([mu_GMM2, sig_GMM2])

#get criterion function value
GMMcrit2 = criterion(params_GMM2, *gmm_args)[0][0]

print('')
print('1c')
print('mu_GMM =', mu_GMM2, ' sig_GMM =', sig_GMM2)
print('Value of GMM criterion:', GMMcrit2)

mean_model, std_model = model_moments(mu_GMM1, sig_GMM1, 150000)
mean_model2, std_model2 = model_moments(mu_GMM2, sig_GMM2, 150000)

print('average income =', mean_data, 'standard deviation of income =', std_data)
print('mean of model (one step) =', mean_model, ', standard deviation of model (one step) =', std_model)
print('mean of model (two steps) =', mean_model2, ', standard deviation of model (two steps) =', std_model2)

plot = True

if plot:  
    fig3 = plt.figure()
    count, bins, ignored = plt.hist(inc, 30, normed=True)

    plt.suptitle('Lognormal distribution model of annual incomes of 2018-2020 MACSS graduates')
    plt.title('GMM with 2 moments', style = 'italic')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Ratio')
    plt.xlim([0, 150000])

    inc_pts = np.linspace(0, 150000, 150000)
    plt.plot(inc_pts, lognorm_pdf(inc_pts, mu_GMM1, sig_GMM1),
         linewidth=2, color='r', label='GMM parameters (one step): $\mu$ = 11.3,$\sigma$ = 0.213')
    plt.plot(inc_pts, lognorm_pdf(inc_pts, mu_GMM2, sig_GMM2),
         linewidth=2, color='g', label='GMM parameters (two steps): $\mu$ = 11.3,$\sigma$ = 0.213')
    plt.legend(loc='upper left')

    output_path = os.path.join(output_dir, '1c') #save image
    plt.savefig(output_path)
    plt.close()

### (d)

#set parameters and arguments
mu_init = 11.3
sig_init = 0.212
params_init = np.array([mu_init, sig_init])
W_hat = np.eye(3)
gmm_args = (inc, 150000, W_hat)
results3 = opt.minimize(criterion3, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM3, sig_GMM3 = results3.x
params_GMM3 = np.array([mu_GMM3, sig_GMM3])

#get criterion function value
GMMcrit3 = criterion3(params_GMM3, *gmm_args)[0][0]

print('')
print('1d')
print('mu_GMM =', mu_GMM3, ' sig_GMM =', sig_GMM3)
print('Value of GMM criterion 3:', GMMcrit3)
print('The three data moments:', data_moments3(inc))
print('The three model moments:', model_moments3(mu_GMM3, sig_GMM3))

# Plot the estimated GMM PDF
plot = True

if plot:  
    fig4 = plt.figure()
    count, bins, ignored = plt.hist(inc, 30, normed=True)

    plt.suptitle('Lognormal distribution model of annual incomes of 2018-2020 MACSS graduates')
    plt.title('GMM with 3 moments', style = 'italic')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Ratio')
    plt.xlim([0, 150000])

    inc_pts = np.linspace(0, 150000, 150000)
    plt.plot(inc_pts, lognorm_pdf(inc_pts, mu_GMM3, sig_GMM3),
         linewidth=2, color='k', label='GMM parameters (3 moments): $\mu$ = 11.3,$\sigma$ = 0.211')
    plt.legend(loc='upper left')

    output_path = os.path.join(output_dir, '1d') #save image
    plt.savefig(output_path)
    plt.close()

### (e)

#set parameters and arguments
err3 = err_vec3(inc, mu_GMM3, sig_GMM3)
VCV2_3 = np.dot(err3, err3.T) / inc.shape[0]
W_hat2 = lin.pinv(VCV2_3)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

params_init = np.array([mu_GMM3, sig_GMM3])
gmm_args = (inc, 150000, W_hat2)

#estimate parameters
results4 = opt.minimize(criterion3, params_init, args=(gmm_args),
                       method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_GMM4, sig_GMM4 = results4.x
params_GMM4 = np.array([mu_GMM4, sig_GMM4])

#get criterion function value
GMMcrit4 = criterion3(params_GMM4, *gmm_args)[0][0]

print('')
print('1e')
print('mu_GMM =', mu_GMM4, ' sig_GMM =', sig_GMM4)
print('Value of GMM criterion 4:', GMMcrit4)

print('The three data moments:', data_moments3(inc))
print('The three model moments (one step):', model_moments3(mu_GMM3, sig_GMM3))
print('The three model moments (two step):', model_moments3(mu_GMM4, sig_GMM4))

# Plot the estimated GMM PDF
plot = True

if plot:  
    fig5 = plt.figure()
    count, bins, ignored = plt.hist(inc, 30, normed=True)

    plt.suptitle('Lognormal distribution model of annual incomes of 2018-2020 MACSS graduates')
    plt.title('GMM with 3 moments', style = 'italic')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Ratio')
    plt.xlim([0, 150000])

    inc_pts = np.linspace(0, 150000, 150000)
    plt.plot(inc_pts, lognorm_pdf(inc_pts, mu_GMM3, sig_GMM3),
         linewidth=2, color='k', label='parameters (one step): $\mu$ = 11.3,$\sigma$ = 0.211')
    plt.plot(inc_pts, lognorm_pdf(inc_pts, mu_GMM4, sig_GMM4),
         linewidth=2, color='g', label='parameters (two steps): $\mu$ = 11.3,$\sigma$ = 0.211')
    plt.legend(loc='upper left')

    output_path = os.path.join(output_dir, '1e') #save image
    plt.savefig(output_path)
    plt.close()

##### Problem 2 #####
### (a)

#read in the data
sick = pd.read_csv('sick.txt', skiprows = 1, names = ['sick', 'age', 'children', 'temp_winter'])

#define functions

#This function generates the error vector for the data and the linear model
def err_vec_lin(xvals, beta0, beta1, beta2, beta3):
    moms_data = np.array(sick['sick'])
    moms_model = beta0 + beta1*xvals['age'] + beta2*xvals['children'] + beta3*xvals['temp_winter']
    err_vec = moms_data - moms_model
    return err_vec

#This function computes the GMM weighted sum of squared moment errors criterion function value
def criterion_lin(params, *args):
    beta_0, beta_1, beta_2, beta_3 = params
    xvals, W = args
    err = err_vec_lin(xvals, *params)
    crit_val = np.dot(np.dot(err.T,W), err)
    return crit_val

#set initial parameter guesses
beta_0_init = 0
beta_1_init = 0
beta_2_init = 0
beta_3_init = 0
params_init = np.array([beta_0_init, beta_1_init, beta_2_init, beta_3_init])
W_hat = np.eye(200)
mle_args = (sick, W_hat)

#estimate parameters
results = opt.minimize(criterion_lin, params_init, args=mle_args, method = 'L-BFGS-B')
b0_GMM, b1_GMM, b2_GMM, b3_GMM = results.x
results.x

#get criterion function value
criterion_sick = criterion_lin(results.x, *mle_args)

print('')
print('2a')
print('b0_GMM=', b0_GMM, ' b1_GMM=', b1_GMM, 'b2_GMM=', b2_GMM, ' b3_GMM=', b3_GMM)
print('Value of GMM criterion:', criterion_sick)