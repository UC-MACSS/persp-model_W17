#Soo Wan Kim
#MACS30100
#Problem Set 4

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
inc = np.loadtxt('incomes.txt') 
#set seed
np.random.seed(seed=1234)

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

#define function that returns lognormal PDF values

def LN_pdf(xvals, mu, sigma):
    pdf_vals = 1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp( - (np.log(xvals) - mu)**2 / (2 * sigma**2))
    
    return pdf_vals

#test to see if function works

#test arguments
xvals_test = np.array([[200.0, 270.0], [180.0, 195.5]])
mu_test = 5.0
sigma_test = 1.0

print("1b")
print("Testing lognormal PDF function:")
print(LN_pdf(xvals_test, mu_test, sigma_test))

### (c)

#define functions

#draws N x S values from a lognormal distribution; N = # observations, S = # simulations
def LN_draws(unif_vals, mu, sigma):
    norm_draws = sts.norm.ppf(unif_vals, loc=mu, scale=sigma)
    ln_draws = np.exp(norm_draws)
    
    return ln_draws

#computes the two data moments for SMM
def data_moments(xvals):
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        std_data = xvals.std()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        std_data = xvals.std(axis=0)
    
    return mean_data, std_data

#computes the vector of moment errors (in percent deviation from the data moment vector) for SMM
def err_vec(data_vals, sim_vals, mu, sigma):
    mean_data, std_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_sim, std_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    std_model = std_sim.mean()
    moms_model = np.array([[mean_model], [std_model]])
    
    err_vec = (moms_model - moms_data) / moms_data
    
    return err_vec

#computes the SMM weighted sum of squared moment errors criterion function value
def criterion(params, *args):
    mu, sigma = params
    xvals, unif_vals, W_hat = args
    sim_vals = LN_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, mu, sigma)
    crit_val = np.dot(np.dot(err.T, W_hat), err) 
    
    return crit_val

#set initial parameter guesses
mu_init1 = 11
sig_init1 = 0.2
params_init1 = np.array([mu_init1, sig_init1])

#set parameters and arguments for SMM

S = 300 #Create S = 300 simulations with N = 200 observations each
N = 200
unif_vals = sts.uniform.rvs(0, 1, size=(N, S))
W_hat1 = np.eye(2) #use identity matrix as weighting matrix 
smm_args1 = (inc, unif_vals, W_hat1)

#get SMM estimates and criterion function value
results1 = opt.minimize(criterion, params_init1, args=(smm_args1),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM1, sig_SMM1 = results1.x
params_SMM1 = np.array([mu_SMM1, sig_SMM1])
crit1 = criterion(params_SMM1, *smm_args1)[0][0]

#get data moments
mean_data, std_data = data_moments(inc)

#get model moments at estimated parameter values
sim_vals1 = LN_draws(unif_vals, mu_SMM1, sig_SMM1)
mean_sim1, std_sim1 = data_moments(sim_vals1)
mean_model1 = mean_sim1.mean()
std_model1 = std_sim1.mean()

print('')
print('1c')
print('mu_SMM1=', mu_SMM1, ' sig_SMM1=', sig_SMM1)
print('Criterion value =', crit1)
print('Data mean of income =', mean_data, ', Data standard deviation of income =', std_data)
print('Model mean of income =', mean_model1, ', Model standard deviation of income =', std_model1)

#plot estimated lognormal PDF against histogram from part (a)
plot = True

if plot:  
    fig2 = plt.figure()
    # Plot histogram
    count, bins, ignored = plt.hist(inc, 30, normed=True)
    # Plot estimated SMM PDF
    inc_pts = np.linspace(0, 150000, 150000)
    plt.plot(inc_pts, LN_pdf(inc_pts, mu_SMM1, sig_SMM1),
         linewidth=2, color='k', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
    plt.legend(loc='upper left')

    plt.suptitle('Lognormal distribution model of annual incomes of 2018-2020 MACSS graduates')
    plt.title('SMM with 2 moments', style = 'italic')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Ratio')
    plt.xlim([0, 150000])

    output_path = os.path.join(output_dir, '1c') #save image
    plt.savefig(output_path)
    plt.close()

### (d)

#get new weighting matrix from variance covariance matrix
err2 = err_vec(inc, sim_vals1, mu_SMM1, sig_SMM1)
VCV2 = np.dot(err2, err2.T) / inc.shape[0]
W_hat2 = lin.pinv(VCV2)  # Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned

smm_args2 = (inc, unif_vals, W_hat2)

#get SMM estimates and criterion function value
results2 = opt.minimize(criterion, params_SMM1, args=(smm_args2),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM2, sig_SMM2 = results2.x
params_SMM2 = np.array([mu_SMM2, sig_SMM2])
crit2 = criterion(params_SMM2, *smm_args2)[0][0]

#get model moments at estimated parameter values
sim_vals2 = LN_draws(unif_vals, mu_SMM2, sig_SMM2)
mean_sim2, std_sim2 = data_moments(sim_vals2)
mean_model2 = mean_sim2.mean()
std_model2 = std_sim2.mean()

print('')
print('1d') 
print('mu_SMM2=', mu_SMM2, ' sig_SMM2=', sig_SMM2)
print('Criterion value =', crit2)
print('Model mean of income =', mean_model2, ', Model standard deviation of income =', std_model2)

#plot estimated lognormal PDF against histogram from part (a) and estimated PDF from part (c)
plot = True

if plot:  
    fig2 = plt.figure()
    # Plot histogram
    count, bins, ignored = plt.hist(inc, 30, normed=True)
    # Plot estimated SMM PDF
    inc_pts = np.linspace(0, 150000, 150000)
    plt.plot(inc_pts, LN_pdf(inc_pts, mu_SMM2, sig_SMM2),
         linewidth=2, color='k', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
    plt.plot(inc_pts, LN_pdf(inc_pts, mu_SMM2, sig_SMM2),
         linewidth=2, color='y', label='2: $\mu_{SMM2}$,$\sigma_{SMM2}$')
    plt.legend(loc='upper left')

    plt.suptitle('Lognormal distribution model of annual incomes of 2018-2020 MACSS graduates')
    plt.title('SMM with 2 moments', style = 'italic')
    plt.xlabel('Income (U.S. dollars)')
    plt.ylabel('Ratio')
    plt.xlim([0, 150000])

    output_path = os.path.join(output_dir, '1d') #save image
    plt.savefig(output_path)
    plt.close()