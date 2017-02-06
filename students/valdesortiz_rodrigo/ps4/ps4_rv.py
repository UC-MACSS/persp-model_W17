'''
----------------------------------------------------------------------
PS4 - Perspectives for Computational Modeling
----------------------------------------------------------------------
Student: Rodrigo Valdes Ortiz
----------------------------------------------------------------------
'''

import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Set random seed
np.random.seed(seed=1234)

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
Ex. 1: Some income data, lognormal distribution, and SMM
------------------------------------------------------------------------
'''
print("**********")
print("Question 1")
print("**********")

# 1.a Histogram
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Annual Income (USD)')
plt.ylabel('Normalized Ratio')
plt.xlim([40000, 150000])  # This gives the xmin and xmax to be plotted"
plt.tight_layout()
# plt.show()
print("1. Part (A)")
output_path = os.path.join(output_dir, 'one_ps4')
plt.savefig(output_path)
plt.close()

def LN_pdf(xvals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Generate pdf values from the lognormal pdf with mean mu and standard
    deviation sigma.
    --------------------------------------------------------------------
    '''

    ln_dist = sts.lognorm(scale = np.exp(mu), s = sigma)
    pdf_vals = ln_dist.pdf(xvals)

    return pdf_vals

print("1. Part (B)")
testing_array = np.array([[200, 270], [180, 1995.5]])
mu_test = 5
sigma_test = 1
print(LN_pdf(testing_array, mu_test, sigma_test))

def log_normal_draws(unif_vals, mu, sigma):
    '''
    --------------------------------------------------------------------
    Defines the drows of the lognormal function,
    given mu and sigma
    --------------------------------------------------------------------
    '''
    normal_draws = sts.norm.ppf(unif_vals, loc = mu, scale = sigma)
    ln_drawns = np.exp(normal_draws)
    return ln_drawns

def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        std_data = xvals.std()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        std_data = xvals.std(axis=0)

    return mean_data, std_data

def err_vec(data_vals, sim_vals, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    '''
    mean_data, std_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [std_data]])
    mean_sim, std_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    std_model = std_sim.mean()
    moms_model = np.array([[mean_model], [std_model]])
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
    '''
    mu, sigma = params
    xvals, unif_vals, W_hat = args
    sim_vals = log_normal_draws(unif_vals, mu, sigma)
    err = err_vec(xvals, sim_vals, simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err)
    return crit_val

print("1. Part (C)")
# Initial parameter for estimation
N = 200
S = 300
unif_vals_2 = sts.uniform.rvs(0, 1, size=(N, S))
mu_init_1 = 11
sig_init_1 = 0.4
params_init_1 = np.array([mu_init_1, sig_init_1])
W_hat1_1 = np.eye(2)
smm_args1_1 = (pts, unif_vals_2, W_hat1_1)
results1_1 = opt.minimize(criterion, params_init_1, args=(smm_args1_1),
                          method='L-BFGS-B',
                          bounds=((1e-10, None), (1e-10, None)))
mu_SMM1_1, sig_SMM1_1 = results1_1.x
print('mu_SMM1_1=', mu_SMM1_1, ' sig_SMM1_1=', sig_SMM1_1)

# Then, tranform this to draws
mean_data, std_data = data_moments(pts)
print('Data mean of incomes =', mean_data, ', Data sigma of incomes =', std_data)
sim_vals_1 = log_normal_draws(unif_vals_2, mu_SMM1_1, sig_SMM1_1)
mean_sim_1, std_sim_1 = data_moments(sim_vals_1)
mean_model_1 = mean_sim_1.mean()
std_model_1 = std_sim_1.mean()
err_1 = err_vec(pts, sim_vals_1, False).reshape(2,)

print('Model mean 1 =', mean_model_1, ', Model sigma 1 =', std_model_1)
# print('Error vector 1 =', err_1)
# print(results1_1)
print("The criterion value is:", results1_1["fun"][0][0])

plt.close()
# Plot data histogram vs. simulated data histogram
count_d, bins_d, ignored_d = plt.hist(pts, 30, normed=True, color='g',label='Data')
xvals = np.linspace(0, 150000, 5000)

plt.plot(xvals, LN_pdf(xvals, mu_SMM1_1, sig_SMM1_1),
         linewidth=2, color='k', label='PDF SMM')
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Annual Income (USD)', fontsize=13)
plt.ylabel('Normalized Ratio', fontsize=13)
plt.xlim([20000, 150000])  # This gives the xmin and xmax to be plotted"
plt.legend(loc='upper right')
plt.tight_layout()
output_path = os.path.join(output_dir, 'two_ps4')
plt.savefig(output_path)
plt.close()
# plt.show()

'''
----------------------------------------------------------------------
Part D
----------------------------------------------------------------------
'''
print("1. Part (D)")

err2_1 = err_vec(pts, sim_vals_1, False)
VCV2_1 = np.dot(err2_1, err2_1.T) / pts.shape[0]
W_hat2_1 = lin.pinv(VCV2_1)  # Use the pseudo-inverse calculated

params_init2_1 = np.array([mu_SMM1_1, sig_SMM1_1])
smm_args2_1 = (pts, unif_vals_2, W_hat2_1)
results2_1 = opt.minimize(criterion, params_init2_1, args=(smm_args2_1),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM2_1, sig_SMM2_1 = results2_1.x
print('mu_SMM2_1=', mu_SMM2_1, ' sig_SMM2_1=', sig_SMM2_1)

print('Data mean of incomes =', mean_data, ', Data sigma of incomes =', std_data)
sim_vals_2 = log_normal_draws(unif_vals_2, mu_SMM2_1, sig_SMM2_1)
mean_sim_2, std_sim_2 = data_moments(sim_vals_2)
mean_model_2 = mean_sim_2.mean()
std_model_2 = std_sim_2.mean()

print('Model mean 2 =', mean_model_2, ', Model sigma 2 =', std_model_2)

print("The criterion value is:", results2_1["fun"][0][0])

# Plot data histogram vs. simulated data histogram
count_d, bins_d, ignored_d = plt.hist(pts, 30, normed=True, color='brown', label='Data')
xvals = np.linspace(0, 150000, 5000)
plt.plot(xvals, LN_pdf(xvals, mu_SMM1_1, sig_SMM1_1),
         linewidth=0.5, color='g', label='PDF SMM 1step')
plt.plot(xvals, LN_pdf(xvals, mu_SMM2_1, sig_SMM2_1),
         linewidth=0.5, color='b', label='PDF SMM 2step')
plt.title('MACSS\'s Alumni Incomes. Class 2018, 2019, and 2020', fontsize=15)
plt.xlabel('Annual Income', fontsize=13)
plt.ylabel('Normalized Ratio', fontsize=13)
plt.xlim([20000, 150000])  # This gives the xmin and xmax to be plotted"
plt.legend(loc='upper right')
plt.tight_layout()
output_path = os.path.join(output_dir, 'three_ps4')
plt.savefig(output_path)
# plt.show()
plt.close()
