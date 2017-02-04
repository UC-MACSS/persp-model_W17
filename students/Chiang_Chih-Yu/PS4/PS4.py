#%%
import pandas as pd
import numpy as np
from numpy import log as ln
from numpy import exp as exp
import numpy.linalg as lin
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as intgr
from scipy.stats import lognorm

np.random.seed(seed=1234)
incomes = np.loadtxt('incomes.txt')
df = {'incomes': incomes}
# print(df)


'''
----------------------------------------------------------------------
Exercise 1a: histogram
----------------------------------------------------------------------
'''
num_obs   = len(df['incomes'])
fig, ax   = plt.subplots()
hist_wgts = (1 / num_obs) * np.ones(num_obs)
num_bins  = 30
plt.hist(df['incomes'], num_bins, weights=hist_wgts, normed=True)
plt.xlim(0, 160000)
plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=15)
plt.xlabel(r'Annual income')
plt.ylabel(r'Percent of students')
plt.savefig('1a',  bbox_inches='tight')
# plt.show()
plt.close()


'''
----------------------------------------------------------------------
Exercise 1b: lognormal pdf function
----------------------------------------------------------------------
'''
def LN_pdf(xvals, mu, sigma):
    pdf_vals = 1/(xvals * sigma * np.sqrt(2 * np.pi)) * exp( - (ln(xvals) - mu) ** 2 / (2 * sigma ** 2))
    return pdf_vals

# Test input
mu    = 5.0
sigma = 1.0
xvals = np.array([[200.0, 270.0], [180.0, 195.5]])

print('1(b). The lognormal PDF value is', LN_pdf(xvals, mu, sigma))


#%%
'''
----------------------------------------------------------------------
Exercise 1c: SMM
----------------------------------------------------------------------
'''
def lognorm_draws(unif_vals, mu, sigma):
    draws    = sts.norm.ppf(unif_vals, loc = mu, scale = sigma)
    draws_ln = np.exp(draws)
    return draws_ln

def moments(xvals):
    if xvals.ndim == 1:
        mean = xvals.mean()
        var  = xvals.var()
    elif xvals.ndim == 2:
        mean = xvals.mean(axis = 0)
        var  = xvals.var(axis  = 0)
    return mean, var

def err_vec(data_vals, sim_vals, mu, sigma, simple):
    mean_data, var_data = moments(data_vals)
    mmt_data            = np.array([[mean_data], [var_data]])

    mean_sim, var_sim   = moments(sim_vals)
    mean_model          = mean_sim.mean()
    var_model           = var_sim.mean()
    mmt_model           = np.array([[mean_model], [var_model]])

    if simple:
        err_vec         = mmt_model - mmt_data
    else:
        err_vec         = (mmt_model - mmt_data) / mmt_data
    return err_vec

def criterion(params, *args):
    mu, sigma = params
    data_vals, unif_vals, W_hat = args
    sim_vals  = lognorm_draws(unif_vals, mu, sigma)
    err       = err_vec(data_vals, sim_vals, mu, sigma, simple = False)
    crit_val  = np.dot(np.dot(err.T, W_hat), err)
    return crit_val

S             = 300
N             = 200
unif_vals     = sts.uniform.rvs(0, 1, size = (N, S))

mu_init_1     = 11.0
sigma_init_1  = 0.2
params_init_1 = np.array([mu_init_1, sigma_init_1])

W_hat_1       = np.eye(2)

smm_args_1    = (df['incomes'], unif_vals, W_hat_1)

# Perform minimization
results_1 = opt.minimize(criterion, params_init_1, args=smm_args_1, method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_SMM_1, sigma_SMM_1 = results_1.x
print('1(c).')
print('mu_SMM_1 =', mu_SMM_1, ' sigma_SMM_1 =', sigma_SMM_1)

# Print data moments
mean_data, var_data = moments(df['incomes'])
print('Data mean_1 =', mean_data)
print('Data variance_1 =', var_data)

# Print model moments
sim_vals_1            = lognorm_draws(unif_vals, mu_SMM_1, sigma_SMM_1)
mean_sim_1, var_sim_1 = moments(sim_vals_1)
mean_mod_1            = mean_sim_1.mean()
var_mod_1             = var_sim_1.mean()
print('Model mean_1 =', mean_mod_1)
print('Model variance_1 =', var_mod_1)

# Compute difference between data and model moments
print('Percentage difference in mean_1 =', (mean_data - mean_mod_1) / mean_data, 'Percentage difference in variance_1 =', (var_data - var_mod_1) / var_data)

# Compute err matrix
err_1 = err_vec(df['incomes'], sim_vals_1, mu_SMM_1, sigma_SMM_1, simple=False).reshape(2, )

# Compute criterion value
crit_1 = criterion(np.array([mu_SMM_1, sigma_SMM_1]), df['incomes'], unif_vals, W_hat_1)
print('Criterion value_1 =', crit_1[0][0])
# print(results_1)

# Plot the histogram of the data
count, bins, ignored = plt.hist(df['incomes'], num_bins, normed=True)
plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=15)
plt.xlabel(r'Annual income')
plt.ylabel(r'Percent of students')

# Plot the estimated SMM PDF
dist_pts = np.linspace(0, 160000, 500)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM_1, sigma_SMM_1),
            linewidth=2, color='k', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
plt.legend(loc='upper left')
plt.savefig('1c',  bbox_inches='tight')
# plt.show()
plt.close()


#%%
'''
----------------------------------------------------------------------
Exercise 1d: 2-step SMM
----------------------------------------------------------------------
'''
mu_init_2     = mu_SMM_1
sigma_init_2  = sigma_SMM_1
params_init_2 = np.array([mu_init_2, sigma_init_2])

# Compute W_hat from the err matrix of the first step
err_1         = err_vec(df['incomes'], sim_vals_1, mu_SMM_1, sigma_SMM_1, simple=False)
VCV_2         = np.dot(err_1, err_1.T) / df['incomes'].shape[0]
W_hat_2       = lin.pinv(VCV_2)
smm_args_2    = (df['incomes'], unif_vals, W_hat_2)

# Perform minimization
results_2 = opt.minimize(criterion, params_init_2, args=smm_args_2, method='TNC', bounds=((1e-10, None), (1e-10, None)))
mu_SMM_2, sigma_SMM_2 = results_2.x
print('1(d).')
print('mu_SMM_2 =', mu_SMM_2, ' sigma_SMM_2 =', sigma_SMM_2)

# Print data moments
mean_data, var_data = moments(df['incomes'])
print('Data mean_2 =', mean_data)
print('Data variance_2 =', var_data)

# Print model moments
sim_vals_2            = lognorm_draws(unif_vals, mu_SMM_2, sigma_SMM_2)
mean_sim_2, var_sim_2 = moments(sim_vals_2)
mean_mod_2            = mean_sim_2.mean()
var_mod_2             = var_sim_2.mean()
print('Model mean_2 =', mean_mod_2)
print('Model variance_2 =', var_mod_2)

# Compute difference between data and model moments
print('Percentage difference in mean_2 =', (mean_data - mean_mod_2) / mean_data, 'Percentage difference in variance_2 =', (var_data - var_mod_2) / var_data)

# Compute err matrix
err_2 = err_vec(df['incomes'], sim_vals_2, mu_SMM_2, sigma_SMM_2, simple=False).reshape(2, )

# Compute criterion value
crit_2 = criterion(np.array([mu_SMM_2, sigma_SMM_2]), df['incomes'], unif_vals, W_hat_2)
print('Criterion value_2 =', crit_2[0][0])
# print(results_2)

# Plot the histogram of the data
count, bins, ignored = plt.hist(df['incomes'], num_bins, normed=True)
plt.title('Annual Income of MACSS Graduates in 2018-2020', fontsize=15)
plt.xlabel(r'Annual income')
plt.ylabel(r'Percent of students')

# Plot the estimated SMM PDF 1 & 2
dist_pts = np.linspace(0, 160000, 500)
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM_1, sigma_SMM_1),
            linewidth=2, color='k', label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
plt.plot(dist_pts, LN_pdf(dist_pts, mu_SMM_2, sigma_SMM_2),
            linewidth=2, color='r', label='2: $\mu_{SMM2}$,$\sigma_{SMM2}$')
plt.legend(loc='upper left')
plt.savefig('1d',  bbox_inches='tight')
# plt.show()
plt.close()
