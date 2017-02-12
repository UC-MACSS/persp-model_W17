# PS4 - Shen Han

import pandas as pd
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as intgr
import seaborn as sns


plot = True

np.random.seed(seed=1234)
incomes = np.loadtxt('incomes.txt')


# 1a.

if plot:
    plt.subplots()
    plt.hist(incomes, 30, normed = True)

    plt.xlim(0, 150000)
    plt.title("Annual Incomes of MACSS Classes of '18, '19, '20")
    plt.xlabel('Annual income')
    plt.ylabel('Percent of students')
    plt.savefig('fig_1a')
    plt.close()

    
# 1b.

def LN_pdf(xvals, mu, sigma):
    pdf_vals = 1/(xvals * sigma * np.sqrt(2 * np.pi)) * np.exp(-(np.log(xvals)
                - mu) ** 2 / (2 * sigma ** 2))
    
    return pdf_vals


mu_test, sigma_test= 5.0, 1.0
xvals_test = np.array([[200.0, 270.0], [180.0, 195.5]])

print('1(b). The lognormal PDF value is ')
print(LN_pdf(xvals_test, mu_test, sigma_test))
print()


# 1c.

def lognorm_draws(unif_vals, mu, sigma):
    draws = sts.norm.ppf(unif_vals, mu, sigma)
    
    return np.exp(draws)


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
    mmt_data = np.array([[mean_data], [var_data]])

    mean_sim, var_sim  = moments(sim_vals)
    mean_model = mean_sim.mean()
    var_model = var_sim.mean()
    mmt_model = np.array([[mean_model], [var_model]])

    if simple:
        err_vec = mmt_model - mmt_data
    else:
        err_vec = (mmt_model - mmt_data) / mmt_data
        
    return err_vec


def criterion(params, *args):
    mu, sigma = params
    data_vals, unif_vals, W_hat = args
    sim_vals = lognorm_draws(unif_vals, mu, sigma)
    err = err_vec(data_vals, sim_vals, mu, sigma, simple = False)
    crit_val = np.dot(np.dot(err.T, W_hat), err)
    
    return crit_val


unif_vals = sts.uniform.rvs(0, 1, size = (200, 300))

params_init_c = np.array([11.3, 0.2])

W_hat_c = np.eye(2)
smm_args_c = (incomes, unif_vals, W_hat_c)

results_c = opt.minimize(criterion, params_init_c, args=smm_args_c,
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM_c, sigma_SMM_c = results_c.x


print('1(c).')

crit_c = criterion(params_init_c, incomes, unif_vals, W_hat_c)
print('At mu_SMM_c = {} and sigma_SMM_c = {}, criterion value = {}'.format(
    mu_SMM_c, sigma_SMM_c, crit_c[0][0]))

mean_data, var_data = moments(incomes)
print('Data mean is: ', mean_data)
print('Data std is: ', np.sqrt(var_data))

sim_vals_c  = lognorm_draws(unif_vals, mu_SMM_c, sigma_SMM_c)
mean_sim_c, var_sim_c = moments(sim_vals_c)
mean_mod_c = mean_sim_c.mean()
var_mod_c  = var_sim_c.mean()
print('Model mean is: ', mean_mod_c)
print('Model std is: ', np.sqrt(var_mod_c))
print()


if plot:
    plt.subplots()
    plt.hist(incomes, 30, normed = True)
    plt.title("Annual Incomes of MACSS Classes of '18, '19, '20")
    plt.xlabel('Annual income')
    plt.ylabel('Density')
    plt.xlim(0, 150000)

    x_linspace = np.linspace(0, 150000, 500)
    plt.plot(x_linspace, LN_pdf(x_linspace, mu_SMM_c, sigma_SMM_c),
              lw = 2, label='SMM: $\mu = {:.2f}, \sigma = {:.2f}$'.format(
                  mu_SMM_c, sigma_SMM_c))
    
    plt.legend(loc='best')
    plt.savefig('fig_1c')
    plt.close()


# 1(d).

mu_init_d = mu_SMM_c
sigma_init_d = sigma_SMM_c
params_init_d = np.array([mu_init_d, sigma_init_d])


err_c = err_vec(incomes, sim_vals_c, mu_SMM_c, sigma_SMM_c, simple=False)
VCV_d  = np.dot(err_c, err_c.T) / incomes.shape[0]
W_hat_d = lin.pinv(VCV_d)
smm_args_d = (incomes, unif_vals, W_hat_d)

results_d = opt.minimize(criterion, params_init_d, args=smm_args_d,
                         method='L-BFGS-B', bounds=((1e-10, None), (1e-10, None)))
mu_SMM_d, sigma_SMM_d = results_d.x

print('1(d).')
      
crit_d = criterion(params_init_d, incomes, unif_vals, W_hat_d)
print('At mu_SMM_c = {} and sigma_SMM_c = {}, criterion value = {}'.format(
    mu_SMM_d, sigma_SMM_d, crit_d[0][0]))

mean_data, var_data = moments(incomes)
print('Data mean =', mean_data)
print('Data std =', np.sqrt(var_data))

sim_vals_d = lognorm_draws(unif_vals, mu_SMM_d, sigma_SMM_d)
mean_sim_d, var_sim_d = moments(sim_vals_d)
mean_mod_d = mean_sim_d.mean()
var_mod_d = var_sim_d.mean()
print('Model mean is: ', mean_mod_d)
print('Model std is: ', np.sqrt(var_mod_d))


if plot:
    plt.hist(incomes, 30, normed = True)
    plt.title("Annual Incomes of MACSS Classes of '18, '19, '20")
    plt.xlabel('Annual income')
    plt.ylabel('Density')
    plt.xlim(0, 150000)

    x_linspace = np.linspace(0, 150000, 500)
    plt.plot(x_linspace, LN_pdf(x_linspace, mu_SMM_c, sigma_SMM_c),
              lw = 2, label='SMM: $\mu = {:.2f}, \sigma = {:.2f}$'.format(
                  mu_SMM_c, sigma_SMM_c))
    plt.plot(x_linspace, LN_pdf(x_linspace, mu_SMM_d, sigma_SMM_d),
              lw = 2, label='2-step SMM: $\mu  = {:.2f}, \sigma = {:.2f}$'.\
             format(mu_SMM_d, sigma_SMM_d))
      
    plt.legend(loc='best')
    plt.savefig('fig_1d')
    plt.close()

